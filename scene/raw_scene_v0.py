#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model_r3dg import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.pose_utils import generate_random_poses_llff_annealing_view, generate_random_poses_dtu_annealing_view
from utils.pose_utils import generate_random_poses_360_annealing_view
from utils.pose_utils import generate_interpolate_poses_360
from scene.cameras import PseudoCamera
import torch
from scene.cameras import Camera
import math
import torch.nn.functional as F
from utils.pseudo_sample_utils import CameraPoseInterpolator

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.interpolator = CameraPoseInterpolator(rotation_weight=1.0, translation_weight=1.0) #伪视角interpolator初始化
        self.pseudo_sampling_num = 40 #针对每个train camera生成的伪视角数量

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}
        self.closest_cameras = {}

        #difix settings
        self.camtoworlds_train=None
        self.camtoworlds_test=None


        print("args.source_path:",args.source_path)
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            if args.source_path.find("vggt")!=-1 or args.source_path.find("interiorverse")!=-1:
                scene_info=sceneLoadTypeCallbacks["Colmap_vggt"](args.source_path, args.images, args.eval, 'vggt',args.n_views)
        else:
            assert False, "Could not recognize scene type!"


        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(self.cameras_extent, 'cameras_extent')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            print("self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)")
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
       
        #pseudo cameras settings,这里要利用interpolator生成两个序列，对每一个train camera,分别生成距离最近邻的train camera以及最近邻
        #的test camera之间的一个插值序列用于后续sam的region alignment
        self.camtoworlds_test= self.get_c2ws_test()
        self.camtoworlds_train = self.get_c2ws_train()
        interpolate_pseudo_test_poses=self.interpolator.interpolate_poses(self.camtoworlds_test,self.camtoworlds_train,self.pseudo_sampling_num) #[N,40,4,4]
        interpolate_pseudo_train_poses,neighbor_indices=self.interpolator.interpolate_training_pose_neighbors(self.camtoworlds_train,self.pseudo_sampling_num) #[N,40,4,4]
        # Convert nested lists to numpy arrays
        interpolate_pseudo_test_poses = np.array(interpolate_pseudo_test_poses)
        interpolate_pseudo_train_poses = np.array(interpolate_pseudo_train_poses)
        # interpolate_pseudo_test_poses: [N_train, 40, 4, 4]
        # interpolate_pseudo_train_poses: [N_train, 40, 4, 4]
        print("interpolate_pseudo_test_poses shape:",interpolate_pseudo_test_poses.shape)
        print("interpolate_pseudo_train_poses shape:",interpolate_pseudo_train_poses.shape)
        self.interpolate_pseudo_test_cams = []
        self.interpolate_pseudo_train_cams = []

        # Use the first train camera as template for intrinsics
        if len(self.train_cameras[1.0]) > 0:
            ref_cam = self.train_cameras[1.0][0]
            FoVx, FoVy = ref_cam.FoVx, ref_cam.FoVy
            width, height = ref_cam.image_width, ref_cam.image_height
        else:
            FoVx = FoVy = width = height = None

        # Convert test pseudo poses to PseudoCamera
        for i in range(interpolate_pseudo_test_poses.shape[0]):
            sequence_cams = []
            for j in range(interpolate_pseudo_test_poses.shape[1]):
                pose = interpolate_pseudo_test_poses[i, j]
                cam = PseudoCamera(
                    R=pose[:3, :3].T,
                    T=pose[:3, 3],
                    FoVx=FoVx,
                    FoVy=FoVy,
                    width=width,
                    height=height,
                    id=(i * interpolate_pseudo_test_poses.shape[1] + j),
                )
                sequence_cams.append(cam)
            self.interpolate_pseudo_test_cams.append(sequence_cams)

        # Convert train pseudo poses to PseudoCamera
        for i in range(interpolate_pseudo_train_poses.shape[0]):
            sequence_cams = []
            for j in range(interpolate_pseudo_train_poses.shape[1]):
                pose = interpolate_pseudo_train_poses[i, j]
                cam = PseudoCamera(
                    R=pose[:3, :3].T,
                    T=pose[:3, 3],
                    FoVx=FoVx,
                    FoVy=FoVy,
                    width=width,
                    height=height,
                    id=(i * interpolate_pseudo_train_poses.shape[1] + j),
                )
                sequence_cams.append(cam)
            self.interpolate_pseudo_train_cams.append(sequence_cams)



        print("self.camtoworlds_test shape:",self.camtoworlds_test.shape)
        print("self.camtoworlds_train shape:",self.camtoworlds_train.shape)

            
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def sample_random_pseudo_cameras(self, id):
        """
        随机从interpolate_pseudo_test_cams[id]和interpolate_pseudo_train_cams[id]中各采样一个PseudoCamera
        :param id: 索引
        :return: (pseudo_test_cam, pseudo_train_cam)
        """
        if id >= len(self.interpolate_pseudo_test_cams) or id >= len(self.interpolate_pseudo_train_cams):
            raise IndexError("id 超出 pseudo camera 序列长度")
        pseudo_test_seq = self.interpolate_pseudo_test_cams[id]
        pseudo_train_seq = self.interpolate_pseudo_train_cams[id]
        pseudo_test_cam = random.choice(pseudo_test_seq)
        pseudo_train_cam = random.choice(pseudo_train_seq)
        return pseudo_test_cam, pseudo_train_cam

    def get_c2ws_train(self, scale=1.0):
        #数组返回形式是一个numpy数组，shape为[N, 4, 4]
        c2ws = []
        for viewpoint_cam in self.train_cameras[scale]:
            c2w = torch.inverse(viewpoint_cam.world_view_transform.T).cpu().numpy()  # [4, 4]
            c2ws.append(c2w)
        return np.stack(c2ws, axis=0)  # [N, 4 ,4]
    
    def get_c2ws_test(self, scale=1.0):
        #数组返回形式是一个numpy数组，shape为[N, 4, 4]
        c2ws = []
        for viewpoint_cam in self.test_cameras[scale]:
            c2w = torch.inverse(viewpoint_cam.world_view_transform.T).cpu().numpy()  # [4, 4]
            c2ws.append(c2w)
        return np.stack(c2ws, axis=0)  # [N, 4, 4]
        
    
    def get_c2ws_pseudo(self, scale=1.0):
        #数组返回形式是一个numpy数组，shape为[N, 4, 4]
        c2ws = []
        for viewpoint_cam in self.pseudo_cameras[scale]:
            c2w = viewpoint_cam.c2w.cpu().numpy()  # [4, 4]
            c2ws.append(c2w)
        return np.stack(c2ws, axis=0)  # [N, 4, 4]

    def get_len_train_cameras(self):
        return len(self.train_cameras[1.0])

    def get_len_pseudo_cameras(self):
        return len(self.pseudo_cameras[1.0])
    
    def get_len_all_cameras(self):
        return len(self.train_cameras[1.0]) + len(self.pseudo_cameras[1.0])

    def getEvalCameras(self, scale=1.0):
        return self.eval_cameras[scale]

    def getPseudoCameras(self, scale=1.0):
        if len(self.pseudo_cameras) == 0:
            return [None]
        else:
            return self.pseudo_cameras[scale]

    def getPseudoCamerasWithClosestViews(self, scale=1.0):
        if len(self.pseudo_cameras) == 0:
            return [None]
        else:
            return self.pseudo_cameras[scale], self.closest_cameras[scale]
    def get_canonical_rays(self, scale: float = 1.0) -> torch.Tensor:
            # NOTE: some datasets do not share the same intrinsic (e.g. DTU)
            # get reference camera
            ref_camera: Camera = self.train_cameras[scale][0]
            # TODO: inject intrinsic
            H, W = ref_camera.image_height, ref_camera.image_width
            cen_x = W / 2
            cen_y = H / 2
            tan_fovx = math.tan(ref_camera.FoVx * 0.5)
            tan_fovy = math.tan(ref_camera.FoVy * 0.5)
            focal_x = W / (2.0 * tan_fovx)
            focal_y = H / (2.0 * tan_fovy)

            x, y = torch.meshgrid(
                torch.arange(W),
                torch.arange(H),
                indexing="xy",
            )
            x = x.flatten()  # [H * W]
            y = y.flatten()  # [H * W]
            camera_dirs = F.pad(
                torch.stack(
                    [
                        (x - cen_x + 0.5) / focal_x,
                        (y - cen_y + 0.5) / focal_y,
                    ],
                    dim=-1,
                ),
                (0, 1),
                value=1.0,
            )  # [H * W, 3]
            # NOTE: it is not normalized
            return camera_dirs.cuda()