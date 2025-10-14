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

from scene.cameras import Camera
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
# from utils.depth_utils import estimate_depth
# from utils.depth_normal_omnidata import estimate_depth_normal_omni,normal_to_geowizard
# from rgbx_guidance import sd_estimate_aovs_normal
from normal_guidance import get_normal_tensor 
# from rgbx_svd_guidance import svd_estimate_aovs
# from geowizard import estimate_normal_depth
import os
import torch

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    gt_semantic_feature = cam_info.semantic_feature
    gt_relevancymap = torch.from_numpy(cam_info.relevancymap).unsqueeze(0).cuda() #[1,H,W] 
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    resized_albedo = PILtoTorch(cam_info.albedo, resolution) if cam_info.albedo is not None else None
    resized_roughness = PILtoTorch(cam_info.roughness, resolution) if cam_info.roughness is not None else None
    resized_metallic = PILtoTorch(cam_info.metallic, resolution) if cam_info.metallic is not None else None
    resized_mask = PILtoTorch(cam_info.mask, resolution) if cam_info.mask is not None else None
    # mask = None if cam_info.mask is None else cv2.resize(cam_info.mask, resolution)
    gt_image = resized_image_rgb[:3, ...]
    # print(f"id: {id}, image shape: {gt_image.shape}")
    gt_albedo = resized_albedo[:3, ...] if resized_albedo is not None else None
    gt_roughness = resized_roughness[:1, ...] if resized_roughness is not None else None
    gt_metallic = resized_metallic[:1, ...] if resized_metallic is not None else None
    gt_mask = resized_mask[:1, ...] if resized_mask is not None else None
    loaded_mask = None

    # depth_filename = os.path.join(args.model_path, f"precompute_depth_normal/depth_{id}.npy")
    normal_filename = os.path.join(args.model_path, f"precompute_depth_normal/normal_{id}.npy")
    precompute_dir = os.path.join(args.model_path, "precompute_depth_normal")
    if not os.path.exists(precompute_dir):
        os.makedirs(precompute_dir)

    #diffuse shading load
    diffuse_filename = os.path.join(args.source_path, "dif_shd",cam_info.image_name + ".npy")
    depth_filename=os.path.join(args.source_path, "depth",cam_info.image_name + ".npy")

    if os.path.exists(diffuse_filename):
        diffuse = np.load(diffuse_filename)
        depth=np.load(depth_filename)
    else:
        diffuse = None
        depth = None

    if diffuse is not None:
        # Convert from [h,w,c] to [c,h,w] and move to cuda
        diffuse = torch.from_numpy(diffuse).permute(2, 0, 1).float().cuda()

    # Depth & Normal
    # if os.path.exists(depth_filename) and os.path.exists(normal_filename):
    if os.path.exists(normal_filename):
        # depth = np.load(depth_filename)
        normal = np.load(normal_filename)
    else:
        # depth, normal = estimate_depth_normal_omni(gt_image.cuda())
        # _, normal = estimate_depth_normal_omni(gt_image.cuda())
        # normal=sd_estimate_aovs_normal(gt_image.cuda(),inference_step=50)
        normal=get_normal_tensor(gt_image.cuda())
        # depth = depth.cpu().numpy()
        normal = normal.cpu().numpy()
        # np.save(depth_filename, depth)
        np.save(normal_filename, normal)
        # depth_visual = (255 * (depth - np.min(depth)) / (np.max(depth) - np.min(depth))).astype(np.uint8)
        normal_visual = ((normal + 1) * 0.5 * 255).astype(np.uint8)
        normal_visual = np.transpose(normal_visual, (1, 2, 0))
        # depth_png_filename = os.path.join(precompute_dir, f"depth_{id}.png")
        normal_png_filename = os.path.join(precompute_dir, f"normal_{id}.png")
        # plt.imsave(depth_png_filename, depth_visual, cmap='viridis')
        plt.imsave(normal_png_filename, normal_visual)

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    
    return Camera(
        colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
        FoVx=cam_info.FovX, FoVy=cam_info.FovY, image=gt_image, albedo=gt_albedo,roughness=gt_roughness,metallic=gt_metallic,
        gt_alpha_mask=loaded_mask,
        uid=id, data_device=args.data_device, image_name=cam_info.image_name,
        depth_image=depth, normal_image=normal,diffuse=diffuse, mask=gt_mask, bounds=cam_info.bounds,
        height=cam_info.height, width=cam_info.width, 
        semantic_feature = gt_semantic_feature,relevancymap=gt_relevancymap
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in tqdm((enumerate(cam_infos))):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
