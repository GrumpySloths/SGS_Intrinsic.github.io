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
import glob
import os
import sys

from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, rotmat2qvec, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import cv2
from tqdm import tqdm
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import re
import torch

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    albedo: np.array
    roughness: np.array
    metallic: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array
    bounds: np.array
    relevancymap: np.array
    semantic_feature: torch.tensor 
    semantic_feature_path: str 
    semantic_feature_name: str 

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    eval_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, semantic_feature_folder,
                      relevancymap_folder,path, rgb_mapping,use_pbr=True):
    cam_infos = []
    print("start reading colmap cameras...")
    for idx, key in enumerate(sorted(cam_extrinsics.keys())):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))  # This R is the transposed camera extrinsic rotation, i.e. the c2w rotation.
        T = np.array(extr.tvec)
        # bounds = np.load(os.path.join(path, 'poses_bounds.npy'))[idx, -2:]

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        
        # Check if image_path matches the pattern "{id}_im_denoised.png"
        albedo = None
        roughness = None
        metallic=None
        match = re.match(r"(\d+)_im_denoised\.png", os.path.basename(image_path))
        if match and use_pbr==True:
            img_id = match.group(1)
            albedo_path = os.path.join(images_folder, f"{img_id}_albedo.png")
            rough_path = os.path.join(images_folder, f"{img_id}_material_rough.png")
            metallic_path = os.path.join(images_folder, f"{img_id}_material_metal.png")
            mask_path = os.path.join(images_folder, f"{img_id}_mask.png")
            if os.path.exists(albedo_path):
                albedo = Image.open(albedo_path)
                # print("Read albedo image:", albedo_path)  # Debug only.
            if os.path.exists(rough_path):
                roughness = Image.open(rough_path)
                metallic = Image.open(metallic_path) if os.path.exists(metallic_path) else None
                mask = Image.open(mask_path) if os.path.exists(mask_path) else None
        elif os.path.exists(os.path.join(path,"images_albedo")) and use_pbr==True:
            albedo_basename = os.path.basename(image_path)
            roughness_basename = os.path.basename(image_path)
            
            # Replace .jpg or .JPG extension with .png
            if albedo_basename.lower().endswith(('.jpg', '.jpeg')):
                albedo_basename = os.path.splitext(albedo_basename)[0] + '.png'
            if roughness_basename.lower().endswith(('.jpg', '.jpeg')):
                roughness_basename = os.path.splitext(roughness_basename)[0] + '.png'
                
            albedo = Image.open(os.path.join(path,"images_albedo", albedo_basename))
            roughness = Image.open(os.path.join(path,"images_roughness", roughness_basename))
            mask=None
            metallic=None
        else:
            albedo = None
            roughness = None
            mask=None
        image_name = os.path.basename(image_path).split(".")[0]
        rgb_path = rgb_mapping[idx]   # os.path.join(images_folder, rgb_mapping[idx])
        rgb_name = os.path.basename(rgb_path).split(".")[0]
        # image = Image.open(rgb_path)
        image = Image.open(image_path)

        if use_pbr:
            print("use_pbr is True,data loading with pbr attributes.")
            semantic_feature = None
            relevancymap = None
            semantic_feature_path = ""
            semantic_feature_name = ""
        else:
            print("use_pbr is False,data loading with semantic features.")
            semantic_feature_path = os.path.join(semantic_feature_folder, image_name) + '_fmap_CxHxW.pt' 
            semantic_feature_name = os.path.basename(semantic_feature_path).split(".")[0]
            semantic_feature = torch.load(semantic_feature_path) 
            relevancymap_path=os.path.join(relevancymap_folder, image_name) + '_relevancy.npy'
            relevancymap = np.load(relevancymap_path) if os.path.exists(relevancymap_path) else None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                image_name=image_name, width=width, height=height, mask=mask, bounds=None,
                albedo=albedo, roughness=roughness, metallic=metallic,
                semantic_feature=semantic_feature,
                relevancymap=relevancymap,
                semantic_feature_path=semantic_feature_path,
                semantic_feature_name=semantic_feature_name)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos


def farthest_point_sampling(points, k):
    """
    Sample k points from input pointcloud data points using Farthest Point Sampling.

    Parameters:
    points: numpy.ndarray
        The input pointcloud data, a numpy array of shape (N, D) where N is the
        number of points and D is the dimensionality of each point.
    k: int
        The number of points to sample.

    Returns:
    sampled_points: numpy.ndarray
        The sampled pointcloud data, a numpy array of shape (k, D).
    """
    N, D = points.shape
    farthest_pts = np.zeros((k, D))
    distances = np.full(N, np.inf)
    farthest = np.random.randint(0, N)
    for i in range(k):
        farthest_pts[i] = points[farthest]
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
    return farthest_pts


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort

def readColmapSceneInfo(path, images, eval, dataset, n_views=0, llffhold=8, rand_ply=False):
    ply_path = os.path.join(path, str(n_views) + "_views/dense/fused.ply")

    try:
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    if (not os.path.exists(ply_path)) or rand_ply: # random init pcd
        print('Init random point cloud.')
        ply_path = os.path.join(path, "sparse/0/points3D_random.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")

        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        print(xyz.max(0), xyz.min(0))

        if dataset == "LLFF":
            pcd_shape = (topk_(xyz, 1, 0)[-1] + topk_(-xyz, 1, 0)[-1])
            num_pts = int(pcd_shape.max() * 50)
            xyz = np.random.random((num_pts, 3)) * pcd_shape * 1.3 - topk_(-xyz, 20, 0)[-1]
        elif dataset=="mipnerf":
            pcd_shape = (topk_(xyz, 1, 0)[-1] + topk_(-xyz, 1, 0)[-1])
            num_pts = int(pcd_shape.max() * 50)
            xyz = np.random.random((num_pts, 3)) * pcd_shape * 1.3 - topk_(-xyz, 20, 0)[-1]
        elif dataset == "dtu":
            pcd_shape = (topk_(xyz, 100, 0)[-1] + topk_(-xyz, 100, 0)[-1])
            num_pts = 10_00
            xyz = np.random.random((num_pts, 3)) * pcd_shape * 1.3 - topk_(-xyz, 100, 0)[-1]
        print(pcd_shape)
        print(f"Generating random point cloud ({num_pts})...")

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else: # dense pcd from n_views images
        pcd = fetchPly(ply_path)

    reading_dir = "images" if images == None else images
    rgb_mapping = [f for f in sorted(glob.glob(os.path.join(path, reading_dir, '*')))
                   if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    cam_extrinsics = {cam_extrinsics[k].name: cam_extrinsics[k] for k in cam_extrinsics}
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                             images_folder=os.path.join(path, reading_dir),  path=path, rgb_mapping=rgb_mapping)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    print("len of cam_infos: ", len(cam_infos))

    eval_cam_infos = []
    
    if eval:
        if dataset == 'LLFF':
            eval_cam_infos = [c for idx, c in enumerate(cam_infos)]
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
            if n_views > 0:
                idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views)
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
                assert len(train_cam_infos) == n_views
        elif dataset == 'dtu':
            train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
            exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
            test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
            if n_views > 0:
                train_idx = train_idx[:n_views]
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_idx]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_idx]
            assert len(train_cam_infos) == n_views
        elif dataset=='mipnerf':
            eval_cam_infos = [c for idx, c in enumerate(cam_infos)]
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
            if n_views > 0:
                idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views)
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
                assert len(train_cam_infos) == n_views
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    
    print("len of train_cam_infos: ", len(train_cam_infos))

    nerf_normalization = getNerfppNorm(train_cam_infos)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           eval_cameras=eval_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapSceneInfo_dust3r(path, images, eval, dataset, n_views=0, llffhold=8):

    cameras_extrinsic_file_train = os.path.join(path, f"sparse_{n_views}/0", "images.txt")
    cameras_intrinsic_file_train = os.path.join(path, f"sparse_{n_views}/0", "cameras.txt")
    cam_extrinsics_train = read_extrinsics_text(cameras_extrinsic_file_train)
    cam_intrinsics_train = read_intrinsics_text(cameras_intrinsic_file_train)
    reading_dir_train = "images"
    rgb_mapping_train = [f for f in sorted(glob.glob(os.path.join(path, reading_dir_train, '*')))
                         if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    cam_extrinsics_train = {cam_extrinsics_train[k].name: cam_extrinsics_train[k] for k in cam_extrinsics_train}
    cam_infos_unsorted_train = readColmapCameras(cam_extrinsics=cam_extrinsics_train, cam_intrinsics=cam_intrinsics_train,
                                 images_folder=os.path.join(path, reading_dir_train), path=path, rgb_mapping=rgb_mapping_train)
    train_cam_infos = sorted(cam_infos_unsorted_train.copy(), key=lambda x: x.image_name)

    cameras_extrinsic_file_test = os.path.join(path, f"sparse_{n_views}/1", "images.txt")
    cameras_intrinsic_file_test = os.path.join(path, f"sparse_{n_views}/1", "cameras.txt")
    cam_extrinsics_test = read_extrinsics_text(cameras_extrinsic_file_test)
    cam_intrinsics_test = read_intrinsics_text(cameras_intrinsic_file_test)
    reading_dir_test = "images"
    rgb_mapping_test = [f for f in sorted(glob.glob(os.path.join(path, reading_dir_test, '*')))
                        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    cam_extrinsics_test = {cam_extrinsics_test[k].name: cam_extrinsics_test[k] for k in cam_extrinsics_test}
    cam_infos_unsorted_test = readColmapCameras(cam_extrinsics=cam_extrinsics_test, cam_intrinsics=cam_intrinsics_test,
                                images_folder=os.path.join(path, reading_dir_test), path=path, rgb_mapping=rgb_mapping_test)
    test_cam_infos = sorted(cam_infos_unsorted_test.copy(), key=lambda x: x.image_name)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, f"sparse_{n_views}/0/points3D.ply")
    bin_path = os.path.join(path, f"sparse_{n_views}/0/points3D.bin")
    txt_path = os.path.join(path, f"sparse_{n_views}/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
        
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           eval_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# 该函数版本的修改在于点云依然使用 sparse_{n_views}/0 下产生的点云，
# 但所有相机参数（外参和内参）统一从 sparse_24/0 文件夹下读取，
# 训练集通过随机采样 n_views 个 camera，其余为 test_camera
def readColmapSceneInfo_dust3r_modify(path, images, eval, dataset, n_views=0, llffhold=8):
    # 固定从 sparse_24/0 读取相机参数
    cameras_extrinsic_file = os.path.join(path, "sparse_24/0", "images.txt")
    cameras_intrinsic_file = os.path.join(path, "sparse_24/0", "cameras.txt")
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = "images"
    rgb_mapping = [f for f in sorted(glob.glob(os.path.join(path, reading_dir, '*')))
                   if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    cam_extrinsics = {cam_extrinsics[k].name: cam_extrinsics[k] for k in cam_extrinsics}
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        path=path,
        rgb_mapping=rgb_mapping
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    # 随机采样 n_views 个训练集，其余为测试集
    total_views = len(cam_infos)
    if n_views > 0 and n_views < total_views:
        idx_all = np.arange(total_views)
        idx_train = np.sort(np.random.choice(idx_all, n_views, replace=False))
        idx_test = np.setdiff1d(idx_all, idx_train)
        train_cam_infos = [cam_infos[i] for i in idx_train]
        # test_cam_infos = [cam_infos[i] for i in idx_train]  #这里只是做一个测试看看模型过拟合后效果是什么样的
        test_cam_infos = [cam_infos[i] for i in idx_test]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 点云依然使用 sparse_{n_views}/0 下的点云
    ply_path = os.path.join(path, f"sparse_{n_views}/0/points3D.ply")
    bin_path = os.path.join(path, f"sparse_{n_views}/0/points3D.bin")
    txt_path = os.path.join(path, f"sparse_{n_views}/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        eval_cameras=[],
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )
    return scene_info

def readColmapSceneInfo_vggt(path, images, eval, dataset, n_views=0, llffhold=8, foundation_model='lseg',use_pbr=True):
    # 固定从 sparse/0 读取相机参数
    cameras_extrinsic_file = os.path.join(path, "sparse", "images.txt")
    cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.txt")
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = "images"

    if foundation_model =='lseg':
        semantic_feature_dir = "rgb_feature_langseg" 
        relevancymap_dir="relevancymap"
    semantic_feature_folder=os.path.join(path, semantic_feature_dir)
    relevancymap_folder=os.path.join(path, relevancymap_dir)

    rgb_mapping = [f for f in sorted(glob.glob(os.path.join(path, reading_dir, '*')))
                    if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    cam_extrinsics = {cam_extrinsics[k].name: cam_extrinsics[k] for k in cam_extrinsics}
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        semantic_feature_folder=semantic_feature_folder,
        relevancymap_folder=relevancymap_folder,
        path=path,
        rgb_mapping=rgb_mapping,
        use_pbr=use_pbr
    )
    def extract_number(filename):
        match = re.search(r'\d+', filename.stem)
        return int(match.group()) if match else float('inf')
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x:extract_number(Path(x.image_name)))

    # 使用llffhold进行采样，%llffhold为0的是测试集，其余为训练集
    total_views = len(cam_infos)
    # if llffhold > 0:
    #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    #     if n_views > 0 and n_views < len(train_cam_infos):
    #         idx_sub = np.linspace(0, len(train_cam_infos) - 1, n_views)
    #         idx_sub = [round(i) for i in idx_sub]
    #         train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
    #         assert len(train_cam_infos) == n_views
    # else:
    #     train_cam_infos = cam_infos
    #     test_cam_infos = []
    test_idx  = np.linspace(1, len(cam_infos) - 2, num=5, dtype=int)
    test_cam_infos = [cam_infos[i] for i in test_idx]
    train_cam_infos = [cam for i, cam in enumerate(cam_infos) if i not in test_idx]
    if n_views > 0:
        idx_sub = np.linspace(0, len(train_cam_infos)-1, num=n_views,dtype=int)
        # idx_sub = [round(i) for i in idx_sub]
        train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
        assert len(train_cam_infos) == n_views
        
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    print("len of train_cam_infos: ", len(train_cam_infos))
    print("len of test_cam_infos: ", len(test_cam_infos))
    # 点云依然使用 sparse_{n_views}/0 下的点云
    ply_path = os.path.join(path, f"sparse/points3D.ply")
    bin_path = os.path.join(path, f"sparse/points3D.bin")
    txt_path = os.path.join(path, f"sparse/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        eval_cameras=[],
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )
    return scene_info

    
def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        skip = 8 if transformsfile == 'transforms_test.json' else 1
        frames = contents["frames"][::skip]
        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem

            # load main image and convert/flatten alpha
            image = Image.open(image_path)
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            mask = norm_data[:, :, 3:4]

            # attempt to load depth if available (kept for backward compatibility but not stored in CameraInfo)
            if skip == 1:
                try:
                    depth_image = np.load('../SparseNeRF/depth_midas_temp_DPT_Hybrid/Blender/' +
                                          image_path.split('/')[-4] + '/' + image_name + '_depth.npy')
                except:
                    depth_image = None
            else:
                depth_image = None

            # Resize image (same behavior as original)
            arr_resized = cv2.resize(arr, (400, 400))
            image_resized = Image.fromarray(np.array(arr_resized * 255.0, dtype=np.byte), "RGB")
            depth_resized = None if depth_image is None else cv2.resize(depth_image, (400, 400))
            mask_resized = None if mask is None else cv2.resize(mask, (400, 400))

            # compute fov values using original image dimensions (before resize)
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            # locate albedo and roughness in the sibling folder of the rgba folder
            # frame["file_path"] is like "./train_007/rgba" -> parent folder is train_007
            rgba_dir = os.path.dirname(os.path.join(path, frame["file_path"]))
            # parent_dir = os.path.dirname(rgba_dir)
            albedo_path = os.path.join(rgba_dir, "albedo_diffrender.png")
            roughness_path = os.path.join(rgba_dir, "roughness_diffrender.png")

            def load_and_flatten(pth):
                if not os.path.exists(pth):
                    return None
                im = Image.open(pth)
                im_data_local = np.array(im.convert("RGBA"))
                norm_local = im_data_local / 255.0
                arr_local = norm_local[:, :, :3] * norm_local[:, :, 3:4] + bg * (1 - norm_local[:, :, 3:4])
                arr_local_resized = cv2.resize(arr_local, (400, 400))
                return Image.fromarray(np.array(arr_local_resized * 255.0, dtype=np.byte), "RGB")

            albedo_img = load_and_flatten(albedo_path)
            roughness_img = load_and_flatten(roughness_path)

            # Build CameraInfo with fields expected by the rest of the code. Fill missing optional fields with None/empty.
            cam_infos.append(CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image_resized,
                albedo=albedo_img,
                roughness=roughness_img,
                metallic=None,
                image_path=image_path,
                image_name=image_name,
                width=image_resized.size[0],
                height=image_resized.size[1],
                mask=mask_resized,
                bounds=None,
                relevancymap=None,
                semantic_feature=None,
                semantic_feature_path="",
                semantic_feature_name=""
            ))
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, n_views=0, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    print("len of train_cam_infos: ", len(train_cam_infos))
    print("len of test_cam_infos: ", len(test_cam_infos))
    #添加稀疏视角设置
    if n_views > 0:
        train_cam_infos = train_cam_infos[:n_views]
        test_cam_infos = test_cam_infos[:n_views]
        assert len(train_cam_infos) == n_views

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        normals = np.random.randn(*xyz.shape)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           eval_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Colmap_dust3r": readColmapSceneInfo_dust3r,
    "Colmap_dust3r_modify": readColmapSceneInfo_dust3r_modify,
    "Colmap_vggt": readColmapSceneInfo_vggt,
    "Blender" : readNerfSyntheticInfo,
}
