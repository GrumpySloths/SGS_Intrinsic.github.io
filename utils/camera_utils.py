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
import PIL
from utils.normal_guidance import get_normal_tensor
import os
import torch

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    gt_semantic_feature = cam_info.semantic_feature
    gt_relevancymap = (
        torch.from_numpy(cam_info.relevancymap).unsqueeze(0).cuda()
        if cam_info.relevancymap is not None
        else None
    )  # [1,H,W]
    if args.resolution in [1, 2, 4, 8]:
        resolution = (
            round(orig_w / (resolution_scale * args.resolution)),
            round(orig_h / (resolution_scale * args.resolution)),
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    resized_albedo = (
        PILtoTorch(cam_info.albedo, resolution) if cam_info.albedo is not None else None
    )
    resized_roughness = (
        PILtoTorch(cam_info.roughness, resolution)
        if cam_info.roughness is not None
        else None
    )
    resized_metallic = (
        PILtoTorch(cam_info.metallic, resolution)
        if cam_info.metallic is not None
        else None
    )
    if isinstance(cam_info.mask, np.ndarray):
        resized_mask = torch.from_numpy(cam_info.mask).unsqueeze(0).float()
    elif isinstance(
        cam_info.mask, PIL.Image.Image
    ):  # Fix: PIL images, including PngImageFile.
        resized_mask = PILtoTorch(cam_info.mask, resolution)
    else:
        resized_mask = None
    # mask = None if cam_info.mask is None else cv2.resize(cam_info.mask, resolution)
    gt_image = resized_image_rgb[:3, ...]
    gt_albedo = resized_albedo[:3, ...] if resized_albedo is not None else None
    gt_roughness = resized_roughness[:1, ...] if resized_roughness is not None else None
    gt_metallic = resized_metallic[:1, ...] if resized_metallic is not None else None
    gt_mask = resized_mask[:1, ...] if resized_mask is not None else None
    loaded_mask = None

    normal_filename = os.path.join(
        args.model_path, f"precompute_depth_normal/normal_{id}.npy"
    )
    precompute_dir = os.path.join(args.model_path, "precompute_depth_normal")
    if not os.path.exists(precompute_dir):
        os.makedirs(precompute_dir)

    depth = None

    # Depth & Normal
    if os.path.exists(normal_filename):
        normal = np.load(normal_filename)
    else:
        normal = get_normal_tensor(gt_image.cuda())
        normal = normal.cpu().numpy()
        np.save(normal_filename, normal)
        normal_visual = ((normal + 1) * 0.5 * 255).astype(np.uint8)
        normal_visual = np.transpose(normal_visual, (1, 2, 0))
        normal_png_filename = os.path.join(precompute_dir, f"normal_{id}.png")
        plt.imsave(normal_png_filename, normal_visual)

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        albedo=gt_albedo,
        roughness=gt_roughness,
        metallic=gt_metallic,
        gt_alpha_mask=loaded_mask,
        uid=id,
        data_device=args.data_device,
        image_name=cam_info.image_name,
        depth_image=depth,
        normal_image=normal,
        mask=gt_mask,
        bounds=cam_info.bounds,
        height=cam_info.height,
        width=cam_info.width,
        semantic_feature=gt_semantic_feature,
        relevancymap=gt_relevancymap,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in tqdm((enumerate(cam_infos))):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
