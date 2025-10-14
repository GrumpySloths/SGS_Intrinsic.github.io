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

import torch
import math
import numpy as np
from typing import NamedTuple
from typing import Dict, List, Optional, Tuple, Union
from .sh_utils import rotation_between_z
import torch.nn.functional as F

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix2(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    z_sign = 1.0

    P = torch.zeros([4, 4])
    P[0, 0] = 1.0 / tanHalfFovX
    P[1, 1] = 1.0 / tanHalfFovY
    P[2, 2] = 1.0
    P[3, 2] = 1.0
    return P

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getExtrinsicMatrix(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getIntrinsicMatrix(H, W, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    P = torch.zeros(3, 3)
    P[0, 0] = W / tanHalfFovX / 2.
    P[1, 1] = H / tanHalfFovY / 2.
    P[2, 2] = 1.
    P[0, 2] = W / 2.
    P[1, 2] = H / 2.
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def rgb_to_srgb(img, clip=True):
    # hdr img
    if isinstance(img, np.ndarray):
        assert img.shape[-1] == 3, img.shape
        img = np.where(img > 0.0031308, np.power(np.maximum(img, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055, 12.92 * img)
        if clip:
            img = np.clip(img, 0.0, 1.0)
        return img
    elif isinstance(img, torch.Tensor):
        # Support [C,H,W] or [B,C,H,W] or [H,W,3]
        if img.dim() == 4:
            # [B, C, H, W]
            assert img.shape[1] == 3, img.shape
            img_out = torch.where(
                img > 0.0031308,
                torch.pow(torch.clamp(img, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
                12.92 * img
            )
            if clip:
                img_out = torch.clamp(img_out, 0.0, 1.0)
            return img_out
        elif img.dim() == 3:
            if img.shape[0] == 3:
                # [C, H, W]
                img_out = torch.where(
                    img > 0.0031308,
                    torch.pow(torch.clamp(img, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
                    12.92 * img
                )
                if clip:
                    img_out = torch.clamp(img_out, 0.0, 1.0)
                return img_out
            else:
                raise AssertionError(f"Unsupported shape for 3D tensor: {img.shape}")
        else:
            raise AssertionError(f"Unsupported tensor dimension: {img.dim()}")
    else:
        raise TypeError("Unsupported input type. Supported types are numpy.ndarray and torch.Tensor.")

def srgb_to_rgb(img):
    # f is LDR
    if isinstance(img, np.ndarray):
        assert len(img.shape) == 3, img.shape
        assert img.shape[2] == 3, img.shape
        img = np.where(img <= 0.04045, img / 12.92, np.power((np.maximum(img, 0.04045) + 0.055) / 1.055, 2.4))
        return img
    elif isinstance(img, torch.Tensor):
        assert len(img.shape) == 3, img.shape
        assert img.shape[0] == 3, img.shape
        img = torch.where(img <= 0.04045, img / 12.92, torch.pow((torch.max(img, torch.tensor(0.04045)) + 0.055) / 1.055, 2.4))
        return img
    else:
        raise TypeError("Unsupported input type. Supported types are numpy.ndarray and torch.Tensor.")
      
def fibonacci_sphere_sampling(normals, sample_num, random_rotate=True):
    pre_shape = normals.shape[:-1]
    if len(pre_shape) > 1:
        normals = normals.reshape(-1, 3)
    delta = np.pi * (3.0 - np.sqrt(5.0))

    # fibonacci sphere sample around z axis
    idx = torch.arange(sample_num, dtype=torch.float, device='cuda')[None]
    z = (1 - 2 * idx / (2 * sample_num - 1)).clamp_min(np.sin(10/180*np.pi))
    rad = torch.sqrt(1 - z ** 2)
    theta = delta * idx
    if random_rotate:
        theta = torch.rand(*pre_shape, 1, device='cuda') * 2 * np.pi + theta
    y = torch.cos(theta) * rad
    x = torch.sin(theta) * rad
    z_samples = torch.stack([x, y, z.expand_as(y)], dim=-2)

    # rotate to normal
    # z_vector = torch.zeros_like(normals)
    # z_vector[..., 2] = 1  # [H, W, 3]
    # rotation_matrix = rotation_between_vectors(z_vector, normals)
    rotation_matrix = rotation_between_z(normals)
    incident_dirs = rotation_matrix @ z_samples
    incident_dirs = F.normalize(incident_dirs, dim=-2).transpose(-1, -2)
    incident_areas = torch.ones_like(incident_dirs)[..., 0:1] * 2 * np.pi
    if len(pre_shape) > 1:
        incident_dirs = incident_dirs.reshape(*pre_shape, sample_num, 3)
        incident_areas = incident_areas.reshape(*pre_shape, sample_num, 1)
    return incident_dirs, incident_areas

def get_envmap_dirs(res: List[int] = [512, 1024]) -> torch.Tensor:
    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
        torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
        indexing="ij",
    )

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]
    return reflvec


def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz


def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
    valid_x, valid_y = torch.meshgrid(valid_x, valid_y, indexing='xy')
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz

def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3)
    xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ torch.inverse(extrinsic_matrix).transpose(0,1)
    xyz_world = xyz_world[...,:3]

    return xyz_world 

def depth_pcd2normal(xyz, offset=None, gt_image=None):
    hd, wd, _ = xyz.shape 
    if offset is not None:
        ix, iy = torch.meshgrid(
            torch.arange(wd), torch.arange(hd), indexing='xy')
        xy = (torch.stack((ix, iy), dim=-1)[1:-1,1:-1]).to(xyz.device)
        p_offset = torch.tensor([[0,1],[0,-1],[1,0],[-1,0]]).float().to(xyz.device)
        new_offset = p_offset[None,None] + offset.reshape(hd, wd, 4, 2)[1:-1,1:-1]
        xys = xy[:,:,None] + new_offset
        xys[..., 0] = 2 * xys[..., 0] / (wd - 1) - 1.0
        xys[..., 1] = 2 * xys[..., 1] / (hd - 1) - 1.0
        sampled_xyzs = torch.nn.functional.grid_sample(xyz.permute(2,0,1)[None], xys.reshape(1, -1, 1, 2))
        sampled_xyzs = sampled_xyzs.permute(0,2,3,1).reshape(hd-2,wd-2,4,3)
        bottom_point = sampled_xyzs[:,:,0]
        top_point = sampled_xyzs[:,:,1]
        right_point = sampled_xyzs[:,:,2]
        left_point = sampled_xyzs[:,:,3]
    else:
        bottom_point = xyz[..., 2:hd,   1:wd-1, :]
        top_point    = xyz[..., 0:hd-2, 1:wd-1, :]
        right_point  = xyz[..., 1:hd-1, 2:wd,   :]
        left_point   = xyz[..., 1:hd-1, 0:wd-2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point 
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(xyz_normal.permute(2,0,1), (1,1,1,1), mode='constant').permute(1,2,0)
    return xyz_normal

def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix, offset=None, gt_image=None):
    # depth: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    # xyz_normal: (H, W, 3)
    xyz_world = depth2point_world(depth, intrinsic_matrix, extrinsic_matrix) # (HxW, 3)        
    xyz_world = xyz_world.reshape(*depth.shape, 3)
    xyz_normal = depth_pcd2normal(xyz_world, offset, gt_image)

    return xyz_normal

def depth_to_world_points(depth_map, intrinsic, extrinsic):
    """
    Convert a depth map to 3D world coordinates.

    Args:
        depth_map (torch.Tensor): [1, H, W] depth map.
        intrinsic (torch.Tensor): [3, 3] camera intrinsic matrix.
        extrinsic (torch.Tensor): [4, 4] camera extrinsic matrix.

    Returns:
        torch.Tensor: [3, H, W] points in world coordinates.
    """
    assert depth_map.dim() == 3 and depth_map.shape[0] == 1, "depth_map must be [1, H, W]"
    device = depth_map.device
    H, W = depth_map.shape[1], depth_map.shape[2]

    # Create meshgrid of pixel coordinates
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    ones = torch.ones_like(x, dtype=depth_map.dtype)
    pixels_hom = torch.stack([x, y, ones], dim=-1).reshape(-1, 3).T  # [3, H*W]

    # Inverse intrinsic
    intrinsic_inv = torch.inverse(intrinsic.to(device))

    # Camera coordinates
    depth_flat = depth_map.view(-1)
    cam_points = intrinsic_inv @ pixels_hom  # [3, H*W]
    cam_points = cam_points * depth_flat  # scale by depth

    # Homogeneous coordinates
    cam_points_hom = torch.cat([cam_points, ones.view(1, -1)], dim=0)  # [4, H*W]

    # Inverse extrinsic to get world coordinates
    extrinsic_inv = torch.inverse(extrinsic.to(device))
    world_points_hom = extrinsic_inv @ cam_points_hom  # [4, H*W]
    world_points = world_points_hom[:3].reshape(3, H, W)

    return world_points