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
from torch import nn
import numpy as np
import cv2
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getExtrinsicMatrix, getIntrinsicMatrix
import torch.nn.functional as F

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, albedo,roughness,metallic,gt_alpha_mask,
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]),
                 scale=1.0, data_device = "cuda", depth_image = None, normal_image=None,diffuse=None,
                 mask = None, bounds=None, height=None, width=None,semantic_feature=None,relevancymap=None):
        super(Camera, self).__init__()

        self.uid = uid  #这里的uid是从0开始的索引，代表其在整个camera list中的位置
        self.colmap_id = colmap_id
        self.R = R  #注意这里传入的R实际上仍然是c2w矩阵的R或者说相机外参R的转置
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.depth_image = depth_image
        self.diffuse=diffuse    
        self.normal_image=normal_image
        self.mask = mask
        self.bounds = bounds
        self.semantic_feature = semantic_feature 
        self.relevancymap=relevancymap

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        #添加pbr属性评估
        self.original_albedo = albedo.clamp(0.0, 1.0).to(self.data_device) if albedo is not None else None
        self.original_roughness = roughness.clamp(0.0, 1.0).to(self.data_device) if roughness is not None else None
        self.original_metallic = metallic.clamp(0.0, 1.0).to(self.data_device) if metallic is not None else None
        self.original_mask= mask.clamp(0.0, 1.0).to(self.data_device) if mask is not None else None
        # print(
        #     f"original_image shape: {self.original_image.shape}, "
        #     f"original_albedo shape: {self.original_albedo.shape if self.original_albedo is not None else None}, "
        #     f"original_roughness shape: {self.original_roughness.shape if self.original_roughness is not None else None}"
        # )   #该行代码主要用于debug
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()
        self.intrinsic_matrix = getIntrinsicMatrix(self.image_height, self.image_width, self.FoVx, self.FoVy).cuda()
        self.extrinsic_matrix = torch.tensor(getExtrinsicMatrix(R, T)).cuda()

    def get_world_directions(self):
        """not used, bug fixed, when the ppx is not in the center"""
        v, u = torch.meshgrid(torch.arange(self.image_height, device='cuda'),
                              torch.arange(self.image_width, device='cuda'), indexing="ij")
        focal_x = self.intrinsic_matrix[0, 0]
        focal_y = self.intrinsic_matrix[1, 1]
        directions = torch.stack([(u - self.intrinsic_matrix[0, 2]) / focal_x,
                                  (v - self.intrinsic_matrix[1, 2]) / focal_y,
                                  torch.ones_like(u)], dim=0)
        directions = F.normalize(directions, dim=0)
        directions = (self.c2w[:3, :3] @ directions.reshape(3, -1)).reshape(3, self.image_height, self.image_width)

        return directions
class PseudoCamera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, width, height, trans=np.array([0.0, 0.0, 0.0]), scale=1.0,id=0 ):
        super(PseudoCamera, self).__init__()

        self.uid=id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.original_image = None  # Placeholder for the image, can be set later
        self.geo_albedo = None  # Placeholder for the albedo, can be set later
        self.geo_roughness = None  # Placeholder for the roughness, can be set later
        self.geo_metallic = None  # Placeholder for the metallic, can be set
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.original_metallic = None  # Placeholder for the metallic, can be set later
        
        self.intrinsic_matrix = getIntrinsicMatrix(self.image_height, self.image_width, self.FoVx, self.FoVy).cuda()
        self.extrinsic_matrix = torch.tensor(getExtrinsicMatrix(R, T)).cuda()

class FixCamera(nn.Module):
    def __init__(self, c2w, K, gt_image,albedo_image):
        super(FixCamera, self).__init__()

        # Handle c2w: convert to tensor if needed, ensure float32, move to cuda, keep original shape
        if isinstance(c2w, np.ndarray):
            c2w = torch.from_numpy(c2w)
        self.c2w = c2w.float().cuda()

        # Handle K: convert to tensor if needed, ensure float32, move to cuda, keep original shape
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K)
        self.K = K.float().cuda()

        # Handle gt_image: PIL to tensor, move to cuda, squeeze
        if 'PIL' in str(type(gt_image)):
            gt_image = torch.from_numpy(np.array(gt_image)).permute(2, 0, 1).float() / 255.0
        if isinstance(gt_image, np.ndarray):
            gt_image = torch.from_numpy(gt_image)
        # if gt_image.dim() == 3:
        #     gt_image = gt_image.unsqueeze(0)
        self.original_image = gt_image.float().cuda()  #[3, H, W]
        self.albedo_image = albedo_image.float().cuda() if albedo_image is not None else None
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

