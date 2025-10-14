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
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, chamfer_dist
from typing import cast,Optional
from scene.config import Config
from torch import Tensor
from functools import reduce
import math
from operator import mul
from bvh import RayTracer
from utils.graphics_utils import fibonacci_sphere_sampling
from tqdm import tqdm


# Constants for spherical harmonics coefficients
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]  

def _get_fourier_features(xyz: Tensor, num_features=3):
    xyz = torch.from_numpy(xyz).to(dtype=torch.float32)
    xyz = xyz - xyz.mean(dim=0, keepdim=True)
    xyz = xyz / torch.quantile(xyz.abs(), 0.97, dim=0) * 0.5 + 0.5
    freqs = torch.repeat_interleave(
        2**torch.linspace(0, num_features-1, num_features, dtype=xyz.dtype, device=xyz.device), 2)
    offsets = torch.tensor([0, 0.5 * math.pi] * num_features, dtype=xyz.dtype, device=xyz.device)
    feat = xyz[..., None] * freqs[None, None] * 2 * math.pi + offsets[None, None]
    print("xyz.shape:", xyz.shape)
    print("feat.shape:", feat.shape)
    feat = torch.sin(feat).reshape(-1, reduce(mul, feat.shape[1:]))
    print("feat after sin and reshape:", feat.shape)
    return feat

def sample_incident_rays(normals, is_training=False, sample_num=24):
    if is_training:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=True)
    else:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=False)

    return incident_dirs, incident_areas  # [N, S, 3], [N, S, 1]

class EmbeddingModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # sh_coeffs = 4**2
        feat_in = 3
        if config.appearance_model_sh:
            feat_in = ((config.sh_degree + 1) ** 2) * 3
        self.mlp = nn.Sequential(
            nn.Linear(config.appearance_embedding_dim + feat_in + 6 * self.config.appearance_n_fourier_freqs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feat_in*2),
        )

    def forward(self, gembedding, aembedding, color, viewdir=None):
        del viewdir  # Viewdirs interface is kept to be compatible with prev. version
        #input_color.shape:[236594,48] 
        color = color.reshape(color.shape[0], -1)
        input_color = color  #这里的input_color并不是gaussian的base_color,不过输入mlp的是gaussian的base_color(可能是为了训练稳定?)
        if not self.config.appearance_model_sh:
            color = color[..., :3]
            # color=color[:,0,:]  # 只取RGB通道
        # print(f'color.shape: {color.shape}, gembedding.shape: {gembedding.shape}, aembedding.shape: {aembedding.shape}')
        # print(f"color.device: {color.device}, gembedding.device: {gembedding.device}, aembedding.device: {aembedding.device}")
        
        inp = torch.cat((color, gembedding, aembedding), dim=-1)
        # print("inp device:", inp.device)
        # print("self.mlp device:", self.mlp[0].weight.device)
        offset, mul = torch.split(self.mlp(inp) * 0.01, [color.shape[-1], color.shape[-1]], dim=-1)
        offset = torch.cat((offset / C0, torch.zeros_like(input_color[..., offset.shape[-1]:])), dim=-1)
        mul = mul.repeat(1, input_color.shape[-1] // mul.shape[-1])
                                #注意这里的mul实际上针对的是r,g,b通道,和sh的阶数是无关的，，即不同阶的sh系数都有对应的r,g,b通道
        return input_color * mul + offset
    
    
class GaussianModel(nn.Module):

    embeddings: Optional[nn.Parameter]
    appearance_embeddings: Optional[nn.Parameter]
    appearance_mlp: Optional[nn.Module]

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        
        self.normal_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-3)

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        
        self.rotation_activation = torch.nn.functional.normalize

        self.material_activation = torch.sigmoid
        #rgbx settings
        self.base_color_activation = lambda x: torch.sigmoid(x) * 0.77 + 0.03
        self.roughness_activation = lambda x: torch.sigmoid(x) * 0.9 + 0.09
        self.inverse_roughness_activation = lambda y: inverse_sigmoid((y-0.09) / 0.9)

    def __init__(self, args, config: Config):
        super().__init__()  # 必须首先调用父类初始化
        self.args = args
        self.config = config  # 保存config引用，后续方法中需要使用
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree
        self.init_point = torch.empty(0)
        self._xyz = torch.empty(0)

        self._normal = torch.empty(0)  # normal
        self.normal_gradient_accum = torch.empty(0)
        
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.confidence = torch.empty(0)
        self._albedo = torch.empty(0)
        self._metallic = torch.empty(0)
        #添加rgbx相关的额外选项,下面的选项在后续pbr阶段会使用到
        self._base_color = torch.empty(0)
        self._roughness = torch.empty(0)
        self._incidents_dc = torch.empty(0) #incidents是去建模间接光去了,这是一个用于建模的可学习参数
        self._incidents_rest = torch.empty(0)
        self._visibility_dc = torch.empty(0)
        self._visibility_rest = torch.empty(0)
            
        self.appearance_n_fourier_freqs=4
        self.appearance_embedding_dim=32
        self.register_parameter("embeddings", cast(nn.Parameter, nn.Parameter(torch.empty(0, 6*self.appearance_n_fourier_freqs, dtype=torch.float32, requires_grad=True))))
        self.register_parameter("appearance_embeddings", cast(nn.Parameter, nn.Parameter(torch.empty(0, self.appearance_embedding_dim, dtype=torch.float32, requires_grad=True))))
        self.appearance_mlp = EmbeddingModel(config)
        self.len_gt_embeddings=0 #用于记录ground truth的embeddings数量
        self.len_pseudo_embeddings=0 #用于记录pseudo_views的embeddings数量

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._normal,
            self.normal_gradient_accum,
            self._albedo,
            self._roughness,
            self._metallic,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale,
         self._normal,
         normal_gradient_accum,
         self._albedo,
         self._roughness,
         self._metallic) = model_args
        if training_args is not None:
            print("training_args is not None, setting up training parameters")
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)
            self.confidence = torch.ones_like(self._opacity, device="cuda") # NOTICE
            self.normal_gradient_accum = normal_gradient_accum

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        w = self.rotation_activation(self._rotation)
        return self.rotation_activation(self._rotation)

    @property
    def get_normal(self):
        return self.normal_activation(self._normal)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_albedo(self):
        return self.material_activation(self._albedo)

    @property
    def get_roughness(self):
        return self.material_activation(self._roughness)

    @property
    def get_metallic(self):
        return self.material_activation(self._metallic)

    @property
    def get_incidents(self):
        """SH"""
        incidents_dc = self._incidents_dc
        incidents_rest = self._incidents_rest
        return torch.cat((incidents_dc, incidents_rest), dim=1)

    @property
    def get_visibility(self):
        """SH"""
        visibility_dc = self._visibility_dc
        visibility_rest = self._visibility_rest
        return torch.cat((visibility_dc, visibility_rest), dim=1)

    @property
    def get_base_color(self):
        return self.base_color_activation(self._base_color)

    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness)
    
    def get_embedding(self, train_image_id=None,type="gt",enable_embedding=True):

        assert train_image_id is not None, "train_image_id must be provided to get_embedding"
        if not enable_embedding:
            return torch.zeros((self.appearance_embedding_dim,), device="cuda", dtype=torch.float32)
        
        if type == "gt":
            return self.appearance_embeddings[train_image_id]
        else:
            return self.appearance_embeddings[train_image_id+self.len_gt_embeddings]
    
    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
        
    def get_inverse_covariance(self, scaling_modifier=1):
        return self.covariance_activation(1 / self.get_scaling,
                                          1 / scaling_modifier,
                                          self.get_rotation)
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).cuda().float()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        features = torch.zeros((fused_point_cloud.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        if self.args.use_color:
            features[:, :3, 0] =  fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        self.init_point = fused_point_cloud

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud)[0], 0.0000001)
        print("dist2.shape", dist2.shape)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        albedo = torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        roughness = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        metallic = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)) #[N,1,3]
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)) #[N,15,3]
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._albedo = nn.Parameter(albedo.requires_grad_(True))
        self._roughness = nn.Parameter(roughness.requires_grad_(True))
        self._metallic = nn.Parameter(metallic.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.confidence = torch.ones_like(opacities, device="cuda")


        fused_normal = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
        fused_normal[:, 0] = 1
        self._normal = nn.Parameter(fused_normal.requires_grad_(True))

        
    #该函数用于初始化PBR阶段所需要的gaussian参数
    def gs_pbr_initialization(self):
        self._base_color = nn.Parameter(torch.zeros_like(self._xyz).requires_grad_(True))
        roughness = torch.zeros_like(self._xyz[..., :1])
        # roughness = self.inverse_roughness_activation(torch.full((self._xyz.shape[0], 1), 0.9, dtype=torch.float, device="cuda"))
        self._roughness = nn.Parameter(roughness.requires_grad_(True))
        incidents = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()

        self._incidents_dc = nn.Parameter(
            incidents[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._incidents_rest = nn.Parameter(
            incidents[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        visibility = torch.zeros((self._xyz.shape[0], 1, 4 ** 2)).float().cuda()
        self._visibility_dc = nn.Parameter(
            visibility[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._visibility_rest = nn.Parameter(
            visibility[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))


    @torch.no_grad()
    def update_visibility(self, sample_num):
        raytracer = RayTracer(self.get_xyz, self.get_scaling, self.get_rotation)
        gaussians_xyz = self.get_xyz
        gaussians_inverse_covariance = self.get_inverse_covariance()
        gaussians_opacity = self.get_opacity[:, 0]
        gaussians_normal = self.get_normal
        incident_visibility_results = []
        incident_dirs_results = []
        incident_areas_results = []
        chunk_size = gaussians_xyz.shape[0] // ((sample_num - 1) // 24 + 1)
        for offset in tqdm(range(0, gaussians_xyz.shape[0], chunk_size), "Update visibility with raytracing."):
            incident_dirs, incident_areas = sample_incident_rays(gaussians_normal[offset:offset + chunk_size], False,
                                                    sample_num)
            trace_results = raytracer.trace_visibility(
                gaussians_xyz[offset:offset + chunk_size, None].expand_as(incident_dirs),
                incident_dirs,
                gaussians_xyz,
                gaussians_inverse_covariance,
                gaussians_opacity,
                gaussians_normal)
            incident_visibility = trace_results["visibility"]
            incident_visibility_results.append(incident_visibility) # [N, S, 1]
            incident_dirs_results.append(incident_dirs)
            incident_areas_results.append(incident_areas)
        incident_visibility_result = torch.cat(incident_visibility_results, dim=0)
        incident_dirs_result = torch.cat(incident_dirs_results, dim=0)
        incident_areas_result = torch.cat(incident_areas_results, dim=0)
        self._visibility_tracing = incident_visibility_result #[N,S,1]
        # print("visibility.shape", incident_visibility_result.shape)
        self._incident_dirs = incident_dirs_result
        self._incident_areas = incident_areas_result
        
        
    #在pbr阶段再对embeddings进行初始化以及优化
    def gs_embeddings_initialization(self):
        # rgbx settings
        # 延迟初始化embeddings：只在训练一段时间、gaussian数量固定后再初始化
        # 1. 检查embeddings是否已经初始化（如shape[0]==0），且当前点数>0
        if self.embeddings.shape[0] == 0 and self._xyz.shape[0] > 0:
            print("Initializing embeddings...")
            embeddings = _get_fourier_features(self._xyz.detach().cpu().numpy(), num_features=self.config.appearance_n_fourier_freqs)
            embeddings.add_(torch.randn_like(embeddings) * 0.0001)
            # 复制到embeddings参数
            self.embeddings.data = embeddings.to("cuda")
            # 2. 将embeddings参数添加到优化器
            # 检查优化器是否已存在embeddings参数组，若无则添加
            
            found = False
            for group in self.optimizer.param_groups:
                # print("group name:", group["name"])
                if group.get("name", "") == "embeddings":
                    found = True
                    group["params"][0] = self.embeddings
                    assert False, "embeddings should not be in optimizer before training"

            self.optimizer.add_param_group({
                'params': [self.embeddings],
                'lr': self.config.embedding_lr,
                'name': "embeddings"
            })

    def set_num_training_images(self, num_images,num_gt_images=0, num_pseudo_images=0):

        self.len_gt_embeddings = num_gt_images  # rgbx settings,用于记录ground truth的embeddings数量
        self.len_pseudo_embeddings = num_pseudo_images  # rgbx settings,用于记录pseudo_views的embeddings数量
        appearance_embeddings=torch.normal(mean=0, std=0.01, size=(num_images, self.appearance_embedding_dim), device="cuda")
        self.appearance_embeddings.data = appearance_embeddings
        # if self.appearance_embeddings is not None:
        #     self._resize_parameter("appearance_embeddings", (num_images, self.appearance_embeddings.shape[1]))
        #     self.appearance_embeddings.data.normal_(0, 0.01)

    def training_setup(self, training_args):

        # self.set_num_training_images(training_args.num_images, training_args.num_gt_images, training_args.num_pseudo_images)

        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.normal_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        # l.append({'params': [self.appearance_embeddings], 'lr': self.config.appearance_embedding_lr, 
        #           "name": "appearance_embeddings", "weight_decay": self.config.appearance_embedding_regularization})
        # l.append({'params': [self.embeddings], 'lr': self.config.embedding_lr, "name": "embeddings"})
        l.append({'params': list(self.appearance_mlp.parameters()), 'lr': self.config.appearance_mlp_lr, "name": "appearance_mlp"})

        if self.args.train_bg:
            l.append({'params': [self.bg_color], 'lr': 0.001, "name": "bg_color"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    # 更新PBR优化器参数
    def update_pbr_optimizer_params(self, training_args):
        # Set default learning rates for rest params if not provided
        if training_args.light_rest_lr < 0:
            training_args.light_rest_lr = training_args.light_lr / 20.0
        if training_args.visibility_rest_lr < 0:
            training_args.visibility_rest_lr = training_args.visibility_lr / 20.0

        self.set_num_training_images(training_args.num_images, training_args.num_gt_images, training_args.num_pseudo_images)

        # Define new parameter groups for PBR-related parameters
        pbr_param_groups = [
            {'params': [self._base_color], 'lr': training_args.base_color_lr, "name": "base_color"},
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
            {'params': [self._incidents_dc], 'lr': training_args.light_lr, "name": "incidents_dc"},
            {'params': [self._incidents_rest], 'lr': training_args.light_rest_lr, "name": "incidents_rest"},
            {'params': [self._visibility_dc], 'lr': training_args.visibility_lr, "name": "visibility_dc"},
            {'params': [self._visibility_rest], 'lr': training_args.visibility_rest_lr, "name": "visibility_rest"},
            {'params': [self.appearance_embeddings], 'lr': self.config.appearance_embedding_lr, 
                  "name": "appearance_embeddings", "weight_decay": self.config.appearance_embedding_regularization}
        ]

        # Remove existing PBR parameter groups if present
        for group in list(self.optimizer.param_groups):
            if group.get("name", "") in [
                "base_color", "roughness", "incidents_dc", "incidents_rest", "visibility_dc", "visibility_rest",
                "appearance_embeddings"
            ]:
                self.optimizer.param_groups.remove(group)

        # Add new parameter groups to the optimizer
        for group in pbr_param_groups:
            self.optimizer.add_param_group(group)
            
    def update_optimizer_lrs(self, position_lr_init=0, normal_lr=0, sh_lr=0, opacity_lr=0, scaling_lr=0, rotation_lr=0):
        lr_map = {
            "xyz": position_lr_init * self.spatial_lr_scale,
            "normal": normal_lr,
            "f_dc": sh_lr,
            "f_rest": sh_lr / 20.0,
            "opacity": opacity_lr,
            "scaling": scaling_lr,
            "rotation": rotation_lr,
        }
        for group in self.optimizer.param_groups:
            name = group.get("name", "")
            if name in lr_map:
                group["lr"] = lr_map[name]

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if iteration<=10_000:
            xyz_lr = self.xyz_scheduler_args(iteration)
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "xyz":
                    param_group['lr'] = xyz_lr
                    return xyz_lr


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()


        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, reset_param=0.05):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * reset_param))
        if len(self.optimizer.state.keys()):
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]
    
    def reset_opacity_origin(self, reset_param=0.01):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * reset_param))
        if len(self.optimizer.state.keys()):
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        albedo = np.stack(
            (
                np.asarray(plydata.elements[0]["albedo_0"]),
                np.asarray(plydata.elements[0]["albedo_1"]),
                np.asarray(plydata.elements[0]["albedo_2"]),
            ),
            axis=1,
        )
        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
        metallic = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis]

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._albedo = nn.Parameter(
            torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._roughness = nn.Parameter(
            torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._metallic = nn.Parameter(
            torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

        fused_normal = torch.zeros((xyz.shape[0], 3), device="cuda")
        fused_normal[:, 0] = 1
        self._normal = nn.Parameter(fused_normal.requires_grad_(True))
        # self.normal_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ['bg_color', 'appearance_embeddings', 'embeddings', 'appearance_mlp']:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, iter):
        if iter > self.args.prune_from_iter:
            valid_points_mask = ~mask
            optimizable_tensors = self._prune_optimizer(valid_points_mask)

            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
            self._normal = optimizable_tensors["normal"]
            #rgbx settings
            # self.embeddings = optimizable_tensors["embeddings"]

            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
            self.normal_gradient_accum = self.normal_gradient_accum[valid_points_mask]

            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]
            self.confidence = self.confidence[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # print("group name:", group["name"])
            if group["name"] in ['bg_color','appearance_embeddings', 'embeddings', 'appearance_mlp']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_normal, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             "normal": new_normal,
            }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._normal = optimizable_tensors["normal"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.confidence = torch.cat([self.confidence, torch.ones(new_opacities.shape, device="cuda")], 0)
        self.normal_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    #这里应该是FSGS论文对于gaussian unpooling的全部算法实现，应该是不会对cuda进行任何改动的，且看代码它也只是在前2000次迭代中使用
    def proximity(self, scene_extent, N = 3):
        dist, nearest_indices = distCUDA2(self.get_xyz)
        selected_pts_mask = torch.logical_and(dist > (5. * scene_extent),
                                              torch.max(self.get_scaling, dim=1).values > (scene_extent))

        new_indices = nearest_indices[selected_pts_mask].reshape(-1).long()
        source_xyz = self._xyz[selected_pts_mask].repeat(1, N, 1).reshape(-1, 3)
        target_xyz = self._xyz[new_indices]
        new_xyz = (source_xyz + target_xyz) / 2
        new_scaling = self._scaling[new_indices]
        new_rotation = torch.zeros_like(self._rotation[new_indices])
        new_rotation[:, 0] = 1
        new_features_dc = torch.zeros_like(self._features_dc[new_indices])
        new_features_rest = torch.zeros_like(self._features_rest[new_indices])
        new_opacity = self._opacity[new_indices]
        new_normal = self._normal[new_indices]

        self.densification_postfix(new_xyz, new_normal, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)



    def densify_and_split(self, grads, grad_threshold, scene_extent, iter, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        dist, _ = distCUDA2(self.get_xyz)
        selected_pts_mask2 = torch.logical_and(dist > (self.args.dist_thres * scene_extent),
                                               torch.max(self.get_scaling, dim=1).values > ( scene_extent))
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask2)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_normal = self._normal[selected_pts_mask].repeat(N, 1)
        self.densification_postfix(new_xyz, new_normal, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, iter)
        

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]


        self.densification_postfix(new_xyz, new_normal, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iter):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent, iter)
        if iter < 2000:
            self.proximity(extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask, iter)
        torch.cuda.empty_cache()

    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1
        self.normal_gradient_accum[update_filter] += torch.norm(self._normal.grad[update_filter], dim=-1, keepdim=True)