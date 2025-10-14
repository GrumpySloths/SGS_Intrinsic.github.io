import math
import torch
import numpy as np
import torch.nn.functional as F
from arguments import OptimizationParams
from scene.gaussian_model_r3dg import GaussianModel
from scene.cameras import Camera,PseudoCamera
from utils.sh_utils import eval_sh
from utils.loss_utils import ssim, bilateral_smooth_loss, second_order_edge_aware_loss, tv_loss, first_order_edge_aware_loss, first_order_loss, first_order_edge_aware_norm_loss
from utils.image_utils import psnr
from utils.graphics_utils import fibonacci_sphere_sampling, rgb_to_srgb, srgb_to_rgb
from .r3dg_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from typing import Optional
from utils.loss_utils import MSGLoss
from gaussian_renderer.render_neilf import GGX_specular

@torch.no_grad()
def render_pbr_selfConsistency(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                scaling_modifier=1.0, override_color=None, is_training=False, dict_params=None,):
    direct_light_env_light = dict_params.get("env_light")

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    intrinsic = viewpoint_camera.intrinsic_matrix

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=True,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        # shs = pc.get_features
        shs=pc.get_shs
    else:
        colors_precomp = override_color

    base_color = pc.get_base_color
    roughness = pc.get_roughness
    normal = pc.get_normal
    incidents = pc.get_incidents  # incident shs
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1) #这是世界坐标系下的view_dirs,这样写
    #是因为它不是逐像素的处理对应元素，而是逐gaussian处理元素，故不需要先构建camera坐标系下的ndc rays再将其转换到世界
    # 坐标系下

    # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_shs.shape[0], 1))

    # brdf_color_batch=rendering_equation_batch(
    #     base_color, roughness, normal, viewdirs, incidents,
    #     direct_light_env_light, visibility_precompute=pc._visibility_tracing, 
    #     incident_dirs_precompute=pc._incident_dirs, incident_areas_precompute=pc._incident_areas
    # )  #shape:[batch,num_pts,3], batch=3
    brdf_color_batch=[]
    for i in range(3):
        brdf_color=rendering_equation_v1(
            base_color, roughness, normal, viewdirs, incidents,
            direct_light_env_light, visibility_precompute=pc._visibility_tracing[i:i+1], 
            incident_dirs_precompute=pc._incident_dirs[i:i+1], incident_areas_precompute=pc._incident_areas[i:i+1]
        )  #shape:[num_pts,3]
        brdf_color_batch.append(brdf_color)
    brdf_color_batch=torch.stack(brdf_color_batch,dim=0) #[batch,num_pts,3]
    # brdf_color_batch: [batch, num_pts, 3] -> [num_pts, 3*batch]
    brdf_color_for_rasterizer = brdf_color_batch.permute(1, 0, 2).reshape(brdf_color_batch.shape[1], -1)
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, weights, radii) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        features=brdf_color_for_rasterizer,
    )
    mask = num_contrib > 0
    rendered_feature = rendered_feature / rendered_opacity.clamp_min(1e-5) * mask  # [9,h,w]
    # bg_color_expanded = bg_color.view(-1, 1, 1).expand(rendered_feature.shape)
    background = torch.zeros(9, dtype=torch.float32, device="cuda")
    rendered_feature = rendered_feature * rendered_opacity + (1 - rendered_opacity) *background[:, None, None]  # [9,h,w]
    #将rendered_feature从[9,h,w]变为[3,3,h,w]
    rendered_feature = rendered_feature.view(3,3,rendered_feature.shape[1],rendered_feature.shape[2])

    return rgb_to_srgb(rendered_feature)



def rendering_equation_batch(base_color, roughness, normals, viewdirs,
                              incidents, direct_light_env_light=None,
                              visibility_precompute=None, incident_dirs_precompute=None, incident_areas_precompute=None):
    incident_dirs, incident_areas = incident_dirs_precompute, incident_areas_precompute

    deg = int(np.sqrt(incidents.shape[1]) - 1)
    # Batch global incident lights: (batch, num_pts, num_sample, 3)
    global_incident_lights_batch = direct_light_env_light.direct_light_with_augmentation_batch(incident_dirs, batch=3)
    local_incident_lights = eval_sh(deg, incidents.transpose(1, 2).view(-1, 1, 3, (deg + 1) ** 2), incident_dirs).clamp_min(0)

    incident_visibility = visibility_precompute
    # Expand visibility for batch if needed
    if incident_visibility is not None and incident_visibility.dim() == 3:
        incident_visibility = incident_visibility.unsqueeze(0).expand(global_incident_lights_batch.shape[0], -1, -1, -1)
    # Batch add
    global_incident_lights = global_incident_lights_batch * incident_visibility
    incident_lights = local_incident_lights.unsqueeze(0) + global_incident_lights

    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)  # (num_pts, num_sample, 1)
    n_d_i = n_d_i.unsqueeze(0)  # (batch, num_pts, num_sample, 1)
    f_d = base_color[:, None] / np.pi  # (num_pts, 1, 3)
    f_d = f_d.unsqueeze(0)  # (batch, num_pts, 1, 3)
    f_s = GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=0.04)  # (num_pts, num_sample, 3)
    f_s = f_s.unsqueeze(0)  # (batch, num_pts, num_sample, 3)

    incident_areas = incident_areas.unsqueeze(0)  # (batch, num_pts, num_sample, 1)

    transport = incident_lights * incident_areas * n_d_i  # (batch, num_pts, num_sample, 3)
    # specular = (f_s * transport).mean(dim=-2)  # (batch, num_pts, 3)
    pbr = ((f_d + f_s) * transport).mean(dim=-2)  # (batch, num_pts, 3)
    # diffuse_light = transport.mean(dim=-2) / np.pi  # (batch, num_pts, 3)

    # extra_results = {
    #     "incident_dirs": incident_dirs,
    #     "incident_lights": incident_lights,
    #     "local_incident_lights": local_incident_lights,
    #     "global_incident_lights": global_incident_lights,
    #     "incident_visibility": incident_visibility,
    #     "diffuse_light": diffuse_light,
    #     "specular": specular,
    # }

    return pbr

    # return pbr, extra_results

def rendering_equation_v1(base_color, roughness, normals, viewdirs,
                              incidents, direct_light_env_light=None,
                              visibility_precompute=None, incident_dirs_precompute=None, incident_areas_precompute=None):
    incident_dirs, incident_areas = incident_dirs_precompute, incident_areas_precompute

    deg = int(np.sqrt(incidents.shape[1]) - 1)
    # global_incident_lights = direct_light_env_light.direct_light_with_augmentation(incident_dirs)
    global_incident_lights = direct_light_env_light.direct_light_with_patch_augmentation(incident_dirs)
    local_incident_lights = eval_sh(deg, incidents.transpose(1, 2).view(-1, 1, 3, (deg + 1) ** 2), incident_dirs).clamp_min(0)
    
    incident_visibility = visibility_precompute
    global_incident_lights = global_incident_lights * incident_visibility
    incident_lights = local_incident_lights + global_incident_lights

    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)
    f_d = base_color[:, None] / np.pi
    f_s = GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=0.04)

    transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    pbr = ((f_d + f_s) * transport).mean(dim=-2)


    return pbr