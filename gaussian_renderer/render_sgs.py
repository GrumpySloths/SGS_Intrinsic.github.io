"""
Adapted from https://github.com/jingsenzhu/IndoorInverseRendering/blob/main/lightnet/models/render/__init__.py
"""

import math
from utils.loss_utils import MSGLoss
import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
from scene.gaussian_model_sgs import GaussianModel
from lighting_optimization.brdf import (
    pdf_ggx,
    eval_ggx,
    GGX_specular_deferred,
    GGX_specular_deferred_v2,
)
from scene.cameras import Camera, PseudoCamera
from .r3dg_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh
from utils.image_utils import psnr
import torch.nn.functional as F
from arguments import OptimizationParams
from utils.graphics_utils import rgb_to_srgb, depth_to_world_points
from utils.loss_utils import ssim, first_order_edge_aware_loss, tv_loss, l1_loss_mask
from pbr import CubemapLight, get_brdf_lut, pbr_shading
import nvdiffrast.torch as dr
from typing import Dict, List
from scene import Scene

msgloss = MSGLoss(device="cuda")


def get_envmap_dirs(res: List[int] = [512, 1024]) -> torch.Tensor:
    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
        torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
        indexing="ij",
    )

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    reflvec = torch.stack(
        (sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1
    )  # [H, W, 3]
    return reflvec


def render_view_sgs(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    is_training=False,
    dict_params=None,
    scene: Scene = None,
    iteration=None,
    **kwargs,
):
    env_light = dict_params.get(
        "env_light"
    )  # Used to model complex indirect lighting effects.
    point_light = dict_params.get(
        "point_light"
    )  # Used to model direct point-light effects.

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

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
        debug=pipe.debug,
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
        shs = pc.get_shs
    else:
        colors_precomp = override_color

    base_color = pc.get_base_color
    roughness = pc.get_roughness
    normal = pc.get_normal
    # shadowdirection=pc.get_shadowdirection

    xyz_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)
    depths = (xyz_homo @ viewpoint_camera.world_view_transform)[:, 2:3]
    depths2 = depths.square()

    features = torch.cat([depths, depths2, normal, base_color, roughness], dim=-1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (
        num_rendered,
        num_contrib,
        rendered_image,
        rendered_opacity,
        rendered_depth,
        rendered_feature,
        rendered_pseudo_normal,
        rendered_surface_xyz,
        weights,
        radii,
    ) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        features=features,
    )

    mask = num_contrib > 0
    rendered_feature = rendered_feature / rendered_opacity.clamp_min(1e-5) * mask
    feature_dict = {}

    (
        rendered_depth,
        rendered_depth2,
        rendered_normal,
        rendered_base_color,
        rendered_roughness,
    ) = rendered_feature.split([1, 1, 3, 3, 1], dim=0)
    feature_dict.update(
        {
            "base_color": rgb_to_srgb(rendered_base_color),
            "roughness": rendered_roughness,
        }
    )

    rendered_var = rendered_depth2 - rendered_depth.square()
    points_world = depth_to_world_points(
        rendered_depth,
        viewpoint_camera.intrinsic_matrix,
        viewpoint_camera.extrinsic_matrix,
    )  # [3,h,w]
    rendered_normal = F.normalize(rendered_normal, dim=0)  # [3,h,w]
    gt_metallic = viewpoint_camera.original_metallic
    # Normalize rendered_shadowdirection.
    wi_world = F.normalize(
        viewpoint_camera.camera_center[:, None, None] - points_world, dim=0
    )  # [3,h,w]
    wo_world_pointlight = point_light.sample_direction(
        vpos=points_world, normal=rendered_normal
    )  # [spp,3,h,w]
    visibility = pc.visibility_enc.forward_defer(
        points_world, point_light.position
    )  # [spp,1,h,w]

    with torch.no_grad():
        perturb_std = 0.01
        points_world_jitter = points_world + torch.normal(
            mean=0.0,
            std=perturb_std,
            size=points_world.shape,
            device=points_world.device,
        )
        point_light_pos_jitter = point_light.position + torch.normal(
            mean=0.0,
            std=perturb_std,
            size=point_light.position.shape,
            device=point_light.position.device,
        )
        visibility_jitter = pc.visibility_enc.forward_defer(
            points_world_jitter, point_light_pos_jitter
        )  # [spp,1,h,w]

    f_d, f_s, _ = GGX_specular_deferred_v2(
        rendered_normal,
        wi_world,
        wo_world_pointlight,
        rendered_base_color,
        rendered_roughness,
        gt_metallic,
        fresnel=0.04,
    )  # [1,3,h,w],[spp+1,3,h,w]

    h, w = rendered_base_color.shape[1:]
    spp = wo_world_pointlight.shape[0]
    wo_pointlight = wo_world_pointlight.permute(0, 2, 3, 1).reshape(spp, -1, 3)
    light_direct = point_light(wo_pointlight)  # [spp,-1,3]
    light_direct = light_direct.view(spp, h, w, 3)
    light_direct = light_direct.permute(0, 3, 1, 2)  # [spp+1,3,h,w]
    pdf_direct = point_light.pdf_direction(
        vpos=points_world, direction=None
    )  # [spp,1,h,w]
    pdf_direct = torch.clamp(pdf_direct, min=1e-3)
    # Compute the cosine term (the angle cosine between the normal and outgoing direction).
    ndl = torch.clamp(
        (rendered_normal.unsqueeze(0) * wo_world_pointlight).sum(1, keepdim=True), min=0
    )  # [spp,1,h,w]

    # Compute the final lighting contribution.
    # light = light_direct * ndl * visibility / pdf_direct  # [spp,3,h,w]
    light = light_direct * ndl / pdf_direct

    # Diffuse and specular components.
    colorDiffuse_direct = torch.mean(f_d * light * visibility, dim=0)  # [3,h,w]
    colorSpec_direct = torch.mean(f_s * light * visibility, dim=0)  # [3,h,w]

    diffuse_light_direct = torch.mean(light, dim=0)  # [3,h,w]
    visibility_direct = torch.mean(visibility, dim=0)  # [1,h,w]
    # Compose the direct-light PBR result.
    pbr_direct = colorDiffuse_direct + colorSpec_direct  # [3,h,w]

    # Compute the environment-light PBR result.
    canonical_rays = (
        scene.get_canonical_rays()
    )  # canonical_rays are camera rays in the canonical coordinate system.
    c2w = torch.inverse(viewpoint_camera.world_view_transform.T)  # [4, 4]
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
    view_dirs = -(
        (
            F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3]
        )  # [HW, 3, 3]
        .sum(dim=-1)
        .reshape(H, W, 3)
    )  # [H, W, 3]
    normal_mask = (rendered_normal != 0).all(
        0, keepdim=True
    )  # Shape [1,800,800]; used to filter valid normals.
    env_light.build_mips()
    normal_map = rendered_normal.permute(1, 2, 0).detach()

    brdf_lut = get_brdf_lut().cuda()  # [1,256,256,2]; this BRDF is not a learnable parameter and supports the PBR renderer.

    pbr_result = pbr_shading(
        light=env_light,
        normals=normal_map,  # [H, W, 3]
        view_dirs=view_dirs,
        mask=normal_mask.permute(1, 2, 0),  # [H, W, 1]
        albedo=rendered_base_color.permute(1, 2, 0),  # [H, W, 3]
        roughness=rendered_roughness.permute(1, 2, 0),  # [H, W, 1]
        metallic=gt_metallic.permute(1, 2, 0)
        if gt_metallic is not None
        else None,  # [H, W, 1]
        tone=dict_params.get("tone", False),
        gamma=dict_params.get("gamma", False),
        occlusion=None,
        brdf_lut=brdf_lut,
    )

    pbr_env = pbr_result["pbr_env"]  # [3, H, W]
    diffuse_env = pbr_result["diffuse_env"]  # [3, H, W]
    specular_env = pbr_result["specular_env"]  # [3, H, W]

    if torch.isnan(pbr_env).any():
        print("pbr_env has nan values!")
        raise ValueError("pbr_env has nan values!")

    pbr_all = pbr_env + pbr_direct

    # Extra outputs.
    extra_results = {
        "specular_direct": colorSpec_direct,
        "diffuse_direct": colorDiffuse_direct,
        "diffuse_light_direct": diffuse_light_direct,
        "visibility_direct": visibility_direct,
        "pbr_env": pbr_env,
        "diffuse_env": diffuse_env,
        "specular_env": specular_env,
        "pbr_direct": pbr_direct,
    }

    rendered_pbr = (
        pbr_all * rendered_opacity + (1 - rendered_opacity) * bg_color[:, None, None]
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results = {
        "render": rendered_image,
        "visibility": visibility,
        "visibility_jitter": visibility_jitter,
        "depth": rendered_depth,
        "depth_var": rendered_var,
        "pbr": rgb_to_srgb(rendered_pbr),
        "normal": rendered_normal,
        "pseudo_normal": rendered_pseudo_normal,
        "surface_xyz": rendered_surface_xyz,
        "opacity": rendered_opacity,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "num_rendered": num_rendered,
        "num_contrib": num_contrib,
    }

    results.update(feature_dict)
    results.update(extra_results)

    return results


def render_view_sgs_light(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    is_training=False,
    dict_params=None,
    scene: Scene = None,
    iteration=None,
    **kwargs,
):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

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
        debug=pipe.debug,
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
        shs = pc.get_shs
    else:
        colors_precomp = override_color

    base_color = pc.get_base_color
    roughness = pc.get_roughness
    normal = pc.get_normal

    xyz_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)
    depths = (xyz_homo @ viewpoint_camera.world_view_transform)[:, 2:3]
    depths2 = depths.square()

    features = torch.cat([depths, depths2, normal, base_color, roughness], dim=-1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (
        num_rendered,
        num_contrib,
        rendered_image,
        rendered_opacity,
        rendered_depth,
        rendered_feature,
        rendered_pseudo_normal,
        rendered_surface_xyz,
        weights,
        radii,
    ) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        features=features,
    )

    mask = num_contrib > 0
    rendered_feature = rendered_feature / rendered_opacity.clamp_min(1e-5) * mask
    feature_dict = {}

    (
        rendered_depth,
        rendered_depth2,
        rendered_normal,
        rendered_base_color,
        rendered_roughness,
    ) = rendered_feature.split([1, 1, 3, 3, 1], dim=0)
    feature_dict.update(
        {
            "base_color": rgb_to_srgb(rendered_base_color),
            "roughness": rendered_roughness,
        }
    )

    return feature_dict


@torch.no_grad()
def render_selfConsistency_sgs(
    viewpoint_camera: Camera,
    render_pkg,
    dict_params=None,
    scene: Scene = None,
    pc: GaussianModel = None,
    **kwargs,
):
    rendered_depth = render_pkg["depth"]
    env_light = dict_params.get(
        "env_light"
    )  # Used to model complex indirect lighting effects.
    point_light = dict_params.get(
        "point_light"
    )  # Used to model direct point-light effects.
    rendered_base_color = render_pkg["base_color"]
    rendered_roughness = render_pkg["roughness"]
    rendered_normal = render_pkg["normal"]
    gt_metallic = viewpoint_camera.original_metallic

    points_world = depth_to_world_points(
        rendered_depth,
        viewpoint_camera.intrinsic_matrix,
        viewpoint_camera.extrinsic_matrix,
    )  # [3,h,w]
    rendered_normal = F.normalize(rendered_normal, dim=0)  # [3,h,w]
    wi_world = F.normalize(
        viewpoint_camera.camera_center[:, None, None] - points_world, dim=0
    )  # [3,h,w]

    # Environment-light pass (fixed).
    canonical_rays = scene.get_canonical_rays() if scene is not None else None
    c2w = torch.inverse(viewpoint_camera.world_view_transform.T)  # [4, 4]
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
    if canonical_rays is not None:
        view_dirs = -(
            (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])
            .sum(dim=-1)
            .reshape(H, W, 3)
        )  # [H, W, 3]
    else:
        view_dirs = None
    normal_mask = (rendered_normal != 0).all(0, keepdim=True)  # [1,H,W]
    env_light.build_mips()
    normal_map = rendered_normal.permute(1, 2, 0).detach()
    brdf_lut = get_brdf_lut().cuda()
    pbr_env_result = pbr_shading(
        light=env_light,
        normals=normal_map,
        view_dirs=view_dirs,
        mask=normal_mask.permute(1, 2, 0),
        albedo=rendered_base_color.permute(1, 2, 0),
        roughness=rendered_roughness.permute(1, 2, 0),
        metallic=gt_metallic.permute(1, 2, 0) if gt_metallic is not None else None,
        tone=dict_params.get("tone", False),
        gamma=dict_params.get("gamma", False),
        occlusion=None,
        brdf_lut=brdf_lut,
    )
    pbr_env = pbr_env_result["pbr_env"].detach()  # [3, H, W]

    # Direct-light pass (random sampling).
    wo_world_pointlight = point_light.sample_direction(
        vpos=points_world, normal=rendered_normal
    )  # [spp,3,h,w]
    rendered_shadowdirection = render_pkg.get("shadowdirection", rendered_normal)
    rendered_shadowdirection = F.normalize(rendered_shadowdirection, dim=0)  # [3,h,w]
    f_d, f_s, _ = GGX_specular_deferred_v2(
        rendered_normal,
        wi_world,
        wo_world_pointlight,
        rendered_base_color,
        rendered_roughness,
        gt_metallic,
        fresnel=0.04,
    )  # f_d: [1,3,h,w], f_s: [spp,3,h,w] (per docs)
    h, w = rendered_base_color.shape[1:]
    spp = wo_world_pointlight.shape[0]
    wo_pointlight = wo_world_pointlight.permute(0, 2, 3, 1).reshape(spp, -1, 3)

    # pc required for visibility
    if pc is None:
        raise ValueError(
            "pc (GaussianModel) must be provided to compute visibility via samples"
        )

    # Render the environment-light branch once, and sample the direct-light branch twice using the perturbed items returned by light_random_sample to compute visibility, ndl, and pdf.
    pbr_samples = []
    sample_count = 2
    for _ in range(sample_count):
        # light_random_sample returns (light_direct_samples, mask_bool, pdf_sample, pert_direction_spatial, pert_positions)
        light_direct_sg, mask_bool, pdf_sample, pert_dir_spatial, pert_positions = (
            point_light.light_random_sample(wo_pointlight, vpos=points_world)
        )
        # reshape light to [spp,3,h,w]
        light_direct = light_direct_sg.view(spp, h, w, 3).permute(
            0, 3, 1, 2
        )  # [spp,3,h,w]

        # Use pdf returned by sampler. Try to reshape to [spp,1,h,w] if needed.
        try:
            pdf_direct = pdf_sample.view(spp, 1, h, w)
        except Exception:
            pdf_direct = pdf_sample
        pdf_direct = torch.clamp(pdf_direct, min=1e-3)

        # Use sampled directions for ndl (perturbed outgoing directions)
        # pert_dir_spatial expected [spp,3,h,w] or reshapeable to that
        try:
            pert_dir = pert_dir_spatial.view(spp, 3, h, w)
        except Exception:
            pert_dir = pert_dir_spatial
        ndl = torch.clamp(
            (rendered_normal.unsqueeze(0) * pert_dir).sum(1, keepdim=True), min=0
        )  # [spp,1,h,w]

        # compute visibility using sampled positions returned by the sampler
        # pert_positions is expected to be compatible with visibility_enc.forward_defer
        visibility_sampled = pc.visibility_enc.forward_defer(
            points_world, pert_positions
        )  # [spp,1,h,w]
        # spp,im_h,im_w=wo_world_pointlight.shape[0],wo_world_pointlight.shape[2],wo_world_pointlight.shape[3]
        # visibility_sampled=torch.ones((spp,1,im_h,im_w),device=points_world.device)

        # shading using sampled light, ndl and pdf
        light = light_direct * ndl / pdf_direct  # [spp,3,h,w]

        colorDiffuse_direct = torch.mean(
            f_d * light * visibility_sampled, dim=0
        )  # [3,h,w]
        colorSpec_direct = torch.mean(
            f_s * light * visibility_sampled, dim=0
        )  # [3,h,w]

        pbr_direct = colorDiffuse_direct + colorSpec_direct  # [3,h,w]
        pbr = pbr_env + pbr_direct
        pbr_samples.append(rgb_to_srgb(pbr))

    pbr_samples = torch.stack(pbr_samples, dim=0)  # [2,3,h,w]
    return pbr_samples


def calculate_loss(
    viewpoint_camera, pc, results, opt, direct_light=None, env_light=None
):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    rendered_image = results["render"]
    rendered_normal = results["normal"]
    rendered_pbr = results["pbr"]
    rendered_base_color = results["base_color"]
    rendered_roughness = results["roughness"]
    rendered_visibility = results["visibility"]
    rendered_visibility_jitter = results["visibility_jitter"]

    gt_image = viewpoint_camera.original_image.cuda()
    gt_albedo = (
        viewpoint_camera.original_albedo.cuda()
        if viewpoint_camera.original_albedo is not None
        else None
    )
    gt_roughness = (
        viewpoint_camera.original_roughness.cuda()
        if viewpoint_camera.original_roughness is not None
        else None
    )
    gt_mask = (
        viewpoint_camera.original_mask.cuda()
        if viewpoint_camera.original_mask is not None
        else None
    )
    Ll1 = F.l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    tb_dict["l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

    # Ll1_pbr = F.l1_loss(rendered_pbr, gt_image)
    # Ll1_pbr=l1_loss_mask(rendered_pbr, gt_image,gt_mask)
    Ll1_pbr = F.l1_loss(rendered_pbr, gt_image)
    # Ll1_pbr = F.l1_loss(rendered_pbr_adapted, gt_image)
    # ssim_val_pbr = ssim(rendered_pbr, gt_image,gt_mask)
    ssim_val_pbr = ssim(rendered_pbr, gt_image)
    tb_dict["l1_pbr"] = Ll1_pbr.item()
    tb_dict["ssim_pbr"] = ssim_val_pbr.item()
    tb_dict["psnr_pbr"] = psnr(rendered_pbr, gt_image).mean().item()
    loss_pbr = (1.0 - opt.lambda_dssim) * Ll1_pbr + opt.lambda_dssim * (
        1.0 - ssim_val_pbr
    )
    loss = loss + opt.lambda_pbr * loss_pbr
    # The current method may still need testing to determine whether the albedo estimate should include this loss.
    Ll1_albedo = F.l1_loss(rendered_base_color, gt_albedo)
    ssim_val_albedo = ssim(rendered_base_color, gt_albedo)
    tb_dict["ssim_albedo"] = ssim_val_albedo.item()
    tb_dict["l1_albedo"] = Ll1_albedo.item()
    loss_albedo = (1.0 - opt.lambda_dssim) * Ll1_albedo + opt.lambda_dssim * (
        1.0 - ssim_val_albedo
    )
    loss_albedo += opt.lambda_msg * msgloss(
        rendered_base_color.unsqueeze(0), gt_albedo.unsqueeze(0), mask=None
    )
    loss += 0.3 * loss_albedo
    # Estimate roughness.
    if rendered_roughness is not None and gt_roughness is not None:
        Ll1_roughness = F.l1_loss(rendered_roughness, gt_roughness)
        ssim_val_roughness = ssim(rendered_roughness, gt_roughness)
        tb_dict["ssim_roughness"] = ssim_val_roughness.item()
        tb_dict["l1_roughness"] = Ll1_roughness.item()
        loss_roughness = (1.0 - opt.lambda_dssim) * Ll1_roughness + opt.lambda_dssim * (
            1.0 - ssim_val_roughness
        )
        loss_roughness += opt.lambda_msg * msgloss(
            rendered_roughness.unsqueeze(0), gt_roughness.unsqueeze(0), mask=None
        )
        loss += 0.3 * loss_roughness

    if opt.lambda_env_smooth > 0:
        # TV smoothness
        envmap_dirs = get_envmap_dirs(res=[512, 1024])  # [H, W, 3]
        envmap = dr.texture(
            env_light.base[None, ...],
            envmap_dirs[None, ...].contiguous(),
            filter_mode="linear",
            boundary_mode="cube",
        )[0]  # [H, W, 3]
        tv_h1 = torch.pow(envmap[1:, :, :] - envmap[:-1, :, :], 2).mean()
        tv_w1 = torch.pow(envmap[:, 1:, :] - envmap[:, :-1, :], 2).mean()
        loss_env_smooth = tv_h1 + tv_w1
        tb_dict["loss_env_smooth"] = loss_env_smooth.item()
        loss = loss + opt.lambda_env_smooth * loss_env_smooth

    if opt.lambda_base_color_smooth > 0:
        # image_mask = viewpoint_camera.image_mask.cuda()
        loss_base_color_smooth = first_order_edge_aware_loss(
            rendered_base_color, gt_image
        )
        # loss_base_color_smooth = second_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
        tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
        loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth

    if opt.lambda_roughness_smooth > 0:
        loss_roughness_smooth = first_order_edge_aware_loss(
            rendered_roughness, gt_image
        )
        # loss_roughness_smooth = second_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
        tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()
        loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth

    if opt.lambda_sgs > 0:
        loss_light_sgs = (
            direct_light.pos_reg_loss() * opt.lambda_pos_reg
            + direct_light.val_reg_loss() * opt.lambda_val_reg
        )
        tb_dict["loss_light_sgs"] = loss_light_sgs.item()
        loss = loss + opt.lambda_sgs * loss_light_sgs

    if opt.lambda_reg_vis_enc > 0:
        loss_visibility_smooth = (
            (rendered_visibility_jitter - rendered_visibility).abs().mean()
        )
        tb_dict["loss_visibility_smooth"] = loss_visibility_smooth.item()
        loss = loss + opt.lambda_reg_vis_enc * loss_visibility_smooth

    if opt.lambda_normal_smooth > 0:
        # loss_normal_smooth = second_order_edge_aware_loss(rendered_normal * image_mask, gt_image)
        loss_normal_smooth = tv_loss(rendered_normal)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        loss = loss + opt.lambda_normal_smooth * loss_normal_smooth

    tb_dict["loss"] = loss.item()
    # Print all entries in tb_dict for debugging/logging
    # for k, v in tb_dict.items():
    #     try:
    #         if isinstance(v, torch.Tensor):
    #             if v.numel() == 1:
    #                 print(f"{k}: {v.item()}")
    #             else:
    #                 print(f"{k}: tensor(shape={tuple(v.shape)})")
    #         else:
    #             print(f"{k}: {v}")
    #     except Exception:
    #         try:
    #             print(f"{k}: {str(v)}")
    #         except Exception:
    #             print(f"{k}: <unprintable value>")

    return loss, tb_dict


def render_sgs(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    opt: OptimizationParams = False,
    is_training=False,
    dict_params=None,
    iteration=None,
    **kwargs,
):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_view_sgs(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        scaling_modifier,
        override_color,
        is_training,
        dict_params,
        iteration=iteration,
        **kwargs,
    )

    if is_training:
        loss, tb_dict = calculate_loss(
            viewpoint_camera,
            pc,
            results,
            opt,
            direct_light=dict_params.get("point_light"),
            env_light=dict_params.get("env_light"),
        )
        results["tb_dict"] = tb_dict
        results["loss"] = loss

    return results
