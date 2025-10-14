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
import os
from tqdm import tqdm
from os import makedirs
import sys
sys.path.append("/home/jiahao/ipsm_relighting_v3")
from scene import Scene
# from scene.gaussian_model_r3dg import GaussianModel
from scene.gaussian_model_gsid import GaussianModel
from gaussian_renderer import render_fn_dict
from gaussian_renderer.render_featuregs import render_featuregs
from torchvision.utils import save_image
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments.config_r3dg import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.direct_light_map import DirectLightMap
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr
from PIL import Image
import torchvision.transforms.functional as tf
import json
import torch.nn as nn
import numpy as np
import sklearn
import sklearn.decomposition        
from models.networks import CNN_decoder
from utils.image_utils import visualize_depth
# from eval_nvs import feature_visualize_saving
import torch.nn.functional as F
from lighting_optimization.lighting import FusedSGGridPointLighting
from pbr import CubemapLight

def feature_visualize_saving(feature):
    fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])
    fmap = nn.functional.normalize(fmap, dim=1)
    pca = sklearn.decomposition.PCA(3, random_state=42)
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()
    transformed = pca.fit_transform(f_samples)
    feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
    feature_pca_components = torch.tensor(pca.components_).float().cuda()
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    del f_samples
    vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
    return vis_feature


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(gt_dir):
        render = Image.open(os.path.join(renders_dir, fname))
        gt = Image.open(os.path.join(gt_dir, fname))
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, pbr_kwargs=None,scene=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")
    pseudo_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pseudo_normal")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    gt_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")
    depth_var_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_var")
    opacity_path = os.path.join(model_path, name, "ours_{}".format(iteration), "opacity")
    feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map")
    gt_feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_feature_map")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    makedirs(pseudo_normal_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(gt_depth_path, exist_ok=True)
    makedirs(depth_var_path, exist_ok=True)
    makedirs(opacity_path, exist_ok=True)
    makedirs(feature_map_path, exist_ok=True)
    makedirs(gt_feature_map_path, exist_ok=True)

    if gaussians.use_pbr:
        base_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "base_color")
        roughness_path = os.path.join(model_path, name, "ours_{}".format(iteration), "roughness")
        gt_metallic_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_metallic")
        shadow_direction_path = os.path.join(model_path, name, "ours_{}".format(iteration), "shadow_direction")
        visibility_direct_path = os.path.join(model_path, name, "ours_{}".format(iteration), "visibility_direct")
        diffuse_light_direct_path = os.path.join(model_path, name, "ours_{}".format(iteration), "diffuse_light_direct")
        diffuse_direct_path = os.path.join(model_path, name, "ours_{}".format(iteration), "diffuse_direct")
        specular_direct_path = os.path.join(model_path, name, "ours_{}".format(iteration), "specular_direct")
        diffuse_env_path = os.path.join(model_path, name, "ours_{}".format(iteration), "diffuse_env")
        specular_env_path = os.path.join(model_path, name, "ours_{}".format(iteration), "specular_env")
        pbr_env_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pbr_env")
        pbr_direct_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pbr_direct")
        pbr_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pbr")
        makedirs(base_color_path, exist_ok=True)
        makedirs(roughness_path, exist_ok=True)
        makedirs(gt_metallic_path, exist_ok=True)
        makedirs(shadow_direction_path, exist_ok=True)
        makedirs(visibility_direct_path, exist_ok=True)
        makedirs(diffuse_light_direct_path, exist_ok=True)
        makedirs(diffuse_direct_path, exist_ok=True)
        makedirs(specular_direct_path, exist_ok=True)
        makedirs(diffuse_env_path, exist_ok=True)
        makedirs(specular_env_path, exist_ok=True)
        makedirs(pbr_env_path, exist_ok=True)
        makedirs(pbr_direct_path, exist_ok=True)
        makedirs(pbr_path, exist_ok=True)

    render_fn = render_fn_dict[args.type]
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render_fn(view, gaussians, pipeline, background, dict_params=pbr_kwargs,scene=scene)
        results_feature = render_featuregs(view, gaussians, pipeline, background)

        feature_map = results_feature["feature_map"]
        gt_feature_map = view.semantic_feature.cuda()
        gt_feature_map_upsampled = F.interpolate(gt_feature_map.unsqueeze(0), size=feature_map.shape[1:], mode='bilinear', align_corners=False).squeeze(0)

        feature_map_vis = feature_visualize_saving(feature_map)
        gt_feature_map_vis = feature_visualize_saving(gt_feature_map_upsampled)

        shadow_direction = results.get("shadowdirection", None)
        if shadow_direction is not None:
            shadow_direction = F.normalize(shadow_direction, dim=0)

        save_image(results["render"], os.path.join(render_path, '{0:05d}.png'.format(idx)))
        save_image(view.original_image.cuda(), os.path.join(gts_path, '{0:05d}.png'.format(idx)))
        save_image(visualize_depth(results["depth"]), os.path.join(depth_path, '{0:05d}.png'.format(idx)))
        save_image(visualize_depth(torch.tensor(view.depth_image).cuda().unsqueeze(0)), os.path.join(gt_depth_path, '{0:05d}.png'.format(idx)))
        save_image((results["depth_var"] / 0.001).clamp_max(1).repeat(3, 1, 1), os.path.join(depth_var_path, '{0:05d}.png'.format(idx)))
        save_image(results["opacity"].repeat(3, 1, 1), os.path.join(opacity_path, '{0:05d}.png'.format(idx)))
        save_image(results["normal"] * 0.5 + 0.5, os.path.join(normal_path, '{0:05d}.png'.format(idx)))
        save_image(results["pseudo_normal"] * 0.5 + 0.5, os.path.join(pseudo_normal_path, '{0:05d}.png'.format(idx)))
        save_image(feature_map_vis.cuda().permute(2, 0, 1), os.path.join(feature_map_path, '{0:05d}.png'.format(idx)))
        save_image(gt_feature_map_vis.cuda().permute(2, 0, 1), os.path.join(gt_feature_map_path, '{0:05d}.png'.format(idx)))

        if gaussians.use_pbr:
            H, W = results["pbr"].shape[1:]
            save_image(results["base_color"], os.path.join(base_color_path, '{0:05d}.png'.format(idx)))
            save_image(results["roughness"].repeat(3, 1, 1), os.path.join(roughness_path, '{0:05d}.png'.format(idx)))
            save_image(view.original_metallic.cuda().repeat(3, 1, 1), os.path.join(gt_metallic_path, '{0:05d}.png'.format(idx)))
            if shadow_direction is not None:
                save_image(shadow_direction * 0.5 + 0.5, os.path.join(shadow_direction_path, '{0:05d}.png'.format(idx)))
            else:
                save_image(torch.zeros(3, H, W), os.path.join(shadow_direction_path, '{0:05d}.png'.format(idx)))
            save_image(results["visibility_direct"].repeat(3, 1, 1), os.path.join(visibility_direct_path, '{0:05d}.png'.format(idx)))
            save_image(results["diffuse_light_direct"], os.path.join(diffuse_light_direct_path, '{0:05d}.png'.format(idx)))
            save_image(results["diffuse_direct"], os.path.join(diffuse_direct_path, '{0:05d}.png'.format(idx)))
            save_image(results["specular_direct"], os.path.join(specular_direct_path, '{0:05d}.png'.format(idx)))
            save_image(results["diffuse_env"], os.path.join(diffuse_env_path, '{0:05d}.png'.format(idx)))
            save_image(results["specular_env"], os.path.join(specular_env_path, '{0:05d}.png'.format(idx)))
            save_image(results["pbr_env"], os.path.join(pbr_env_path, '{0:05d}.png'.format(idx)))
            save_image(results["pbr_direct"], os.path.join(pbr_direct_path, '{0:05d}.png'.format(idx)))
            save_image(results["pbr"], os.path.join(pbr_path, '{0:05d}.png'.format(idx)))

def render_sets(dataset : ModelParams, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
        scene = Scene(dataset, gaussians, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if args.checkpoint:
            print("Create Gaussians from checkpoint {}".format(args.checkpoint))
            iteration = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)
        elif scene.loaded_iter:
            gaussians.load_ply(os.path.join(dataset.model_path,
                                            "point_cloud",
                                            "iteration_" + str(scene.loaded_iter),
                                            "point_cloud.ply"))
            iteration = scene.loaded_iter
        else:
            gaussians.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)
            iteration = scene.loaded_iter

        pbr_kwargs = dict()
        if iteration is not None and gaussians.use_pbr:
        
            direct_pointlights=FusedSGGridPointLighting()#初始化直接光点光源
            direct_pointlights.create_from_ckpt(os.path.join(scene.model_path,"point_light_chkpnt10000.pth"), restore_optimizer=False)
            #将direct_pointlights移动到GPU上
            direct_pointlights = direct_pointlights.cuda()
            # direct_pointlights.training_setup(opt)
            # env_light = CubemapLight(base_res=256,scale=1.5).cuda()
            env_light = CubemapLight(base_res=256,scale=1.5)
            env_light.create_from_ckpt(os.path.join(scene.model_path,"env_light_chkpnt10000.pth"), restore_optimizer=False)
            env_light = env_light.cuda()
            env_light.train()
            # env_light.training_setup(opt)
            pbr_kwargs["point_light"] = direct_pointlights
            pbr_kwargs["env_light"] = env_light
            
            # if args.checkpoint:
            #     env_checkpoint = os.path.dirname(args.checkpoint) + "/env_light_" + os.path.basename(args.checkpoint)
            #     print("Trying to load global incident light from ", env_checkpoint)
            #     if os.path.exists(env_checkpoint):
            #         direct_env_light.create_from_ckpt(env_checkpoint, restore_optimizer=True)
            #         print("Successfully loaded!")
            #     else:
            #         print("Failed to load!")
            #     pbr_kwargs["env_light"] = direct_env_light

        
        if not skip_train:
             render_set(dataset.model_path, "train", iteration, scene.getTrainCameras(), gaussians, pipeline, background, pbr_kwargs,scene=scene)

        if not skip_test:
             render_set(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians, pipeline, background, pbr_kwargs,scene=scene)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('-t', '--type', choices=['render', 'normal', 'neilf','gsid'], default='render')
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), pipeline.extract(args), args.skip_train, args.skip_test)