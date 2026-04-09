import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from collections import defaultdict
from random import randint
import sys
from utils.loss_utils import ssim, inter_view_loss, inter_view_loss_semantic
from gaussian_renderer import render_fn_dict
from gaussian_renderer.render_sgs import render_view_sgs, render_view_sgs_light
from scene import Scene
from scene.gaussian_model_sgs import GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr, visualize_depth, save_image_grid_with_masks
from utils.system_utils import prepare_output_and_logger
from argparse import ArgumentParser
from arguments.config import ModelParams, PipelineParams, OptimizationParams
from lighting_optimization.lighting import FusedSGGridPointLighting
from pbr import CubemapLight
from utils.graphics_utils import rgb_to_srgb
from torchvision.utils import save_image, make_grid
from lpipsPyTorch import lpips
from models.networks import CNN_decoder
from gaussian_renderer.render_featuregs import render_featuregs
from eval_nvs import feature_visualize_saving
from PIL import Image
from utils.sam_region_infer import get_video_masks
from gaussian_renderer.render_sgs import render_selfConsistency_sgs
from utils.rgbx_guidance import sd_estimate_aovs_batch
from utils.loss_utils import (
    MSGLoss,
    self_albedo_lightinvariance_loss,
    inter_view_albedo_loss,
)


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    is_pbr=False,
    is_blender=False,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    """
    Setup Gaussians
    """
    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
    scene = Scene(dataset, gaussians)
    # sam_align_dir is a temporary directory used to store segmentation results for the training and virtual-view video.
    sam_align_dir = os.path.join(args.model_path, "sam_align_tmp")
    sam_align_visualize_dir = os.path.join(args.model_path, "sam_align_visualize")
    os.makedirs(sam_align_dir, exist_ok=True)
    os.makedirs(sam_align_visualize_dir, exist_ok=True)
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)

    gaussians.training_setup(opt)

    """
    Setup PBR components
    """
    pbr_kwargs = dict()
    if is_pbr:
        feature_out_dim = 512
        feature_in_dim = int(feature_out_dim / 16)
        cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
        msgloss = MSGLoss(device="cuda")

        direct_pointlights = FusedSGGridPointLighting(
            num_lights=[4, 4, 4]
        )  # Initialize direct point lights.
        # Move direct_pointlights to the GPU.
        direct_pointlights = direct_pointlights.cuda()
        direct_pointlights.training_setup(opt)
        env_light = CubemapLight(base_res=256, scale=1.5).cuda()
        env_light.train()
        env_light.training_setup(opt)
        pbr_kwargs["point_light"] = direct_pointlights
        pbr_kwargs["env_light"] = env_light
        path = os.path.join(args.source_path, "category_features.pt")
        if os.path.exists(path):
            category_features = torch.load(path).cuda()
        else:
            print(f"category_features not found at {path}, setting to None")
            category_features = None
        gaussians.category_features = category_features
    else:
        feature_out_dim = 512
        feature_in_dim = int(feature_out_dim / 16)
        cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
        cnn_decoder.training_setup()
        path = os.path.join(args.source_path, "category_features.pt")
        if os.path.exists(path):
            category_features = torch.load(path).cuda()
        else:
            print(f"category_features not found at {path}, setting to None")
            category_features = None
        gaussians.category_features = category_features

    """ Prepare render function and bg"""
    render_fn = render_fn_dict[args.type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    """ Training """
    viewpoint_stack = None
    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(
        range(first_iter + 1, opt.iterations + 1),
        desc="Training progress",
        initial=first_iter,
        total=opt.iterations,
    )

    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        loss = 0
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        pseudo_test_cam, pseudo_train_cam = scene.sample_random_pseudo_cameras(
            viewpoint_cam.uid
        )

        # Render
        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        pbr_kwargs["iteration"] = iteration - first_iter
        render_pkg = render_fn(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            opt=opt,
            is_training=True,
            dict_params=pbr_kwargs,
            iteration=iteration,
            feature_decoder=cnn_decoder,
            scene=scene,
        )

        # Pseudo-view training starts.
        if iteration >= 2000 and is_pbr == False and is_blender == False:
            save_image(render_pkg["render"], os.path.join(sam_align_dir, "001.png"))
            render_pkg_pseudo_test = render_featuregs(
                pseudo_test_cam, gaussians, pipe, background
            )
            render_pkg_pseudo_train = render_featuregs(
                pseudo_train_cam, gaussians, pipe, background
            )
            save_image(
                render_pkg_pseudo_test["render"], os.path.join(sam_align_dir, "002.png")
            )
            save_image(
                render_pkg_pseudo_train["render"],
                os.path.join(sam_align_dir, "000.png"),
            )

            feature_map_train = cnn_decoder(
                render_pkg["feature_map"]
            )  # The decoded feature map matches the shape of gt_featuremap.
            feature_norm_train = torch.nn.functional.normalize(feature_map_train, dim=0)
            relevancymap_train = torch.tensordot(
                gaussians.category_features, feature_norm_train, dims=([1], [0])
            )  # [numclass,360,480]
            feature_map_pseudo_train = cnn_decoder(
                render_pkg_pseudo_train["feature_map"]
            )  # The decoded feature map matches the shape of gt_featuremap.
            feature_norm_pseudo_train = torch.nn.functional.normalize(
                feature_map_pseudo_train, dim=0
            )
            relevancymap_pseudo_train = torch.tensordot(
                gaussians.category_features, feature_norm_pseudo_train, dims=([1], [0])
            )  # [numclass,360,480]
            feature_map_pseudo_test = cnn_decoder(
                render_pkg_pseudo_test["feature_map"]
            )  # The decoded feature map matches the shape of gt_featuremap.
            feature_norm_pseudo_test = torch.nn.functional.normalize(
                feature_map_pseudo_test, dim=0
            )
            relevancymap_pseudo_test = torch.tensordot(
                gaussians.category_features, feature_norm_pseudo_test, dims=([1], [0])
            )  # [numclass,360,480]

            sam_masks = get_video_masks(sam_align_dir)  # [3,1,h,w]
            if iteration % 100 == 0:
                print("Generating intermediate SAM training video results")
                renders = torch.stack(
                    [
                        render_pkg_pseudo_train["render"],
                        render_pkg["render"],
                        render_pkg_pseudo_test["render"],
                    ],
                    dim=0,
                )  # [3,3,h,w]
                save_image_grid_with_masks(
                    renders,
                    sam_masks,
                    os.path.join(sam_align_visualize_dir, f"{iteration:06d}_grid.png"),
                )
            feature_align = torch.stack(
                [
                    render_pkg_pseudo_train["feature_map"],
                    render_pkg["feature_map"],
                    render_pkg_pseudo_test["feature_map"],
                ],
                dim=0,
            )  # [3,c,h,w]
            semantic_align = torch.stack(
                [
                    relevancymap_pseudo_train,
                    relevancymap_train,
                    relevancymap_pseudo_test,
                ],
                dim=0,
            )  # [3,numclass,h,w]

            loss_sam_align = inter_view_loss_semantic(
                feature_align, sam_masks, semantic_align
            )
            loss += 0.3 * loss_sam_align

        if iteration % 3 == 0 and is_pbr:
            save_image(render_pkg["pbr"], os.path.join(sam_align_dir, "001.png"))
            with torch.no_grad():
                render_pkg_pseudo_test = render_view_sgs(
                    pseudo_test_cam,
                    gaussians,
                    pipe,
                    background,
                    scene=scene,
                    dict_params=pbr_kwargs,
                )
                render_pkg_pseudo_train = render_view_sgs(
                    pseudo_train_cam,
                    gaussians,
                    pipe,
                    background,
                    scene=scene,
                    dict_params=pbr_kwargs,
                )
            render_pkg_pseudo_test_light = render_view_sgs_light(
                pseudo_test_cam, gaussians, pipe, background, scene=scene
            )
            render_pkg_pseudo_train_light = render_view_sgs_light(
                pseudo_train_cam, gaussians, pipe, background, scene=scene
            )

            save_image(
                render_pkg_pseudo_test["pbr"], os.path.join(sam_align_dir, "002.png")
            )
            save_image(
                render_pkg_pseudo_train["pbr"], os.path.join(sam_align_dir, "000.png")
            )

            sam_masks = get_video_masks(sam_align_dir)  # [3,1,h,w]
            if iteration % 100 == 0:
                print("Generating intermediate SAM training video results")
                renders = torch.stack(
                    [
                        render_pkg_pseudo_train["pbr"],
                        render_pkg["pbr"],
                        render_pkg_pseudo_test["pbr"],
                    ],
                    dim=0,
                )  # [3,3,h,w]
                save_image_grid_with_masks(
                    renders,
                    sam_masks,
                    os.path.join(sam_align_visualize_dir, f"{iteration:06d}_grid.png"),
                )
            albedo_align = torch.stack(
                [
                    render_pkg_pseudo_train_light["base_color"],
                    render_pkg["base_color"],
                    render_pkg_pseudo_test_light["base_color"],
                ],
                dim=0,
            )  # [3,c,h,w]
            loss_albedo_invariance = inter_view_albedo_loss(albedo_align, sam_masks)
            loss += 0.4 * loss_albedo_invariance

        if iteration % 3 == 1 and is_pbr and iteration >= 8000:
            # Self-invariance constraint for albedo.
            pbr_render_batch = render_selfConsistency_sgs(
                viewpoint_cam,
                render_pkg,
                dict_params=pbr_kwargs,
                scene=scene,
                pc=gaussians,
            ).detach()  # [2,3,h,w]
            save_image(pbr_render_batch[0], os.path.join(sam_align_dir, "000.png"))
            save_image(render_pkg["pbr"], os.path.join(sam_align_dir, "001.png"))
            save_image(pbr_render_batch[1], os.path.join(sam_align_dir, "002.png"))
            sam_masks = get_video_masks(sam_align_dir)  # [3,1,h,w]
            pbr_render_final = torch.stack(
                [pbr_render_batch[0], render_pkg["pbr"], pbr_render_batch[1]], dim=0
            )  # [3,3,h,w]
            if iteration % 100 == 1:
                print("Generating intermediate SAM training video results")
                save_image_grid_with_masks(
                    pbr_render_final,
                    sam_masks,
                    os.path.join(sam_align_visualize_dir, f"{iteration:06d}_grid.png"),
                )
            albedo_estimate_batch = sd_estimate_aovs_batch(
                pbr_render_batch
            ).detach()  # [2,3,h,w]
            albedo_train_view = render_pkg["base_color"]
            loss_albedo_invariance_self = self_albedo_lightinvariance_loss(
                albedo_estimate_batch, albedo_train_view, sam_masks[0].float(), msgloss
            )
            loss += 0.4 * loss_albedo_invariance_self

        viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        tb_dict = render_pkg["tb_dict"]
        # print(f"iteration: {iteration}, psnr:{tb_dict['psnr']},psnr_pbr:{tb_dict['psnr_pbr']}")
        loss += render_pkg["loss"]
        loss.backward()

        with torch.no_grad():
            if pipe.save_training_vis:
                save_training_vis(
                    viewpoint_cam,
                    gaussians,
                    background,
                    render_fn,
                    pipe,
                    opt,
                    first_iter,
                    iteration,
                    pbr_kwargs,
                    scene=scene,
                    is_blender=is_blender,
                )
            # Progress bar
            pbar_dict = {"num": gaussians.get_xyz.shape[0]}
            # if is_pbr:
            #     pbar_dict["light_mean"] = direct_env_light.get_env.mean().item()
            #     pbar_dict["env"] = direct_env_light.H
            for k in tb_dict:
                if k in ["psnr", "psnr_pbr"]:
                    ema_dict_for_log[k] = 0.4 * tb_dict[k] + 0.6 * ema_dict_for_log[k]
                    pbar_dict[k] = f"{ema_dict_for_log[k]:.{7}f}"
            # if iteration % 10 == 0:
            progress_bar.set_postfix(pbar_dict)

            # Log and save
            training_report(
                tb_writer,
                iteration,
                tb_dict,
                scene,
                render_fn,
                pipe=pipe,
                bg_color=background,
                dict_params=pbr_kwargs,
            )

            # densification

            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter, render_pkg["weights"]
                )
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    densify_grad_normal_threshold = (
                        opt.densify_grad_normal_threshold
                        if iteration > opt.normal_densify_from_iter
                        else 99999
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        densify_grad_normal_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            gaussians.step()

            if not is_pbr:
                # featuregs decoder step
                cnn_decoder.step()

            for component in pbr_kwargs.values():
                try:
                    component.step()
                except:
                    pass

            # save checkpoints
            if iteration % args.save_interval == 0 or iteration == args.iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if (
                iteration % args.checkpoint_interval == 0
                or iteration == args.iterations
            ):
                torch.save(
                    (gaussians.capture(), iteration),
                    os.path.join(scene.model_path, "chkpnt" + str(iteration) + ".pth"),
                )
                # Save the visibility encoder as well.
                if is_pbr:
                    gaussians.visibility_enc.save(
                        os.path.join(
                            scene.model_path,
                            "visibility_enc_chkpnt" + str(iteration) + ".pth",
                        )
                    )
                # Save the CNN decoder as well.
                if not is_pbr:
                    torch.save(
                        cnn_decoder.state_dict(),
                        scene.model_path + "/decoder_chkpnt" + str(iteration) + ".pth",
                    )
                for com_name, component in pbr_kwargs.items():
                    try:
                        component.save_ckpt(
                            os.path.join(
                                scene.model_path,
                                f"{com_name}_chkpnt" + str(iteration) + ".pth",
                            ),
                            iteration,
                        )
                        # torch.save((component.model.state_dict(), component.optimizer.state_dict(), iteration),
                        #            os.path.join(scene.model_path, f"{com_name}_chkpnt" + str(iteration) + ".pth"))
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    except:
                        pass

                    print("[ITER {}] Saving {} Checkpoint".format(iteration, com_name))

    if dataset.eval:
        eval_render(
            scene,
            gaussians,
            render_fn,
            pipe,
            background,
            opt,
            pbr_kwargs,
            cnn_decoder,
            is_blender=is_blender,
        )


def training_report(
    tb_writer,
    iteration,
    tb_dict,
    scene: Scene,
    renderFunc,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    opt: OptimizationParams = None,
    is_training=False,
    **kwargs,
):
    if tb_writer:
        for key in tb_dict:
            tb_writer.add_scalar(f"train_loss_patches/{key}", tb_dict[key], iteration)

    # Report test and samples of training set
    if iteration % args.test_interval == 0:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {"name": "train", "cameras": scene.getTrainCameras()},
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                psnr_pbr_test = 0.0
                for idx, viewpoint in enumerate(
                    tqdm(
                        config["cameras"],
                        desc="Evaluating " + config["name"],
                        leave=False,
                    )
                ):
                    render_pkg = renderFunc(
                        viewpoint,
                        scene.gaussians,
                        pipe,
                        bg_color,
                        scaling_modifier,
                        override_color,
                        opt,
                        is_training,
                        iteration=iteration,
                        scene=scene,
                        **kwargs,
                    )

                    image = render_pkg["render"]
                    gt_image = viewpoint.original_image.cuda()

                    opacity = torch.clamp(render_pkg["opacity"], 0.0, 1.0)
                    depth = render_pkg["depth"]
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    normal = torch.clamp(
                        render_pkg.get("normal", torch.zeros_like(image)) / 2
                        + 0.5 * opacity,
                        0.0,
                        1.0,
                    )

                    # BRDF
                    base_color = torch.clamp(
                        render_pkg.get("base_color", torch.zeros_like(image)), 0.0, 1.0
                    )
                    roughness = torch.clamp(
                        render_pkg.get("roughness", torch.zeros_like(depth)), 0.0, 1.0
                    )
                    image_pbr = render_pkg.get("pbr", torch.zeros_like(image))

                    grid = torchvision.utils.make_grid(
                        torch.stack(
                            [
                                image,
                                image_pbr,
                                gt_image,
                                opacity.repeat(3, 1, 1),
                                depth.repeat(3, 1, 1),
                                normal,
                                base_color,
                                roughness.repeat(3, 1, 1),
                            ],
                            dim=0,
                        ),
                        nrow=3,
                    )

                    if tb_writer and (idx < 2):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            grid[None],
                            global_step=iteration,
                        )

                    l1_test += F.l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    psnr_pbr_test += psnr(image_pbr, gt_image).mean().double()

                psnr_test /= len(config["cameras"])
                psnr_pbr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {} PSNR_PBR {}".format(
                        iteration, config["name"], l1_test, psnr_test, psnr_pbr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr_pbr",
                        psnr_pbr_test,
                        iteration,
                    )
                if iteration == args.iterations:
                    with open(
                        os.path.join(args.model_path, config["name"] + "_loss.txt"), "w"
                    ) as f:
                        f.write(
                            "L1 {} PSNR {} PSNR_PBR {}".format(
                                l1_test, psnr_test, psnr_pbr_test
                            )
                        )

        torch.cuda.empty_cache()


def save_training_vis(
    viewpoint_cam,
    gaussians,
    background,
    render_fn,
    pipe,
    opt,
    first_iter,
    iteration,
    pbr_kwargs,
    cnn_decoder=None,
    scene=None,
    is_blender=False,
):
    os.makedirs(os.path.join(args.model_path, "visualize"), exist_ok=True)
    with torch.no_grad():
        if (
            iteration % pipe.save_training_vis_iteration == 0
            or iteration == first_iter + 1
        ):
            render_pkg = render_fn(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                opt=opt,
                is_training=False,
                dict_params=pbr_kwargs,
                iteration=iteration,
                scene=scene,
            )

            # Only compute feature renders/visualizations when not running in blender mode
            if not is_blender:
                render_pkg_feature = render_featuregs(
                    viewpoint_cam, gaussians, pipe, background
                )
                feature_map = render_pkg_feature["feature_map"]
                # gt_feature_map = viewpoint_cam.semantic_feature.cuda()
                # gt_feature_map_upsampled = F.interpolate(gt_feature_map.unsqueeze(0), size=feature_map.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
                feature_map_vis = feature_visualize_saving(feature_map)
                # gt_feature_map_vis = feature_visualize_saving(gt_feature_map_upsampled)

            visualization_list = [
                render_pkg["render"],
                viewpoint_cam.original_image.cuda(),
                # visualize_depth(render_pkg["depth"]),
                # visualize_depth(torch.tensor(viewpoint_cam.depth_image).cuda().unsqueeze(0)),
                (render_pkg["depth_var"] / 0.001).clamp_max(1).repeat(3, 1, 1),
                render_pkg["opacity"].repeat(3, 1, 1),
                render_pkg["normal"] * 0.5 + 0.5,
                render_pkg["pseudo_normal"] * 0.5 + 0.5,
            ]

            # append feature visualizations only when not in blender mode
            if not is_blender:
                visualization_list.extend(
                    [
                        feature_map_vis.cuda().permute(2, 0, 1),
                        # gt_feature_map_vis.cuda().permute(2,0,1),
                        visualize_depth(render_pkg["depth"]),
                        # visualize_depth(torch.tensor(viewpoint_cam.depth_image).cuda().unsqueeze(0)),
                    ]
                )

            # keep PBR visualizations as before (uses gaussians/use_pbr implicitly)
            if getattr(gaussians, "use_pbr", False):
                H, W = render_pkg["pbr"].shape[1:]
                visualization_list.extend(
                    [
                        render_pkg["base_color"],
                        render_pkg["roughness"].repeat(3, 1, 1),
                        render_pkg["visibility_direct"].repeat(3, 1, 1),
                        render_pkg["diffuse_light_direct"],
                        render_pkg["diffuse_direct"],
                        render_pkg["specular_direct"],
                        render_pkg["diffuse_env"],
                        render_pkg["specular_env"],
                        render_pkg["pbr_env"],
                        render_pkg["pbr_direct"],
                        render_pkg["pbr"],
                    ]
                )

            grid = torch.stack(visualization_list, dim=0)
            grid = make_grid(grid, nrow=4)
            scale = grid.shape[-2] / 800
            grid = F.interpolate(
                grid[None], (int(grid.shape[-2] / scale), int(grid.shape[-1] / scale))
            )[0]
            save_image(
                grid, os.path.join(args.model_path, "visualize", f"{iteration:06d}.png")
            )


def eval_render(
    scene,
    gaussians,
    render_fn,
    pipe,
    background,
    opt,
    pbr_kwargs,
    cnn_decoder=None,
    is_blender=False,
):
    print("is_blender result:", is_blender)

    def evaluate_and_save(cameras, split_name):
        eval_dir = os.path.join(args.model_path, f"eval_{split_name}")
        vis_dir = os.path.join(eval_dir, "visualize")
        os.makedirs(vis_dir, exist_ok=True)
        gt_feature_map_path = os.path.join(eval_dir, "gt_feature")
        feature_map_path = os.path.join(eval_dir, "render_feature")
        os.makedirs(gt_feature_map_path, exist_ok=True)
        os.makedirs(feature_map_path, exist_ok=True)

        psnr_total = 0.0
        ssim_total = 0.0
        lpips_total = 0.0

        progress_bar = tqdm(
            range(0, len(cameras)),
            desc=f"Evaluating {split_name}",
            initial=0,
            total=len(cameras),
        )

        with torch.no_grad():
            for idx in progress_bar:
                viewpoint = cameras[idx]
                render_pkg = render_fn(
                    viewpoint,
                    gaussians,
                    pipe,
                    background,
                    opt=opt,
                    is_training=False,
                    dict_params=pbr_kwargs,
                    iteration=args.iterations,
                    scene=scene,
                )

                # Only compute feature renders/visualizations when not running in blender mode
                if not is_blender:
                    render_pkg_feature = render_featuregs(
                        viewpoint, gaussians, pipe, background
                    )
                    feature_map = render_pkg_feature["feature_map"]
                    # gt_feature_map = viewpoint.semantic_feature.cuda()
                    # gt_feature_map_upsampled = F.interpolate(gt_feature_map.unsqueeze(0), size=feature_map.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
                    feature_map_vis = feature_visualize_saving(feature_map)
                    gt_feature_map_vis = None
                    # gt_feature_map_vis = feature_visualize_saving(gt_feature_map_upsampled)
                else:
                    feature_map_vis = None
                    gt_feature_map_vis = None

                shadow_direction = render_pkg.get("shadowdirection", None)
                if shadow_direction is not None:
                    shadow_direction = F.normalize(shadow_direction, dim=0)

                save_image(
                    render_pkg["render"], os.path.join(vis_dir, f"{idx:05d}_render.png")
                )
                save_image(
                    viewpoint.original_image.cuda(),
                    os.path.join(vis_dir, f"{idx:05d}_gt.png"),
                )
                save_image(
                    visualize_depth(render_pkg["depth"]),
                    os.path.join(vis_dir, f"{idx:05d}_depth.png"),
                )
                # save_image(visualize_depth(torch.tensor(viewpoint.depth_image).cuda().unsqueeze(0)), os.path.join(vis_dir, f"{idx:05d}_gt_depth.png"))
                save_image(
                    (render_pkg["depth_var"] / 0.001).clamp_max(1).repeat(3, 1, 1),
                    os.path.join(vis_dir, f"{idx:05d}_depth_var.png"),
                )
                save_image(
                    render_pkg["opacity"].repeat(3, 1, 1),
                    os.path.join(vis_dir, f"{idx:05d}_opacity.png"),
                )
                save_image(
                    render_pkg["normal"] * 0.5 + 0.5,
                    os.path.join(vis_dir, f"{idx:05d}_normal.png"),
                )
                save_image(
                    render_pkg["pseudo_normal"] * 0.5 + 0.5,
                    os.path.join(vis_dir, f"{idx:05d}_pseudo_normal.png"),
                )

                if (
                    not is_blender
                    and feature_map_vis is not None
                    and gt_feature_map_vis is not None
                ):
                    save_image(
                        feature_map_vis.cuda().permute(2, 0, 1),
                        os.path.join(vis_dir, f"{idx:05d}_feature_map.png"),
                    )
                    save_image(
                        gt_feature_map_vis.cuda().permute(2, 0, 1),
                        os.path.join(vis_dir, f"{idx:05d}_gt_feature_map.png"),
                    )

                    Image.fromarray(
                        (feature_map_vis.cpu().numpy() * 255).astype(np.uint8)
                    ).save(
                        os.path.join(
                            feature_map_path, "{0:05d}_feature_vis.png".format(idx)
                        )
                    )
                    Image.fromarray(
                        (gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)
                    ).save(
                        os.path.join(
                            gt_feature_map_path, "{0:05d}_feature_vis.png".format(idx)
                        )
                    )

                is_pbr = getattr(gaussians, "use_pbr", False)
                if is_pbr:
                    H, W = render_pkg["pbr"].shape[1:]
                    save_image(
                        render_pkg["base_color"],
                        os.path.join(vis_dir, f"{idx:05d}_base_color.png"),
                    )
                    save_image(
                        render_pkg["roughness"].repeat(3, 1, 1),
                        os.path.join(vis_dir, f"{idx:05d}_roughness.png"),
                    )
                    if viewpoint.original_metallic is not None:
                        save_image(
                            viewpoint.original_metallic.cuda().repeat(3, 1, 1),
                            os.path.join(vis_dir, f"{idx:05d}_gt_metallic.png"),
                        )
                    if shadow_direction is not None:
                        save_image(
                            shadow_direction * 0.5 + 0.5,
                            os.path.join(vis_dir, f"{idx:05d}_shadowdirection.png"),
                        )
                    else:
                        save_image(
                            torch.zeros(3, H, W),
                            os.path.join(vis_dir, f"{idx:05d}_shadowdirection.png"),
                        )
                    save_image(
                        render_pkg["visibility_direct"].repeat(3, 1, 1),
                        os.path.join(vis_dir, f"{idx:05d}_visibility_direct.png"),
                    )
                    save_image(
                        render_pkg["diffuse_light_direct"],
                        os.path.join(vis_dir, f"{idx:05d}_diffuse_light_direct.png"),
                    )
                    save_image(
                        render_pkg["diffuse_direct"],
                        os.path.join(vis_dir, f"{idx:05d}_diffuse_direct.png"),
                    )
                    save_image(
                        render_pkg["specular_direct"],
                        os.path.join(vis_dir, f"{idx:05d}_specular_direct.png"),
                    )
                    save_image(
                        render_pkg["diffuse_env"],
                        os.path.join(vis_dir, f"{idx:05d}_diffuse_env.png"),
                    )
                    save_image(
                        render_pkg["specular_env"],
                        os.path.join(vis_dir, f"{idx:05d}_specular_env.png"),
                    )
                    save_image(
                        render_pkg["pbr_env"],
                        os.path.join(vis_dir, f"{idx:05d}_pbr_env.png"),
                    )
                    save_image(
                        render_pkg["pbr_direct"],
                        os.path.join(vis_dir, f"{idx:05d}_pbr_direct.png"),
                    )
                    save_image(
                        render_pkg["pbr"], os.path.join(vis_dir, f"{idx:05d}_pbr.png")
                    )

                if is_pbr:
                    image = render_pkg["pbr"]
                else:
                    image = render_pkg["render"]
                image = torch.clamp(image, 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                psnr_total += psnr(image, gt_image).mean().double()
                ssim_total += ssim(image, gt_image).mean().double()
                lpips_total += lpips(image, gt_image, net_type="vgg").mean().double()

        psnr_total /= len(cameras)
        ssim_total /= len(cameras)
        lpips_total /= len(cameras)
        with open(os.path.join(eval_dir, "eval.txt"), "w") as f:
            f.write(f"psnr: {psnr_total}\n")
            f.write(f"ssim: {ssim_total}\n")
            f.write(f"lpips: {lpips_total}\n")
        print(
            f"\n[ITER {args.iterations}] Evaluating {split_name}: PSNR {psnr_total} SSIM {ssim_total} LPIPS {lpips_total}"
        )

    test_cameras = scene.getTestCameras()
    train_cameras = scene.getTrainCameras()
    evaluate_and_save(test_cameras, "test")
    evaluate_and_save(train_cameras, "train")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--gui", action="store_true", default=False, help="use gui")
    parser.add_argument(
        "-t", "--type", choices=["render", "normal", "neilf", "sgs"], default="render"
    )
    parser.add_argument("--test_interval", type=int, default=1500)
    parser.add_argument("--save_interval", type=int, default=20_000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=20_000)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--is_blender", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    is_pbr = args.type in ["neilf", "sgs"]
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        is_pbr=is_pbr,
        is_blender=args.is_blender,
    )

    # All done
    print("\nTraining complete.")
