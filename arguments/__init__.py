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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images_8"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.n_views = 0
        self.env_resolution = 16
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.use_confidence = False
        self.use_color = True
        self.sample_num = 32 
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 10_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 10_000

        self.normal_lr = 0.01
        self.lambda_normal_render_depth = 0.0


        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 10000
        self.densify_from_iter = 500
        self.prune_from_iter = 500
        self.densify_until_iter = 10_000
        self.densify_grad_threshold = 0.0005
        self.prune_threshold = 0.005
        self.start_sample_pseudo = 2000
        self.end_sample_pseudo = 9500
        self.sample_pseudo_interval = 1
        self.dist_thres = 10.
        self.depth_weight = 0.05
        self.normal_weight= 0.05
        self.albedo_weight= 0.05
        self.depth_pseudo_weight = 0.5
        self.normal_pseudo_weight=0.5
        self.albedo_pseudo_weight = 0.5
        self.rgbx_pseudo_weight = 0.5
        self.sd_pseudo_weight = 0.5
        self.warp_sds_pseudo_weight = 0.2
        self.warp_sds_pseudo_weight_2 = 2.0
        self.sds_pseudo_weight = 2.0
        self.pixel_pseudo_weight = 2.0
        self.guidance_scale = 7.5
        self.num_inference_steps = 25
        #r3dg settings
        self.lambda_env_smooth = 0.01
        self.lambda_base_color = 0.0
        self.lambda_base_color_smooth = 1.0
        self.lambda_roughness_smooth = 0.2
        self.lambda_light_smooth = 1.0
        self.lambda_light=0.01
        self.lambda_pbr = 1
        self.lambda_normal_smooth = 0.01
        
        self.base_color_lr = 0.01
        self.roughness_lr = 0.01
        self.light_lr = 0.001
        self.light_rest_lr = 0.0001
        self.light_init = 3.0
        self.visibility_lr = 0.0025
        self.visibility_rest_lr = 0.0025
        self.env_lr = 0.1
        self.env_rest_lr = 0.001
         
        self.reset_param = 0.05
        self.sh_interval = 500

        self.num_images=0  # RGBX settings: total number of train and pseudo views.
        self.num_gt_images=0  # RGBX settings: total number of ground-truth views.
        self.num_pseudo_images=0  # RGBX settings: total number of pseudo views.

        super().__init__(parser, "Optimization Parameters")

class AblateParams(ParamGroup):
    def __init__(self, parser):
        self.add_sd_guidance = False
        self.add_warp_sds_guidance = False
        self.add_warp_sds_guidance_2 = False
        self.add_sds_guidance = False
        self.use_sdsi = False
        self.add_pixel_guidance = False
        self.use_lora = False
        self.use_lora_2 = False
        self.rand_ply = False
        self.origin = False
        super().__init__(parser, "Ablation Parameters")

class AblateParamsSDS(ParamGroup):
    def __init__(self, parser):
        self.add_sd_guidance = False
        self.add_warp_sds_guidance = False
        self.add_warp_sds_guidance_2 = False
        self.add_sds_guidance = False
        self.add_sds_guidance_ori = False
        self.use_sdsi = False
        self.add_pixel_guidance = False
        self.use_lora = False
        self.use_lora_2 = False
        self.rand_ply = False
        super().__init__(parser, "Ablation Parameters")

class AblateParamsDTU(ParamGroup):
    def __init__(self, parser):
        self.add_sd_guidance = False
        self.add_warp_sds_guidance = False
        self.add_warp_sds_guidance_2 = False
        self.add_sds_guidance = False
        self.use_sdsi = False
        self.add_pixel_guidance = False
        self.use_lora = False
        self.use_lora_2 = False
        self.rand_ply = False
        super().__init__(parser, "Ablation Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
