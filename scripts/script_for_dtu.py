import os
import csv
import json
import yaml


if __name__ == '__main__':

    exp_setting = './configs/dtu.yaml'

    with open(exp_setting, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)

    dataset = result['dataset']
    n_views = result['n_views']
    cuda_idx = result['cuda_idx']
    use_wsds = result['use_wsds']
    use_wsds_2 = result['use_wsds_2']
    use_pixel = result['use_pixel']
    rand_ply = result['rand_ply']
    sh_interval = result['sh_interval']
    reset_param = result['reset_param']
    densify_grad_threshold = result['densify_grad_threshold']
    depth_weight = result['depth_weight']
    depth_pseudo_weight = result['depth_pseudo_weight']
    opacity_reset_interval = result['opacity_reset_interval']
    lambda_dssim = result['lambda_dssim']
    sample_pseudo_interval = result['sample_pseudo_interval']
    pixel_pseudo_weight = result['pixel_pseudo_weight']
    warp_sds_pseudo_weight = result['warp_sds_pseudo_weight']
    warp_sds_pseudo_weight_2 = result['warp_sds_pseudo_weight_2']
    resolution = result['resolution']
    warp_sds_guidance_scale = result['warp_sds_guidance_scale']

    iterations = result['iterations']
    sd_guidance_scale = result['sd_guidance_scale']
    num_inference_steps = result['num_inference_steps']
    lora_start_iter = result['lora_start_iter']
    sd_guidance_start_iter = result['sd_guidance_start_iter']
    warp_sds_guidance_start_iter = result['warp_sds_guidance_start_iter']
    pixel_guidance_start_iter = result['pixel_guidance_start_iter']
    start_sample_pseudo = result['start_sample_pseudo']

    scene_dir = '<dataset_scene_dir>'
    scene_out_dir = '<output_dir>'

    '''
    Training
    '''

    pycode = f'CUDA_VISIBLE_DEVICES={str(cuda_idx)} python train_dtu_mask.py -s {str(scene_dir)} -m {str(scene_out_dir)} --eval --n_views {str(n_views)} --iterations {str(iterations)} --depth_weight {str(depth_weight)} --depth_pseudo_weight {str(depth_pseudo_weight)} --opacity_reset_interval {str(opacity_reset_interval)} --start_sample_pseudo {str(start_sample_pseudo)} --sample_pseudo_interval {str(sample_pseudo_interval)} --images images_{str(resolution)} --lambda_dssim {str(lambda_dssim)} --reset_param {str(reset_param)} --sh_interval {str(sh_interval)} --densify_grad_threshold {str(densify_grad_threshold)} --my_debug'

    if use_wsds or use_wsds_2:
        pycode = pycode + f' --guidance_scale {warp_sds_guidance_scale}  --warp_sds_guidance_start_iter {warp_sds_guidance_start_iter}'
        if use_wsds:
            pycode = pycode + f' --add_warp_sds_guidance --warp_sds_pseudo_weight {warp_sds_pseudo_weight}'
        if use_wsds_2:
            pycode = pycode + f' --add_warp_sds_guidance_2 --warp_sds_pseudo_weight_2 {warp_sds_pseudo_weight_2}'
            
    if use_pixel:
        pycode = pycode + f' --add_pixel_guidance --pixel_guidance_start_iter {pixel_guidance_start_iter} --pixel_pseudo_weight {pixel_pseudo_weight}'

    if rand_ply:
        pycode = pycode + ' --rand_ply'

    print(pycode)
    os.system(pycode)

    '''
    Rendering
    '''

    pycode = f'CUDA_VISIBLE_DEVICES={str(cuda_idx)} python render.py --eval --source_path {scene_dir} --model_path {scene_out_dir} --iteration {str(iterations)} --images images_{str(resolution)} --n_views {str(n_views)}'
    print(pycode)
    state = os.system(pycode)

    '''
    Evaluation
    '''

    mask_dir = os.path.join(scene_dir, 'mask')
    pycode = f'CUDA_VISIBLE_DEVICES={str(cuda_idx)} python metrics_dtu.py --model_paths {scene_out_dir} --mask_paths {mask_dir}'
    print(pycode)
    state = os.system(pycode)
