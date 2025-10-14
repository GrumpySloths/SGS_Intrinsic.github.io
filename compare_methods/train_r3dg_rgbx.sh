#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
name=scene_0_bp
dataset_name=interiorverse
version="v5"
# for n_views in 3 6 12
for n_views in 12
do
    # Train 3DGS model
    # python train_r3dg.py --eval \
    #     -s ../datasets/${dataset_name}/${name}/ \
    #     -m ../outputs/${dataset_name}/${name}_3dgs_nv${n_views}_${version} \
    #     --lambda_normal_render_depth 0.01 \
    #     --lambda_normal_smooth 0.01 \
    #     --lambda_depth_smooth 0.01 \
    #     --lambda_mask_entropy 0.1 \
    #     --densification_interval 200 \
    #     --save_training_vis \
    #     --densify_grad_normal_threshold 6e-8 \
    #     --lambda_depth_var 2e-2 \
    #     --n_views ${n_views} \
    #     --iterations 7000 \

    # # # # Evaluate 3DGS model
    # python eval_nvs.py --eval \
    #     -m ../outputs/${dataset_name}/${name}_3dgs_nv${n_views}_${version} \
    #     -c ../outputs/${dataset_name}/${name}_3dgs_nv${n_views}_${version}/chkpnt7000.pth \
    #     --n_views ${n_views}

    #Train NeILF model using 3DGS checkpoint
    # python train_r3dg.py --eval \
    #     -s ../datasets/${dataset_name}/${name}/ \
    #     -m ../outputs/${dataset_name}/${name}_neilf_nv${n_views}_gsid_mixture_${version}_multiview2 \
    #     -c ../outputs/${dataset_name}/${name}_3dgs_nv${n_views}_v5/chkpnt7000.pth \
    #     --save_training_vis \
    #     --position_lr_init 0 \
    #     --position_lr_final 0 \
    #     --normal_lr 0 \
    #     --sh_lr 0 \
    #     --opacity_lr 0 \
    #     --scaling_lr 0 \
    #     --rotation_lr 0 \
    #     --pointlight_lr 0.05 \
    #     --gsid_env_lr 0.01 \
    #     --iterations 10000 \
    #     --env_resolution 128 \
    #     --lambda_base_color_smooth 1 \
    #     --lambda_roughness_smooth 0.2 \
    #     --lambda_geo_diffuse 0.25 \
    #     --lambda_light_smooth 1 \
    #     --lambda_light 0.01 \
    #     --lambda_pbr 1.0 \
    #     --lambda_env_smooth 0.01 \
    #     -t gsid --sample_num 24 \
    #     --save_training_vis_iteration 200 \
    #     --lambda_env_smooth 0.01 \
    #     --n_views ${n_views}

    # Evaluate NeILF model
    python eval_nvs.py --eval \
        -m ../outputs/${dataset_name}/${name}_neilf_nv${n_views}_gsid_mixture_${version}_multiview2 \
        -c ../outputs/${dataset_name}/${name}_neilf_nv${n_views}_gsid_mixture_${version}_multiview/chkpnt10000.pth \
        -t gsid \
        --n_views ${n_views}
done