#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT_DIR="${ROOT_DIR}/datasets"
OUTPUT_DIR="${ROOT_DIR}/outputs"

export CUDA_VISIBLE_DEVICES=7
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"

name=kitchen
dataset_name=mipnerf360_vggt_v3
version="v0"
for n_views in 24
do
    # Train 3DGS model
    python "${ROOT_DIR}/train.py" --eval \
        -s ${DATA_ROOT_DIR}/${dataset_name}/${name}/ \
        -m ${OUTPUT_DIR}/${dataset_name}/${name}_3dgs_nv${n_views}_${version} \
        --lambda_normal_render_depth 0.01 \
        --lambda_normal_smooth 0.01 \
        --lambda_depth_smooth 0.01 \
        --lambda_mask_entropy 0.1 \
        --densification_interval 200 \
        --save_training_vis \
        --densify_grad_normal_threshold 5e-7 \
        --lambda_depth_var 2e-2 \
        --n_views ${n_views} \
        --iterations 7000 \

    # Evaluate 3DGS model
    python "${ROOT_DIR}/eval_nvs.py" --eval \
        -m ${OUTPUT_DIR}/${dataset_name}/${name}_3dgs_nv${n_views}_${version} \
        -c ${OUTPUT_DIR}/${dataset_name}/${name}_3dgs_nv${n_views}_${version}/chkpnt7000.pth \
        --n_views ${n_views}

    #Train NeILF model using 3DGS checkpoint
    python "${ROOT_DIR}/train.py" --eval \
        -s ${DATA_ROOT_DIR}/${dataset_name}/${name}/ \
        -m ${OUTPUT_DIR}/${dataset_name}/${name}_neilf_nv${n_views}_sgs_${version} \
        -c ${OUTPUT_DIR}/${dataset_name}/${name}_3dgs_nv${n_views}_${version}/chkpnt7000.pth \
        --save_training_vis \
        --position_lr_init 0 \
        --position_lr_final 0 \
        --normal_lr 0 \
        --sh_lr 0 \
        --opacity_lr 0 \
        --scaling_lr 0 \
        --rotation_lr 0 \
        --pointlight_lr 0.05 \
        --sgs_env_lr 0.01 \
        --iterations 10000 \
        --env_resolution 128 \
        --lambda_base_color_smooth 1 \
        --lambda_roughness_smooth 0.2 \
        --lambda_geo_diffuse 0.25 \
        --lambda_light_smooth 1 \
        --lambda_light 0.01 \
        --lambda_pbr 1.0 \
        --lambda_env_smooth 0.01 \
        -t sgs --sample_num 24 \
        --save_training_vis_iteration 200 \
        --lambda_env_smooth 0.01 \
        --n_views ${n_views}

    # Evaluate NeILF model
    python "${ROOT_DIR}/eval_nvs.py" --eval \
        -m ${OUTPUT_DIR}/${dataset_name}/${name}_neilf_nv${n_views}_sgs_${version} \
        -c ${OUTPUT_DIR}/${dataset_name}/${name}_neilf_nv${n_views}_sgs_${version}/chkpnt10000.pth \
        -t sgs \
        --n_views ${n_views}
done
