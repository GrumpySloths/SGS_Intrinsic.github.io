#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT_DIR="${ROOT_DIR}/datasets"
OUTPUT_DIR="${ROOT_DIR}/outputs"

export CUDA_VISIBLE_DEVICES=5,6,7
name=scene_0_bp
dataset_name=interiorverse
version="v8"
n_views=12

# Define the different pointlight_lr values and their corresponding GPUs.
declare -a pointlight_lrs=(0.05 0.01 0.005)
declare -a gpus=(5 6 7)

for i in "${!pointlight_lrs[@]}"; do
    pointlight_lr=${pointlight_lrs[$i]}
    gpu=${gpus[$i]}
    output_dir="${OUTPUT_DIR}/${dataset_name}/${name}_neilf_nv${n_views}_sgs_vis_plr${pointlight_lr}_${version}"
    # Create output_dir if it does not exist.
    mkdir -p ${output_dir}
    checkpoint="${OUTPUT_DIR}/${dataset_name}/${name}_3dgs_nv${n_views}_v5/chkpnt7000.pth"

    CUDA_VISIBLE_DEVICES=${gpu} nohup python "${ROOT_DIR}/train.py" --eval \
        -s ${DATA_ROOT_DIR}/${dataset_name}/${name}/ \
        -m ${output_dir} \
        -c ${checkpoint} \
        --save_training_vis \
        --position_lr_init 0 \
        --position_lr_final 0 \
        --normal_lr 0 \
        --sh_lr 0 \
        --opacity_lr 0 \
        --scaling_lr 0 \
        --rotation_lr 0 \
        --pointlight_lr ${pointlight_lr} \
        --iterations 15000 \
        --env_resolution 128 \
        --lambda_base_color_smooth 1 \
        --lambda_roughness_smooth 0.2 \
        --lambda_geo_diffuse 0.25 \
        --lambda_light_smooth 1 \
        --lambda_light 0.01 \
        --lambda_pbr 1.0 \
        -t sgs --sample_num 24 \
        --save_training_vis_iteration 200 \
        --lambda_env_smooth 0.01 \
        --n_views ${n_views} \
        > ${output_dir}/train.log 2>&1 &
done

wait
echo "All training jobs have been launched in parallel."
