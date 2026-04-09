#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT_DIR="${ROOT_DIR}/datasets"
OUTPUT_DIR="${ROOT_DIR}/outputs"

export CUDA_VISIBLE_DEVICES=3
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"

dataset_name=interiorverse_vggt
version="v0"
n_views=12

# helper to run a command and continue on error
run_or_continue() {
    "$@"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[WARNING] command failed (rc=$rc): $*"
        return 1
    fi
    return 0
}

for i in $(seq 1 16); do
    name=scene_${i}
    echo "===== START ${name} (nv${n_views}) ====="

    # Train 3DGS model
    run_or_continue python train_r3dg.py --eval \
        -s ../datasets/${dataset_name}/${name}/ \
        -m ../outputs/${dataset_name}/${name}_3dgs_nv${n_views}_${version} \
        --lambda_normal_render_depth 0.01 \
        --lambda_normal_smooth 0.01 \
        --lambda_depth_smooth 0.01 \
        --lambda_mask_entropy 0.1 \
        --densification_interval 200 \
        --save_training_vis \
        --densify_grad_normal_threshold 6e-8 \
        --lambda_depth_var 2e-2 \
        --n_views ${n_views} \
        --iterations 7000 || { echo "Skipping remaining steps for ${name} due to train_3dgs failure."; continue; }

    # Evaluate 3DGS model
    run_or_continue python eval_nvs.py --eval \
        -m ../outputs/${dataset_name}/${name}_3dgs_nv${n_views}_${version} \
        -c ../outputs/${dataset_name}/${name}_3dgs_nv${n_views}_${version}/chkpnt7000.pth \
        --n_views ${n_views} || echo "Eval 3DGS failed for ${name}, continuing."

    # Train NeILF model using 3DGS checkpoint
    run_or_continue python "${ROOT_DIR}/train.py" --eval \
        -s ${DATA_ROOT_DIR}/${dataset_name}/${name}/ \
        -m ${OUTPUT_DIR}/${dataset_name}/${name}_neilf_nv${n_views}_sgs_${version}_self \
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
        --n_views ${n_views} || echo "Train NeILF failed for ${name}, continuing."

    # Evaluate NeILF model
    run_or_continue python "${ROOT_DIR}/eval_nvs.py" --eval \
        -m ${OUTPUT_DIR}/${dataset_name}/${name}_neilf_nv${n_views}_sgs_${version}_self \
        -c ${OUTPUT_DIR}/${dataset_name}/${name}_neilf_nv${n_views}_sgs_${version}_self/chkpnt10000.pth \
        -t sgs \
        --n_views ${n_views} || echo "Eval NeILF failed for ${name}, continuing."

    echo "===== DONE ${name} ====="
done

echo "All scenes processed."
