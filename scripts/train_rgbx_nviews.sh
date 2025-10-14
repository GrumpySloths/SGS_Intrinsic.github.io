#!/bin/bash

# dataset and GPU settings
dataset_name=interiorverse
gpus=(4 5 6 7)
num_gpus=${#gpus[@]}
num_scenes=16

# n_views settings
# n_views_list=(0 3 6 12)
n_views_list=(3 6 12)

# 生成场景名和输出名
scenes=()
output_names=()
n_views_values=()
for n_views in "${n_views_list[@]}"; do
    for ((i=0; i<$num_scenes; i++)); do
        scenes+=("scene_${i}")
        output_names+=("scene_${i}_rgbx_debug_nviews${n_views}")
        n_views_values+=("$n_views")
    done
done

echo "Scenes: ${scenes[@]}"
echo "Output names: ${output_names[@]}"
echo "n_views values: ${n_views_values[@]}"

# 封装多GPU任务分配和运行的函数
run_multi_gpu_jobs() {
    local -n scenes_ref=$1
    local -n outputs_ref=$2
    local -n nviews_ref=$3
    local script=$4
    local extra_args=$5

    local total_jobs=${#scenes_ref[@]}
    local job_idx=0

    for ((; job_idx<$total_jobs; job_idx++)); do
        while true; do
            for gpu in "${gpus[@]}"; do
                # 检查该GPU上是否有python进程
                gpu_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F, -v id=$gpu '{gsub(/^ +| +$/, "", $1); gsub(/^ +| +$/, "", $2); if ($1 == id) print $2}')
                running=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader | grep python | awk -v uuid="$gpu_uuid" -F, '{gsub(/^ +| +$/, "", $1); if ($1 == uuid) print 1}')
                if [ -z "$running" ]; then
                    scene=${scenes_ref[$job_idx]}
                    output_name=${outputs_ref[$job_idx]}
                    n_views=${nviews_ref[$job_idx]}
                    output_dir="output/$dataset_name/$output_name"
                    #如果output_dir已经存在,先删除它
                    # if [ -d "$output_dir" ]; then
                    #     echo "Output directory $output_dir already exists. Deleting it."
                    #     rm -rf "$output_dir"
                    # fi
                    # echo "Creating output directory: $output_dir"
                    # 创建输出目录
                    mkdir -p "$output_dir"
                    log_file="$output_dir/output_${scene}.log"
                    echo "log_file: $log_file"
                    echo "Launching $scene with n_views=$n_views on GPU $gpu"
                    CUDA_VISIBLE_DEVICES=$gpu python $script \
                        -s ./datasets/$dataset_name/$scene \
                        -m $output_dir \
                        --n_views $n_views \
                        $extra_args \
                        > $log_file 2>&1 &
                    sleep 40
                    break 2
                fi
            done
            sleep 40
        done
    done
}

# 需要传递的参数（去掉--n_views，单独传递）
extra_args="--eval \
    --iterations 30000 \
    --end_sample_pseudo 30000 \
    --pbr_iteration 10000 \
    --metallic \
    --depth_weight 0.05 \
    --depth_pseudo_weight 0.5 \
    --opacity_reset_interval 10000 \
    --start_sample_pseudo 2000 \
    --sample_pseudo_interval 3 \
    --lambda_dssim 0.2 \
    --lambda_normal_render_depth 0.01 \
    --reset_param 0.01 \
    --sh_interval 500 \
    --densify_grad_threshold 0.0005 \
    "
#    --my_debug"

# 调用多GPU运行函数
run_multi_gpu_jobs scenes output_names n_views_values train_modify_rgbx.py "$extra_args"
