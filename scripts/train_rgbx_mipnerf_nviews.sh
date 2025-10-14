#!/bin/bash
# filepath: /home/jiahao/ipsm_relighting/scripts/train_rgbx_mipnerf_nviews.sh
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"

# dataset and GPU settings
dataset_name=mipnerf360_vggt
gpus=(0 5 6 7 4 3 1 2)
num_gpus=${#gpus[@]}

# specific scenes for mipnerf360
scene_names=(bonsai garden kitchen room)
num_scenes=${#scene_names[@]}

# n_views settings
n_views_list=(3 6 12)

# 存储后台进程PID
pids=()

# 信号处理函数 - 捕获Ctrl+C信号
cleanup() {
    echo "收到中断信号，正在终止所有训练进程..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "终止进程 $pid"
            kill -TERM "$pid"  # 发送TERM信号
        fi
    done
    
    # 等待进程优雅退出
    sleep 3
    
    # 强制终止仍在运行的进程
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "强制终止进程 $pid"
            kill -KILL "$pid"
        fi
    done
    
    echo "所有进程已终止"
    exit 1
}

# 注册信号处理
trap cleanup SIGINT SIGTERM

# 生成场景名和输出名
scenes=()
output_names=()
n_views_values=()
for n_views in "${n_views_list[@]}"; do
    for scene in "${scene_names[@]}"; do
        scenes+=("$scene")
        output_names+=("${scene}_nviews_${n_views}_v2")
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
                    
                    # 创建输出目录
                    mkdir -p "$output_dir"
                    log_file="$output_dir/record.log"
                    
                    echo "Training $scene with n_views=$n_views on GPU $gpu..."
                    echo "Output directory: $output_dir"
                    echo "Log file: $log_file"
                    
                    CUDA_VISIBLE_DEVICES=$gpu python $script \
                        -s ./datasets/$dataset_name/$scene \
                        -m $output_dir \
                        --n_views $n_views \
                        $extra_args \
                        > $log_file 2>&1 &
                    
                    # 记录后台进程PID
                    pids+=($!)
                    
                    sleep 40
                    break 2
                fi
            done
            sleep 40
        done
    done
}

# 训练参数（参考train_rgbx_interiorverse_v1.sh中debug=false的参数）
extra_args="--eval \
    --iterations 20000 \
    --end_sample_pseudo 20000 \
    --pbr_iteration 10000 \
    --depth_weight 0.05 \
    --depth_pseudo_weight 0.5 \
    --opacity_reset_interval 10000 \
    --start_sample_pseudo 2000 \
    --sample_pseudo_interval 3 \
    --lambda_dssim 0.2 \
    --lambda_normal_render_depth 0.01 \
    --reset_param 0.01 \
    --sh_interval 500 \
    --densify_grad_threshold 0.0005"

# 调用多GPU运行函数
run_multi_gpu_jobs scenes output_names n_views_values train_modify_rgbx.py "$extra_args"

# 等待所有进程完成
wait

echo "所有训练任务完成"