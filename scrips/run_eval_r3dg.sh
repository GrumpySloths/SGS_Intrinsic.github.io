#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
DATA_ROOT_DIR="${ROOT_DIR}/datasets"
OUTPUT_DIR="${ROOT_DIR}/outputs"
DATASETS=(
    interiorverse
)

SCENES=(
    scene_0
    scene_1
    scene_2
    scene_3
    scene_4
)

N_VIEWS=(
    3
    6
    12
)

# Global array to store background process PIDs
BACKGROUND_PIDS=()

# Signal handler for cleanup
cleanup() {
    echo ""
    echo "======================================================="
    echo "Received interrupt signal (Ctrl+C). Cleaning up..."
    echo "======================================================="
    
    if [ ${#BACKGROUND_PIDS[@]} -gt 0 ]; then
        echo "Terminating ${#BACKGROUND_PIDS[@]} background processes..."
        for pid in "${BACKGROUND_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "Terminating process $pid..."
                kill -TERM "$pid" 2>/dev/null
            fi
        done
        
        # Wait a moment for graceful termination
        sleep 3
        
        # Force kill any remaining processes
        for pid in "${BACKGROUND_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "Force killing process $pid..."
                kill -KILL "$pid" 2>/dev/null
            fi
        done
    fi
    
    echo "Cleanup completed. Exiting..."
    exit 1
}

# Set up signal traps
trap cleanup SIGINT SIGTERM

# Function to get the id of an available GPU
get_available_gpu() {
    local mem_threshold=800
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
    $2 < threshold { print $1; exit }
    '
}

# Function: Run R3DG training on specified GPU
run_r3dg_on_gpu() {
    local GPU_ID=$1
    local DATASET=$2
    local SCENE=$3
    local N_VIEW=$4
    
    local SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/
    local MODEL_PATH_3DGS=${OUTPUT_DIR}/${DATASET}/${SCENE}_3dgs_nv${N_VIEW}
    local MODEL_PATH_NEILF=${OUTPUT_DIR}/${DATASET}/${SCENE}_neilf_nv${N_VIEW}_gt
    
    # Create necessary directories
    mkdir -p ${MODEL_PATH_3DGS}
    mkdir -p ${MODEL_PATH_NEILF}
    
    echo "======================================================="
    echo "Starting R3DG process: ${DATASET}/${SCENE} (${N_VIEW} views) on GPU ${GPU_ID}"
    echo "======================================================="
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 3DGS training on GPU ${GPU_ID}..."
    # Train 3DGS model
    CUDA_VISIBLE_DEVICES=${GPU_ID} python "${ROOT_DIR}/train.py" --eval \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH_3DGS} \
        --lambda_normal_render_depth 0.01 \
        --lambda_normal_smooth 0.01 \
        --lambda_mask_entropy 0.1 \
        --densification_interval 100 \
        --save_training_vis \
        --densify_grad_normal_threshold 1e-8 \
        --lambda_depth_var 2e-2 \
        --n_views ${N_VIEW} \
        > ${MODEL_PATH_3DGS}/train_3dgs.log 2>&1

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 3DGS training completed. Starting NeILF training on GPU ${GPU_ID}..."
    # Train NeILF model using 3DGS checkpoint
    CUDA_VISIBLE_DEVICES=${GPU_ID} python "${ROOT_DIR}/train.py" --eval \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH_NEILF} \
        -c ${MODEL_PATH_3DGS}/chkpnt10000.pth \
        --save_training_vis \
        --position_lr_init 0 \
        --position_lr_final 0 \
        --normal_lr 0 \
        --sh_lr 0 \
        --opacity_lr 0 \
        --scaling_lr 0 \
        --rotation_lr 0 \
        --iterations 20000 \
        --lambda_base_color_smooth 1 \
        --lambda_roughness_smooth 0.2 \
        --lambda_light_smooth 1 \
        --lambda_light 0.01 \
        -t neilf --sample_num 32 \
        --save_training_vis_iteration 200 \
        --lambda_env_smooth 0.01 \
        --n_views ${N_VIEW} \
        > ${MODEL_PATH_NEILF}/train_neilf.log 2>&1

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] NeILF training completed. Starting evaluation on GPU ${GPU_ID}..."
    # Evaluate NeILF model
    CUDA_VISIBLE_DEVICES=${GPU_ID} python "${ROOT_DIR}/eval_nvs.py" --eval \
        -m ${MODEL_PATH_NEILF} \
        -c ${MODEL_PATH_NEILF}/chkpnt20000.pth \
        -t neilf \
        --n_views ${N_VIEW} \
        > ${MODEL_PATH_NEILF}/eval_neilf.log 2>&1
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] All tasks completed. Logs saved in respective directories."
    echo "======================================================="
    echo "R3DG task completed: ${DATASET}/${SCENE} (${N_VIEW} views) on GPU ${GPU_ID}"
    echo "======================================================="
}

# Main loop
total_tasks=$((${#DATASETS[@]} * ${#SCENES[@]} * ${#N_VIEWS[@]}))
current_task=0

for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do
            current_task=$((current_task + 1))
            echo "Processing R3DG task $current_task / $total_tasks"

            # Get available GPU
            GPU_ID=$(get_available_gpu)

            # If no GPU is available, wait for a while and retry
            while [ -z "$GPU_ID" ]; do
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] No GPU available, waiting 30 seconds before retrying..."
                sleep 30
                GPU_ID=$(get_available_gpu)
            done

            # Run the task in the background and store its PID
            (run_r3dg_on_gpu $GPU_ID "$DATASET" "$SCENE" "$N_VIEW") &
            BACKGROUND_PIDS+=($!)

            # Wait for 120 seconds before trying to start the next task
            sleep 120 
        done
    done
done

# Wait for all background tasks to complete
wait

echo "======================================================="
echo "All R3DG tasks completed! Processed $total_tasks tasks in total."
echo "======================================================="
