#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
DATA_ROOT_DIR="${ROOT_DIR}"
OUTPUT_DIR="${ROOT_DIR}/output"
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

# Function: Run task on specified GPU
run_on_gpu() {
    local GPU_ID=$1
    local DATASET=$2
    local SCENE=$3
    local N_VIEW=$4
    
    SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}
    MODEL_PATH=${OUTPUT_DIR}/${DATASET}/${SCENE}_${N_VIEW}views
    
    # Create necessary directories
    mkdir -p ${MODEL_PATH}
    
    echo "======================================================="
    echo "Starting process: ${DATASET}/${SCENE} (${N_VIEW} views) on GPU ${GPU_ID}"
    echo "======================================================="
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training on GPU ${GPU_ID}..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python "${ROOT_DIR}/train.py" \
        --source_path ${SOURCE_PATH} \
        --model_path ${MODEL_PATH} \
        --eval \
        --n_views ${N_VIEW} \
        --images images \
        --sample_pseudo_interval 1 \
        > ${MODEL_PATH}/train.log 2>&1
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed. Log saved in ${MODEL_PATH}/train.log"
    echo "======================================================="
    echo "Task completed: ${DATASET}/${SCENE} (${N_VIEW} views) on GPU ${GPU_ID}"
    echo "======================================================="
}

# Main loop
total_tasks=$((${#DATASETS[@]} * ${#SCENES[@]} * ${#N_VIEWS[@]}))
current_task=0

for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do
            current_task=$((current_task + 1))
            echo "Processing task $current_task / $total_tasks"

            # Get available GPU
            GPU_ID=$(get_available_gpu)

            # If no GPU is available, wait for a while and retry
            while [ -z "$GPU_ID" ]; do
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] No GPU available, waiting 30 seconds before retrying..."
                sleep 30
                GPU_ID=$(get_available_gpu)
            done

            # Run the task in the background and store its PID
            (run_on_gpu $GPU_ID "$DATASET" "$SCENE" "$N_VIEW") &
            BACKGROUND_PIDS+=($!)

            # Wait for 30 seconds before trying to start the next task
            sleep 120 
        done
    done
done

# Wait for all background tasks to complete
wait

echo "======================================================="
echo "All tasks completed! Processed $total_tasks tasks in total."
echo "======================================================="
