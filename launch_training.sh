#!/bin/bash
# Launch script for multi-GPU GRPO training
# Usage: ./launch_training.sh [num_gpus] [options]

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
NUM_GPUS=${1:-2}  # Default to 2 GPUs if not specified
MASTER_PORT=${MASTER_PORT:-29500}

echo -e "${GREEN}🚀 Multi-GPU GRPO Training Launcher${NC}"
echo "========================================="
echo -e "${YELLOW}Number of GPUs: $NUM_GPUS${NC}"
echo -e "${YELLOW}Master Port: $MASTER_PORT${NC}"
echo ""

# Check if running with specific number of GPUs
if [ "$NUM_GPUS" -eq 1 ]; then
    echo -e "${GREEN}Running in single-GPU mode${NC}"
    python train.py "${@:2}"
elif [ "$NUM_GPUS" -eq 2 ] || [ "$NUM_GPUS" -eq 3 ] || [ "$NUM_GPUS" -eq 4 ]; then
    echo -e "${GREEN}Running in multi-GPU mode with $NUM_GPUS GPUs${NC}"
    
    # Use torchrun for distributed training
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        train.py \
        --num_gpus $NUM_GPUS \
        "${@:2}"
else
    echo -e "${RED}Error: Invalid number of GPUs. Please specify 1, 2, 3, or 4.${NC}"
    exit 1
fi

# Example commands:
# ./launch_training.sh 2                                    # Run with 2 GPUs
# ./launch_training.sh 3 --run_sft                         # Run with 3 GPUs and SFT pre-training  
# ./launch_training.sh 4 --gradient_checkpointing          # Run with 4 GPUs and gradient checkpointing
# ./launch_training.sh 2 --use_mixed_precision            # Run with 2 GPUs and mixed precision