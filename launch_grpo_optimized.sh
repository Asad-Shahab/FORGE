#!/bin/bash

# Optimized GRPO Training Launcher for Complex Reward Function
# This script uses reduced batch sizes and memory optimizations
# to handle the complex reward function with NL->FOL and Prover9

echo -e "\033[0;32m🚀 Optimized GRPO Training with Complex Rewards\033[0m"
echo "========================================="

# Get number of GPUs
NUM_GPUS=${1:-4}
echo -e "\033[1;33mNumber of GPUs: $NUM_GPUS\033[0m"

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false

# Reduced batch size for memory efficiency
# With complex rewards, we need smaller batches
BATCH_SIZE=2  # Reduced from 4
GRAD_ACCUM=6  # Increased from 3 to maintain effective batch size

# Other optimized settings
MAX_STEPS=500  # Reduced for testing
LOGGING_STEPS=5
SAVE_STEPS=50
NUM_GENERATIONS=2  # Reduced from 4

echo -e "\033[1;33mSettings:\033[0m"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Max Steps: $MAX_STEPS"
echo "  Num Generations: $NUM_GENERATIONS"
echo ""

# Launch command based on number of GPUs
if [ "$NUM_GPUS" -eq 1 ]; then
    echo -e "\033[0;32mRunning in single-GPU mode\033[0m"
    python train.py \
        --sft_model_path sft_models/final \
        --grpo_dataset dataset/proverqa_simplified.json \
        --grpo_batch_size $BATCH_SIZE \
        --grpo_gradient_accumulation_steps $GRAD_ACCUM \
        --grpo_max_steps $MAX_STEPS \
        --num_generations $NUM_GENERATIONS \
        --grpo_learning_rate 5e-6 \
        --temperature 1.0 \
        --grpo_logging_steps $LOGGING_STEPS \
        --grpo_save_steps $SAVE_STEPS \
        --grpo_output_dir ./grpo_models_optimized \
        --gradient_checkpointing \
        --use_mixed_precision \
        --max_seq_length 2500 \
        --max_examples 100
else
    echo -e "\033[0;32mRunning in multi-GPU mode with $NUM_GPUS GPUs\033[0m"
    
    # Set master port to avoid conflicts
    MASTER_PORT=$((29500 + RANDOM % 100))
    echo -e "\033[1;33mMaster Port: $MASTER_PORT\033[0m"
    echo ""
    
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        train.py \
        --sft_model_path sft_models/final \
        --grpo_dataset dataset/proverqa_simplified.json \
        --grpo_batch_size $BATCH_SIZE \
        --grpo_gradient_accumulation_steps $GRAD_ACCUM \
        --grpo_max_steps $MAX_STEPS \
        --num_generations $NUM_GENERATIONS \
        --grpo_learning_rate 5e-6 \
        --temperature 1.0 \
        --grpo_logging_steps $LOGGING_STEPS \
        --grpo_save_steps $SAVE_STEPS \
        --grpo_output_dir ./grpo_models_optimized \
        --gradient_checkpointing \
        --use_mixed_precision \
        --max_seq_length 2500 \
        --max_examples 100
fi

echo ""
echo -e "\033[0;32m✅ Training launcher completed\033[0m"