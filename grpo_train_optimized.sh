#!/bin/bash
#SBATCH -p h100
#SBATCH --gres=gpu:4
#SBATCH --mem=250gb
#SBATCH -t 5-00:00:00
#SBATCH -o h100_optimized_%j.log
#SBATCH -e h100_optimized_%j.err

# =====================================================
# Optimized GRPO Training with Complex Reward Function
# =====================================================
# This version includes optimizations for:
# - Loading Llama model only on rank 0
# - Reduced batch sizes for memory efficiency
# - Timeout controls for Prover9
# - Better memory management
# =====================================================

echo "========================================="
echo "Starting Optimized GRPO Training"
echo "Time: $(date)"
echo "========================================="

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0

# Prevent OOM by limiting memory fraction
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.7"

# Clear cache before starting
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Source bashrc to load aliases and custom configurations
source ~/.bashrc

# Change to project directory
cd /scratch/ashahab4/no_unsloth/logical-reasoning-training

# Set cache directories
export HF_HOME=/scratch/ashahab4/no_unsloth/logical-reasoning-training/cache/huggingface
export TRANSFORMERS_CACHE=/scratch/ashahab4/no_unsloth/logical-reasoning-training/cache/transformers
export HF_DATASETS_CACHE=/scratch/ashahab4/no_unsloth/logical-reasoning-training/cache/datasets
export TORCH_HOME=/scratch/ashahab4/no_unsloth/logical-reasoning-training/cache/torch
export WANDB_DIR=/scratch/ashahab4/no_unsloth/logical-reasoning-training/cache/wandb
export WANDB_CACHE_DIR=/scratch/ashahab4/no_unsloth/logical-reasoning-training/cache/wandb/cache

# Load CUDA module
module load cuda 

# Load conda
source /scratch/ashahab4/miniforge3/bin/activate

# Activate environment
conda activate logic

echo "Environment activated: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not found')"
echo ""

# =====================================================
# OPTIMIZED TRAINING CONFIGURATION
# =====================================================
# These settings are optimized for complex reward computation
# with NL->FOL conversion and Prover9 verification

echo "Configuration:"
echo "--------------"
echo "Batch Size: 2 (reduced for memory)"
echo "Gradient Accumulation: 6 (maintains effective batch size of 12)"
echo "Number of Generations: 2 (reduced from 4)"
echo "Max Sequence Length: 2500 (reduced from 3000)"
echo "LoRA Rank: 32"
echo "Learning Rate: 5e-6"
echo "Max Steps: 1000 (for initial testing)"
echo ""

# Run optimized training with complex rewards
./launch_training.sh 4 \
    --sft_model_path sft_models/final \
    --grpo_batch_size 2 \
    --grpo_gradient_accumulation_steps 6 \
    --grpo_learning_rate 5e-6 \
    --num_generations 2 \
    --temperature 1.0 \
    --max_seq_length 2500 \
    --lora_rank 32 \
    --gradient_checkpointing \
    --use_mixed_precision \
    --grpo_max_steps 1000 \
    --grpo_logging_steps 10 \
    --grpo_save_steps 100 \
    --grpo_output_dir ./grpo_models_optimized_complex

# Alternative configuration for longer training (uncomment to use)
# This configuration is for when initial testing succeeds
: '
./launch_training.sh 4 \
    --sft_model_path sft_models/final \
    --grpo_batch_size 1 \
    --grpo_gradient_accumulation_steps 12 \
    --grpo_learning_rate 3e-6 \
    --num_generations 2 \
    --temperature 1.2 \
    --max_seq_length 2000 \
    --lora_rank 32 \
    --gradient_checkpointing \
    --use_mixed_precision \
    --use_8bit_optimizer \
    --grpo_max_steps 5000 \
    --grpo_logging_steps 50 \
    --grpo_save_steps 500 \
    --grpo_output_dir ./grpo_models_optimized_long
'

echo ""
echo "========================================="
echo "Training completed at: $(date)"
echo "========================================="

# Print GPU memory usage after training
echo ""
echo "Final GPU Memory Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Clean up any remaining cache
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true