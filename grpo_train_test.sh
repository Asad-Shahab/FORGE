#!/bin/bash
#SBATCH -p h100
#SBATCH --gres=gpu:4
#SBATCH --mem=250gb
#SBATCH -t 0-02:00:00
#SBATCH -o h100_test_opt_%j.log
#SBATCH -e h100_test_opt_%j.err

# =====================================================
# TEST VERSION - Quick test with minimal resources
# =====================================================
# This is for testing the complex reward function setup
# Uses very conservative settings to ensure it works
# =====================================================

echo "========================================="
echo "TEST RUN - Optimized GRPO Training"
echo "Time: $(date)"
echo "========================================="

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.6"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0

# Additional debugging
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=WARN

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
# ULTRA CONSERVATIVE TEST CONFIGURATION
# =====================================================
echo "TEST Configuration:"
echo "------------------"
echo "Batch Size: 1 (minimal memory)"
echo "Gradient Accumulation: 4"
echo "Number of Generations: 1 (single generation)"
echo "Max Sequence Length: 2000 (reduced)"
echo "Max Examples: 50 (small dataset)"
echo "Max Steps: 20 (quick test)"
echo "Test Mode: Enabled"
echo ""

# Run test with --test_mode flag for quick validation
./launch_training.sh 4 \
    --sft_model_path sft_models/final \
    --grpo_batch_size 1 \
    --grpo_gradient_accumulation_steps 4 \
    --grpo_learning_rate 5e-6 \
    --num_generations 1 \
    --temperature 1.0 \
    --max_seq_length 2000 \
    --max_examples 50 \
    --lora_rank 32 \
    --gradient_checkpointing \
    --use_mixed_precision \
    --grpo_max_steps 20 \
    --grpo_logging_steps 2 \
    --grpo_save_steps 10 \
    --grpo_output_dir ./grpo_models_test \
    --test_mode

echo ""
echo "========================================="
echo "Test completed at: $(date)"
echo "========================================="

# Print GPU memory usage after training
echo ""
echo "Final GPU Memory Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test run completed successfully!"
    echo "You can now run the full training with grpo_train_optimized.sh"
else
    echo ""
    echo "❌ Test run failed. Check the error logs above."
fi