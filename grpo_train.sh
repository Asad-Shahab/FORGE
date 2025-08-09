#!/bin/bash
#SBATCH -p h100
#SBATCH --gres=gpu:4
#SBATCH --mem=250gb
#SBATCH -t 5-00:00:00
#SBATCH -o h100_test_%j.log
#SBATCH -e h100_test_%j.err

# Source bashrc to load aliases and custom configurations
source ~/.bashrc

# Change to project directory
cd /scratch/ashahab4/no_unsloth/logical-reasoning-training

# Set cache directories
export HF_HOME=/scratch/ashahab4/no_unsloth/logical-reasoning-training/cache/huggingface

module load cuda 

# Load conda
source /scratch/ashahab4/miniforge3/bin/activate

# Activate environment
conda activate logic

# Run training
./launch_training.sh 4 \
    --sft_model_path sft_models/final \
    --grpo_batch_size 2 \
    --grpo_gradient_accumulation_steps 2 \
    --grpo_learning_rate 3e-6 \
    --num_generations 8 \
    --temperature 1.2 \
    --max_seq_length 3000 \
    --lora_rank 64 \
    --use_mixed_precision \
    --grpo_max_steps 6000 \
    --grpo_logging_steps 50 \
    --grpo_save_steps 500 \
    --grpo_output_dir ./grpo_models_4gpu_option3 \
    --grpo_dataset dataset/proverqa_simplified.json
