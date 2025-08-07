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
./launch_training.sh 4 --sft_model_path sft_models/final --use_mixed_precision