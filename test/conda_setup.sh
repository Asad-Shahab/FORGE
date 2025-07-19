#!/bin/bash
# Conda environment setup with cache configuration
# Usage: source conda_setup.sh

# Activate your conda environment
# conda activate your_env_name

# Set cache directories to current project
export PROJECT_DIR="/gpfs/fs2/scratch/ashahab4/logical-reasoning-training/test"
export CACHE_DIR="$PROJECT_DIR/cache"

# ML Library caches
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export HF_HOME="$CACHE_DIR/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR/huggingface/hub"
export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export TENSORBOARD_LOG_DIR="$CACHE_DIR/tensorboard"
export WANDB_DIR="$CACHE_DIR/wandb"
export WANDB_CACHE_DIR="$CACHE_DIR/wandb/cache"
export NLTK_DATA="$CACHE_DIR/nltk_data"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"

# Create cache directories
mkdir -p "$CACHE_DIR/{pip,huggingface/hub,transformers,datasets,torch,tensorboard,wandb/cache,nltk_data,matplotlib}"

echo "✅ Cache directories configured for project: $PROJECT_DIR"
echo "📁 All ML caches will be stored in: $CACHE_DIR"

# Optional: Set offline mode
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
