#!/usr/bin/env python3
"""
Standalone cache directory setup for ML workflows
Import this before importing ML libraries to configure cache directories
"""

import os
from pathlib import Path

def setup_cache_directories():
    """Setup all ML-related cache directories to current directory"""
    
    # Get current directory
    current_dir = Path.cwd()
    cache_base = current_dir / "cache"
    
    # Create cache directories
    cache_dirs = {
        "pip": cache_base / "pip",
        "huggingface": cache_base / "huggingface", 
        "transformers": cache_base / "transformers",
        "torch": cache_base / "torch",
        "datasets": cache_base / "datasets",
        "tensorboard": cache_base / "tensorboard",
        "wandb": cache_base / "wandb",
        "nltk": cache_base / "nltk_data",
        "matplotlib": cache_base / "matplotlib",
    }
    
    # Create all cache directories
    for name, path in cache_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created cache directory: {name} -> {path}")
    
    # Set environment variables
    env_vars = {
        # Pip cache
        "PIP_CACHE_DIR": str(cache_dirs["pip"]),
        
        # Hugging Face ecosystem
        "HF_HOME": str(cache_dirs["huggingface"]),
        "HUGGINGFACE_HUB_CACHE": str(cache_dirs["huggingface"] / "hub"),
        "TRANSFORMERS_CACHE": str(cache_dirs["transformers"]),
        "HF_DATASETS_CACHE": str(cache_dirs["datasets"]),
        
        # PyTorch
        "TORCH_HOME": str(cache_dirs["torch"]),
        
        # TensorBoard logs
        "TENSORBOARD_LOG_DIR": str(cache_dirs["tensorboard"]),
        
        # Weights & Biases
        "WANDB_DIR": str(cache_dirs["wandb"]),
        "WANDB_CACHE_DIR": str(cache_dirs["wandb"] / "cache"),
        
        # NLTK data
        "NLTK_DATA": str(cache_dirs["nltk"]),
        
        # Matplotlib
        "MPLCONFIGDIR": str(cache_dirs["matplotlib"]),
        
        # Disable some online features (set to "1" for offline mode)
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
    }
    
    # Set environment variables
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"🔧 Set {var} = {value}")
    
    print(f"\n📁 All caches redirected to: {cache_base}")
    
    return cache_dirs

def check_cache_sizes():
    """Check current cache directory sizes"""
    current_dir = Path.cwd()
    cache_base = current_dir / "cache"
    
    if not cache_base.exists():
        print("❌ Cache directory not found. Run setup first.")
        return
    
    print(f"\n📊 Cache Directory Sizes:")
    print("-" * 40)
    
    total_size = 0
    for item in cache_base.iterdir():
        if item.is_dir():
            size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
            size_mb = size / (1024 * 1024)
            total_size += size
            print(f"{item.name:15} : {size_mb:8.1f} MB")
    
    total_mb = total_size / (1024 * 1024)
    total_gb = total_mb / 1024
    print("-" * 40)
    print(f"{'TOTAL':15} : {total_mb:8.1f} MB ({total_gb:.2f} GB)")

def clear_specific_cache(cache_name):
    """Clear a specific cache directory"""
    import shutil
    current_dir = Path.cwd()
    cache_base = current_dir / "cache"
    cache_path = cache_base / cache_name
    
    if not cache_path.exists():
        print(f"❌ Cache directory '{cache_name}' not found at {cache_path}")
        return
    
    try:
        shutil.rmtree(cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        print(f"🗑️  Cleared cache: {cache_name}")
    except Exception as e:
        print(f"❌ Error clearing cache {cache_name}: {e}")

if __name__ == "__main__":
    print("🗂️  Setting up ML cache directories...")
    setup_cache_directories()
    check_cache_sizes()
    print("\n✅ Cache setup complete!")