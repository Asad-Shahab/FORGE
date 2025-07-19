#!/usr/bin/env python3
"""
Comprehensive cache management utility for ML workflows
Usage: python cache_manager.py [command]
"""

import os
import sys
import shutil
import argparse
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
        
        # Disable some online features
        "HF_HUB_OFFLINE": "0",  # Set to "1" for fully offline mode
        "TRANSFORMERS_OFFLINE": "0",  # Set to "1" for offline mode
    }
    
    # Set environment variables
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"🔧 Set {var} = {value}")
    
    print(f"\n📁 All caches redirected to: {cache_base}")
    print(f"💾 Total cache directory size will be tracked in: {cache_base}")
    
    return cache_dirs

def check_cache_sizes():
    """Check current cache directory sizes"""
    current_dir = Path.cwd()
    cache_base = current_dir / "cache"
    
    if not cache_base.exists():
        print("❌ Cache directory not found. Run 'python cache_manager.py setup' first.")
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

def generate_bashrc_export():
    """Generate bash export commands for permanent setup"""
    current_dir = Path.cwd()
    cache_base = current_dir / "cache"
    
    bash_exports = f"""#!/bin/bash
# Conda environment setup with cache configuration
# Usage: source conda_setup.sh

# Activate your conda environment
# conda activate your_env_name

# Set cache directories to current project
export PROJECT_DIR="{current_dir}"
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
mkdir -p "$CACHE_DIR/{{pip,huggingface/hub,transformers,datasets,torch,tensorboard,wandb/cache,nltk_data,matplotlib}}"

echo "✅ Cache directories configured for project: $PROJECT_DIR"
echo "📁 All ML caches will be stored in: $CACHE_DIR"

# Optional: Set offline mode
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
"""
    
    # Save to file
    cache_script = current_dir / "conda_setup.sh"
    with open(cache_script, 'w') as f:
        f.write(bash_exports)
    
    # Make executable
    os.chmod(cache_script, 0o755)
    
    print(f"\n📝 Bash export script saved to: {cache_script}")
    print("💡 To make permanent, run: source conda_setup.sh")
    print("💡 Or add contents to ~/.bashrc")

def show_cache_status():
    """Show current cache configuration and status"""
    print("🔍 Current Cache Configuration:")
    print("=" * 50)
    
    cache_vars = [
        "PIP_CACHE_DIR",
        "HF_HOME", 
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "WANDB_DIR",
        "NLTK_DATA",
        "MPLCONFIGDIR"
    ]
    
    for var in cache_vars:
        value = os.environ.get(var, "❌ Not set")
        print(f"{var:20} : {value}")
    
    print("=" * 50)
    
    # Check if our cache directory exists
    current_dir = Path.cwd()
    cache_base = current_dir / "cache"
    
    if cache_base.exists():
        print(f"✅ Local cache directory: {cache_base}")
        check_cache_sizes()
    else:
        print(f"❌ Local cache directory not found: {cache_base}")
        print("💡 Run 'python cache_manager.py setup' to create")

def cleanup_caches():
    """Interactive cache cleanup"""
    current_dir = Path.cwd()
    cache_base = current_dir / "cache"
    
    if not cache_base.exists():
        print("❌ No local cache directory found")
        return
    
    print("🗑️  Cache Cleanup Options:")
    print("1. Clear all caches (keeps directory structure)")
    print("2. Clear specific cache")
    print("3. Show cache sizes only")
    print("4. Cancel")
    
    choice = input("Choose option (1-4): ").strip()
    
    if choice == "1":
        confirm = input("⚠️  This will clear ALL caches. Continue? (y/N): ")
        if confirm.lower() == 'y':
            for item in cache_base.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                    item.mkdir(exist_ok=True)
                    print(f"🗑️  Cleared: {item.name}")
            print("✅ All caches cleared")
    
    elif choice == "2":
        # List available caches
        caches = [d.name for d in cache_base.iterdir() if d.is_dir()]
        if not caches:
            print("❌ No cache directories found")
            return
            
        print("Available caches:")
        for i, cache in enumerate(caches, 1):
            print(f"{i}. {cache}")
        
        try:
            idx = int(input("Choose cache to clear: ")) - 1
            if 0 <= idx < len(caches):
                clear_specific_cache(caches[idx])
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Invalid input")
    
    elif choice == "3":
        check_cache_sizes()
    
    else:
        print("Cancelled")

def monitor_cache_growth():
    """Monitor cache directory growth"""
    import time
    
    current_dir = Path.cwd()
    cache_base = current_dir / "cache"
    
    if not cache_base.exists():
        print("❌ No cache directory found")
        return
    
    print("📊 Cache Growth Monitor (Ctrl+C to stop)")
    print("Checking every 30 seconds...")
    
    try:
        while True:
            print(f"\n⏰ {time.strftime('%H:%M:%S')}")
            check_cache_sizes()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

def main():
    parser = argparse.ArgumentParser(description="ML Cache Management Utility")
    parser.add_argument("command", nargs="?", default="status",
                       choices=["setup", "status", "cleanup", "conda", "monitor"],
                       help="Command to execute")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        print("🛠️  Setting up cache directories...")
        setup_cache_directories()
        generate_bashrc_export()
        print("\n🎯 Next steps:")
        print("1. Source conda_setup.sh for permanent setup: source conda_setup.sh")
        print("2. Or just run your Python scripts - cache will be configured automatically")
        
    elif args.command == "status":
        show_cache_status()
        
    elif args.command == "cleanup":
        cleanup_caches()
        
    elif args.command == "conda":
        generate_bashrc_export()
        
    elif args.command == "monitor":
        monitor_cache_growth()
    
    else:
        print("❌ Unknown command")
        parser.print_help()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments, show status
        show_cache_status()
    else:
        main()