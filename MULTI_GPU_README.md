# Multi-GPU GRPO Training Setup

## 🚀 Quick Start

### Prerequisites
1. Install required packages:
```bash
pip install torch transformers accelerate trl wandb peft datasets bitsandbytes
```

2. Install Prover9 (for logical verification):
```bash
# Ubuntu/Debian
sudo apt-get install prover9

# Or build from source
wget https://www.cs.unm.edu/~mccune/prover9/download/LADR-2009-11A.tar.gz
tar xzf LADR-2009-11A.tar.gz
cd LADR-2009-11A
make all
sudo make install

# or use brew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install prover9
```

3. Set up Hugging Face authentication:
```bash
huggingface-cli login
```

### File Structure
```
project/
├── setup/
│   ├── setup_models.py      # Fixed model setup with multi-GPU support
│   └── setup_cache.py        # Cache directory management
├── reward/
│   └── reward.py            # Reward calculation functions
├── verification/
│   └── prover9_integration.py  # Prover9 verification
├── train.py                 # Main GRPO training script (fixed)
├── train_sft.py            # SFT pre-training script
├── launch_training.sh      # Launch script for multi-GPU
├── gpu_configs.py          # GPU configuration templates
├── test_multi_gpu.py       # Test script for setup verification
└── dataset/
    ├── proverqa_simplified.json
    └── sft/
        └── sft_with_reasoning.json
```

## 🔧 Testing Your Setup

Before training, test your multi-GPU setup:

### Test single GPU:
```bash
python test_multi_gpu.py
```

### Test 2 GPUs:
```bash
torchrun --nproc_per_node=2 test_multi_gpu.py
```

### Test 3 GPUs:
```bash
torchrun --nproc_per_node=3 test_multi_gpu.py
```

### Test 4 GPUs:
```bash
torchrun --nproc_per_node=4 test_multi_gpu.py
```

## 🎯 Training Commands

### Using the Launch Script (Recommended)

The launch script automatically handles distributed training setup:

```bash
# Make the script executable
chmod +x launch_training.sh

# Train with 2 GPUs (default)
./launch_training.sh 2

# Train with 3 GPUs
./launch_training.sh 3

# Train with 4 GPUs
./launch_training.sh 4

# With SFT pre-training
./launch_training.sh 2 --run_sft

# With custom settings
./launch_training.sh 2 --grpo_batch_size 6 --use_mixed_precision
```

### Direct Commands

If you prefer to run directly:

#### Single GPU:
```bash
python train.py --grpo_batch_size 4 --use_mixed_precision
```

#### 2 GPUs:
```bash
torchrun --standalone --nproc_per_node=2 train.py \
    --num_gpus 2 \
    --grpo_batch_size 4 \
    --use_mixed_precision
```

#### 3 GPUs:
```bash
torchrun --standalone --nproc_per_node=3 train.py \
    --num_gpus 3 \
    --grpo_batch_size 3 \
    --use_mixed_precision
```

#### 4 GPUs:
```bash
torchrun --standalone --nproc_per_node=4 train.py \
    --num_gpus 4 \
    --grpo_batch_size 4 \
    --grpo_gradient_accumulation_steps 1 \
    --use_mixed_precision
```

## 📊 Recommended Configurations

Use the configuration generator to get optimal settings:

```bash
# Get config for 2 GPUs
python gpu_configs.py 2

# Get config for 3 GPUs
python gpu_configs.py 3

# Get config for 4 GPUs
python gpu_configs.py 4
```

### Key Parameters by GPU Count:

| GPUs | Batch Size | Grad Accum | Effective Batch | Max Steps | LoRA Rank |
|------|------------|------------|-----------------|-----------|-----------|
| 2    | 4          | 2          | 16              | 1250      | 32        |
| 3    | 3          | 2          | 18              | 1000      | 32        |
| 4    | 4          | 1          | 16              | 800       | 64        |

## 💡 Memory Optimization

If you encounter OOM errors:

1. **Enable gradient checkpointing:**
```bash
./launch_training.sh 2 --gradient_checkpointing
```

2. **Use 8-bit optimizer:**
```bash
./launch_training.sh 2 --use_8bit_optimizer
```

3. **Reduce batch size:**
```bash
./launch_training.sh 2 --grpo_batch_size 2 --grpo_gradient_accumulation_steps 4
```

4. **Reduce sequence length:**
```bash
./launch_training.sh 2 --max_seq_length 2048
```

5. **Use memory-optimized config:**
```bash
./launch_training.sh 2 \
    --grpo_batch_size 2 \
    --grpo_gradient_accumulation_steps 4 \
    --max_seq_length 2048 \
    --lora_rank 16 \
    --gradient_checkpointing \
    --use_8bit_optimizer
```

## 🐛 Troubleshooting

### Issue: "NCCL error"
**Solution:** Ensure all GPUs are visible and NCCL backend is properly installed:
```bash
# Check GPU visibility
nvidia-smi

# Set NCCL debug for more info
export NCCL_DEBUG=INFO
```

### Issue: "Prover9 not found"
**Solution:** Install Prover9 and ensure it's in PATH:
```bash
which prover9  # Should show the path
```

### Issue: "CUDA out of memory"
**Solution:** Use memory optimization options above or reduce the number of generations:
```bash
./launch_training.sh 2 --num_generations 2
```

### Issue: "Different processes have different model weights"
**Solution:** This is expected with LoRA + multi-GPU. The trainer handles synchronization.

### Issue: "wandb not logging"
**Solution:** Only the main process logs to wandb. Check if you're logged in:
```bash
wandb login
```

## 📈 Monitoring Training

1. **WandB Dashboard:** Training metrics are logged to Weights & Biases
2. **Console Output:** Progress is printed to console
3. **Checkpoints:** Models are saved periodically to `--grpo_output_dir`

## 🔄 Resuming Training

To resume from a checkpoint:
```bash
./launch_training.sh 2 --sft_model_path ./grpo_models/checkpoint-500
```

## 🎓 Full Pipeline Example

Complete training pipeline with SFT pre-training and GRPO:

```bash
# Step 1: Run SFT pre-training (single GPU is fine)
python train_sft.py --epochs 3 --output_dir ./sft_models

# Step 2: Run GRPO training with pre-trained model
./launch_training.sh 4 \
    --sft_model_path ./sft_models/final \
    --grpo_max_steps 1000 \
    --use_mixed_precision \
    --grpo_output_dir ./grpo_models_final
```

## 📝 Notes

- **VLLM Limitation:** Multi-GPU GRPO uses standard generation instead of VLLM due to compatibility issues
- **Data Sharding:** The dataset is automatically sharded across GPUs
- **Synchronization:** Model weights are synchronized across GPUs at each step
- **Effective Batch Size:** Total batch = per_gpu_batch × num_gpus × gradient_accumulation_steps

## 🚨 Important Warnings

1. **Do NOT use `device_map="auto"`** in multi-GPU mode
2. **Do NOT manually set CUDA_VISIBLE_DEVICES** when using the launch script
3. **Ensure all GPUs have the same model/memory** for best performance
4. **Kill zombie processes** between runs: `pkill -f train.py`

## 📧 Support

For issues or questions:
1. Check the test script output: `torchrun --nproc_per_node=N test_multi_gpu.py`
2. Enable debug logging: `export TORCH_DISTRIBUTED_DEBUG=DETAIL`
3. Check GPU utilization: `nvidia-smi -l 1`