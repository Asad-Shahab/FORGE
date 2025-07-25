# Logical Reasoning Pipeline with Reinforcement Learning

A comprehensive pipeline for training and evaluating logical reasoning capabilities in Large Language Models using First-Order Logic verification and reinforcement learning techniques.

## 🚀 Setup

### Prerequisites

- **Python 3.8+**
- **CUDA-compatible GPU** (recommended: 16GB+ VRAM)
- **Conda or virtual environment**
- **Git**

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd logical-reasoning-pipeline

conda create --name logic python=3.11 -y
conda activate logic

# Install vLLM first (this will install PyTorch with proper NCCL setup)
pip install vllm

# Install remaining packages
pip install accelerate transformers

# Configure Accelerate (for multi-GPU training)
accelerate config

# Test everything
python -c "
import torch
import accelerate
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
print('Accelerate version:', accelerate.__version__)
print('vLLM import test...', end=' ')
import vllm
print('✓')
"
```

### 2. HuggingFace Authentication

```bash
# Login to HuggingFace (required for model access)
pip install -U "huggingface_hub[cli]"
huggingface-cli login
# Enter your HuggingFace token when prompted
```

### 3. Cache Configuration *(Optional - for university/cluster environments)*

If you're running on a university cluster or environment with restricted permissions:

```bash
# Setup local cache directories (optional)
python -c "from setup.cache_manager import setup_cache_directories; setup_cache_directories()"
```

For manual cache setup:
```bash
export HF_HOME="./cache/huggingface"
export TRANSFORMERS_CACHE="./cache/transformers" 
mkdir -p ./cache/huggingface ./cache/transformers
```

### 4. Verify Installation

Test each component to ensure proper setup:

#### Test Prover9 Installation
```bash
python -c "from verification.prover9_integration import test_prover9_installation; print('✅ Prover9 OK' if test_prover9_installation() else '❌ Prover9 FAILED')"
```

#### Test Model Loading
```bash
python -c "
from setup.setup_models import setup_qwen3, setup_llama_lora
print('Testing model loading...')
try:
    qwen_model, qwen_tokenizer = setup_qwen3()
    print('✅ Qwen3-8B loaded successfully')
    llama_model, llama_tokenizer = setup_llama_lora() 
    print('✅ Llama NL-to-FOL loaded successfully')
    print('🎉 All models ready!')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"
```

#### Test Reward System
```bash
python -c "
from reward.reward import LogicalReasoningReward
reward_calc = LogicalReasoningReward()
print('✅ Reward system loaded successfully')
print(f'Weights: {reward_calc.weights}')
"
```

### 5. Run the Pipeline

```bash
# Execute the complete logical reasoning pipeline
python pipeline_with_rewards.py
```

### 6. Train with Accelerate

```bash
accelerate config      # set number of GPUs (e.g. 4 H100s)
accelerate launch train.py [args]
```

### Expected Directory Structure

After successful setup, your directory should look like:

```
logical-reasoning-training/
├── pipeline_with_rewards.py          # Main pipeline
├── pipeline_dev.py                   # Pipeline on dev dataset
├── train.py                          # Main GRPO Training file
├── dataset/
│   ├── dev/
│   ├── sft/
│   ├── __init__.py
│   ├── dev_split.py
│   ├── proverqa_simplified.json      # Processed dataset
│   └── proverqa_processor.py         # Dataset processing
├── reward/
│   ├── __init__.py
│   └── reward.py                     # Reward calculation
├── setup/
│   ├── __init__.py
│   ├── setup_models.py               # Model loading
│   └── cache_manager.py              # Cache management (optional)
├── verification/
│   ├── __init__.py
│   └── prover9_integration.py        # FOL verification
└── cache/                            # Local cache (optional)
    ├── huggingface/
    └── transformers/
```

## 🔧 Troubleshooting

### Common Issues

**HuggingFace Authentication Error:**
- Ensure you have a valid HuggingFace account and token
- Verify token has access to required models (Qwen, Llama)

**CUDA/GPU Issues:**
- Check GPU availability: `nvidia-smi`
- Ensure CUDA version compatibility with PyTorch
- For CPU-only usage, modify model loading parameters

**Prover9 Installation Issues:**
- **Linux:** `sudo apt-get install prover9`
- **macOS:** `brew install prover9`  
- **Windows:** Use WSL or Docker
- Verify installation: `which prover9`

**Memory Issues:**
- Reduce batch sizes in model configuration
- Use gradient checkpointing for large models
- Consider using smaller model variants

**Dataset Download Issues:**
- Check internet connection
- Verify HuggingFace credentials
- Manually download dataset if automated process fails

### System Requirements

- **Minimum:** 8GB RAM, 4GB GPU VRAM
- **Recommended:** 32GB RAM, 16GB+ GPU VRAM  
- **Storage:** 50GB+ free space for models and cache

### University/Cluster Environments *(Optional)*

If running on university clusters or restricted environments:

1. **Use local cache** (Step 4) to avoid permission issues
2. **Request GPU access** for model training/inference
3. **Check quota limits** for storage and compute
4. **Use job scheduling** systems (SLURM, PBS) if required

## 📋 Verification Checklist

Before proceeding, ensure:

- [ ] All dependencies installed successfully
- [ ] HuggingFace authentication configured
- [ ] Prover9 returns "OK" in verification test
- [ ] Models load without errors
- [ ] Dataset file `proverqa_simplified.json` created
- [ ] Main pipeline runs and displays LLM output + reward breakdown

## 🎯 Next Steps

Once setup is complete, you can:

1. **Explore the pipeline** with different logical reasoning problems
2. **Analyze reward components** and model performance
3. **Generate training data** for reinforcement learning
4. **Implement GRPO training** for model improvement

---

*For detailed usage instructions, training procedures, and advanced configuration, see the additional sections below.*
