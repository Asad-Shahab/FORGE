# Training Pipeline Guide

This guide explains how to use the refactored training pipeline for logical reasoning tasks.

## Overview

The training pipeline has been split into two main components:

1. **`train_sft.py`** - Standalone Supervised Fine-Tuning (SFT) script
2. **`train.py`** - Main GRPO training pipeline with optional SFT integration

Both scripts follow the architectural patterns from `pipeline_dev.py` and integrate with existing modules in `setup/` and `reward/` directories.

## Architecture Changes

### Key Improvements
- **Modular Design**: Separated SFT and GRPO training for flexibility
- **Reusable Components**: Imports from `setup/setup_models.py` and `reward/reward.py`
- **CLI Support**: Both scripts accept command-line arguments
- **Pipeline Integration**: `train.py` can call `train_sft.py` automatically
- **Consistent Patterns**: Follows `pipeline_dev.py` conventions for prompts and chat templates

### Directory Structure
```
logical-reasoning-training/
├── train_sft.py          # Standalone SFT training
├── train.py              # Main GRPO pipeline
├── setup/
│   └── setup_models.py   # Model setup functions (imported)
├── reward/
│   └── reward.py         # Reward calculation (imported)
└── dataset/
    ├── sft/
    │   └── sft_with_reasoning.json    # SFT dataset
    └── proverqa_simplified.json       # GRPO dataset
```

## Usage Examples

### 1. Standalone SFT Training

Train only the SFT model:

```bash
# Basic SFT training
python train_sft.py

# Custom configuration
python train_sft.py \
    --dataset dataset/sft/sft_with_reasoning.json \
    --max_examples 200 \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --output_dir ./my_sft_models
```

### 2. GRPO Training Only

Train GRPO model from base model (no SFT pre-training):

```bash
# Basic GRPO training
python train.py

# Custom configuration
python train.py \
    --grpo_dataset dataset/proverqa_simplified.json \
    --grpo_max_steps 500 \
    --grpo_batch_size 2 \
    --grpo_learning_rate 5e-6 \
    --num_generations 4 \
    --grpo_output_dir ./my_grpo_models
```

### 3. Complete Pipeline (SFT + GRPO)

Run SFT pre-training followed by GRPO training:

```bash
# Full pipeline with automatic SFT
python train.py --run_sft

# Full pipeline with custom settings
python train.py \
    --run_sft \
    --sft_epochs 2 \
    --sft_steps 100 \
    --grpo_max_steps 1000 \
    --max_examples 500 \
    --sft_output_dir ./sft_models \
    --grpo_output_dir ./grpo_models
```

### 4. GRPO with Pre-trained SFT Model

Use an existing SFT model for GRPO training:

```bash
python train.py \
    --sft_model_path ./sft_models/final \
    --grpo_max_steps 1000
```

## Configuration Options

### SFT Training (`train_sft.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | `dataset/sft/sft_with_reasoning.json` | SFT training dataset |
| `--max_examples` | `None` | Limit number of training examples |
| `--epochs` | `3` | Number of training epochs |
| `--max_steps` | `-1` | Max steps (overrides epochs if > 0) |
| `--batch_size` | `1` | Per device batch size |
| `--learning_rate` | `2e-4` | Learning rate |
| `--output_dir` | `./sft_models` | Output directory |

### GRPO Training (`train.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--run_sft` | `False` | Run SFT pre-training first |
| `--sft_model_path` | `None` | Path to pre-trained SFT model |
| `--grpo_dataset` | `dataset/proverqa_simplified.json` | GRPO training dataset |
| `--grpo_max_steps` | `1000` | GRPO training steps |
| `--grpo_batch_size` | `1` | GRPO batch size |
| `--grpo_learning_rate` | `5e-6` | GRPO learning rate |
| `--num_generations` | `4` | Generations per prompt |
| `--temperature` | `1.0` | Sampling temperature |

## Technical Details

### Chat Template Format

Both scripts use the same chat template format following `pipeline_dev.py`:

```
<initial_reasoning>
Your initial analysis here...
</initial_reasoning>

<steps>
Step 1: Logical reasoning step
Step 2: Another reasoning step
...
</steps>

<answer>
A
</answer>
```

### Reward Functions

The GRPO training uses three reward functions from `reward/reward.py`:

1. **Format Compliance** (10% weight) - Checks for proper tag structure
2. **Answer Correctness** (35% weight) - Verifies correct A/B/C answer
3. **Logical Validity** (55% weight) - Evaluates reasoning quality

### Model Setup

Both scripts use `setup_qwen3()` from `setup/setup_models.py` for consistent model initialization with:
- Qwen3-8B base model with 4-bit quantization
- LoRA adaptation with rank 32
- Proper cache and authentication handling

## Output Structure

### SFT Training Output
```
./sft_models/
├── checkpoint-50/     # Intermediate checkpoints
├── checkpoint-100/
└── final/            # Final model
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files...
```

### GRPO Training Output
```
./grpo_models/
├── checkpoint-100/    # Intermediate checkpoints
├── checkpoint-200/
└── final/            # Final model
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files...
```

## Loading Trained Models

After training, load models with:

```python
from unsloth import FastLanguageModel

# Load the final model
model, tokenizer = FastLanguageModel.from_pretrained(
    "./grpo_models/final",  # or "./sft_models/final"
    max_seq_length=2048,
    dtype=None,
)

# Use for inference
model = model.to("cuda")
```

## Error Handling

The scripts include error handling for:
- Missing dataset files
- GPU memory issues
- Model loading failures
- Authentication problems

Check logs for specific error messages and solutions.

## Performance Recommendations

### For SFT Training:
- Use `--max_examples 100-500` for initial experiments
- Increase `--batch_size` if GPU memory allows
- Monitor loss curves to avoid overfitting

### For GRPO Training:
- Start with `--grpo_max_steps 500` for testing
- Use `--num_generations 4` for good reward signal
- Monitor reward components for balanced training

## Troubleshooting

### Common Issues:

1. **GPU Memory Error**: Reduce batch size or sequence length
2. **Authentication Error**: Check HuggingFace token setup
3. **Dataset Loading Error**: Verify dataset file paths
4. **Import Error**: Ensure all dependencies are installed

### Performance Tips:

1. Use gradient accumulation for larger effective batch sizes
2. Monitor reward balance during GRPO training
3. Save checkpoints frequently for long training runs
4. Use `--max_examples` for quick experiments

## Integration with Existing Code

The refactored scripts maintain compatibility with:
- `pipeline_dev.py` prompt formats
- `setup/setup_models.py` model initialization
- `reward/reward.py` reward calculation
- Existing dataset structures

This ensures seamless integration with your existing logical reasoning pipeline.