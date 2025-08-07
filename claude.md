# Logical Reasoning GRPO Training Pipeline - Project Documentation

## 🎯 Project Overview

This project implements a two-stage training pipeline for logical reasoning:
1. **SFT (Supervised Fine-Tuning)**: Initial training on reasoning examples
2. **GRPO (Group Relative Policy Optimization)**: Reinforcement learning with logical verification rewards

The system fine-tunes Qwen3-8B for logical reasoning, uses Llama-3.1-8B for NL→FOL conversion, and validates logic with Prover9.

## 📁 Project Structure

```
logical-reasoning-training/
├── setup/
│   ├── setup_models.py         # Model loading utilities (Qwen3, Llama)
│   └── setup_cache.py          # Cache directory management
├── reward/
│   └── reward.py               # Reward calculation (answer correctness, logical validity)
├── verification/
│   └── prover9_integration.py  # Prover9 FOL verification
├── dataset/
│   ├── proverqa_simplified.json    # GRPO training data
│   └── sft/
│       └── sft_with_reasoning.json # SFT training data
├── train.py                    # Main GRPO training script (multi-GPU)
├── train_sft.py               # SFT pre-training script
├── launch_training.sh         # Multi-GPU launch script
├── gpu_configs.py             # GPU configuration templates
├── test_multi_gpu.py          # Multi-GPU setup verification
└── README.md                  # User documentation
```

## 🔧 Core Components

### 1. **Model Setup (`setup/setup_models.py`)**

**Key Functions:**
- `setup_qwen3(device, use_quantization)`: Loads Qwen3-8B base model
  - Supports 4-bit quantization (single GPU only)
  - Returns model and tokenizer
  
- `setup_llama_lora(device)`: Loads Llama-3.1-8B with NL→FOL LoRA adapter
  - Used for converting reasoning to First-Order Logic
  - Returns model and tokenizer

**Important Variables:**
- `model_name = "Qwen/Qwen3-8B"`: Base model for reasoning
- `lora_weights = "fvossel/Llama-3.1-8B-Instruct-nl-to-fol"`: FOL conversion adapter

### 2. **Reward System (`reward/reward.py`)**

**Class: `LogicalReasoningReward`**

Calculates composite rewards with three components:
```python
weights = {
    'answer_correctness': 0.35,   # 35% - Correct answer (A/B/C)
    'logical_validity': 0.55,     # 55% - Prover9 validation
    'format_compliance': 0.1      # 10% - Output structure
}
```

**Key Methods:**
- `calculate_composite_reward()`: Main reward calculation
- `calculate_answer_correctness()`: Checks if answer matches expected
- `calculate_logical_validity()`: Uses Prover9 verification results
- `calculate_format_compliance()`: Checks for required XML tags

### 3. **FOL Verification (`verification/prover9_integration.py`)**

**Key Functions:**
- `convert_fol_to_prover9(fol_formula)`: Converts FOL symbols to Prover9 syntax
- `verify_reasoning_with_prover9(fol_statements)`: Main verification pipeline
- `parse_prover9_output(output)`: Extracts proof validity

**Conversion Rules:**
- `∀` → `all`
- `∃` → `exists`
- `¬` → `-`
- `∧` → `&`
- `∨` → `|`
- `→` → `->`

### 4. **Training Scripts**

#### **SFT Training (`train_sft.py`)**

**Key Parameters:**
```python
--epochs 3                    # Training epochs
--batch_size 2               # Per-device batch size
--gradient_accumulation_steps 4  # Gradient accumulation
--learning_rate 2e-4         # Learning rate
--max_seq_length 2048        # Max token length
--lora_rank 32              # LoRA adapter rank
```

**Training Flow:**
1. Loads dataset from JSON
2. Formats with chat template
3. Applies LoRA adapters
4. Trains with AdamW optimizer
5. Saves to `./sft_models/final/`

#### **GRPO Training (`train.py`)**

**Key Parameters:**
```python
--num_gpus 2                      # Number of GPUs to use
--sft_model_path ./sft_models/final  # Pre-trained SFT model
--grpo_max_steps 1000            # Maximum training steps
--grpo_batch_size 4              # Per-device batch size
--num_generations 4              # Generations per prompt
--temperature 1.0                # Generation temperature
--grpo_learning_rate 5e-6        # Learning rate
```

**Training Pipeline:**
1. **Generation**: Model generates multiple responses per prompt
2. **Parse**: Extract reasoning steps and answer
3. **Convert**: Llama converts reasoning to FOL
4. **Verify**: Prover9 validates logic
5. **Reward**: Calculate composite reward
6. **Update**: GRPO updates model based on rewards

## 🏗️ Architecture Details

### Chat Template Format

The model expects/generates responses in this format:
```xml
<initial_reasoning>
Initial analysis of the problem...
</initial_reasoning>

<steps>
Step 1: First logical statement
Step 2: Second logical statement
...
Conclusion: Final deduction
</steps>

<answer>
A/B/C
</answer>
```

### Multi-GPU Strategy

- **Distribution**: Uses PyTorch DDP (DistributedDataParallel)
- **Data Sharding**: Dataset automatically sharded across GPUs
- **Synchronization**: Gradients synchronized after each batch
- **Effective Batch Size**: `batch_size * num_gpus * gradient_accumulation`

### Memory Optimizations

1. **Gradient Checkpointing**: `--gradient_checkpointing`
2. **Mixed Precision (BF16)**: `--use_mixed_precision`
3. **8-bit Optimizer**: `--use_8bit_optimizer`
4. **LoRA**: Only trains adapter weights, not full model

## 📊 Configuration Templates

### 2 GPU Configuration
```python
{
    "grpo_batch_size": 4,
    "grpo_gradient_accumulation_steps": 2,
    "effective_batch_size": 16,  # 4 * 2 * 2
    "grpo_max_steps": 1250,
    "lora_rank": 32
}
```

### 3 GPU Configuration
```python
{
    "grpo_batch_size": 3,
    "grpo_gradient_accumulation_steps": 2,
    "effective_batch_size": 18,  # 3 * 3 * 2
    "grpo_max_steps": 1000,
    "lora_rank": 32
}
```

### 4 GPU Configuration
```python
{
    "grpo_batch_size": 4,
    "grpo_gradient_accumulation_steps": 1,
    "effective_batch_size": 16,  # 4 * 4 * 1
    "grpo_max_steps": 800,
    "lora_rank": 64
}
```

## 🔄 Training Workflow

### Complete Pipeline
```bash
# Step 1: SFT Pre-training
python train_sft.py --epochs 3 --output_dir ./sft_models

# Step 2: GRPO Fine-tuning
./launch_training.sh 2 --sft_model_path ./sft_models/final
```

### Key Functions Call Chain

1. **GRPO Generation Phase:**
   ```
   train.py:main()
   → setup_model_for_grpo()
   → format_grpo_dataset()
   → train_grpo_model_distributed()
     → GRPOTrainer.generate()
   ```

2. **Reward Calculation Phase:**
   ```
   compute_full_pipeline_reward()
   → parse_qwen_solution()
   → convert_reasoning_to_fol() [Llama]
   → verify_reasoning_with_prover9()
   → calculate_composite_reward()
   ```

3. **Model Update Phase:**
   ```
   GRPOTrainer.train()
   → compute_rewards()
   → policy_gradient_update()
   → optimizer.step()
   ```

## 🛠️ Common Modifications

### 1. **Change Base Model**
In `setup/setup_models.py`:
```python
model_name = "Qwen/Qwen3-8B"  # Change to any HF model
```

### 2. **Adjust Reward Weights**
In `reward/reward.py`:
```python
self.weights = {
    'answer_correctness': 0.35,  # Modify these
    'logical_validity': 0.55,
    'format_compliance': 0.1
}
```

### 3. **Modify Generation Parameters**
In `train.py`:
```python
temperature=1.0,  # Increase for more diversity
top_p=0.95,      # Nucleus sampling
top_k=50,        # Top-k sampling
```

### 4. **Change LoRA Targets**
In `train.py` and `train_sft.py`:
```python
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    # Add more layers here
]
```

### 5. **Custom Dataset Format**
Dataset JSON structure:
```json
[
  {
    "question": "Logical reasoning question",
    "answer": "A/B/C",
    "initial_reasoning": "Optional reasoning",
    "steps": "Optional step-by-step solution"
  }
]
```

## 🐛 Debugging Tips

### Check GPU Utilization
```bash
nvidia-smi -l 1  # Monitor GPU usage
```

### Enable Debug Logging
```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
```

### Test Components Individually
```python
# Test reward calculation
from reward.reward import LogicalReasoningReward
reward_calc = LogicalReasoningReward()

# Test Prover9
from verification.prover9_integration import test_prover9_installation
test_prover9_installation()
```

### Common Issues

1. **OOM Error**: Reduce batch size or enable gradient checkpointing
2. **NCCL Error**: Check GPU visibility with `nvidia-smi`
3. **Import Error**: Ensure running from project root
4. **Prover9 Not Found**: Add to PATH or install with `apt-get install prover9`

## 📈 Monitoring Metrics

### WandB Tracking
- `reward/answer_correctness`: Should increase over time
- `reward/logical_validity`: Key metric for reasoning quality
- `reward/format_compliance`: Should quickly reach ~1.0
- `loss`: GRPO policy loss, should decrease

### Expected Performance
- **SFT Loss**: Should drop from ~3.0 to ~0.5
- **GRPO Rewards**: Should increase from ~0.0 to ~5.0+
- **Answer Accuracy**: Target 70%+ on validation set

## 🔮 Future Enhancements

Potential improvements to implement:
1. **VLLM Integration**: For faster generation (currently disabled for multi-GPU)
2. **FSDP Support**: For model sharding across GPUs
3. **Curriculum Learning**: Start with easier examples
4. **Reward Shaping**: More sophisticated reward functions
5. **Validation Set**: Add evaluation during training
6. **Model Merging**: Merge LoRA weights back to base model

## 📝 Notes for Claude Code

When making modifications:
1. **Imports**: All imports assume running from project root
2. **Device Handling**: Check for `device` parameter in multi-GPU mode
3. **Accelerator**: Use for distributed operations
4. **Logging**: Only log from main process (`accelerator.is_main_process`)
5. **Saving**: Models save to `{output_dir}/final/`
6. **Testing**: Run `test_multi_gpu.py` after major changes

This codebase is designed for H100 GPUs (80GB) but works on other NVIDIA GPUs with adjustments to batch size and sequence length.