# FORGE: FOL-Optimized Reasoning with GRPO Enhancement

A training pipeline for reducing reasoning hallucinations in Large Language Models using First-Order Logic (FOL) verification and reinforcement learning (GRPO).

## Research Overview

LLMs can generate plausible but logically invalid reasoning chains, inserting unsupported steps ("hallucinations"). This project implements a two-stage training approach to address this:

1. **SFT (Supervised Fine-Tuning)**: Train the model to produce structured proofs
2. **GRPO (Group Relative Policy Optimization)**: Reward-guided training using formal verification

### Key Observation

> **Right answers are common; correct reasoning is not.**
>
> Across our experiments, models selected the correct answer ~75% of the time, but only ~40% of generated proofs were logically valid (free of hallucinated steps).

### Preliminary Results (Work in Progress)

- **SFT Stage**: Model successfully learned to produce structured proofs in the required format
- **GRPO Stage**: Showed small improvements (~4%) in reasoning quality on validation sets
- Results are preliminary due to compute constraints

### Known Challenges

1. **NL-to-FOL Conversion**: Translation errors cause false negatives in Prover9 checking
2. **Prover9 Timeouts**: Proofs with more than 20 steps often timeout
3. **Generation Speed**: Long reasoning chains (3000 tokens) slow down GRPO training

### Future Directions

- **Chunked Verification**: Break long proofs into smaller sub-proofs
- **Alternative Provers**: Explore Z3, Lean for faster/more scalable verification
- **Cross-Domain Transfer**: Test if logical proof training improves math and general QA

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Base Model | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Reasoning and proof generation |
| NL-to-FOL | [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) + [LoRA adapter](https://huggingface.co/fvossel/Llama-3.1-8B-Instruct-nl-to-fol) | Convert natural language to First-Order Logic |
| Prover | [Prover9](https://www.cs.unm.edu/~mccune/prover9/) | Formal verification of logical validity |

---

## Installation

### 1. Prover9 Setup

Prover9 is required for logical verification. Install it first:

**Ubuntu/Debian:**
```bash
sudo apt-get install prover9
```

**macOS:**
```bash
brew install prover9
```

**Build from source:**
```bash
wget https://www.cs.unm.edu/~mccune/prover9/download/LADR-2009-11A.tar.gz
tar xzf LADR-2009-11A.tar.gz
cd LADR-2009-11A
make all
sudo make install
```

Verify installation:
```bash
which prover9
```

### 2. Environment Setup

```bash
git clone https://github.com/Asad-Shahab/FORGE.git
cd FORGE

conda create --name logic python=3.11 -y
conda activate logic

pip install -r requirements.txt
```

### 3. HuggingFace Authentication

```bash
huggingface-cli login
```

### 4. Verify Installation

```bash
# Test Prover9
python -c "from verification.prover9_integration import test_prover9_installation; print('Prover9 OK' if test_prover9_installation() else 'Prover9 FAILED')"

# Test model loading
python -c "from setup.setup_models import setup_qwen3; m, t = setup_qwen3(); print('Models OK')"
```

### 5. Cache Configuration (Optional)

For cluster environments with restricted permissions:

```bash
export HF_HOME="./cache/huggingface"
export TRANSFORMERS_CACHE="./cache/transformers"
mkdir -p ./cache/huggingface ./cache/transformers
```

---

## Training

### Quick Start

```bash
# Step 1: SFT pre-training
python train_sft.py --epochs 3 --output_dir ./sft_models

# Step 2: GRPO training
./launch_training.sh 2 --sft_model_path ./sft_models/final
```

### SFT Training Options

```bash
python train_sft.py \
    --dataset dataset/sft/sft_with_reasoning.json \
    --max_examples 200 \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --output_dir ./sft_models
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | `dataset/sft/sft_with_reasoning.json` | SFT training dataset |
| `--max_examples` | `None` | Limit number of training examples |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `1` | Per device batch size |
| `--learning_rate` | `2e-4` | Learning rate |
| `--output_dir` | `./sft_models` | Output directory |

### GRPO Training Options

```bash
python train.py \
    --sft_model_path ./sft_models/final \
    --grpo_dataset dataset/proverqa_simplified.json \
    --grpo_max_steps 500 \
    --grpo_batch_size 2 \
    --num_generations 4 \
    --grpo_learning_rate 5e-6 \
    --grpo_output_dir ./grpo_models
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sft_model_path` | `None` | Path to pre-trained SFT model |
| `--grpo_max_steps` | `1000` | GRPO training steps |
| `--grpo_batch_size` | `1` | Per-GPU batch size |
| `--num_generations` | `4` | Generations per prompt |
| `--temperature` | `1.0` | Sampling temperature |
| `--grpo_learning_rate` | `5e-6` | GRPO learning rate |

---

## Multi-GPU Training

### Using the Launch Script

```bash
chmod +x launch_training.sh

./launch_training.sh 2                              # 2 GPUs
./launch_training.sh 4 --gradient_checkpointing     # 4 GPUs with memory optimization
./launch_training.sh 2 --run_sft                    # With SFT pre-training
```

### Direct Commands

```bash
# 2 GPUs
torchrun --standalone --nproc_per_node=2 train.py \
    --num_gpus 2 \
    --grpo_batch_size 4 \
    --use_mixed_precision

# 4 GPUs
torchrun --standalone --nproc_per_node=4 train.py \
    --num_gpus 4 \
    --grpo_batch_size 4 \
    --grpo_gradient_accumulation_steps 1 \
    --use_mixed_precision
```

### Memory Optimization

If you encounter OOM errors:

```bash
./launch_training.sh 2 \
    --grpo_batch_size 2 \
    --grpo_gradient_accumulation_steps 4 \
    --max_seq_length 2048 \
    --lora_rank 16 \
    --gradient_checkpointing \
    --use_8bit_optimizer
```

---

## Reward Function

The GRPO training uses three reward components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Format Compliance | 10% | Checks for proper XML tag structure |
| Answer Correctness | 35% | Verifies correct A/B/C answer |
| Logical Validity | 55% | Prover9 verification of reasoning |

### Expected Output Format

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

---

## Loading Trained Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./grpo_models/final")
tokenizer = AutoTokenizer.from_pretrained("./grpo_models/final")

model = model.to("cuda")
```

---

## Troubleshooting

**NCCL error:**
```bash
nvidia-smi                    # Check GPU visibility
export NCCL_DEBUG=INFO        # Enable debug logging
```

**Prover9 not found:**
```bash
which prover9                 # Should show the path
```

**CUDA out of memory:**
- Reduce `--grpo_batch_size`
- Enable `--gradient_checkpointing`
- Reduce `--num_generations`
- Use `--use_8bit_optimizer`

**WandB not logging:**
```bash
wandb login
```

### Important Notes

- Do NOT use `device_map="auto"` in multi-GPU mode
- Do NOT manually set `CUDA_VISIBLE_DEVICES` when using the launch script
- Kill zombie processes between runs: `pkill -f train.py`
- Effective batch size = per_gpu_batch x num_gpus x gradient_accumulation_steps

---

## References

This work builds on:

- [LogicInference](https://arxiv.org/abs/2203.15099) - Dataset for teaching logical inference to seq2seq models
- [FOLIO](https://arxiv.org/abs/2209.00840) - Natural language reasoning with first-order logic
- [LINC](https://arxiv.org/abs/2310.15164) - Neurosymbolic approach combining LLMs with FOL provers
- [SatLM](https://arxiv.org/abs/2305.09656) - Satisfiability-aided language models using declarative prompting
- [ProverQA](https://arxiv.org/abs/2502.06563) - Large language models meet symbolic provers for logical reasoning evaluation

