#!/usr/bin/env python3
"""
GRPO Training Pipeline for Logical Reasoning with Multi-GPU Support
Supports 2-4 H100 GPUs with configurable settings
"""

import argparse
import json
import os
import re
import torch
import torch.distributed as dist
import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import warnings

# Import from existing modules - Fix paths
from setup.setup_models import setup_qwen3, setup_llama_lora
from reward.reward import LogicalReasoningReward
from verification.prover9_integration import verify_reasoning_with_prover9, test_prover9_installation
from train_sft import main as train_sft_main

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

def setup_distributed_training(num_gpus=None):
    """Setup distributed training environment
    
    Args:
        num_gpus: Number of GPUs to use (None = use all available)
    
    Returns:
        rank, world_size, device
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    # Set CUDA_VISIBLE_DEVICES if limiting GPUs
    if num_gpus < torch.cuda.device_count():
        visible_devices = ','.join(str(i) for i in range(num_gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
        print(f"🎯 Limited to {num_gpus} GPUs: {visible_devices}")
    
    # Initialize distributed training if not already done
    if not dist.is_initialized():
        # Set default environment variables for distributed training
        if 'RANK' not in os.environ:
            os.environ['RANK'] = '0'
        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = str(num_gpus)
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = '0'
        
        # For multi-node training (optional)
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
    
    # Get device info
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    return local_rank, world_size, device

def setup_chat_template(tokenizer):
    """Setup chat template following pipeline patterns"""
    reasoning_start = "<initial_reasoning>"
    reasoning_end = "</initial_reasoning>"
    steps_start = "<steps>"
    steps_end = "</steps>"
    answer_start = "<answer>"
    answer_end = "</answer>"

    system_prompt = f"""You are an expert in logical reasoning. Analyze the given problem step by step. First, provide your initial reasoning between {reasoning_start} and {reasoning_end}. Then, provide your logical reasoning steps between {steps_start} and {steps_end}. Finally, provide your answer (A, B, or C) between {answer_start} and {answer_end}."""

    chat_template = \
        "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
        "{% else %}"\
        "{{ system_prompt + eos_token }}"\
        "{% set loop_messages = messages %}"\
        "{% endif %}"\
        "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
        "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
        "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
        "{% endfor %}"\
        "{% if add_generation_prompt %}{{ reasoning_start }}"\
        "{% endif %}"

    chat_template = chat_template\
        .replace("system_prompt", f"'{system_prompt}'")\
        .replace("reasoning_start", f"'{reasoning_start}'")
    
    tokenizer.chat_template = chat_template
    return tokenizer, system_prompt

def load_grpo_dataset(dataset_path, max_examples=None):
    """Load GRPO dataset"""
    print(f"📂 Loading GRPO dataset from {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if max_examples:
        data = data[:max_examples]
        print(f"📊 Limited to {max_examples} examples")
    
    print(f"📊 Loaded {len(data)} training examples")
    return data

def format_grpo_dataset(data, system_prompt, tokenizer):
    """Format dataset for GRPO training"""
    formatted_data = []
    for item in data:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["question"]}
        ]
        
        # Create prompt using chat template
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # GRPO expects these specific fields
        formatted_data.append({
            "prompt": prompt,
            "expected_answer": item["answer"],  
        })
    
    return Dataset.from_list(formatted_data)

def parse_qwen_solution(qwen_output):
    """Parse Qwen's solution to extract reasoning and answer"""
    
    # Extract answer
    answer_match = re.search(r'<answer>(.*?)</answer>', qwen_output, re.DOTALL | re.IGNORECASE)
    answer = answer_match.group(1).strip() if answer_match else ""
    
    # Extract steps
    steps_match = re.search(r'<steps>(.*?)</steps>', qwen_output, re.DOTALL | re.IGNORECASE)
    reasoning = steps_match.group(1).strip() if steps_match else ""
    
    # Parse reasoning into individual statements
    reasoning_statements = []
    if reasoning:
        sentences = re.split(r'\.(?=\s*[A-Z]|Therefore|So |Thus |Hence )', reasoning)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                sentence = re.sub(r'^\d+\.\s*', '', sentence)
                sentence = re.sub(r'^-\s*', '', sentence)
                sentence = sentence.strip(' .,')
                
                if sentence:
                    reasoning_statements.append(sentence)
        
        if len(reasoning_statements) < 2:
            lines = reasoning.split('\n')
            reasoning_statements = []
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:
                    line = re.sub(r'^\d+\.\s*', '', line)
                    line = re.sub(r'^-\s*', '', line)
                    reasoning_statements.append(line)
    
    return reasoning_statements, answer

def convert_reasoning_to_fol(reasoning_statements, llama_model, llama_tokenizer):
    """Convert reasoning statements to FOL using Llama"""

    improved_system_prompt = """You are an expert at converting natural language statements into First-Order Logic (FOL). Follow these strict rules:

SYMBOLS: Use only ∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (if and only if)

NAMING RULES:
- All names/entities must be lowercase (natalie, john, stone)
- Use simple, clear predicate names (Weak(x), Resilient(x), Fearless(x))
- Be consistent with predicate names throughout

SIMPLICITY RULES:
- Keep expressions as simple as possible
- Avoid unnecessary quantifiers when dealing with specific individuals
- Use direct predicates: Weak(natalie) instead of exists x (Weakness(x) & Has(natalie, x))

VARIABLE RULES:
- Never use unbound variables
- If you use ∃x or ∀x, make sure x appears in the formula
- For specific people, use their name directly as a constant

Start your answer with '𝜙=' followed by the FOL formula. Do not include any other text."""

    def translate_nl_to_fol(text):
        def formatting_func(text):
            return llama_tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": improved_system_prompt
                    },
                    {"role": "user", "content": text},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        
        prompt = formatting_func(text)
        inputs = llama_tokenizer(prompt, return_tensors="pt", padding=True)
        
        device = next(llama_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = llama_model.generate(
                **inputs, 
                max_new_tokens=300, 
                temperature=0.1, 
                do_sample=True
            )
        
        result = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
    
    fol_statements = []
    
    for statement in reasoning_statements:
        try:
            fol_result = translate_nl_to_fol(statement)
            if "𝜙=" in fol_result:
                fol_formula = fol_result.split("𝜙=")[-1].strip()
            else:
                fol_formula = fol_result.strip()
            
            fol_statements.append((statement, fol_formula))
            
        except Exception as e:
            fol_statements.append((statement, f"Error: {e}"))
    
    return fol_statements

def setup_reward_functions(tokenizer, accelerator, device=None):
    """Setup reward functions with full pipeline integration
    
    Args:
        tokenizer: Tokenizer for chat template
        accelerator: Accelerator for distributed training
        device: Device to load Llama model on
    """
    print("🔧 Loading Llama model for NL→FOL conversion...")
    
    # Load Llama model for NL→FOL conversion
    try:
        # For distributed training, load on specific device
        llama_model, llama_tokenizer = setup_llama_lora(device=device)
        print("✅ Llama model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load Llama model: {e}")
        raise
    
    # Initialize reward calculator
    reward_calculator = LogicalReasoningReward()
    print("✅ Reward calculator initialized")
    
    def compute_full_pipeline_reward(prompts, completions, expected_answer, **kwargs):
        """Compute rewards using the full pipeline with NL→FOL and Prover9"""
        scores = []
        
        for i, completion in enumerate(completions):
            try:
                response = completion
                expected = expected_answer[i] if isinstance(expected_answer, list) else expected_answer
                
                # Parse Qwen's solution
                reasoning_statements, predicted_answer = parse_qwen_solution(response)
                
                if not reasoning_statements:
                    # No reasoning found - assign penalty
                    scores.append(-1.0)
                    continue
                
                # Convert to FOL using Llama
                fol_statements = convert_reasoning_to_fol(
                    reasoning_statements, llama_model, llama_tokenizer
                )
                
                # Verify with Prover9
                verification_result = verify_reasoning_with_prover9(
                    fol_statements, 
                    f"GRPO_Training_Example_{i}"
                )
                
                # Calculate composite reward
                reward_components = reward_calculator.calculate_composite_reward(
                    response=response,
                    expected_answer=expected,
                    prover9_result=verification_result,
                    reasoning_steps=reasoning_statements
                )

                # Log only from main process
                if i < 5 and accelerator.is_main_process:
                    wandb.log({
                        f"reward/answer_correctness": reward_components.answer_correctness,
                        f"reward/logical_validity": reward_components.logical_validity,
                        f"reward/format_compliance": reward_components.format_compliance,
                        f"reward/prover9_valid": verification_result.get('valid', False),
                    })
                
                # Scale reward for GRPO (typically 0-10 range works well)
                scaled_reward = reward_components.total_reward * 10.0
                scores.append(scaled_reward)
                
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"⚠️ Error in reward computation: {e}")
                scores.append(-1.0)  # Penalty for failed processing
        
        return scores
    
    # Return single comprehensive reward function
    return [compute_full_pipeline_reward]

def setup_model_for_grpo(model_path=None, max_seq_length=3000, lora_rank=32, device=None):
    """Setup model for GRPO training with multi-GPU support
    
    Args:
        model_path: Path to pre-trained model
        max_seq_length: Maximum sequence length
        lora_rank: LoRA rank for parameter efficiency
        device: Device to load model on (for distributed training)
    """
    print("🚀 Setting up model for GRPO training...")
    
    if model_path and os.path.exists(model_path):
        print(f"📂 Loading pre-trained model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Don't use device_map="auto" for distributed training
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        if device is not None:
            model = model.to(device)
    else:
        print("🔧 Loading base model for GRPO training...")
        model_name = "Qwen/Qwen3-8B"

        # Only use quantization in single-GPU mode
        use_quantization = device is None or torch.cuda.device_count() == 1
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model without device_map for distributed training
        if use_quantization:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            except:
                bnb_config = None
        else:
            bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )
        
        if device is not None:
            model = model.to(device)

        # Apply LoRA for training using peft
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def train_grpo_model_distributed(model, tokenizer, dataset, reward_functions, args, device):
    """Train GRPO model with multi-GPU support
    
    Note: VLLM with multi-GPU in GRPO is complex. We'll use standard generation instead.
    """
    print("🎯 Starting GRPO training with multi-GPU support...")
    
    # Setup accelerator with proper distributed config
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grpo_gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
        mixed_precision="bf16" if args.use_mixed_precision else None,
    )
    
    # Create output directory
    os.makedirs(args.grpo_output_dir, exist_ok=True)
    
    max_prompt_length = 1000
    max_completion_length = args.max_seq_length - max_prompt_length

    # For multi-GPU, we need to use standard generation instead of VLLM
    # VLLM requires complex setup for multi-GPU that conflicts with TRL
    
    # Training configuration
    training_args = GRPOConfig(
        # Generation parameters (no VLLM for multi-GPU)
        temperature=args.temperature,
        top_p=0.95,
        top_k=50,
        
        # Training parameters
        learning_rate=args.grpo_learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_torch" if not args.use_8bit_optimizer else "adamw_8bit",
        
        # Batch and steps - scale by world size
        per_device_train_batch_size=args.grpo_batch_size,
        gradient_accumulation_steps=args.grpo_gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_steps=args.grpo_max_steps,
        
        # Generation parameters
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        
        # Logging and saving
        logging_steps=args.grpo_logging_steps,
        save_steps=args.grpo_save_steps,
        output_dir=args.grpo_output_dir,
        report_to="wandb" if accelerator.is_main_process else None,
        save_total_limit=2,
        
        # Distributed training
        ddp_find_unused_parameters=True,
        dataloader_num_workers=2,
        
        # Memory optimization
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.use_mixed_precision,
    )
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=dataset,
        accelerator=accelerator,
    )

    # Train
    trainer.train()

    # Save final model (only from main process)
    if accelerator.is_main_process:
        final_dir = os.path.join(args.grpo_output_dir, "final")
        accelerator.unwrap_model(model).save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"✅ Model saved to {final_dir}")
    
    accelerator.wait_for_everyone()
    
    return os.path.join(args.grpo_output_dir, "final")

def main():
    """Main training pipeline with multi-GPU support"""
    parser = argparse.ArgumentParser(description="GRPO Training Pipeline for Logical Reasoning")
    
    # Multi-GPU settings
    parser.add_argument("--num_gpus", type=int, default=None,
                      help="Number of GPUs to use (default: all available)")
    parser.add_argument("--use_fsdp", action="store_true",
                      help="Use Fully Sharded Data Parallel (better for large models)")
    
    # Pipeline control
    parser.add_argument("--run_sft", action="store_true",
                      help="Run SFT pre-training first")
    parser.add_argument("--sft_model_path", type=str, default=None,
                      help="Path to pre-trained SFT model (skips SFT if provided)")
    
    # Dataset settings
    parser.add_argument("--grpo_dataset", type=str, default="dataset/proverqa_simplified.json",
                      help="Path to GRPO dataset")
    parser.add_argument("--sft_dataset", type=str, default="dataset/sft/sft_with_reasoning.json",
                      help="Path to SFT dataset")
    parser.add_argument("--max_examples", type=int, default=None,
                      help="Maximum number of examples to use")
    
    # Model settings
    parser.add_argument("--max_seq_length", type=int, default=3000,
                      help="Maximum sequence length")
    parser.add_argument("--lora_rank", type=int, default=32,
                      help="LoRA rank")
    
    # SFT parameters
    parser.add_argument("--sft_epochs", type=int, default=1,
                      help="Number of SFT epochs")
    parser.add_argument("--sft_steps", type=int, default=100,
                      help="Number of SFT steps")
    parser.add_argument("--sft_output_dir", type=str, default="./sft_models",
                      help="SFT output directory")
    
    # GRPO parameters
    parser.add_argument("--grpo_max_steps", type=int, default=1250,
                      help="Maximum GRPO training steps")
    parser.add_argument("--grpo_batch_size", type=int, default=4,
                      help="Per-device batch size for GRPO")
    parser.add_argument("--grpo_gradient_accumulation_steps", type=int, default=3,
                      help="Gradient accumulation steps")
    parser.add_argument("--grpo_learning_rate", type=float, default=5e-6,
                      help="GRPO learning rate")
    parser.add_argument("--num_generations", type=int, default=4,
                      help="Number of generations per prompt")
    parser.add_argument("--temperature", type=float, default=1.0,
                      help="Generation temperature")
    parser.add_argument("--grpo_logging_steps", type=int, default=10,
                      help="Logging steps")
    parser.add_argument("--grpo_save_steps", type=int, default=100,
                      help="Save steps")
    parser.add_argument("--grpo_output_dir", type=str, default="./grpo_models",
                      help="GRPO output directory")
    
    # Optimization settings
    parser.add_argument("--gradient_checkpointing", action="store_true",
                      help="Use gradient checkpointing to save memory")
    parser.add_argument("--use_mixed_precision", action="store_true",
                      help="Use mixed precision training (bf16)")
    parser.add_argument("--use_8bit_optimizer", action="store_true",
                      help="Use 8-bit optimizer to save memory")
    
    # Random seed
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    print("🧠 GRPO Training Pipeline for Logical Reasoning")
    print("=" * 60)
    
    # Setup distributed training
    local_rank, world_size, device = setup_distributed_training(args.num_gpus)
    
    print(f"🌐 Distributed Training Setup:")
    print(f"  World Size: {world_size}")
    print(f"  Local Rank: {local_rank}")
    print(f"  Device: {device}")
    print()
    
    # Test Prover9 installation before starting
    if local_rank == 0:  # Only test from main process
        print("🔍 Testing Prover9 installation...")
        if not test_prover9_installation():
            print("❌ Prover9 not available - training cannot continue")
            print("Please install Prover9 to use logical verification rewards")
            return
        print("✅ Prover9 ready")

    # Initialize wandb only on main process
    if local_rank == 0:
        print("🔧 Initializing wandb...")
        wandb.init(
            project="logical-reasoning-grpo",
            name=f"grpo-training-{world_size}gpu",
            config=vars(args)
        )
        print("✅ Wandb initialized")
    
    sft_model_path = args.sft_model_path
    
    # Run SFT training if requested and no pre-trained model provided
    if args.run_sft and not sft_model_path and local_rank == 0:
        print("🚀 Starting SFT pre-training phase...")
        
        # Create SFT args namespace
        sft_args = argparse.Namespace(
            dataset=args.sft_dataset,
            max_examples=args.max_examples,
            max_seq_length=args.max_seq_length,
            lora_rank=args.lora_rank,
            epochs=args.sft_epochs,
            max_steps=args.sft_steps,
            batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            warmup_steps=10,
            logging_steps=5,
            save_steps=50,
            output_dir=args.sft_output_dir
        )
        
        sft_model_path = train_sft_main(sft_args)
        print(f"✅ SFT pre-training completed: {sft_model_path}")
    
    # Synchronize all processes
    if world_size > 1:
        torch.distributed.barrier()
    
    # Load GRPO dataset
    if local_rank == 0:
        grpo_data = load_grpo_dataset(args.grpo_dataset, args.max_examples)
    else:
        grpo_data = None
    
    # Setup model for GRPO (each process loads its own)
    model, tokenizer = setup_model_for_grpo(
        sft_model_path, 
        args.max_seq_length, 
        args.lora_rank,
        device=device if world_size > 1 else None
    )
    
    # Setup chat template
    tokenizer, system_prompt = setup_chat_template(tokenizer)
    
    # Format GRPO dataset (only on main process, then broadcast)
    if local_rank == 0:
        print("📝 Formatting dataset for GRPO training...")
        grpo_dataset = format_grpo_dataset(grpo_data, system_prompt, tokenizer)
        print(f"📊 Formatted {len(grpo_dataset)} examples")
    else:
        grpo_dataset = None
    
    # For multi-GPU, the dataset will be automatically sharded by the trainer
    if world_size > 1:
        # Broadcast dataset info
        if local_rank == 0:
            dataset_size = len(grpo_dataset)
        else:
            dataset_size = None
        
        if world_size > 1:
            dataset_size = torch.tensor([dataset_size if dataset_size else 0], device=device)
            dist.broadcast(dataset_size, src=0)
            dataset_size = dataset_size.item()
        
        # Each process needs the full dataset (trainer will shard it)
        if local_rank != 0:
            # Re-load and format dataset on other processes
            grpo_data = load_grpo_dataset(args.grpo_dataset, args.max_examples)
            grpo_dataset = format_grpo_dataset(grpo_data, system_prompt, tokenizer)

    # Setup reward functions with full pipeline
    print(f"🎯 [Rank {local_rank}] Setting up reward functions...")
    
    # Create a temporary accelerator for reward setup
    temp_accelerator = Accelerator()
    reward_functions = setup_reward_functions(
        tokenizer, 
        temp_accelerator,
        device=device if world_size > 1 else None
    )
    print(f"✅ [Rank {local_rank}] Configured reward functions")
    
    # Train GRPO model
    final_model_path = train_grpo_model_distributed(
        model, tokenizer, grpo_dataset, reward_functions, args, device
    )
    
    if local_rank == 0:
        print(f"\n🎉 Training pipeline completed successfully!")
        print(f"📁 Final model saved to: {final_model_path}")
        print(f"💡 To use this model, load it with:")
        print("    from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"    model = AutoModelForCausalLM.from_pretrained('{final_model_path}')")
        print(f"    tokenizer = AutoTokenizer.from_pretrained('{final_model_path}')")
        
        wandb.finish()

    return final_model_path

if __name__ == "__main__":
    main()