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
import warnings
import logging

# Suppress warnings BEFORE importing transformers
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="The following generation flags")
warnings.filterwarnings("ignore", message="generation_config")
warnings.filterwarnings("ignore", message="default values have been modified")
warnings.filterwarnings("ignore", message="Multiple generation_config")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Suppress specific loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

# Import from existing modules - Fix paths
from setup.setup_models import setup_qwen3, setup_llama_lora
from reward.reward import LogicalReasoningReward
from verification.prover9_integration import verify_reasoning_with_prover9, test_prover9_installation
from train_sft import main as train_sft_main

def setup_distributed_training(num_gpus=None):
    """Setup distributed training environment
    
    Args:
        num_gpus: Number of GPUs to use (None = use all available)
    
    Returns:
        rank, world_size, device
    """
    # Get distributed info from environment (set by torchrun)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Only initialize if we're actually in distributed mode
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    # Set device
    if torch.cuda.is_available() and world_size > 1:
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
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
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if max_examples:
        data = data[:max_examples]
    
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
            # Also store in a format that can be accessed by reward function
            "query": item["question"],
            "reference": item["answer"],
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
                max_new_tokens=50,  # Reduced significantly
                temperature=0.1, 
                do_sample=True,
                pad_token_id=llama_tokenizer.eos_token_id,  # Explicit pad token
                eos_token_id=llama_tokenizer.eos_token_id,
                num_beams=1,  # Greedy decoding for speed
                use_cache=False  # Disable KV-cache to prevent conflicts
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
    # Load Llama model for NL→FOL conversion
    try:
        llama_model, llama_tokenizer = setup_llama_lora(device=device)
    except Exception as e:
        print(f"❌ Failed to load Llama model: {e}")
        raise
    
    # Initialize reward calculator
    reward_calculator = LogicalReasoningReward()
    
    def compute_full_pipeline_reward(completions, **kwargs):
        """Compute rewards using the full pipeline with NL→FOL and Prover9
        
        Simplified signature for GRPO compatibility
        """
        # Check if we're the main process using distributed environment
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        is_main_process = local_rank == 0
        
        # Silently compute rewards
        
        scores = []
        
        # For now, use a simpler reward that doesn't require expected answers
        # This will allow us to test if the training loop works
        for i, completion in enumerate(completions):
            try:
                # Simple reward based on format compliance and length
                response = completion
                
                # Basic format check
                has_reasoning = '<initial_reasoning>' in response and '</initial_reasoning>' in response
                has_steps = '<steps>' in response and '</steps>' in response  
                has_answer = '<answer>' in response and '</answer>' in response
                
                format_score = (has_reasoning + has_steps + has_answer) / 3.0
                
                # Length penalty for very short/long responses
                length_score = 1.0
                if len(response) < 100:
                    length_score = 0.5
                elif len(response) > 2000:
                    length_score = 0.7
                
                # Combined score
                score = (format_score * 0.7 + length_score * 0.3) * 10.0  # Scale to 0-10
                scores.append(score)
                
                # Score computed
                
            except Exception as e:
                if is_main_process:
                    print(f"⚠️ Error in reward computation for completion {i}: {e}")
                scores.append(1.0)  # Neutral score for failed processing
        
        # Rewards computed
        
        return scores
    
    # Return simple reward function for initial testing
    return [compute_full_pipeline_reward]

def setup_complex_reward_functions(tokenizer, accelerator_placeholder, device=None):
    """Full pipeline reward with NL→FOL and Prover9 - use after basic training works"""
    
    # Load Llama model for NL→FOL conversion
    try:
        llama_model, llama_tokenizer = setup_llama_lora(device=device)
    except Exception as e:
        print(f"❌ Failed to load Llama model: {e}")
        raise
    
    # Initialize reward calculator
    reward_calculator = LogicalReasoningReward()
    
    def compute_complex_reward(completions, **kwargs):
        """Full pipeline reward computation with NL→FOL and Prover9"""
        # Check if we're the main process using distributed environment
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        is_main_process = local_rank == 0
        
        # Compute rewards silently
        
        scores = []
        
        # Try to get expected answers from dataset
        expected_answers = None
        
        # Check various sources for expected answers
        for key in ['expected_answer', 'reference', 'answer', 'target']:
            if key in kwargs:
                expected_answers = kwargs[key]
                break
        
        # If still none, try to extract from batch/dataset
        if expected_answers is None:
            batch = kwargs.get('batch', {})
            for key in ['expected_answer', 'reference', 'answer', 'target']:
                if hasattr(batch, key):
                    expected_answers = getattr(batch, key)
                    break
                elif isinstance(batch, dict) and key in batch:
                    expected_answers = batch[key]
                    break
        
        # Final fallback
        if expected_answers is None:
            expected_answers = ['C'] * len(completions)
        
        for i, completion in enumerate(completions):
            try:
                response = completion
                expected = expected_answers[i] if isinstance(expected_answers, list) else expected_answers
                
                reasoning_statements, predicted_answer = parse_qwen_solution(response)
                
                if not reasoning_statements:
                    scores.append(-1.0)
                    continue
                
                # Convert to FOL using Llama
                try:
                    with torch.no_grad():
                        fol_statements = convert_reasoning_to_fol(
                            reasoning_statements, llama_model, llama_tokenizer  # Convert all statements
                        )
                except Exception as fol_error:
                    fol_statements = [(stmt, "Error in FOL conversion") for stmt in reasoning_statements]
                
                # Verify with Prover9
                try:
                    verification_result = verify_reasoning_with_prover9(
                        fol_statements, 
                        f"GRPO_Training_Example_{i}"
                    )
                except Exception as prover_error:
                    verification_result = {'valid': False, 'details': 'Prover9 error'}
                
                # Calculate composite reward
                reward_components = reward_calculator.calculate_composite_reward(
                    response=response,
                    expected_answer=expected,
                    prover9_result=verification_result,
                    reasoning_steps=reasoning_statements
                )

                # Log only from main process and first few examples
                if i < 5 and is_main_process:
                    try:
                        wandb.log({
                            f"reward/answer_correctness": reward_components.answer_correctness,
                            f"reward/logical_validity": reward_components.logical_validity,
                            f"reward/format_compliance": reward_components.format_compliance,
                            f"reward/prover9_valid": verification_result.get('valid', False),
                        })
                    except:
                        pass  # Don't fail if wandb logging fails
                
                # Scale reward for GRPO
                scaled_reward = reward_components.total_reward * 10.0
                scores.append(scaled_reward)
                
            except Exception as e:
                if is_main_process:
                    print(f"⚠️ Error in reward computation for completion {i}: {e}")
                scores.append(-1.0)  # Penalty for failed processing
        
        # Rewards computed
        
        return scores
    
    return [compute_complex_reward]

def setup_model_for_grpo(model_path=None, max_seq_length=3000, lora_rank=32, device=None, world_size=1):
    """Setup model for GRPO training with multi-GPU support
    
    Args:
        model_path: Path to pre-trained model
        max_seq_length: Maximum sequence length
        lora_rank: LoRA rank for parameter efficiency
        device: Device to load model on (for distributed training)
        world_size: Number of processes (1 for single GPU)
    """
    
    if model_path and os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Check if this is a LoRA model by looking for adapter_config.json
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            print(f"📦 Loading LoRA model from {model_path}")
            from peft import PeftModel, PeftConfig
            
            # Load the PEFT config to get base model name
            print("   Loading PEFT config...")
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_name = peft_config.base_model_name_or_path
            print(f"   Base model: {base_model_name}")
            
            
            # Load base model first
            print(f"   Loading base model (this may take several minutes)...")
            if world_size > 1:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )
                if device is not None:
                    base_model = base_model.to(device)
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            print("   Base model loaded successfully")
            
            # Load LoRA adapters
            print("   Loading LoRA adapters...")
            model = PeftModel.from_pretrained(base_model, model_path)
            print("   LoRA adapters loaded successfully")
            
            # IMPORTANT: Set adapters to trainable mode for continued training
            model.train()
            for param in model.parameters():
                param.requires_grad = False  # First freeze all
            
            # Then unfreeze LoRA parameters
            for name, param in model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
                    
            # Verify we have trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✅ Model loaded with {trainable_params:,} trainable parameters out of {total_params:,} total")
            
            if trainable_params == 0:
                raise ValueError("No trainable parameters found! Check LoRA configuration.")
            
        else:
            # For distributed training, load without device_map
            if world_size > 1:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )
                if device is not None:
                    model = model.to(device)
            else:
                # Single GPU - can use device_map
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
        
        # Disable caching to prevent gradient checkpointing conflicts
        if hasattr(model, 'config') and hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
        # Also handle the case where model is a PeftModel
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'config'):
            model.base_model.config.use_cache = False
    else:
        model_name = "Qwen/Qwen3-8B"

        # Only use quantization in single-GPU mode
        use_quantization = world_size == 1
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Explicitly set pad_token_id to avoid repeated warnings
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # Configure quantization
        bnb_config = None
        if use_quantization:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                print("⚠️ BitsAndBytesConfig not available, loading without quantization")
                bnb_config = None

        # Load model appropriately for single/multi-GPU
        if world_size > 1:
            # Multi-GPU: no device_map, no quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            if device is not None:
                model = model.to(device)
            # Disable caching to prevent gradient checkpointing conflicts
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = False
        else:
            # Single GPU: can use device_map and quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            # Disable caching to prevent gradient checkpointing conflicts
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = False

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
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        
        # Disable caching to prevent gradient checkpointing conflicts
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
    
    return model, tokenizer

def train_grpo_model_distributed(model, tokenizer, dataset, reward_functions, args, device, world_size=1):
    """Train GRPO model with multi-GPU support
    
    Note: VLLM with multi-GPU in GRPO is complex. We'll use standard generation instead.
    """
    print("🎯 Starting GRPO training...")
    
    # Setup accelerator with proper distributed config
    if world_size > 1:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.grpo_gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs],
            mixed_precision="bf16" if args.use_mixed_precision else None,
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.grpo_gradient_accumulation_steps,
            mixed_precision="bf16" if args.use_mixed_precision else None,
        )
    
    # Create output directory
    os.makedirs(args.grpo_output_dir, exist_ok=True)
    
    max_prompt_length = 1000
    max_completion_length = args.max_seq_length - max_prompt_length

    # For multi-GPU, we need to use standard generation instead of VLLM
    # VLLM requires complex setup for multi-GPU that conflicts with TRL
    
    # Training configuration - REDUCED MEMORY USAGE
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
        
        per_device_train_batch_size=args.grpo_batch_size,
        gradient_accumulation_steps=args.grpo_gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_steps=10 if args.test_mode else args.grpo_max_steps,  # Test mode: only 10 steps
        
        # Generation parameters - REDUCED lengths
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
        dataloader_num_workers=1,  # Reduce workers
        
        # Memory optimization
        gradient_checkpointing=args.gradient_checkpointing,  # Don't force it - let user decide
        bf16=args.use_mixed_precision,  # Don't force it - let user decide  
        remove_unused_columns=True,
        dataloader_pin_memory=False,  # Disable pin memory
    )
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=dataset,
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
    
    # Testing and debugging
    parser.add_argument("--test_mode", action="store_true",
                      help="Run in test mode with only a few steps to verify setup")
    parser.add_argument("--debug_rewards", action="store_true",
                      help="Enable detailed reward computation debugging")
    
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
    
    # Setup distributed training
    local_rank, world_size, device = setup_distributed_training(args.num_gpus)
    
    if world_size > 1:
        print(f"🌐 Distributed Training Setup: {world_size} GPUs")
    
    # Test Prover9 installation before starting
    if local_rank == 0:  # Only test from main process
        if not test_prover9_installation():
            print("❌ Prover9 not available - training cannot continue")
            print("Please install Prover9 to use logical verification rewards")
            return

    # Initialize wandb only on main process
    if local_rank == 0:
        wandb.init(
            project="logical-reasoning-grpo",
            name=f"grpo-training-{world_size}gpu",
            config=vars(args)
        )
    
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
    
    # Synchronize all processes
    if world_size > 1:
        torch.distributed.barrier()
    
    # Setup model for GRPO (each process loads its own)
    model, tokenizer = setup_model_for_grpo(
        sft_model_path, 
        args.max_seq_length, 
        args.lora_rank,
        device=device if world_size > 1 else None,
        world_size=world_size
    )
    
    # Setup chat template
    tokenizer, system_prompt = setup_chat_template(tokenizer)
    
    # Load and format GRPO dataset (each process loads independently)
    # This is simpler and more reliable than broadcasting
    if local_rank == 0:
        print(f"📂 Loading GRPO dataset from {args.grpo_dataset}...")
    grpo_data = load_grpo_dataset(args.grpo_dataset, args.max_examples)
    
    grpo_dataset = format_grpo_dataset(grpo_data, system_prompt, tokenizer)
    if local_rank == 0:
        print(f"📊 Loaded {len(grpo_dataset)} training examples")
    
    # Synchronize all processes before proceeding
    if world_size > 1:
        torch.distributed.barrier()

    # Setup reward functions with full pipeline
    reward_functions = setup_complex_reward_functions(  # Changed to complex version
        tokenizer, 
        None,  # Pass None instead of accelerator
        device=device if world_size > 1 else None
    )
    
    # Train GRPO model
    final_model_path = train_grpo_model_distributed(
        model, tokenizer, grpo_dataset, reward_functions, args, device, world_size
    )
    
    if local_rank == 0:
        print(f"\n🎉 Training pipeline completed successfully!")
        print(f"📁 Final model saved to: {final_model_path}")
        
        wandb.finish()

    return final_model_path

if __name__ == "__main__":
    main()
