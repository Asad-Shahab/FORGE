#!/usr/bin/env python3
"""
Standalone SFT Training for Logical Reasoning
Multi-GPU compatible version
"""

import argparse
import json
import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader

# Import from existing setup modules - Fix the import path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setup.setup_models import setup_qwen3

def setup_chat_template(tokenizer):
    """Setup chat template following pipeline patterns"""
    reasoning_start = "<initial_reasoning>"
    reasoning_end = "</initial_reasoning>"
    steps_start = "<steps>"
    steps_end = "</steps>"
    answer_start = "<answer>"
    answer_end = "</answer>"

    system_prompt = f"""You are an expert in logical reasoning. Analyze the given problem step by step. First, provide your initial reasoning between {reasoning_start} and {reasoning_end}. Then, provide your logical reasoning steps between {steps_start} and {steps_end}. Finally, provide your answer (A, B, or C) between {answer_start} and {answer_end}.

For the steps section, write simple, clear statements in natural language. Each statement should be on its own line. Do not use "Premise 1:", "Premise 2:" or formal logic notation. Just state the facts and conclusions directly, like:
<Statement 1>
<Statement 2>
...
<Conclusion>"""

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

def load_sft_dataset(dataset_path, max_examples=None):
    """Load and format SFT dataset"""
    print(f"📂 Loading SFT dataset from {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if max_examples:
        data = data[:max_examples]
        print(f"📊 Limited to {max_examples} examples")
    
    print(f"📊 Loaded {len(data)} training examples")
    return data

def format_sft_examples(data, tokenizer, system_prompt):
    """Format dataset for SFT training with proper responses"""
    reasoning_start = "<initial_reasoning>"
    reasoning_end = "</initial_reasoning>"
    steps_start = "<steps>"
    steps_end = "</steps>"
    answer_start = "<answer>"
    answer_end = "</answer>"
    
    formatted_data = []
    
    for item in data:
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["question"]}
        ]
        
        # Create properly formatted response using existing reasoning
        if "initial_reasoning" in item and item["initial_reasoning"]:
            initial_reasoning = item["initial_reasoning"]
        else:
            initial_reasoning = "Let me analyze this logical reasoning problem step by step."
        
        if "steps" in item and item["steps"]:
            steps = item["steps"]
        else:
            steps = "Step 1: Analyze the given information\nStep 2: Apply logical rules\nStep 3: Draw conclusions"
        
        perfect_response = f"""{reasoning_start}
{initial_reasoning}
{reasoning_end}

{steps_start}
{steps}
{steps_end}

{answer_start}
{item["answer"]}
{answer_end}"""

        # Add assistant response
        messages.append({"role": "assistant", "content": perfect_response})
        
        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)

def setup_model_for_sft(max_seq_length=2048, lora_rank=32, device=None):
    """Setup model for SFT training with multi-GPU support"""
    print("🚀 Setting up model for SFT training...")
    
    # Determine if we're in distributed mode
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    use_quantization = world_size == 1  # Only use quantization for single GPU
    
    # Use setup function from setup/setup_models.py
    model, tokenizer = setup_qwen3(device=device, use_quantization=use_quantization)

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
    
    # Enable gradient checkpointing if requested
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    return model, tokenizer

def train_sft_model(model, tokenizer, dataset, args):
    """Train SFT model with configuration"""
    print("🎯 Starting SFT training...")

    # Setup accelerator with proper config for multi-GPU
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16" if args.use_mixed_precision else None,
        kwargs_handlers=[ddp_kwargs] if int(os.environ.get('WORLD_SIZE', 1)) > 1 else None,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    def collate_fn(batch):
        texts = [example["text"] for example in batch]
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    # Create dataloader with proper batch size for distributed training
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=0.01
    )

    # Prepare for distributed training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        num_batches = 0
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                # Gradient clipping
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.detach().float()
                num_batches += 1
                global_step += 1

            # Logging
            if accelerator.is_main_process and (step + 1) % args.logging_steps == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch+1}/{args.epochs} | Step {step+1} | Loss: {avg_loss:.4f}")
            
            # Saving
            if accelerator.is_main_process and args.save_steps > 0 and (global_step % args.save_steps == 0):
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                print(f"💾 Saved checkpoint to {save_dir}")
            
            # Max steps limit
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
        
        # End of epoch logging
        if accelerator.is_main_process:
            avg_loss = epoch_loss / num_batches
            print(f"✅ Epoch {epoch+1}/{args.epochs} completed | Average Loss: {avg_loss:.4f}")
    
    # Save final model
    accelerator.wait_for_everyone()
    
    final_dir = os.path.join(args.output_dir, "final")
    if accelerator.is_main_process:
        os.makedirs(final_dir, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"✅ SFT training complete! Model saved to {final_dir}")
    
    accelerator.wait_for_everyone()
    return final_dir

def main(args=None):
    """Main SFT training function"""
    if args is None:
        parser = argparse.ArgumentParser(description="SFT Training for Logical Reasoning")
        
        # Dataset arguments
        parser.add_argument("--dataset", type=str, default="dataset/sft/sft_with_reasoning.json",
                          help="Path to SFT training dataset")
        parser.add_argument("--max_examples", type=int, default=None,
                          help="Maximum number of training examples")
        
        # Model arguments
        parser.add_argument("--max_seq_length", type=int, default=2048,
                          help="Maximum sequence length")
        parser.add_argument("--lora_rank", type=int, default=32,
                          help="LoRA rank")
        
        # Training arguments
        parser.add_argument("--epochs", type=int, default=3,
                          help="Number of training epochs")
        parser.add_argument("--max_steps", type=int, default=-1,
                          help="Maximum training steps (overrides epochs if > 0)")
        parser.add_argument("--batch_size", type=int, default=1,
                          help="Per device batch size")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                          help="Gradient accumulation steps")
        parser.add_argument("--learning_rate", type=float, default=2e-4,
                          help="Learning rate")
        parser.add_argument("--warmup_steps", type=int, default=10,
                          help="Warmup steps")
        
        # Logging and saving
        parser.add_argument("--logging_steps", type=int, default=10,
                          help="Logging steps")
        parser.add_argument("--save_steps", type=int, default=100,
                          help="Save steps")
        parser.add_argument("--output_dir", type=str, default="./sft_models",
                          help="Output directory")
        
        # Optimization
        parser.add_argument("--use_mixed_precision", action="store_true",
                          help="Use mixed precision training")
        
        args = parser.parse_args()
    
    print("🧠 SFT Training for Logical Reasoning")
    print("=" * 50)
    
    # Check if we're in distributed mode
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if world_size > 1:
        print(f"🌐 Running in distributed mode: Rank {local_rank}/{world_size}")
        device = torch.device(f'cuda:{local_rank}')
    else:
        print("🖥️ Running in single GPU/CPU mode")
        device = None
    
    # Load dataset
    raw_data = load_sft_dataset(args.dataset, args.max_examples)
    
    # Setup model
    model, tokenizer = setup_model_for_sft(args.max_seq_length, args.lora_rank, device)
    
    # Setup chat template
    tokenizer, system_prompt = setup_chat_template(tokenizer)
    
    # Format dataset
    print("📝 Formatting dataset for SFT training...")
    formatted_dataset = format_sft_examples(raw_data, tokenizer, system_prompt)
    print(f"📊 Formatted {len(formatted_dataset)} examples")
    
    # Train model
    final_model_path = train_sft_model(model, tokenizer, formatted_dataset, args)
    
    if local_rank == 0:
        print(f"\n🎉 SFT training completed successfully!")
        print(f"📁 Model saved to: {final_model_path}")
        print(f"💡 To use this model, load it with:")
        print("    from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"    model = AutoModelForCausalLM.from_pretrained('{final_model_path}')")
        print(f"    tokenizer = AutoTokenizer.from_pretrained('{final_model_path}')")
    
    return final_model_path

if __name__ == "__main__":
    main()