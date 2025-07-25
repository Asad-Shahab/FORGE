#!/usr/bin/env python3
"""
Standalone SFT Training for Logical Reasoning
Follows architectural patterns from pipeline_dev.py
"""

import argparse
import json
import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# Import from existing setup modules
from setup.setup_models import setup_qwen3

def setup_chat_template(tokenizer):
    """Setup chat template following pipeline_dev.py patterns"""
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
    """Load and format SFT dataset following pipeline_dev.py patterns"""
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

def setup_model_for_sft(max_seq_length=2048, lora_rank=32):
    """Setup model for SFT training using setup functions"""
    print("🚀 Setting up model for SFT training...")
    
    # Use setup function from setup/setup_models.py
    model, tokenizer = setup_qwen3()

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

def train_sft_model(model, tokenizer, dataset, args):
    """Train SFT model with configuration"""
    print("🎯 Starting SFT training...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training configuration
    training_args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        optim="adamw_8bit",
        seed=3407,
        output_dir=args.output_dir,
        report_to="none",
        save_total_limit=2,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    
    # Train
    trainer.train()
    
    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print(f"✅ SFT training complete! Model saved to {final_dir}")
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
        
        args = parser.parse_args()
    
    print("🧠 SFT Training for Logical Reasoning")
    print("=" * 50)
    
    # Load dataset
    raw_data = load_sft_dataset(args.dataset, args.max_examples)
    
    # Setup model
    model, tokenizer = setup_model_for_sft(args.max_seq_length, args.lora_rank)
    
    # Setup chat template
    tokenizer, system_prompt = setup_chat_template(tokenizer)
    
    # Format dataset
    print("📝 Formatting dataset for SFT training...")
    formatted_dataset = format_sft_examples(raw_data, tokenizer, system_prompt)
    print(f"📊 Formatted {len(formatted_dataset)} examples")
    
    # Train model
    final_model_path = train_sft_model(model, tokenizer, formatted_dataset, args)
    
    print(f"\n🎉 SFT training completed successfully!")
    print(f"📁 Model saved to: {final_model_path}")
    print(f"💡 To use this model, load it with:")
    print("    from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"    model = AutoModelForCausalLM.from_pretrained('{final_model_path}')")
    print(f"    tokenizer = AutoTokenizer.from_pretrained('{final_model_path}')")
    
    return final_model_path

if __name__ == "__main__":
    main()
