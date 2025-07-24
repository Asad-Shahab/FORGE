#!/usr/bin/env python3
"""
GRPO Training Pipeline for Logical Reasoning
Integrates SFT pre-training and GRPO training with full pipeline reward functions
Includes NL→FOL conversion and Prover9 verification during training
"""

import argparse
import json
import os
import re
import torch
import wandb
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

# Import from existing modules
from setup.setup_models import setup_qwen3, setup_llama_lora
from reward.reward import LogicalReasoningReward
from verification.prover9_integration import verify_reasoning_with_prover9, test_prover9_installation
from train_sft import main as train_sft_main

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

def load_grpo_dataset(dataset_path, max_examples=None):
    """Load GRPO dataset following pipeline_dev.py patterns"""
    print(f"📂 Loading GRPO dataset from {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if max_examples:
        data = data[:max_examples]
        print(f"📊 Limited to {max_examples} examples")
    
    print(f"📊 Loaded {len(data)} training examples")
    return data

def format_grpo_dataset(data, system_prompt):
    """Format dataset for GRPO training"""
    formatted_data = []
    for item in data:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["question"]}
        ]
        
        formatted_data.append({
            "messages": messages,
            "expected_answer": item["answer"],
            "question": item["question"]
        })
    
    return Dataset.from_list(formatted_data)

def parse_qwen_solution(qwen_output):
    """Parse Qwen's solution to extract reasoning and answer (from pipeline_dev.py)"""
    
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
    """Convert reasoning statements to FOL using Llama (from pipeline_dev.py)"""

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

EXAMPLES:
"John is tall" → Tall(john)
"If someone is weak, they are not resilient" → all x (Weak(x) → ¬Resilient(x))
"Mary is either smart or funny" → Smart(mary) ∨ Funny(mary)
"Everyone who studies passes" → all x (Studies(x) → Passes(x))

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

def setup_reward_functions(tokenizer):
    """Setup reward functions with full pipeline integration"""
    print("🔧 Loading Llama model for NL→FOL conversion...")
    
    # Load Llama model for NL→FOL conversion
    try:
        llama_model, llama_tokenizer = setup_llama_lora()
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
                response = completion[0]["content"]
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

                if i < 5:  # Log first 5 examples per batch to avoid spam
                    wandb.log({
                        f"reward/answer_correctness": reward_components.answer_correctness,
                        f"reward/logical_validity": reward_components.logical_validity, 
                        f"reward/format_compliance": reward_components.format_compliance,
                        f"reward/prover9_valid": verification_result.get('valid', False),
                    })
                
                # Scale reward for GRPO (typically 0-10 range works well)
                scaled_reward = reward_components.total_reward * 10.0
                scores.append(scaled_reward)
                
                # Optional: Print progress for first few examples
                if i < 3:  # Only print first 3 to avoid spam
                    print(f"  Example {i}: Predicted={predicted_answer}, Expected={expected}, "
                          f"Valid={verification_result.get('valid', False)}, "
                          f"Reward={scaled_reward:.2f}")
                
            except Exception as e:
                print(f"  ❌ Error processing completion {i}: {e}")
                scores.append(-1.0)  # Penalty for failed processing
        
        return scores
    
    # Return single comprehensive reward function
    return [compute_full_pipeline_reward]

def setup_model_for_grpo(model_path=None, max_seq_length=3000, lora_rank=32):
    """Setup model for GRPO training"""
    print("🚀 Setting up model for GRPO training...")
    
    if model_path and os.path.exists(model_path):
        print(f"📂 Loading pre-trained model from {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
    else:
        print("🔧 Loading base model for GRPO training...")
        # Use setup function from setup/setup_models.py
        model, tokenizer = setup_qwen3()
        
        # Apply LoRA for training
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    
    return model, tokenizer

def train_grpo_model(model, tokenizer, dataset, reward_functions, args):
    """Train GRPO model with reward functions"""
    print("🎯 Starting GRPO training with full pipeline rewards...")
    
    # Create output directory
    os.makedirs(args.grpo_output_dir, exist_ok=True)
    
    # VLLM sampling parameters
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
    
    # Training configuration
    training_args = GRPOConfig(

        ddp_find_unused_parameters=False # For multiple-GPU

        # Sampling parameters
        vllm_sampling_params=vllm_sampling_params,
        temperature=args.temperature,
        
        # Training parameters
        learning_rate=args.grpo_learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        
        # Batch and steps
        per_device_train_batch_size=args.grpo_batch_size,
        gradient_accumulation_steps=args.grpo_gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_steps=args.grpo_max_steps,
        
        # Generation parameters
        max_prompt_length=1000,
        max_completion_length=2000,
        
        # Logging and saving
        logging_steps=args.grpo_logging_steps,
        save_steps=args.grpo_save_steps,
        output_dir=args.grpo_output_dir,
        report_to="wandb",
        save_total_limit=2,
    )
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Print example to verify format
    print("\n=== Example GRPO prompt ===")
    example = dataset[0]
    prompt = tokenizer.apply_chat_template(
        example['messages'], 
        tokenize=False, 
        add_generation_prompt=True
    )
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print(f"Expected answer: {example['expected_answer']}")
    print("=" * 50)
    
    # Train
    trainer.train()
    
    # Save final model
    final_dir = os.path.join(args.grpo_output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print(f"✅ GRPO training complete! Model saved to {final_dir}")
    return final_dir

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="GRPO Training Pipeline for Logical Reasoning")
    
    # Pipeline control
    parser.add_argument("--run_sft", action="store_true",
                      help="Run SFT pre-training first")
    parser.add_argument("--sft_model_path", type=str, default=None,
                      help="Path to pre-trained SFT model (skips SFT if provided)")
    
    args = parser.parse_args()
    
    # Set default parameters
    grpo_dataset = "dataset/proverqa_simplified.json"
    sft_dataset = "dataset/sft/sft_with_reasoning.json"
    max_examples = None
    max_seq_length = 3000
    lora_rank = 32
    
    # SFT parameters
    sft_epochs = 1
    sft_steps = 100
    sft_output_dir = "./sft_models"
    
    # GRPO parameters
    grpo_max_steps = 1250
    grpo_batch_size = 2
    grpo_gradient_accumulation_steps = 2
    grpo_learning_rate = 5e-6
    num_generations = 4
    temperature = 1.0
    grpo_logging_steps = 10
    grpo_save_steps = 100
    grpo_output_dir = "./grpo_models"
    
    print("🧠 GRPO Training Pipeline for Logical Reasoning")
    print("=" * 60)
    
    # Test Prover9 installation before starting
    print("🔍 Testing Prover9 installation...")
    if not test_prover9_installation():
        print("❌ Prover9 not available - training cannot continue")
        print("Please install Prover9 to use logical verification rewards")
        return
    print("✅ Prover9 ready")

    print("🔧 Initializing wandb...")
    wandb.init(
        project="logical-reasoning-grpo",
        name=f"grpo-training",
        config={
            "grpo_max_steps": grpo_max_steps,
            "grpo_batch_size": grpo_batch_size,
            "grpo_learning_rate": grpo_learning_rate,
            "num_generations": num_generations,
            "temperature": temperature,
            "max_seq_length": max_seq_length,
            "lora_rank": lora_rank,
        }
    )
    print("✅ Wandb initialized")
    
    sft_model_path = args.sft_model_path
    
    # Run SFT training if requested and no pre-trained model provided
    if args.run_sft and not sft_model_path:
        print("🚀 Starting SFT pre-training phase...")
        
        # Create SFT args
        class SFTArgs:
            def __init__(self):
                self.dataset = sft_dataset
                self.max_examples = max_examples
                self.max_seq_length = max_seq_length
                self.lora_rank = lora_rank
                self.epochs = sft_epochs
                self.max_steps = sft_steps
                self.batch_size = 1
                self.gradient_accumulation_steps = 1
                self.learning_rate = 2e-4
                self.warmup_steps = 10
                self.logging_steps = 5
                self.save_steps = 50
                self.output_dir = sft_output_dir
        
        sft_args = SFTArgs()
        sft_model_path = train_sft_main(sft_args)
        print(f"✅ SFT pre-training completed: {sft_model_path}")
    
    # Load GRPO dataset
    grpo_data = load_grpo_dataset(grpo_dataset, max_examples)
    
    # Setup model for GRPO
    model, tokenizer = setup_model_for_grpo(
        sft_model_path, max_seq_length, lora_rank
    )
    
    # Setup chat template
    tokenizer, system_prompt = setup_chat_template(tokenizer)
    
    # Format GRPO dataset
    print("📝 Formatting dataset for GRPO training...")
    grpo_dataset = format_grpo_dataset(grpo_data, system_prompt)
    print(f"📊 Formatted {len(grpo_dataset)} examples")
    
    # Setup reward functions with full pipeline
    print("🎯 Setting up reward functions with full pipeline...")
    reward_functions = setup_reward_functions(tokenizer)
    print(f"✅ Configured {len(reward_functions)} reward functions with NL→FOL and Prover9 verification")
    
    # Create GRPO args object
    class GRPOArgs:
        def __init__(self):
            self.grpo_max_steps = grpo_max_steps
            self.grpo_batch_size = grpo_batch_size
            self.grpo_gradient_accumulation_steps = grpo_gradient_accumulation_steps
            self.grpo_learning_rate = grpo_learning_rate
            self.num_generations = num_generations
            self.temperature = temperature
            self.grpo_logging_steps = grpo_logging_steps
            self.grpo_save_steps = grpo_save_steps
            self.grpo_output_dir = grpo_output_dir
            self.max_seq_length = max_seq_length
    
    grpo_args = GRPOArgs()
    
    # Train GRPO model
    final_model_path = train_grpo_model(model, tokenizer, grpo_dataset, reward_functions, grpo_args)
    
    print(f"\n🎉 Training pipeline completed successfully!")
    print(f"📁 Final model saved to: {final_model_path}")
    print(f"💡 To use this model, load it with:")
    print(f"    from unsloth import FastLanguageModel")
    print(f"    model, tokenizer = FastLanguageModel.from_pretrained('{final_model_path}')")
    
    wandb.finish()

    return final_model_path

if __name__ == "__main__":
    main()