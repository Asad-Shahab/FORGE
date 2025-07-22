#!/usr/bin/env python3
"""
GRPO Training for Logical Reasoning - Adapted from Colab Template
Trains Qwen3-8B on ProverQA dataset using logical reasoning rewards
"""

import re
import torch
import json
import numpy as np
from datasets import load_dataset, Dataset
from transformers import TextStreamer
from pathlib import Path

# Import your existing components
from setup.setup_models import setup_qwen3, setup_llama_lora
from verification.prover9_integration import verify_reasoning_with_prover9
from reward.reward import LogicalReasoningReward

# Unsloth and training imports
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

def setup_model_and_tokenizer():
    """Setup Qwen3-8B model with Unsloth optimizations"""
    
    max_seq_length = 2048
    lora_rank = 32
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",  # Using available model
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.7,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    return model, tokenizer, max_seq_length

def setup_chat_template(tokenizer):
    """Setup logical reasoning chat template"""
    
    reasoning_start = "<initial_reasoning>"
    reasoning_end = "</initial_reasoning>"
    steps_start = "<steps>"
    steps_end = "</steps>"
    solution_start = "<answer>"
    solution_end = "</answer>"
    
    system_prompt = f"""You are an expert in logical reasoning. Analyze the given logical problem step by step.
Start with brief initial reasoning in {reasoning_start}{reasoning_end} tags.
Then provide detailed step-by-step reasoning in {steps_start}{steps_end} tags.
Finally, provide your answer (A, B, or C) in {solution_start}{solution_end} tags."""
    
    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        f"{{ '{system_prompt}' + eos_token }}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        f"{{ '{reasoning_start}' }}"
        "{% endif %}"
    ).replace(f"'{system_prompt}'", f"'{system_prompt}'").replace(f"'{reasoning_start}'", f"'{reasoning_start}'")
    
    tokenizer.chat_template = chat_template
    
    return {
        'reasoning_start': reasoning_start,
        'reasoning_end': reasoning_end,
        'steps_start': steps_start,
        'steps_end': steps_end,
        'solution_start': solution_start,
        'solution_end': solution_end,
        'system_prompt': system_prompt
    }

def load_proverqa_dataset():
    """Load and format ProverQA dataset for GRPO training"""
    
    print("📊 Loading ProverQA simplified dataset...")
    
    # Load your processed dataset
    with open('proverqa_simplified.json', 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} logical reasoning problems")
    
    # Convert to GRPO format
    formatted_data = []
    for item in data:
        formatted_item = {
            'prompt': [
                {'role': 'system', 'content': 'You are an expert in logical reasoning. Analyze the given logical problem step by step.\nStart with brief initial reasoning in <initial_reasoning></initial_reasoning> tags.\nThen provide detailed step-by-step reasoning in <steps></steps> tags.\nFinally, provide your answer (A, B, or C) in <answer></answer> tags.'},
                {'role': 'user', 'content': item['question']}
            ],
            'answer': item['answer']
        }
        formatted_data.append(formatted_item)
    
    return Dataset.from_list(formatted_data)

def create_format_checker(template_info):
    """Create format compliance reward function"""
    
    def check_format_compliance(completions, **kwargs):
        """Check if response follows required format"""
        scores = []
        
        for completion in completions:
            score = 0.0
            response = completion[0]["content"]
            
            # Check for all required tags
            required_tags = [
                ('initial_reasoning', template_info['reasoning_start'], template_info['reasoning_end']),
                ('steps', template_info['steps_start'], template_info['steps_end']),
                ('answer', template_info['solution_start'], template_info['solution_end'])
            ]
            
            tag_scores = []
            for tag_name, start_tag, end_tag in required_tags:
                if start_tag in response and end_tag in response:
                    tag_scores.append(1.0)
                else:
                    tag_scores.append(0.0)
            
            # Perfect format gets full score
            if all(tag_scores):
                score = 3.0
            else:
                # Partial credit for partial compliance
                score = sum(tag_scores) * 0.5
            
            scores.append(score)
        
        return scores
    
    return check_format_compliance

def create_answer_checker():
    """Create answer correctness reward function"""
    
    def check_answer_correctness(prompts, completions, answer, **kwargs):
        """Check if extracted answer matches expected answer"""
        scores = []
        
        # Extract answer pattern
        answer_pattern = re.compile(r'<answer>\s*([ABC])\s*</answer>', re.IGNORECASE)
        
        for completion, expected in zip(completions, answer):
            response = completion[0]["content"]
            
            # Extract answer from response
            match = answer_pattern.search(response)
            if match:
                predicted_answer = match.group(1).upper()
                expected_answer = expected.upper()
                
                # Award points for correct answer
                if predicted_answer == expected_answer:
                    scores.append(5.0)  # High reward for correctness
                else:
                    scores.append(-2.0)  # Penalty for wrong answer
            else:
                scores.append(-3.0)  # Penalty for no answer found
        
        return scores
    
    return check_answer_correctness

def create_logic_validator():
    """Create logical validity reward function using existing infrastructure"""
    
    def check_logical_validity(prompts, completions, **kwargs):
        """Validate logical reasoning using Prover9 (simplified for GRPO)"""
        scores = []
        
        # Load existing models for FOL conversion (simplified)
        try:
            llama_model, llama_tokenizer = setup_llama_lora()
            logic_available = True
        except:
            logic_available = False
            print("⚠️ Logical validation unavailable, using simplified scoring")
        
        steps_pattern = re.compile(r'<steps>(.*?)</steps>', re.DOTALL | re.IGNORECASE)
        
        for completion in completions:
            response = completion[0]["content"]
            
            # Extract reasoning steps
            steps_match = steps_pattern.search(response)
            if steps_match:
                reasoning_text = steps_match.group(1).strip()
                
                if logic_available:
                    # Simplified logical validation
                    # In full implementation, would convert to FOL and verify with Prover9
                    # For now, use heuristics
                    score = evaluate_reasoning_quality(reasoning_text)
                else:
                    # Fallback: basic reasoning quality check
                    score = evaluate_reasoning_quality(reasoning_text)
                
                scores.append(score)
            else:
                scores.append(-1.0)  # No reasoning found
        
        return scores
    
    return check_logical_validity

def evaluate_reasoning_quality(reasoning_text):
    """Evaluate reasoning quality with heuristics"""
    
    score = 0.0
    
    # Check for logical connectors
    logical_words = ['therefore', 'thus', 'hence', 'because', 'since', 'given', 'if', 'then']
    for word in logical_words:
        if word.lower() in reasoning_text.lower():
            score += 0.2
    
    # Check for step structure
    lines = [line.strip() for line in reasoning_text.split('\n') if line.strip()]
    if len(lines) >= 2:
        score += 1.0
    
    # Check length (not too short, not too verbose)
    if 50 < len(reasoning_text) < 500:
        score += 0.5
    
    return min(score, 2.0)  # Cap at 2.0

def filter_dataset_by_length(dataset, tokenizer, max_length):
    """Filter dataset to avoid truncation issues"""
    
    print("📏 Filtering dataset by length...")
    
    def get_length(example):
        text = tokenizer.apply_chat_template(
            example['prompt'], 
            add_generation_prompt=True, 
            tokenize=True
        )
        return {'length': len(text)}
    
    # Add length column
    dataset_with_length = dataset.map(get_length)
    
    # Filter to 90th percentile
    lengths = dataset_with_length['length']
    max_allowed = int(np.quantile(lengths, 0.9))
    
    filtered_dataset = dataset_with_length.filter(
        lambda x: x['length'] <= min(max_allowed, max_length)
    )
    
    print(f"✅ Filtered from {len(dataset)} to {len(filtered_dataset)} examples")
    print(f"📊 Max prompt length: {max_allowed}")
    
    return filtered_dataset

def setup_grpo_training(model, tokenizer, dataset, max_seq_length):
    """Setup GRPO training configuration"""
    
    template_info = setup_chat_template(tokenizer)
    
    # Calculate lengths
    max_prompt_length = int(max_seq_length * 0.3)  # 30% for prompt
    max_completion_length = max_seq_length - max_prompt_length
    
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
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=4,  # Generate 4 responses per question
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=500,  # Adjust based on your needs
        save_steps=100,
        report_to="none",
        output_dir="grpo_logical_reasoning_outputs",
    )
    
    # Create reward functions
    reward_functions = [
        create_format_checker(template_info),
        create_answer_checker(), 
        create_logic_validator(),
    ]
    
    # Setup trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=dataset,
    )
    
    return trainer, training_args

def main():
    """Main training function"""
    
    print("🚀 GRPO Logical Reasoning Training")
    print("=" * 60)
    
    # Setup model
    print("🔧 Setting up model and tokenizer...")
    model, tokenizer, max_seq_length = setup_model_and_tokenizer()
    
    # Load dataset
    print("📊 Loading ProverQA dataset...")
    dataset = load_proverqa_dataset()
    
    # Filter dataset by length
    dataset = filter_dataset_by_length(dataset, tokenizer, max_seq_length // 2)
    
    # Setup training
    print("⚙️ Setting up GRPO training...")
    trainer, training_args = setup_grpo_training(model, tokenizer, dataset, max_seq_length)
    
    # Start training
    print("🎯 Starting GRPO training...")
    print(f"📊 Dataset size: {len(dataset)}")
    print(f"🔄 Max steps: {training_args.max_steps}")
    print(f"🎲 Generations per question: {training_args.num_generations}")
    
    trainer.train()
    
    # Save model
    print("💾 Saving trained model...")
    model.save_lora("grpo_logical_reasoning_lora")
    
    print("✅ Training complete!")
    return model, tokenizer

def test_trained_model(model, tokenizer):
    """Test the trained model on a sample problem"""
    
    test_question = """Jayce earns academic credentials. If someone earns academic credentials and gains practical skills, then they can participate in excavations. Jayce analyzes artifacts. Jayce does not develop expertise. Kennedy analyzes artifacts. Jayce either investigates historical sites or uncovers hidden archaeological treasures (or both). If Jayce either participates in excavations or conducts historical research (but not both), then he has field experience. Kaleb does not develop expertise. If Kennedy pursues historical interests, then he either gains practical skills or develops expertise (or both). Jayce either uncovers hidden archaeological treasures or joins archaeological digs, but not both. Anyone who holds a PhD in archaeology or has field experience can join archaeological digs. If Jayce pursues historical interests, then he either gains practical skills or develops expertise (or both). Kennedy does not hold a PhD in archaeology. If Jayce reads Turkish texts or visits historical sites, then he is likely to study Seljuk history. Leila pursues historical interests. Jayce either analyzes artifacts or conducts historical research, but not both. Jayce pursues historical interests. If Kennedy is fascinated by medieval cultures, then he will study Seljuk history. Leila either uncovers hidden archaeological treasures or joins archaeological digs, but not both. Jayce does not hold a PhD in archaeology.

Question: Based on the above information, is the following statement true, false, or uncertain? If Jayce studies Seljuk history or explores Anatolian landscapes, then he uncovers hidden archaeological treasures.

Options:
A) True
B) False
C) Uncertain"""
    
    messages = [
        {"role": "system", "content": "You are an expert in logical reasoning. Analyze the given logical problem step by step.\nStart with brief initial reasoning in <initial_reasoning></initial_reasoning> tags.\nThen provide detailed step-by-step reasoning in <steps></steps> tags.\nFinally, provide your answer (A, B, or C) in <answer></answer> tags."},
        {"role": "user", "content": test_question}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    print("\n🧪 Testing trained model:")
    print("-" * 50)
    
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=50,
        max_tokens=1024,
    )
    
    output = model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=model.load_lora("grpo_logical_reasoning_lora"),
    )[0].outputs[0].text
    
    print(output)
    print("-" * 50)

if __name__ == "__main__":
    model, tokenizer = main()
    
    # Optional: Test the trained model
    test_trained_model(model, tokenizer)