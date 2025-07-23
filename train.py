#!/usr/bin/env python3
"""
GRPO Training for Logical Reasoning
Direct adaptation of the Colab notebook for ProverQA dataset
"""

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, Dataset
import json
import re
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

# ============= SETUP MODEL =============
max_seq_length = 2048
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-unsloth-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank*2,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ============= SETUP CHAT TEMPLATE =============
reasoning_start = "<initial_reasoning>"
reasoning_end = "</initial_reasoning>"
steps_start = "<steps>"
steps_end = "</steps>"
answer_start = "<answer>"
answer_end = "</answer>"

system_prompt = f"""You are an expert in logical reasoning.
Analyze the given problem step by step.
First, provide your initial analysis between {reasoning_start} and {reasoning_end}.
Then, provide your logical reasoning steps between {steps_start} and {steps_end}.
Finally, provide your answer (A, B, or C) between {answer_start} and {answer_end}."""

chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"

chat_template = chat_template\
    .replace("'{system_prompt}'", f"'{system_prompt}'")\
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")
tokenizer.chat_template = chat_template

# ============= LOAD AND FORMAT DATASET =============
def format_dataset(dataset_path):
    """Format ProverQA dataset for GRPO training"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
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

# Load dataset
dataset = format_dataset("dataset/proverqa_simplified.json")
print(f"Loaded {len(dataset)} problems")

# ============= REWARD FUNCTIONS =============

# Format checking regex
answer_regex = re.compile(
    rf"{answer_end}.*?{answer_start}([ABC]).*?{answer_end}",
    flags=re.MULTILINE | re.DOTALL
)

def check_format_exact(completions, **kwargs):
    """Check if response follows exact format (3 points)"""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 0.0
        
        # Check all required tags
        if reasoning_start in response and reasoning_end in response:
            score += 1.0
        if steps_start in response and steps_end in response:
            score += 1.0
        if answer_start in response and answer_end in response:
            score += 1.0
            
        scores.append(score)
    return scores

def check_answer_correctness(prompts, completions, expected_answer, **kwargs):
    """Check if answer is correct (5 points max)"""
    scores = []
    
    for i, completion in enumerate(completions):
        response = completion[0]["content"]
        
        # Extract answer
        answer_match = re.search(rf'{answer_start}([ABC]){answer_end}', response)
        if not answer_match:
            # Fallback: look for A, B, or C at the end
            answer_match = re.search(r'([ABC])\s*$', response)
        
        if answer_match:
            predicted = answer_match.group(1)
            correct = expected_answer[i]
            
            if predicted == correct:
                scores.append(5.0)  # Full points for correct answer
            else:
                scores.append(-2.0)  # Penalty for wrong answer
        else:
            scores.append(-3.0)  # Penalty for no answer
    
    return scores

def check_reasoning_quality(completions, **kwargs):
    """Check reasoning quality based on structure (2 points max)"""
    scores = []
    
    for completion in completions:
        response = completion[0]["content"]
        score = 0.0
        
        # Extract reasoning and steps
        reasoning_match = re.search(
            rf'{reasoning_start}(.*?){reasoning_end}', 
            response, re.DOTALL
        )
        steps_match = re.search(
            rf'{steps_start}(.*?){steps_end}', 
            response, re.DOTALL
        )
        
        # Award points for non-empty reasoning
        if reasoning_match and len(reasoning_match.group(1).strip()) > 20:
            score += 1.0
            
        # Award points for structured steps
        if steps_match:
            steps_text = steps_match.group(1).strip()
            # Count logical connectors
            logic_words = ['if', 'then', 'therefore', 'because', 'since', 'thus']
            logic_count = sum(1 for word in logic_words if word in steps_text.lower())
            if logic_count >= 2:
                score += 1.0
        
        scores.append(score)
    
    return scores

# Combine all reward functions
reward_functions = [
    check_format_exact,
    check_answer_correctness,
    check_reasoning_quality
]

# ============= TRAINING CONFIGURATION =============

# VLLM sampling parameters
vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=3407,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

# Training arguments
training_args = GRPOConfig(
    # Sampling parameters
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    
    # Training parameters
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    
    # Batch and steps
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Effective batch size = 4
    num_generations=4,  # Generate 4 responses per prompt
    max_steps=1000,     # Adjust based on dataset size
    
    # Generation parameters
    max_prompt_length=max_seq_length // 2,
    max_completion_length=max_seq_length // 2,
    
    # Logging and saving
    logging_steps=10,
    save_steps=100,
    output_dir="./grpo_logical_reasoning",
    report_to="none",  # Disable wandb for cluster
)

# ============= OPTIONAL: Pre Fine-tuning =============
# Following the notebook pattern, we can do a small SFT run first
# to teach the model our format

def create_sft_examples(dataset, n_examples=100):
    """Create a few examples with perfect format for pre-training"""
    sft_data = []
    
    for i in range(min(n_examples, len(dataset))):
        item = dataset[i]
        
        # Create a well-formatted response
        perfect_response = f"""{reasoning_start}
I need to analyze this logical reasoning problem step by step.
Let me identify the key elements and relationships.
{reasoning_end}

{steps_start}
Step 1: Identify the given information
Step 2: Apply logical rules
Step 3: Draw conclusions based on the analysis
Therefore, the answer is {item['expected_answer']}.
{steps_end}

{answer_start}
{item['expected_answer']}
{answer_end}"""
        
        messages = item['messages'] + [{"role": "assistant", "content": perfect_response}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        sft_data.append({"text": text})
    
    return Dataset.from_list(sft_data)

# Optional: Run SFT first
if True:  # Set to True to run pre-fine-tuning
    from trl import SFTTrainer, SFTConfig
    
    sft_dataset = create_sft_examples(dataset, n_examples=100)
    
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=sft_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=5,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=5,
            optim="adamw_8bit",
            seed=3407,
            output_dir="./sft_pretrain",
        ),
    )
    
    print("Running SFT pre-training...")
    sft_trainer.train()
    print("SFT complete!")

# ============= GRPO TRAINING =============

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_functions,
    args=training_args,
    train_dataset=dataset,
)

# Print example to verify format
print("\n=== Example prompt ===")
example = dataset[0]
prompt = tokenizer.apply_chat_template(
    example['messages'], 
    tokenize=False, 
    add_generation_prompt=True
)
print(prompt)
print("Expected answer:", example['expected_answer'])

# Train!
print("\n=== Starting GRPO Training ===")
trainer.train()

# ============= SAVE MODEL =============
model.save_pretrained("grpo_logical_reasoning_final")
tokenizer.save_pretrained("grpo_logical_reasoning_final")

print("Training complete! Model saved to grpo_logical_reasoning_final/")