
import json
import os
import random
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Use only one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    "./sft_models/final",
    device_map="auto",   # automatically places everything on your 1 GPU
    torch_dtype="auto"   # picks the right dtype (fp16/bf16/fp32 depending on weights)
)
tokenizer = AutoTokenizer.from_pretrained("./sft_models/final")

# Load dataset
dataset = load_dataset("yale-nlp/FOLIO", split="validation")

# System prompt
system_prompt = (
    "You are an expert in logical reasoning. Analyze the given premises and conclusion step by step. "
    "First, provide your initial reasoning between <initial_reasoning> and </initial_reasoning>. "
    "Then, provide your logical reasoning steps between <steps> and </steps>. "
    "Finally, provide your answer (A for True, B for False, or C for Uncertain) between <answer> and </answer>."
)

# Function to extract answer from tags
def extract_answer(text):
    # Find all matches and take the last one (the model's actual answer)
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None

# Function to map LLM output (A, B, C) to dataset labels (True, False, Uncertain)
def map_answer(llm_answer):
    mapping = {
        'A': 'True',
        'B': 'False', 
        'C': 'Uncertain'
    }
    return mapping.get(llm_answer, None)

# Sample 10 random questions
dataset_list = list(dataset)
random_questions = random.sample(dataset_list, min(50, len(dataset_list)))

correct = 0
total = len(random_questions)

# Test on each sampled question
for i, item in enumerate(random_questions, 1):
    premises = item['premises']
    conclusion = item['conclusion']
    expected_answer = item['label']

    # Construct the question from premises and conclusion
    question = f"Premises:\n{premises}\n\nConclusion:\n{conclusion}"
    
    # Prepend the system prompt to the question
    prompt = system_prompt + "\n\n" + question

    # Tokenize and move to GPU
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=4048, do_sample=False)
    complete_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer from tags
    extracted_answer = extract_answer(complete_output)
    
    # Map LLM answer to dataset format
    mapped_answer = map_answer(extracted_answer)
    
    # Compare answers
    is_correct = mapped_answer == expected_answer
    if is_correct:
        correct += 1
    
    print(f"Question {i}:")
    #print(f"Complete output: {complete_output}")
    print(f"Expected: {expected_answer}")
    print(f"LLM Output: {extracted_answer}")
    print(f"Mapped: {mapped_answer}")
    print(f"Correct: {is_correct}")
    print("-" * 50)

# Final score
score = correct / total
print(f"\nFinal Score: {correct}/{total} = {score:.2%}")


