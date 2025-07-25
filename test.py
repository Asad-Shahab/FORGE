import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use only one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the model
model = AutoModelForCausalLM.from_pretrained('./grpo_models/final')
tokenizer = AutoTokenizer.from_pretrained('./grpo_models/final')

# Load dataset
with open('dataset/dev/hard.json', 'r') as f:
    dataset = json.load(f)

# Test on each question
for item in dataset:
    question = item['question']
    expected_answer = item['answer']
    
    # Tokenize and generate
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    # Decode the complete output
    complete_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Complete output: {complete_output}")
    print(f"Expected answer: {expected_answer}")
    print("-" * 50)

