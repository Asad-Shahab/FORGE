import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model repository identifiers
base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
lora_weights = "fvossel/Llama-3.1-8B-Instruct-nl-to-fol"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load the base model
model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True, device_map="auto")

# Load the LoRA adapter weights
model = PeftModel.from_pretrained(model, lora_weights, device_map="auto")

# Define formatting function for preparing the input prompt
def formatting_func(text):
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant that translates Natural Language (NL) text "
                    "into First-Order Logic (FOL) using only the given quantors and junctors: "
                    "∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), "
                    "↔ (if and only if), ⊕ (xor). "
                    "Start your answer with '𝜙=' followed by the FOL-formula. Do not include any other text."
                ),
            },
            {"role": "user", "content": text},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

# Example NL input
input_text = "All dogs are animals."

# Use formatting_func to create the model input
prompt = formatting_func(input_text)

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt", padding=True)

# Generate output
outputs = model.generate(**inputs, max_new_tokens=100)

# Decode and print the result
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
