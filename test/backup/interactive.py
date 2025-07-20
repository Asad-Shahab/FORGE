#!/usr/bin/env python3
"""
Interactive testing script for both models
Run: python interactive_test.py
"""

import torch
from setup_models import setup_qwen3, setup_llama_lora, check_gpu

def interactive_qwen3(model, tokenizer):
    """Interactive testing for Qwen3 model"""
    print("🦥 Qwen3-8B Interactive Mode")
    print("Enter prompts for text generation (type 'back' to return to main menu)")
    
    def generate_text(prompt, max_length=150):
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    while True:
        prompt = input("\nQwen3 Prompt: ").strip()
        if prompt.lower() == 'back':
            break
        if not prompt:
            continue
            
        try:
            result = generate_text(prompt)
            print(f"Generated: {result[len(prompt):].strip()}")
        except Exception as e:
            print(f"❌ Error: {e}")

def interactive_nl_to_fol(model_lora, tokenizer_lora):
    """Interactive testing for NL-to-FOL model"""
    print("🦙 NL-to-FOL Interactive Mode")
    print("Enter natural language sentences to convert to First-Order Logic (type 'back' to return)")
    
    def formatting_func(text):
        return tokenizer_lora.apply_chat_template(
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
    
    def translate_nl_to_fol(input_text, max_tokens=150):
        prompt = formatting_func(input_text)
        inputs = tokenizer_lora(prompt, return_tensors="pt", padding=True)
        
        # Move inputs to the same device as the model
        device = next(model_lora.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model_lora.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                temperature=0.1, 
                do_sample=True,
                repetition_penalty=1.1
            )
        
        result = tokenizer_lora.decode(outputs[0], skip_special_tokens=True)
        return result
    
    # Show example translations first
    examples = [
        "All dogs are animals.",
        "Some birds can fly.",
        "If it's sunny, then people go outside.",
        "No cats are dogs."
    ]
    
    print("\n📚 Example translations:")
    for example in examples:
        try:
            result = translate_nl_to_fol(example, max_tokens=50)
            if "𝜙=" in result:
                fol_part = result.split("𝜙=")[-1].strip()
                print(f"  '{example}' → 𝜙={fol_part}")
        except Exception as e:
            print(f"  '{example}' → Error: {e}")
    
    print("\n💡 Logical symbols: ∀ (forall), ∃ (exists), ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (iff), ⊕ (xor)")
    
    while True:
        nl_input = input("\nNL sentence: ").strip()
        if nl_input.lower() == 'back':
            break
        if not nl_input:
            continue
            
        try:
            result = translate_nl_to_fol(nl_input)
            if "𝜙=" in result:
                fol_part = result.split("𝜙=")[-1].strip()
                print(f"FOL: 𝜙={fol_part}")
            else:
                print(f"Translation: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_test_nl_to_fol(model_lora, tokenizer_lora):
    """Batch test multiple NL sentences"""
    print("🧪 Batch Testing NL-to-FOL")
    
    test_sentences = [
        "All humans are mortal.",
        "Some students are smart.",
        "Every dog has an owner.",
        "No fish can fly.",
        "If someone studies hard, they will pass.",
        "All cats are either black or white.",
        "There exists a perfect number.",
        "Every teacher who is patient succeeds.",
        "Some books are interesting and educational.",
        "If it rains and the ground is dry, then the ground becomes wet."
    ]
    
    def formatting_func(text):
        return tokenizer_lora.apply_chat_template(
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
    
    def translate_nl_to_fol(input_text):
        prompt = formatting_func(input_text)
        inputs = tokenizer_lora(prompt, return_tensors="pt", padding=True)
        
        # Move inputs to the same device as the model
        device = next(model_lora.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model_lora.generate(
                **inputs, 
                max_new_tokens=100, 
                temperature=0.1, 
                do_sample=True
            )
        
        result = tokenizer_lora.decode(outputs[0], skip_special_tokens=True)
        return result
    
    print("Running batch translation tests...\n")
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"{i:2d}. Input: {sentence}")
        try:
            result = translate_nl_to_fol(sentence)
            if "𝜙=" in result:
                fol_part = result.split("𝜙=")[-1].strip()
                print(f"    FOL: 𝜙={fol_part}")
            else:
                print(f"    Output: {result}")
        except Exception as e:
            print(f"    ❌ Error: {e}")
        print()

def main_menu():
    """Main interactive menu"""
    print("\n" + "="*50)
    print("🚀 H100 GPU Interactive Model Testing")
    print("="*50)
    print("1. Interactive Qwen3-8B Text Generation")
    print("2. Interactive NL-to-FOL Translation") 
    print("3. Batch Test NL-to-FOL")
    print("4. Check GPU Status")
    print("5. Exit")
    print("="*50)
    
    choice = input("Choose option (1-5): ").strip()
    return choice

def main():
    """Main execution function"""
    print("🔄 Loading models...")
    
    # Check GPU first
    check_gpu()
    
    # Load models
    qwen_model, qwen_tokenizer = setup_qwen3()
    llama_model, llama_tokenizer = setup_llama_lora()
    
    print("✅ All models loaded successfully!\n")
    
    while True:
        choice = main_menu()
        
        if choice == '1':
            interactive_qwen3(qwen_model, qwen_tokenizer)
        elif choice == '2':
            interactive_nl_to_fol(llama_model, llama_tokenizer)
        elif choice == '3':
            batch_test_nl_to_fol(llama_model, llama_tokenizer)
        elif choice == '4':
            check_gpu()
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()