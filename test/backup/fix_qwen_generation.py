#!/usr/bin/env python3
"""
Fixed Qwen generation that handles earlier compatibility issues
"""

import torch
from setup_models import setup_qwen3, setup_llama_lora

def generate_qwen_text_fixed(model, tokenizer, prompt, max_new_tokens=200):
    """Fixed Qwen text generation that avoids attention issues"""
    
    # Try to disable flash attention if causing issues
    try:
        # Method 1: Use model.generate with specific parameters for stability
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=True,
                # Disable flash attention options that might cause issues
                attn_implementation="eager"  # Use eager attention instead of flash_attention_2
            )
        
        # Decode only the new tokens
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
        
    except Exception as e:
        print(f"⚠️  Standard generation failed: {e}")
        
        # Method 2: Fallback to basic token-by-token generation
        try:
            return generate_text_fallback(model, tokenizer, prompt, max_new_tokens)
        except Exception as e2:
            print(f"❌ Fallback generation also failed: {e2}")
            return None

def generate_text_fallback(model, tokenizer, prompt, max_new_tokens=200):
    """Fallback generation method for compatibility issues"""
    print("🔄 Using fallback generation method...")
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    input_length = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        # Simple generation without advanced attention mechanisms
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_length=input_length + max_new_tokens,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            use_cache=False,  # Disable cache to avoid attention issues
            num_return_sequences=1
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def install_xformers_fix():
    """Attempt to fix optional dependency installation"""
    import subprocess
    import sys
    
    print("🔧 Attempting to fix optional dependency installation...")
    
    try:
        # Try to install/upgrade package
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "xformers", "--upgrade", "--no-deps"
        ])
        print("✅ Installation attempted")
        return True
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False

def test_qwen_generation():
    """Test Qwen generation with the fixed method"""
    
    print("🚀 Testing Fixed Qwen Generation")
    print("=" * 40)
    
    # Load Qwen model
    print("📥 Loading Qwen model...")
    qwen_model, qwen_tokenizer = setup_qwen3()
    
    # Simple test prompt
    test_prompt = "Generate a simple logic problem: All cats are animals. Fluffy is a cat. Question: Is Fluffy an animal? Answer:"
    
    print(f"📝 Test prompt: {test_prompt}")
    print("\n🔄 Generating response...")
    
    # Try fixed generation
    result = generate_qwen_text_fixed(qwen_model, qwen_tokenizer, test_prompt, max_new_tokens=100)
    
    if result:
        print(f"✅ Generated text:")
        print(f"{result}")
        print("\n🎉 Generation successful!")
        return True
    else:
        print("❌ Generation failed")
        return False

def create_simple_problem_manually():
    """Create a simple problem manually if Qwen generation fails"""
    
    simple_problems = [
        {
            "context": "All birds can fly. Tweety is a bird. Flying animals are free.",
            "question": "Is Tweety free?",
            "reasoning": [
                "All birds can fly",
                "Tweety is a bird", 
                "Therefore, Tweety can fly",
                "Flying animals are free",
                "Therefore, Tweety is free"
            ],
            "answer": "A"
        },
        {
            "context": "All students study. Some students are smart. John is a student.",
            "question": "Is John smart?",
            "reasoning": [
                "All students study",
                "John is a student",
                "Therefore, John studies", 
                "Some students are smart",
                "We don't know if John is one of the smart students"
            ],
            "answer": "C"
        },
        {
            "context": "No cats are dogs. Fluffy is a cat. Dogs bark.",
            "question": "Does Fluffy bark?",
            "reasoning": [
                "No cats are dogs",
                "Fluffy is a cat",
                "Therefore, Fluffy is not a dog",
                "Dogs bark",
                "Since Fluffy is not a dog, Fluffy does not bark"
            ],
            "answer": "B"
        }
    ]
    
    import random
    problem = random.choice(simple_problems)
    
    print("📝 Using pre-defined logic problem:")
    print(f"Context: {problem['context']}")
    print(f"Question: {problem['question']}")
    print(f"Answer: {problem['answer']}")
    
    return problem

def solve_problem_with_fol(problem_data):
    """Convert problem reasoning to FOL"""
    
    print("\n🔄 Loading Llama model for FOL conversion...")
    llama_model, llama_tokenizer = setup_llama_lora()
    
    from fol_problem_solver import translate_nl_to_fol
    
    print("\n🔄 Converting reasoning to First-Order Logic:")
    print("-" * 50)
    
    fol_results = []
    for i, statement in enumerate(problem_data['reasoning'], 1):
        print(f"\n{i}. Statement: {statement}")
        
        try:
            fol_result = translate_nl_to_fol(llama_model, llama_tokenizer, statement)
            if "𝜙=" in fol_result:
                fol_formula = fol_result.split("𝜙=")[-1].strip()
                print(f"   FOL: 𝜙={fol_formula}")
                fol_results.append((statement, fol_formula))
            else:
                print(f"   FOL: {fol_result}")
                fol_results.append((statement, fol_result))
        except Exception as e:
            print(f"   ❌ Error: {e}")
            fol_results.append((statement, f"Error: {e}"))
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"Problem Answer: {problem_data['answer']}")
    print(f"Reasoning Statements: {len(problem_data['reasoning'])}")
    print(f"FOL Conversions: {len(fol_results)}")
    
    return fol_results

def main():
    """Main function with error handling"""
    
    print("🚀 FOL Logic Problem Solver (Fixed Version)")
    print("=" * 50)
    
    # First, try to test Qwen generation
    print("1️⃣ Testing Qwen generation...")
    
    qwen_works = test_qwen_generation()
    
    if not qwen_works:
        print("\n⚠️  Qwen generation has issues. Using manual problem instead.")
        print("This can happen due to optional dependency issues on some systems.")

        # Ask user if they want to attempt a dependency fix
        fix_choice = input("\nAttempt installation fix? (y/N): ").strip().lower()
        if fix_choice == 'y':
            install_xformers_fix()
            print("🔄 Restart Python and try again after the fix.")
            return
        
        # Use manual problem
        print("\n2️⃣ Using pre-defined problem...")
        problem_data = create_simple_problem_manually()
        
        # Convert to FOL
        print("\n3️⃣ Converting to FOL...")
        fol_results = solve_problem_with_fol(problem_data)
        
        print("\n✅ Demo completed with manual problem!")
    
    else:
        print("\n✅ Qwen generation works! You can use the full system.")
        print("Run: python fol_problem_solver.py")

if __name__ == "__main__":
    main()