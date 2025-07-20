#!/usr/bin/env python3
"""
FOL Logic Problem Generator and Solver
Generates simple logic problems with Qwen and converts reasoning to FOL with Llama
"""

import re
import torch
from setup_models import setup_qwen3, setup_llama_lora, check_gpu

def generate_qwen_text(model, tokenizer, prompt, max_length=300):
    """Generate text using Qwen model"""
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
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def translate_nl_to_fol(model_lora, tokenizer_lora, text):
    """Convert natural language to FOL using Llama model"""
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
    
    prompt = formatting_func(text)
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

def create_logic_problem_prompt():
    """Create a prompt for generating simple logic problems"""
    prompt = """Generate a simple logic problem with the following format:

Create a short context (2-3 sentences) involving basic logical relationships like:
- All X are Y
- Some X are Y  
- If X then Y
- X is a Y

Then ask a question that can be answered as True, False, or Uncertain.

Provide your reasoning step by step, then give your final answer.

Format your response exactly like this:
Context: [2-3 simple sentences with logical relationships]

Question: [A clear question about the context]

<reasoning>
[Step by step logical reasoning, one statement per line]
</reasoning>

<answer>
[A, B, or C where A=True, B=False, C=Uncertain]
</answer>

Example topic: animals, professions, objects, or basic properties.

Generate a logic problem now:"""
    
    return prompt

def parse_qwen_output(text):
    """Parse reasoning and answer from Qwen output"""
    # Extract reasoning section
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    
    # Extract answer section
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""
    
    # Split reasoning into individual statements
    reasoning_statements = []
    if reasoning:
        lines = reasoning.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('-') and len(line) > 10:  # Filter out short lines and dashes
                # Clean up the line
                line = re.sub(r'^\d+\.\s*', '', line)  # Remove numbering
                line = re.sub(r'^-\s*', '', line)      # Remove dashes
                reasoning_statements.append(line)
    
    return reasoning_statements, answer

def generate_and_solve_problem(qwen_model, qwen_tokenizer, llama_model, llama_tokenizer):
    """Generate a logic problem and convert reasoning to FOL"""
    
    print("🧠 Generating Logic Problem with Qwen...")
    print("=" * 60)
    
    # Generate the problem
    prompt = create_logic_problem_prompt()
    qwen_output = generate_qwen_text(qwen_model, qwen_tokenizer, prompt, max_length=400)
    
    # Extract the generated part (remove the prompt)
    if prompt in qwen_output:
        generated_part = qwen_output.replace(prompt, "").strip()
    else:
        generated_part = qwen_output
    
    print("📝 Generated Problem:")
    print(generated_part)
    print("\n" + "=" * 60)
    
    # Parse reasoning and answer
    reasoning_statements, answer = parse_qwen_output(generated_part)
    
    if not reasoning_statements:
        print("❌ Could not extract reasoning statements")
        return
    
    print("🔍 Extracted Reasoning Statements:")
    for i, stmt in enumerate(reasoning_statements, 1):
        print(f"{i}. {stmt}")
    
    print(f"\n🎯 Final Answer: {answer}")
    print("\n" + "=" * 60)
    
    # Convert each reasoning statement to FOL
    print("🔄 Converting Reasoning to First-Order Logic...")
    print()
    
    fol_results = []
    for i, statement in enumerate(reasoning_statements, 1):
        print(f"Statement {i}: {statement}")
        
        try:
            fol_result = translate_nl_to_fol(llama_model, llama_tokenizer, statement)
            if "𝜙=" in fol_result:
                fol_formula = fol_result.split("𝜙=")[-1].strip()
                print(f"FOL: 𝜙={fol_formula}")
                fol_results.append((statement, fol_formula))
            else:
                print(f"FOL: {fol_result}")
                fol_results.append((statement, fol_result))
        except Exception as e:
            print(f"❌ Error converting to FOL: {e}")
            fol_results.append((statement, f"Error: {e}"))
        
        print()
    
    print("=" * 60)
    print("📊 Summary:")
    print(f"Problem Answer: {answer}")
    print(f"Reasoning Statements: {len(reasoning_statements)}")
    print(f"FOL Conversions: {len(fol_results)}")
    
    return {
        'problem': generated_part,
        'reasoning': reasoning_statements,
        'answer': answer,
        'fol_results': fol_results
    }

def interactive_problem_solver():
    """Interactive logic problem solver"""
    print("🚀 Loading models...")
    
    # Load models
    qwen_model, qwen_tokenizer = setup_qwen3()
    llama_model, llama_tokenizer = setup_llama_lora()
    
    print("✅ Models loaded successfully!\n")
    
    while True:
        print("🎯 Logic Problem Generator and Solver")
        print("1. Generate new logic problem")
        print("2. Manual reasoning conversion")
        print("3. Exit")
        
        choice = input("\nChoose option (1-3): ").strip()
        
        if choice == '1':
            print()
            result = generate_and_solve_problem(qwen_model, qwen_tokenizer, llama_model, llama_tokenizer)
            
            # Ask if user wants to save results
            save_choice = input("\nSave results to file? (y/N): ").strip().lower()
            if save_choice == 'y':
                save_results(result)
        
        elif choice == '2':
            print("\n🔄 Manual Reasoning Conversion")
            text = input("Enter reasoning statement: ").strip()
            if text:
                try:
                    fol_result = translate_nl_to_fol(llama_model, llama_tokenizer, text)
                    if "𝜙=" in fol_result:
                        fol_formula = fol_result.split("𝜙=")[-1].strip()
                        print(f"FOL: 𝜙={fol_formula}")
                    else:
                        print(f"FOL: {fol_result}")
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        elif choice == '3':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")
        
        print("\n" + "-" * 40 + "\n")

def save_results(result):
    """Save results to a file"""
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logic_problem_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to: {filename}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")

def run_batch_problems(num_problems=3):
    """Generate multiple problems for testing"""
    print("🚀 Loading models...")
    
    # Load models
    qwen_model, qwen_tokenizer = setup_qwen3()
    llama_model, llama_tokenizer = setup_llama_lora()
    
    print("✅ Models loaded successfully!\n")
    
    all_results = []
    
    for i in range(num_problems):
        print(f"\n🔢 Problem {i+1}/{num_problems}")
        print("=" * 60)
        
        result = generate_and_solve_problem(qwen_model, qwen_tokenizer, llama_model, llama_tokenizer)
        if result:
            all_results.append(result)
        
        if i < num_problems - 1:
            print("\n" + "🔄 Generating next problem...\n")
    
    # Save batch results
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"batch_logic_problems_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Batch results saved to: {filename}")
    except Exception as e:
        print(f"\n❌ Error saving batch file: {e}")
    
    return all_results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        num = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        run_batch_problems(num)
    else:
        interactive_problem_solver()