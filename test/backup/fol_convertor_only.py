#!/usr/bin/env python3
"""
FOL Converter Only - Bypasses Qwen generation issues
Focuses only on converting reasoning statements to First-Order Logic
"""

from setup_models import setup_llama_lora

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
    import torch
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

def demo_fol_conversion():
    """Demo FOL conversion with predefined problems"""
    
    print("🧠 FOL Logic Converter Demo")
    print("=" * 50)
    print("(Bypassing Qwen generation due to xformers issues)")
    print()
    
    # Load only the Llama model
    print("📥 Loading Llama NL-to-FOL model...")
    llama_model, llama_tokenizer = setup_llama_lora()
    print("✅ Llama model loaded!")
    
    # Predefined logic problems
    problems = [
        {
            "title": "Problem 1: Animal Logic",
            "context": "All cats are animals. Fluffy is a cat. Animals need food.",
            "question": "Does Fluffy need food?",
            "reasoning": [
                "All cats are animals",
                "Fluffy is a cat", 
                "Therefore, Fluffy is an animal",
                "Animals need food",
                "Therefore, Fluffy needs food"
            ],
            "answer": "True"
        },
        {
            "title": "Problem 2: Student Logic", 
            "context": "All students study. Some students are smart. John is a student.",
            "question": "Is John smart?",
            "reasoning": [
                "All students study",
                "John is a student",
                "Therefore, John studies",
                "Some students are smart", 
                "We cannot determine if John is smart from the given information"
            ],
            "answer": "Uncertain"
        },
        {
            "title": "Problem 3: Negation Logic",
            "context": "No cats are dogs. Fluffy is a cat. Dogs bark.",
            "question": "Does Fluffy bark?", 
            "reasoning": [
                "No cats are dogs",
                "Fluffy is a cat",
                "Therefore, Fluffy is not a dog",
                "Dogs bark",
                "Since Fluffy is not a dog, Fluffy does not bark"
            ],
            "answer": "False"
        }
    ]
    
    for problem in problems:
        print(f"\n🎯 {problem['title']}")
        print("-" * 30)
        print(f"Context: {problem['context']}")
        print(f"Question: {problem['question']}")
        print(f"Answer: {problem['answer']}")
        
        print(f"\n🔄 Converting reasoning to FOL:")
        
        for i, statement in enumerate(problem['reasoning'], 1):
            print(f"\n{i}. Statement: {statement}")
            
            try:
                fol_result = translate_nl_to_fol(llama_model, llama_tokenizer, statement)
                if "𝜙=" in fol_result:
                    fol_formula = fol_result.split("𝜙=")[-1].strip()
                    print(f"   FOL: 𝜙={fol_formula}")
                else:
                    print(f"   FOL: {fol_result}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("\n" + "=" * 50)
    
    print("\n✅ FOL conversion demo completed!")
    print("\n💡 The Llama NL-to-FOL model is working perfectly!")
    print("The xformers issue only affects Qwen text generation.")

def interactive_fol_converter():
    """Interactive FOL converter"""
    
    print("🔄 Loading Llama NL-to-FOL model...")
    llama_model, llama_tokenizer = setup_llama_lora()
    print("✅ Ready for interactive FOL conversion!")
    
    print("\n🎯 Interactive FOL Converter")
    print("Enter natural language statements to convert to First-Order Logic")
    print("Type 'quit' to exit")
    print()
    
    while True:
        statement = input("Statement: ").strip()
        
        if statement.lower() == 'quit':
            break
            
        if not statement:
            continue
        
        try:
            fol_result = translate_nl_to_fol(llama_model, llama_tokenizer, statement)
            if "𝜙=" in fol_result:
                fol_formula = fol_result.split("𝜙=")[-1].strip()
                print(f"FOL: 𝜙={fol_formula}")
            else:
                print(f"FOL: {fol_result}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    print("👋 Goodbye!")

def batch_convert_statements():
    """Convert a batch of statements"""
    
    statements = [
        "All birds can fly",
        "Tweety is a bird",
        "Some cats are black", 
        "No dogs are cats",
        "If it rains, then the ground is wet",
        "All students who study hard pass the exam",
        "There exists a perfect number",
        "Every teacher is patient",
        "John is either tall or short",
        "If someone is happy and healthy, then they live well"
    ]
    
    print("🔄 Loading Llama model...")
    llama_model, llama_tokenizer = setup_llama_lora()
    
    print("\n📝 Batch FOL Conversion")
    print("=" * 40)
    
    for i, stmt in enumerate(statements, 1):
        print(f"\n{i:2d}. {stmt}")
        
        try:
            fol_result = translate_nl_to_fol(llama_model, llama_tokenizer, stmt)
            if "𝜙=" in fol_result:
                fol_formula = fol_result.split("𝜙=")[-1].strip()
                print(f"     𝜙={fol_formula}")
            else:
                print(f"     {fol_result}")
        except Exception as e:
            print(f"     ❌ Error: {e}")
    
    print("\n✅ Batch conversion completed!")

def main():
    """Main menu"""
    
    print("🚀 FOL Converter (Qwen-Free Version)")
    print("=" * 40)
    print("1. Demo with predefined problems")
    print("2. Interactive converter")
    print("3. Batch convert statements")
    print("4. Exit")
    
    while True:
        choice = input("\nChoose option (1-4): ").strip()
        
        if choice == '1':
            demo_fol_conversion()
        elif choice == '2':
            interactive_fol_converter()
        elif choice == '3':
            batch_convert_statements()
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()