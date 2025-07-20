#!/usr/bin/env python3
"""
Simple example of FOL problem generation and solving
Run this to see a quick demonstration
"""

from setup_models import setup_qwen3, setup_llama_lora
from fol_problem_solver import generate_and_solve_problem

def simple_demo():
    """Simple demonstration of the FOL problem solver"""
    
    print("🚀 Simple FOL Logic Problem Demo")
    print("=" * 50)
    print("Loading models (this may take a moment)...")
    
    # Load both models
    qwen_model, qwen_tokenizer = setup_qwen3()
    llama_model, llama_tokenizer = setup_llama_lora()
    
    print("✅ Models loaded!\n")
    
    # Generate and solve a problem
    result = generate_and_solve_problem(qwen_model, qwen_tokenizer, llama_model, llama_tokenizer)
    
    if result:
        print("\n🎉 Demo completed successfully!")
        print("\nThe system:")
        print("1. Generated a logic problem using Qwen")
        print("2. Extracted reasoning statements") 
        print("3. Converted each statement to First-Order Logic using Llama")
        print("4. Provided the final answer")
    else:
        print("\n❌ Demo encountered issues")

def manual_example():
    """Manual example with predefined problem"""
    
    print("🧪 Manual FOL Example")
    print("=" * 30)
    
    # Example problem (simpler than what Qwen generates)
    example_problem = """
Context: All cats are animals. Fluffy is a cat. Animals need food.

Question: Does Fluffy need food?

<reasoning>
All cats are animals.
Fluffy is a cat.
Therefore, Fluffy is an animal.
Animals need food.
Therefore, Fluffy needs food.
</reasoning>

<answer>
A
</answer>
"""
    
    print("📝 Example Problem:")
    print(example_problem)
    
    # Load just the Llama model for FOL conversion
    print("\n🔄 Loading Llama model for FOL conversion...")
    llama_model, llama_tokenizer = setup_llama_lora()
    
    # Reasoning statements to convert
    statements = [
        "All cats are animals",
        "Fluffy is a cat", 
        "Therefore, Fluffy is an animal",
        "Animals need food",
        "Therefore, Fluffy needs food"
    ]
    
    print("\n🔄 Converting to First-Order Logic:")
    print("-" * 40)
    
    from fol_problem_solver import translate_nl_to_fol
    
    for i, stmt in enumerate(statements, 1):
        print(f"\n{i}. Statement: {stmt}")
        try:
            fol_result = translate_nl_to_fol(llama_model, llama_tokenizer, stmt)
            if "𝜙=" in fol_result:
                fol_formula = fol_result.split("𝜙=")[-1].strip()
                print(f"   FOL: 𝜙={fol_formula}")
            else:
                print(f"   FOL: {fol_result}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n✅ Manual example completed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        manual_example()
    else:
        simple_demo()