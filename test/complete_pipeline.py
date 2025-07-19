#!/usr/bin/env python3
"""
Complete FOL + Prover9 Pipeline
End-to-end system: NL → FOL → Prover9 Verification
"""

from setup_models import setup_llama_lora
from prover9_integration import verify_reasoning_with_prover9, test_prover9_installation

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

def complete_verification_pipeline(statements, problem_name="Logic Problem"):
    """Complete pipeline: NL statements → FOL → Prover9 verification"""
    
    print(f"🚀 Complete Verification Pipeline: {problem_name}")
    print("=" * 60)
    
    # Step 1: Check Prover9
    print("1️⃣ Checking Prover9...")
    if not test_prover9_installation():
        print("❌ Prover9 not available. Cannot verify reasoning.")
        return None
    print("✅ Prover9 ready")
    
    # Step 2: Load Llama model
    print("\n2️⃣ Loading NL-to-FOL model...")
    llama_model, llama_tokenizer = setup_llama_lora()
    print("✅ Model loaded")
    
    # Step 3: Convert NL to FOL
    print(f"\n3️⃣ Converting {len(statements)} statements to FOL...")
    fol_statements = []
    
    for i, statement in enumerate(statements, 1):
        print(f"\n  {i}. Converting: {statement}")
        
        try:
            fol_result = translate_nl_to_fol(llama_model, llama_tokenizer, statement)
            if "𝜙=" in fol_result:
                fol_formula = fol_result.split("𝜙=")[-1].strip()
            else:
                fol_formula = fol_result.strip()
            
            fol_statements.append((statement, fol_formula))
            print(f"     FOL: {fol_formula}")
            
        except Exception as e:
            print(f"     ❌ Error: {e}")
            fol_statements.append((statement, f"Error: {e}"))
    
    # Step 4: Verify with Prover9
    print(f"\n4️⃣ Verifying reasoning with Prover9...")
    
    if fol_statements:
        verification_result = verify_reasoning_with_prover9(fol_statements, problem_name)
        
        # Final result
        print("\n" + "🏆" * 20)
        print(f"🎯 FINAL RESULT: {problem_name}")
        print(f"📊 Statements processed: {len(statements)}")
        print(f"🔄 FOL conversions: {len([f for s, f in fol_statements if not f.startswith('Error')])}")
        print(f"✅ Reasoning validity: {'VALID' if verification_result['valid'] else 'INVALID'}")
        print(f"📋 Prover9 says: {verification_result['details']}")
        print("🏆" * 20)
        
        return verification_result
    else:
        print("❌ No valid FOL statements to verify")
        return None

def run_predefined_tests():
    """Run tests with predefined logic problems"""
    
    test_problems = [
        {
            "name": "Classic Syllogism",
            "statements": [
                "All humans are mortal",
                "Socrates is a human", 
                "Therefore, Socrates is mortal"
            ],
            "expected": True  # Should be valid
        },
        {
            "name": "Animal Classification",
            "statements": [
                "All cats are animals",
                "All animals need food",
                "Fluffy is a cat",
                "Therefore, Fluffy needs food"
            ],
            "expected": True  # Should be valid
        },
        {
            "name": "Invalid Generalization",
            "statements": [
                "Some birds can fly",
                "Tweety is a bird",
                "Therefore, Tweety can fly"
            ],
            "expected": False  # Should be invalid (hasty generalization)
        },
        {
            "name": "Negation Logic",
            "statements": [
                "No cats are dogs",
                "Fluffy is a cat",
                "Therefore, Fluffy is not a dog"
            ],
            "expected": True  # Should be valid
        }
    ]
    
    results = []
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n{'🔬' * 20}")
        print(f"TEST {i}/{len(test_problems)}: {problem['name']}")
        print(f"Expected: {'VALID' if problem['expected'] else 'INVALID'}")
        print('🔬' * 20)
        
        result = complete_verification_pipeline(problem['statements'], problem['name'])
        
        if result:
            actual_valid = result['valid']
            expected_valid = problem['expected']
            
            test_passed = actual_valid == expected_valid
            
            print(f"\n🧪 Test Result: {'✅ PASS' if test_passed else '❌ FAIL'}")
            print(f"   Expected: {'VALID' if expected_valid else 'INVALID'}")
            print(f"   Actual:   {'VALID' if actual_valid else 'INVALID'}")
            
            results.append({
                'name': problem['name'],
                'passed': test_passed,
                'expected': expected_valid,
                'actual': actual_valid
            })
        else:
            print("🧪 Test Result: ❌ FAIL (No result)")
            results.append({
                'name': problem['name'],
                'passed': False,
                'expected': problem['expected'],
                'actual': None
            })
    
    # Summary
    print(f"\n{'📊' * 20}")
    print("TEST SUMMARY")
    print('📊' * 20)
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    for result in results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{status} - {result['name']}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    return results

def interactive_pipeline():
    """Interactive mode for the complete pipeline"""
    
    print("🎯 Interactive FOL Verification Pipeline")
    print("Enter your reasoning statements one by one")
    print("Type 'done' when finished, 'quit' to exit")
    print()
    
    while True:
        statements = []
        
        print("📝 Enter reasoning statements:")
        while True:
            stmt = input(f"Statement {len(statements)+1}: ").strip()
            
            if stmt.lower() == 'quit':
                return
            elif stmt.lower() == 'done':
                break
            elif stmt:
                statements.append(stmt)
        
        if not statements:
            print("❌ No statements entered")
            continue
        
        # Get problem name
        problem_name = input("Problem name (optional): ").strip()
        if not problem_name:
            problem_name = f"User Problem {len(statements)} statements"
        
        # Run pipeline
        result = complete_verification_pipeline(statements, problem_name)
        
        # Ask to continue
        print("\n" + "-" * 40)
        choice = input("Verify another problem? (y/N): ").strip().lower()
        if choice != 'y':
            break

def main():
    """Main menu"""
    
    print("🚀 Complete FOL + Prover9 Verification Pipeline")
    print("=" * 50)
    print("1. Run predefined tests")
    print("2. Interactive verification")
    print("3. Single quick test")
    print("4. Exit")
    
    while True:
        choice = input("\nChoose option (1-4): ").strip()
        
        if choice == '1':
            run_predefined_tests()
            
        elif choice == '2':
            interactive_pipeline()
            
        elif choice == '3':
            # Quick single test
            quick_statements = [
                "All roses are flowers",
                "This plant is a rose", 
                "Therefore, this plant is a flower"
            ]
            complete_verification_pipeline(quick_statements, "Quick Test")
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()