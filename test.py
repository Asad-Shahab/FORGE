#!/usr/bin/env python3
"""
Test script to verify Prover9 pipeline works with a simple example
"""

from setup.setup_models import setup_llama_lora
from verification.prover9_integration import verify_reasoning_with_prover9, test_prover9_installation

def test_simple_reasoning():
    """Test with a simple logical reasoning example"""
    
    print("🧪 Testing Prover9 Pipeline with Complex Multi-Step Example")
    print("=" * 60)
    
    # Test Prover9 installation first
    if not test_prover9_installation():
        print("❌ Cannot proceed without Prover9")
        return False
    
    # Load Llama model
    print("\n🔄 Loading Llama model...")
    try:
        llama_model, llama_tokenizer = setup_llama_lora()
        print("✅ Llama model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load Llama model: {e}")
        return False
    
    # Complex test reasoning with 7 steps (fixed)
    test_reasoning = """<initial_reasoning>
This is a multi-step reasoning problem involving scientists, logical thinking, 
decision-making, and success. We need to trace through several conditional 
statements and specific facts to reach the conclusion.
</initial_reasoning>

<steps>
If someone is a scientist, then they are logical.
If someone is logical, then they make good decisions.
All people who make good decisions are successful.
John is a scientist.
Mary is either a scientist or an artist.
Mary is not an artist.
Therefore, both John and Mary are successful.
</steps>

<answer>A</answer>"""
    
    print("\n📝 Test Reasoning:")
    print("-" * 30)
    print(test_reasoning)
    print("-" * 30)
    
    # Parse the reasoning (simplified version of parse_qwen_solution)
    import re
    
    steps_match = re.search(r'<steps>(.*?)</steps>', test_reasoning, re.DOTALL | re.IGNORECASE)
    if not steps_match:
        print("❌ Could not extract steps")
        return False
    
    reasoning = steps_match.group(1).strip()
    
    # Split into individual statements
    reasoning_statements = []
    lines = reasoning.split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) > 5:
            reasoning_statements.append(line)
    
    print(f"\n🔍 Extracted {len(reasoning_statements)} reasoning statements:")
    for i, stmt in enumerate(reasoning_statements, 1):
        print(f"  {i}. {stmt}")
    
    # Convert to FOL using Llama (simplified version of convert_reasoning_to_fol)
    def translate_nl_to_fol(text):
        improved_system_prompt = """You are an expert at converting natural language statements into First-Order Logic (FOL). Follow these strict rules:

SYMBOLS: Use only ∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (if and only if), ⊕ (xor)

NAMING RULES:
- All names/entities must be lowercase (fluffy, john, stone)
- Use simple, clear predicate names (Cat(x), Animal(x))
- Be consistent with predicate names throughout

SIMPLICITY RULES:
- Keep expressions as simple as possible
- Avoid unnecessary quantifiers when dealing with specific individuals
- Use direct predicates: Cat(fluffy) instead of exists x (CatType(x) & Has(fluffy, x))

VARIABLE RULES:
- Never use unbound variables
- If you use ∃x or ∀x, make sure x appears in the formula
- For specific people/things, use their name directly as a constant

EXAMPLES:
"John is tall" → Tall(john)
"All scientists are logical" → all x (Scientist(x) → Logical(x))
"Mary is a scientist" → Scientist(mary)
"John is either a doctor or a teacher" → Doctor(john) ∨ Teacher(john)
"Mary is not an artist" → ¬Artist(mary)

Start your answer with '𝜙=' followed by the FOL formula. Do not include any other text."""
        
        prompt = llama_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": improved_system_prompt},
                {"role": "user", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        
        inputs = llama_tokenizer(prompt, return_tensors="pt", padding=True)
        device = next(llama_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = llama_model.generate(
                **inputs, 
                max_new_tokens=100, 
                temperature=0.1, 
                do_sample=True
            )
        
        result = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
    
    # Convert each statement to FOL
    print("\n🔄 Converting to FOL...")
    fol_statements = []
    
    for statement in reasoning_statements:
        try:
            fol_result = translate_nl_to_fol(statement)
            if "𝜙=" in fol_result:
                fol_formula = fol_result.split("𝜙=")[-1].strip()
            else:
                fol_formula = fol_result.strip()
            
            fol_statements.append((statement, fol_formula))
            print(f"✅ '{statement}' → {fol_formula}")
            
        except Exception as e:
            print(f"❌ '{statement}' → Error: {e}")
            fol_statements.append((statement, f"Error: {e}"))
    
    if not fol_statements:
        print("❌ No FOL statements generated")
        return False
    
    # Test with Prover9
    print(f"\n🔍 Testing with Prover9...")
    verification_result = verify_reasoning_with_prover9(fol_statements, "Complex Scientist Logic Test")
    
    # Final result
    success = verification_result.get('valid', False)
    print(f"\n🏆 FINAL RESULT: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print("🎉 The pipeline works correctly!")
    else:
        print("⚠️  The pipeline needs debugging")
        print(f"Details: {verification_result.get('details', 'Unknown error')}")
    
    return success

if __name__ == "__main__":
    import torch
    test_simple_reasoning()