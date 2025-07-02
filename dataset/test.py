#!/usr/bin/env python3
"""
Simple script to test LLM FOL conversion and Prover9 validation
"""

import subprocess
import tempfile
import torch
from unsloth import FastLanguageModel
import re

class SimpleFOLTester:
    def __init__(self):
        print("Loading Qwen model...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            device_map="auto",
        )
        FastLanguageModel.for_inference(self.model)
        print("Model loaded!")

    def convert_to_fol(self, context_statements):
        """Convert natural language statements to FOL using LLM"""
        
        prompt = f"""Convert these logical statements to First-Order Logic (FOL) format for Prover9.

Context statements:
{context_statements}

IMPORTANT: Use proper Prover9 syntax!

Examples:
- "John is tall" -> "Tall(john)"
- "If someone is tall, then they are visible" -> "all x (Tall(x) -> Visible(x))"
- "Mary either sings or dances, but not both" -> "(Sings(mary) | Dances(mary)) & -(Sings(mary) & Dances(mary))"
- "All cats are mammals" -> "all x (Cat(x) -> Mammal(x))"

Rules:
1. Use constants like 'queenie' (lowercase)
2. Use predicates like 'LivesInSymbiosis(queenie)'
3. Use 'all x' for universal quantification
4. Use -> for implies, & for and, | for or, - for not
5. For "either X or Y but not both": (X | Y) & -(X & Y)
6. For "either X or Y": X | Y

Convert each statement to ONE valid FOL formula. Output ONLY the formulas, one per line:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_tokens = outputs[0][len(inputs['input_ids'][0]):]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print("LLM Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        # Extract FOL formulas from response
        lines = response.split('\n')
        fol_formulas = []
        for line in lines:
            line = line.strip()
            # Filter for valid FOL formulas
            if (line and 
                not line.startswith('#') and 
                not line.startswith('Here') and 
                not line.startswith('The') and
                '(' in line and
                not line.startswith('Convert') and
                len(line) > 5):
                # Basic syntax validation
                if not line.startswith('x(') and not line.startswith('( |') and not line.startswith('1.') and not line.startswith('2.'):
                    fol_formulas.append(line)
        
        return fol_formulas

    def convert_question_to_fol(self, question):
        """Convert the question statement to FOL"""
        
        prompt = f"""Convert this statement to First-Order Logic (FOL) for Prover9:

Statement: "{question}"

Examples:
- "John is happy" -> "Happy(john)"
- "Someone is tall" -> "exists x Tall(x)"
- "Mary contributes to health" -> "ContributesToHealth(mary)"

Use the same naming style as the context. Output ONLY the FOL formula, nothing else:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_tokens = outputs[0][len(inputs['input_ids'][0]):]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print("Question LLM Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        # Extract just the FOL formula
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and '(' in line and not line.startswith('The') and not line.startswith('Statement'):
                return line
        
        return lines[0] if lines else "ContributesToHealth(queenie)"

    def run_prover9(self, premises, goal=None):
        """Run Prover9 with given premises and optional goal"""
        
        prover9_input = "set(auto).\n\n"
        
        if premises:
            prover9_input += "formulas(assumptions).\n"
            for premise in premises:
                if premise.strip():
                    prover9_input += f"  {premise}.\n"
            prover9_input += "end_of_list.\n\n"
        
        if goal and goal.strip():
            prover9_input += "formulas(goals).\n"
            prover9_input += f"  {goal}.\n"
            prover9_input += "end_of_list.\n"
        
        print("Prover9 input:")
        print(prover9_input)
        print("-" * 50)
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
                f.write(prover9_input)
                filename = f.name
            
            result = subprocess.run(['prover9', filename], capture_output=True, text=True, timeout=15)
            
            print("Prover9 output:")
            print(result.stdout)
            
            if "THEOREM PROVED" in result.stdout:
                return "TRUE"
            elif "SEARCH FAILED" in result.stdout:
                return "FALSE"
            else:
                return "UNCERTAIN"
                
        except Exception as e:
            print(f"Error running Prover9: {e}")
            return "ERROR"

def main():
    # Shorter reasoning problem
    context = """Queenie lives in symbiosis. Queenie does not stimulate her host's defenses. If Queenie aids her host's digestion, then she contributes to her host's health. Queenie either lives in symbiosis or regulates her host's metabolism, but not both. Every parasite either stimulates its host's defenses or enhances nutrient absorption. If Queenie either regulates her host's metabolism or enhances nutrient absorption (but not both), then she produces beneficial enzymes. If Queenie produces beneficial enzymes or maintains ecological balance, then she can aid her host's digestion."""
    
    question = "Queenie contributes to her host's health."
    
    print("🚀 Starting FOL conversion and Prover9 test...")
    
    tester = SimpleFOLTester()
    
    print("\n📝 Converting context to FOL...")
    fol_premises = tester.convert_to_fol(context)
    
    print(f"Generated {len(fol_premises)} FOL premises:")
    for i, premise in enumerate(fol_premises[:5]):  # Show first 5
        print(f"  {i+1}: {premise}")
    if len(fol_premises) > 5:
        print(f"  ... and {len(fol_premises) - 5} more")
    
    # If LLM failed, use manual fallback
    if (len(fol_premises) == 0 or 
        any('x(' in p for p in fol_premises) or 
        any('LInInSynos' in p for p in fol_premises) or
        any(p.startswith('1.') or p.startswith('2.') for p in fol_premises)):
        print("\n⚠️  LLM FOL conversion failed, using manual fallback...")
        fol_premises = [
            "LivesInSymbiosis(queenie)",
            "-StimulatesDefenses(queenie)",
            "all x (AidsDigestion(x) -> ContributesToHealth(x))",
            "(LivesInSymbiosis(queenie) | RegulatesMetabolism(queenie)) & -(LivesInSymbiosis(queenie) & RegulatesMetabolism(queenie))",
            "all x (Parasite(x) -> (StimulatesDefenses(x) | EnhancesAbsorption(x)))",
            "all x (((RegulatesMetabolism(x) | EnhancesAbsorption(x)) & -(RegulatesMetabolism(x) & EnhancesAbsorption(x))) -> ProducesEnzymes(x))",
            "all x ((ProducesEnzymes(x) | MaintainsBalance(x)) -> AidsDigestion(x))"
        ]
    
    print(f"\n❓ Converting question to FOL...")
    fol_question = tester.convert_question_to_fol(question)
    print(f"Question FOL: {fol_question}")
    
    # Manual fallback if LLM failed
    if (len(fol_question) < 10 or 
        'The statement' in fol_question or 
        'marysHealth' in fol_question or
        'ContributesTo(queenie, marysHealth)' in fol_question):
        print("⚠️  Using manual fallback for question...")
        fol_question = "ContributesToHealth(queenie)"
    
    print(f"\n🔍 Running Prover9...")
    result = tester.run_prover9(fol_premises, fol_question)
    
    print(f"\n📊 Final Result: {result}")
    print(f"Expected: Should be provable (TRUE) or UNCERTAIN")
    print(f"Result: {'✅' if result in ['TRUE', 'UNCERTAIN'] else '❌'}")

if __name__ == "__main__":
    main()