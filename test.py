#!/usr/bin/env python3
"""
Standalone script to test per-set validation approach using existing Prover9 integration
"""

import re
import subprocess
import tempfile
import os
from typing import List, Dict, Tuple

# Example problem with multiple reasoning sets
EXAMPLE_PROBLEM = {
    "question": """Maya creates art. If Maya engages in performance, then she can either create art or inspire audiences, but not both. Maya engages in performance. If Maya inspires audiences, then she gains recognition. Artists who gain recognition become successful.

Question: Based on the above information, is the following statement true, false, or uncertain? Maya is successful.

Options:
A) True
B) False
C) Uncertain""",
    
    "steps": """fact1: Maya creates art.
fact2: Maya engages in performance.
rule: If Maya engages in performance, then she can either create art or inspire audiences, but not both.
conclusion: Maya does not inspire audiences.

fact1: Maya does not inspire audiences.
rule: If Maya inspires audiences, then she gains recognition.
conclusion: Maya does not gain recognition.

fact1: Maya does not gain recognition.
rule: Artists who gain recognition become successful.
conclusion: Maya is not successful.""",
    
    "answer": "B"
}

# Import your existing Prover9 functions
def convert_fol_to_prover9(fol_formula):
    """Convert mathematical FOL notation to Prover9 syntax"""
    
    # Remove the φ= prefix if present
    if fol_formula.startswith('𝜙='):
        fol_formula = fol_formula[2:].strip()
    elif fol_formula.startswith('φ='):
        fol_formula = fol_formula[2:].strip()
    
    # Convert XOR to equivalent expression: A ⊕ B → (A | B) & -(A & B)
    xor_conversions = [
        (r'(\w+)\s*⊕\s*(\w+)', r'((\1 | \2) & -(\1 & \2))'),
        (r'(\w+)\s+XOR\s+(\w+)', r'((\1 | \2) & -(\1 & \2))'),
    ]
    
    result = fol_formula
    for pattern, replacement in xor_conversions:
        result = re.sub(pattern, replacement, result)
    
    # Prover9 syntax conversions
    conversions = [
        (r'∀(\w+)', r'all \1'),
        (r'∃(\w+)', r'exists \1'),
        (r'∧', r' & '),
        (r'∨', r' | '),
        (r'¬', r'-'),
        (r'→', r' -> '),
        (r'↔', r' <-> '),
        (r'\s+', r' '),
    ]
    
    for pattern, replacement in conversions:
        result = re.sub(pattern, replacement, result)
    
    return result.strip()

def create_prover9_input(premises, conclusion, problem_name="logic_problem"):
    """Create Prover9 input file content"""
    
    prover9_input = f"""% {problem_name}
% Generated FOL verification

formulas(assumptions).
"""
    
    # Add premises
    for premise in premises:
        if premise.strip():
            prover9_input += f"{premise}.\n"
    
    prover9_input += """end_of_list.

formulas(goals).
"""
    
    # Add conclusion
    if conclusion and conclusion.strip():
        prover9_input += f"{conclusion}.\n"
    
    prover9_input += """end_of_list.
"""
    
    return prover9_input

def run_prover9(input_content, timeout=10):
    """Run Prover9 with the given input"""
    
    try:
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write(input_content)
            input_file = f.name
        
        # Create temporary output file
        output_file = input_file.replace('.in', '.out')
        
        # Run Prover9
        cmd = ['prover9', '-f', input_file]
        
        result = subprocess.run(
            cmd,
            stdout=open(output_file, 'w'),
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True
        )
        
        # Read output
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                output = f.read()
        else:
            output = ""
        
        # Clean up
        try:
            os.unlink(input_file)
            os.unlink(output_file)
        except:
            pass
        
        return {
            'returncode': result.returncode,
            'stdout': output,
            'stderr': result.stderr if result.stderr else "",
            'success': result.returncode == 0
        }
        
    except subprocess.TimeoutExpired:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': 'Prover9 timeout',
            'success': False
        }
    except FileNotFoundError:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': 'Prover9 not found. Please ensure Prover9 is installed and in PATH.',
            'success': False
        }
    except Exception as e:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': f'Error running Prover9: {e}',
            'success': False
        }

def parse_prover9_output(output):
    """Parse Prover9 output to determine if proof was found"""
    
    if not output:
        return False, "No output from Prover9"
    
    # Check for proof indicators
    proof_found = False
    proof_info = ""
    
    if "THEOREM PROVED" in output:
        proof_found = True
        proof_info = "Theorem proved successfully"
    elif "SEARCH FAILED" in output:
        proof_found = False
        proof_info = "Proof search failed"
    elif "PROOF" in output and "PROVED" in output:
        proof_found = True
        proof_info = "Proof found"
    elif "proof" in output.lower():
        proof_found = True
        proof_info = "Proof found (partial match)"
    else:
        proof_found = False
        proof_info = "No clear proof indicator found"
    
    return proof_found, proof_info

def parse_reasoning_sets(steps_text: str) -> List[Dict]:
    """Parse reasoning steps into individual fact-rule-conclusion sets"""
    sets = []
    current_set = {"facts": [], "rules": [], "conclusions": []}
    
    lines = steps_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_set["facts"] or current_set["rules"] or current_set["conclusions"]:
                sets.append(current_set)
                current_set = {"facts": [], "rules": [], "conclusions": []}
            continue
            
        if line.startswith("fact"):
            fact_content = line.split(":", 1)[1].strip()
            current_set["facts"].append(fact_content)
        elif line.startswith("rule"):
            rule_content = line.split(":", 1)[1].strip()
            current_set["rules"].append(rule_content)
        elif line.startswith("conclusion"):
            conclusion_content = line.split(":", 1)[1].strip()
            current_set["conclusions"].append(conclusion_content)
    
    # Add final set if exists
    if current_set["facts"] or current_set["rules"] or current_set["conclusions"]:
        sets.append(current_set)
    
    return sets

def convert_to_fol_mock(reasoning_set: Dict) -> Tuple[List[str], List[str]]:
    """
    Mock NL to FOL conversion - returns (premises, conclusions)
    In practice, you'd call your llama model here
    """
    premises = []
    conclusions = []
    
    # Convert facts and rules to premises
    for fact in reasoning_set["facts"]:
        if "Maya creates art" in fact:
            premises.append("creates_art(maya)")
        elif "Maya engages in performance" in fact:
            premises.append("engages_performance(maya)")
        elif "Maya does not inspire audiences" in fact:
            premises.append("-inspires_audiences(maya)")
        elif "Maya does not gain recognition" in fact:
            premises.append("-gains_recognition(maya)")
    
    for rule in reasoning_set["rules"]:
        if "If Maya engages in performance" in rule and "either create art or inspire audiences" in rule:
            # XOR: Maya performs -> (creates_art XOR inspires_audiences)
            premises.append("engages_performance(maya) -> ((creates_art(maya) | inspires_audiences(maya)) & -(creates_art(maya) & inspires_audiences(maya)))")
        elif "If Maya inspires audiences, then she gains recognition" in rule:
            premises.append("inspires_audiences(maya) -> gains_recognition(maya)")
        elif "Artists who gain recognition become successful" in rule:
            premises.append("all X (gains_recognition(X) -> successful(X))")
    
    # Convert conclusions to goals
    for conclusion in reasoning_set["conclusions"]:
        if "Maya does not inspire audiences" in conclusion:
            conclusions.append("-inspires_audiences(maya)")
        elif "Maya does not gain recognition" in conclusion:
            conclusions.append("-gains_recognition(maya)")
        elif "Maya is not successful" in conclusion:
            conclusions.append("-successful(maya)")
    
    return premises, conclusions

def validate_reasoning_set(reasoning_set: Dict) -> bool:
    """Validate a single reasoning set with Prover9"""
    
    # Convert to FOL
    premises, conclusions = convert_to_fol_mock(reasoning_set)
    
    if not premises or not conclusions:
        return False
    
    # Convert to Prover9 syntax
    prover9_premises = [convert_fol_to_prover9(p) for p in premises]
    prover9_conclusions = [convert_fol_to_prover9(c) for c in conclusions]
    
    # Create Prover9 input
    prover9_input = create_prover9_input(prover9_premises, prover9_conclusions[0])  # Use first conclusion as goal
    
    print(f"Prover9 input:\n{prover9_input}")
    
    # Run Prover9
    result = run_prover9(prover9_input)
    
    if result['stdout']:
        proof_found, proof_info = parse_prover9_output(result['stdout'])
        print(f"Prover9 result: {proof_info}")
        return proof_found
    else:
        print(f"Prover9 error: {result['stderr']}")
        return False

def validate_reasoning_sets(steps: str) -> Tuple[List[bool], float]:
    """
    Validate each reasoning set and return binary scores
    """
    # Parse into sets
    reasoning_sets = parse_reasoning_sets(steps)
    print(f"Found {len(reasoning_sets)} reasoning sets")
    
    validation_results = []
    
    for i, reasoning_set in enumerate(reasoning_sets):
        print(f"\n--- Set {i+1} ---")
        print(f"Facts: {reasoning_set['facts']}")
        print(f"Rules: {reasoning_set['rules']}")
        print(f"Conclusions: {reasoning_set['conclusions']}")
        
        # Validate with Prover9
        is_valid = validate_reasoning_set(reasoning_set)
        validation_results.append(is_valid)
        print(f"Valid: {is_valid}")
    
    # Calculate overall score
    overall_score = sum(validation_results) / len(validation_results) if validation_results else 0.0
    
    return validation_results, overall_score

def main():
    print("=== Standalone Per-Set Validation Test ===")
    print(f"Question: {EXAMPLE_PROBLEM['question'][:100]}...")
    print(f"Expected Answer: {EXAMPLE_PROBLEM['answer']}")
    print(f"\nSteps:\n{EXAMPLE_PROBLEM['steps']}")
    
    # Validate each set
    results, score = validate_reasoning_sets(EXAMPLE_PROBLEM['steps'])
    
    print(f"\n=== RESULTS ===")
    print(f"Set-by-set validation: {results}")
    print(f"Overall score: {score:.2f}")
    print(f"Valid sets: {sum(results)}/{len(results)}")

if __name__ == "__main__":
    main()

def validate_with_prover9(reasoning_set: Dict) -> bool:
    """
    Validate reasoning set with Prover9
    Returns True if conclusion can be proven from facts and rules
    """
    try:
        # Convert each part to FOL
        assumptions = []
        goals = []
        
        # Add facts as assumptions
        for fact in reasoning_set["facts"]:
            if "Maya creates art" in fact:
                assumptions.append("creates_art(maya).")
            elif "Maya engages in performance" in fact:
                assumptions.append("engages_performance(maya).")
            elif "Maya does not inspire audiences" in fact:
                assumptions.append("-inspires_audiences(maya).")
            elif "Maya does not gain recognition" in fact:
                assumptions.append("-gains_recognition(maya).")
        
        # Add rules as assumptions
        for rule in reasoning_set["rules"]:
            if "If Maya engages in performance" in rule and "either create art or inspire audiences" in rule:
                # If Maya performs, then (creates_art XOR inspires_audiences)
                assumptions.append("engages_performance(maya) -> (creates_art(maya) & -inspires_audiences(maya) | -creates_art(maya) & inspires_audiences(maya)).")
            elif "If Maya inspires audiences, then she gains recognition" in rule:
                assumptions.append("inspires_audiences(maya) -> gains_recognition(maya).")
            elif "Artists who gain recognition become successful" in rule:
                assumptions.append("all X (gains_recognition(X) -> successful(X)).")
        
        # Add conclusions as goals to prove
        for conclusion in reasoning_set["conclusions"]:
            if "Maya does not inspire audiences" in conclusion:
                goals.append("-inspires_audiences(maya).")
            elif "Maya does not gain recognition" in conclusion:
                goals.append("-gains_recognition(maya).")
            elif "Maya is not successful" in conclusion:
                goals.append("-successful(maya).")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            # Write assumptions
            f.write("formulas(assumptions).\n")
            for assumption in assumptions:
                f.write(f"  {assumption}\n")
            f.write("end_of_list.\n\n")
            
            # Write goals
            f.write("formulas(goals).\n")
            for goal in goals:
                f.write(f"  {goal}\n")
            f.write("end_of_list.\n")
            
            input_file = f.name
        
        print(f"Prover9 input:\n{open(input_file).read()}")
        
        # Run Prover9
        result = subprocess.run(
            ['prover9', '-f', input_file],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Clean up
        os.unlink(input_file)
        
        # Check if proof was found
        proof_found = "PROOF" in result.stdout
        if not proof_found:
            print(f"Prover9 output: {result.stdout[:200]}...")
        
        return proof_found
        
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"Prover9 error: {e}")
        return False

def validate_reasoning_sets(steps: str) -> Tuple[List[bool], float]:
    """
    Validate each reasoning set and return binary scores
    """
    # Parse into sets
    reasoning_sets = parse_reasoning_sets(steps)
    print(f"Found {len(reasoning_sets)} reasoning sets")
    
    validation_results = []
    
    for i, reasoning_set in enumerate(reasoning_sets):
        print(f"\n--- Set {i+1} ---")
        print(f"Facts: {reasoning_set['facts']}")
        print(f"Rules: {reasoning_set['rules']}")
        print(f"Conclusions: {reasoning_set['conclusions']}")
        
        # Validate with Prover9 (FOL conversion happens inside)
        is_valid = validate_with_prover9(reasoning_set)
        validation_results.append(is_valid)
        print(f"Valid: {is_valid}")
    
    # Calculate overall score
    overall_score = sum(validation_results) / len(validation_results) if validation_results else 0.0
    
    return validation_results, overall_score

def main():
    print("=== Standalone Per-Set Validation Test ===")
    print(f"Question: {EXAMPLE_PROBLEM['question'][:100]}...")
    print(f"Expected Answer: {EXAMPLE_PROBLEM['answer']}")
    print(f"\nSteps:\n{EXAMPLE_PROBLEM['steps']}")
    
    # Validate each set
    results, score = validate_reasoning_sets(EXAMPLE_PROBLEM['steps'])
    
    print(f"\n=== RESULTS ===")
    print(f"Set-by-set validation: {results}")
    print(f"Overall score: {score:.2f}")
    print(f"Valid sets: {sum(results)}/{len(results)}")

if __name__ == "__main__":
    main()