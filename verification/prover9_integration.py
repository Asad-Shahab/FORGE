#!/usr/bin/env python3
"""
Prover9 Integration for FOL Verification
Converts FOL formulas to Prover9 syntax and verifies logical reasoning
"""

import re
import os
import subprocess
import tempfile
from pathlib import Path
from setup.setup_models import setup_llama_lora

def convert_fol_to_prover9(fol_formula):
    """Convert mathematical FOL notation to Prover9 syntax"""
    
    # Remove the φ= prefix if present
    if fol_formula.startswith('𝜙='):
        fol_formula = fol_formula[2:].strip()
    elif fol_formula.startswith('φ='):
        fol_formula = fol_formula[2:].strip()
    
    # Prover9 syntax conversions
    conversions = [
        # Quantifiers
        (r'∀(\w+)', r'all \1'),
        (r'∃(\w+)', r'exists \1'),
        
        # Logical operators
        (r'∧', r' & '),
        (r'∨', r' | '),
        (r'¬', r'-'),
        (r'→', r' -> '),
        (r'↔', r' <-> '),
        (r'⊕', r' XOR '),  # XOR might need special handling
        
        # Clean up extra spaces
        (r'\s+', r' '),
    ]
    
    result = fol_formula
    for pattern, replacement in conversions:
        result = re.sub(pattern, replacement, result)
    
    return result.strip()

def extract_premises_and_conclusion(fol_statements):
    """Extract premises and conclusion from reasoning statements"""
    
    premises = []
    conclusion = None
    
    for i, (statement, fol) in enumerate(fol_statements):
        # Convert to Prover9 syntax
        prover9_fol = convert_fol_to_prover9(fol)
        
        # Check if this is a conclusion (usually contains "therefore" or is the last statement)
        is_conclusion = (
            "therefore" in statement.lower() or 
            "thus" in statement.lower() or
            "hence" in statement.lower() or
            i == len(fol_statements) - 1  # Last statement
        )
        
        if is_conclusion and conclusion is None:
            conclusion = prover9_fol
        else:
            premises.append(prover9_fol)
    
    return premises, conclusion

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
        
        print(f"🔄 Running: {' '.join(cmd)}")
        
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
        
        # Read stderr
        stderr = result.stderr if result.stderr else ""
        
        # Clean up
        try:
            os.unlink(input_file)
            os.unlink(output_file)
        except:
            pass
        
        return {
            'returncode': result.returncode,
            'stdout': output,
            'stderr': stderr,
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

def verify_reasoning_with_prover9(fol_statements, problem_name="Logic Problem"):
    """Main function to verify reasoning using Prover9"""
    
    print(f"🔍 Verifying '{problem_name}' with Prover9")
    print("=" * 60)
    
    # Extract premises and conclusion
    premises, conclusion = extract_premises_and_conclusion(fol_statements)
    
    print("📝 Prover9 Input:")
    print(f"Premises ({len(premises)}):")
    for i, premise in enumerate(premises, 1):
        print(f"  {i}. {premise}")
    
    print(f"Conclusion: {conclusion}")
    print()
    
    # Create Prover9 input
    prover9_input = create_prover9_input(premises, conclusion, problem_name)
    
    print("📄 Generated Prover9 Input File:")
    print("-" * 30)
    print(prover9_input)
    print("-" * 30)
    
    # Run Prover9
    result = run_prover9(prover9_input)
    
    print(f"🔧 Prover9 Status: {'✅ Success' if result['success'] else '❌ Failed'}")
    
    if result['stderr']:
        print(f"⚠️  Stderr: {result['stderr']}")
    
    if result['stdout']:
        print("📊 Prover9 Output:")
        print("-" * 30)
        print(result['stdout'])
        print("-" * 30)
        
        # Parse output
        proof_found, proof_info = parse_prover9_output(result['stdout'])
        
        print(f"🎯 Proof Status: {'✅ VALID' if proof_found else '❌ INVALID'}")
        print(f"📋 Details: {proof_info}")
    else:
        print("❌ No output from Prover9")
        proof_found = False
        proof_info = "No output"
    
    print("=" * 60)
    
    return {
        'valid': proof_found,
        'details': proof_info,
        'prover9_input': prover9_input,
        'prover9_output': result['stdout'],
        'premises': premises,
        'conclusion': conclusion
    }

def test_prover9_installation():
    """Test if Prover9 is properly installed"""
    
    print("🔧 Testing Prover9 Installation")
    print("-" * 30)
    
    try:
        result = subprocess.run(['prover9', '--help'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0 or 'prover9' in result.stdout.lower():
            print("✅ Prover9 is installed and accessible")
            return True
        else:
            print("❌ Prover9 responded but may have issues")
            print(f"Output: {result.stdout[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  Prover9 help command timed out")
        return False
    except FileNotFoundError:
        print("❌ Prover9 not found in PATH")
        print("💡 Please install Prover9 and ensure it's in your PATH")
        return False
    except Exception as e:
        print(f"❌ Error testing Prover9: {e}")
        return False

def demo_prover9_verification():
    """Demo Prover9 verification with sample problems"""
    
    # Test Prover9 installation first
    if not test_prover9_installation():
        print("\n❌ Cannot proceed without Prover9")
        return False
    
    print("\n🧠 Loading Llama NL-to-FOL model...")
    llama_model, llama_tokenizer = setup_llama_lora()
    
    # Use local translate_nl_to_fol function
    
    # Sample problems
    problems = [
        {
            "name": "Simple Animal Logic",
            "statements": [
                "All cats are animals",
                "Fluffy is a cat",
                "Therefore, Fluffy is an animal"
            ]
        },
        {
            "name": "Mortal Humans", 
            "statements": [
                "All humans are mortal",
                "Socrates is a human",
                "Therefore, Socrates is mortal"
            ]
        },
        {
            "name": "Invalid Reasoning",
            "statements": [
                "Some birds can fly",
                "Tweety is a bird", 
                "Therefore, Tweety can fly"  # This should be invalid
            ]
        }
    ]
    
    for problem in problems:
        print(f"\n🎯 Problem: {problem['name']}")
        print("=" * 40)
        
        # Convert statements to FOL
        fol_statements = []
        
        for statement in problem['statements']:
            print(f"📝 Converting: {statement}")
            
            try:
                fol_result = translate_nl_to_fol(llama_model, llama_tokenizer, statement)
                if "𝜙=" in fol_result:
                    fol_formula = fol_result.split("𝜙=")[-1].strip()
                else:
                    fol_formula = fol_result.strip()
                
                fol_statements.append((statement, fol_formula))
                print(f"   FOL: {fol_formula}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                fol_statements.append((statement, f"Error: {e}"))
        
        print()
        
        # Verify with Prover9
        if fol_statements:
            verification_result = verify_reasoning_with_prover9(fol_statements, problem['name'])
            
            print(f"🏆 Final Result: {'VALID REASONING' if verification_result['valid'] else 'INVALID REASONING'}")
        
        print("\n" + "🔄" * 20 + "\n")
    
    return True

def interactive_prover9_verification():
    """Interactive Prover9 verification"""
    
    if not test_prover9_installation():
        return
    
    print("\n🔄 Loading Llama model...")
    llama_model, llama_tokenizer = setup_llama_lora()
    
    # Use local translate_nl_to_fol function
    
    print("\n🎯 Interactive Prover9 Verification")
    print("Enter reasoning statements (one per line)")
    print("Type 'done' when finished, 'quit' to exit")
    print()
    
    while True:
        statements = []
        fol_statements = []
        
        print("📝 Enter your reasoning statements:")
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
        
        # Convert to FOL
        print("\n🔄 Converting to FOL...")
        for statement in statements:
            try:
                fol_result = translate_nl_to_fol(llama_model, llama_tokenizer, statement)
                if "𝜙=" in fol_result:
                    fol_formula = fol_result.split("𝜙=")[-1].strip()
                else:
                    fol_formula = fol_result.strip()
                
                fol_statements.append((statement, fol_formula))
                print(f"✅ {statement} → {fol_formula}")
                
            except Exception as e:
                print(f"❌ {statement} → Error: {e}")
                fol_statements.append((statement, f"Error: {e}"))
        
        # Verify with Prover9
        if fol_statements:
            verification_result = verify_reasoning_with_prover9(fol_statements, "User Input")
        
        print("\n" + "-" * 40)
        choice = input("Try another problem? (y/N): ").strip().lower()
        if choice != 'y':
            break

def main():
    """Main menu for Prover9 integration"""
    
    print("🚀 Prover9 FOL Verification System")
    print("=" * 40)
    print("1. Test Prover9 installation")
    print("2. Demo verification with sample problems")
    print("3. Interactive verification")
    print("4. Exit")
    
    while True:
        choice = input("\nChoose option (1-4): ").strip()
        
        if choice == '1':
            test_prover9_installation()
        elif choice == '2':
            demo_prover9_verification()
        elif choice == '3':
            interactive_prover9_verification()
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()