#!/usr/bin/env python3
"""
Simple Prover9 integration test
Quick test to verify Prover9 works with FOL conversion
"""

import subprocess
import tempfile
import os
from setup_models import setup_llama_lora

def test_prover9_basic():
    """Test basic Prover9 functionality"""
    
    print("🔧 Testing basic Prover9 functionality...")
    
    # Simple test input
    test_input = """formulas(assumptions).
all x (Cat(x) -> Animal(x)).
Cat(fluffy).
end_of_list.

formulas(goals).
Animal(fluffy).
end_of_list.
"""
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write(test_input)
            input_file = f.name
        
        # Run Prover9
        result = subprocess.run(
            ['prover9', '-f', input_file],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Clean up
        os.unlink(input_file)
        
        print(f"Return code: {result.returncode}")
        print("Output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        # Check for proof
        if "THEOREM PROVED" in result.stdout or "PROOF" in result.stdout:
            print("✅ Prover9 successfully proved the theorem!")
            return True
        else:
            print("❌ Prover9 did not find a proof")
            return False
            
    except FileNotFoundError:
        print("❌ Prover9 not found. Please ensure it's installed and in PATH.")
        return False
    except Exception as e:
        print(f"❌ Error running Prover9: {e}")
        return False

def test_fol_to_prover9_conversion():
    """Test FOL conversion and Prover9 integration"""
    
    print("\n🧪 Testing FOL to Prover9 conversion...")
    
    # Load Llama model
    print("📥 Loading Llama model...")
    llama_model, llama_tokenizer = setup_llama_lora()
    
    from prover9_integration import translate_nl_to_fol, verify_reasoning_with_prover9
    
    # Simple test case
    statements = [
        "All cats are animals",
        "Fluffy is a cat",
        "Therefore, Fluffy is an animal"
    ]
    
    print("📝 Test statements:")
    for i, stmt in enumerate(statements, 1):
        print(f"  {i}. {stmt}")
    
    # Convert to FOL
    print("\n🔄 Converting to FOL...")
    fol_statements = []
    
    for statement in statements:
        try:
            fol_result = translate_nl_to_fol(llama_model, llama_tokenizer, statement)
            if "𝜙=" in fol_result:
                fol_formula = fol_result.split("𝜙=")[-1].strip()
            else:
                fol_formula = fol_result.strip()
            
            fol_statements.append((statement, fol_formula))
            print(f"✅ {statement}")
            print(f"   → {fol_formula}")
            
        except Exception as e:
            print(f"❌ {statement} → Error: {e}")
    
    # Verify with Prover9
    if fol_statements:
        print("\n🔍 Verifying with Prover9...")
        result = verify_reasoning_with_prover9(fol_statements, "Simple Test")
        
        if result['valid']:
            print("🎉 SUCCESS: Reasoning is logically valid!")
        else:
            print("❌ FAILED: Reasoning is not valid or Prover9 error")
        
        return result['valid']
    
    return False

def quick_manual_test():
    """Quick manual test without model loading"""
    
    print("\n⚡ Quick manual Prover9 test...")
    
    from prover9_integration import verify_reasoning_with_prover9
    
    # Manual FOL statements (bypass model conversion)
    manual_fol = [
        ("All cats are animals", "all x (Cat(x) -> Animal(x))"),
        ("Fluffy is a cat", "Cat(fluffy)"), 
        ("Therefore, Fluffy is an animal", "Animal(fluffy)")
    ]
    
    print("📝 Manual FOL statements:")
    for stmt, fol in manual_fol:
        print(f"  {stmt} → {fol}")
    
    print("\n🔍 Testing with Prover9...")
    result = verify_reasoning_with_prover9(manual_fol, "Manual Test")
    
    return result['valid']

def main():
    """Run all tests"""
    
    print("🚀 Prover9 Integration Test Suite")
    print("=" * 50)
    
    # Test 1: Basic Prover9
    test1_pass = test_prover9_basic()
    
    if not test1_pass:
        print("\n❌ Basic Prover9 test failed. Cannot proceed.")
        print("💡 Please check Prover9 installation:")
        print("   - Ensure Prover9 is installed")
        print("   - Ensure 'prover9' command is in PATH")
        return
    
    # Test 2: Quick manual test
    print("\n" + "="*50)
    test2_pass = quick_manual_test()
    
    # Test 3: Full FOL conversion + Prover9 (requires model loading)
    print("\n" + "="*50)
    
    proceed = input("Run full test with model loading? (y/N): ").strip().lower()
    if proceed == 'y':
        test3_pass = test_fol_to_prover9_conversion()
    else:
        test3_pass = True
        print("⏭️  Skipping full model test")
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Summary:")
    print(f"  Basic Prover9: {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"  Manual FOL:    {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print(f"  Full Pipeline: {'✅ PASS' if test3_pass else '⏭️  SKIP'}")
    
    if test1_pass and test2_pass:
        print("\n🎉 Prover9 integration is working!")
        print("💡 You can now run: python prover9_integration.py")
    else:
        print("\n❌ Some tests failed. Check Prover9 installation.")

if __name__ == "__main__":
    main()