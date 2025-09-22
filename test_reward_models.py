#!/usr/bin/env python3
"""
Test Script for Reward Models - Debug GRPO Logic Validity Checker
Based on train.py but focused on debugging reward computation pipeline
"""

import argparse
import json
import os
import re
import torch
import warnings
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from existing modules
from setup.setup_models import setup_qwen3, setup_llama_lora
from reward.reward import LogicalReasoningReward, RewardComponents
from verification.prover9_integration import verify_reasoning_with_prover9, test_prover9_installation

class RewardModelTester:
    """Comprehensive tester for reward models and logic validity checker"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        print("🤖 Loading models...")
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.llama_model = None
        self.llama_tokenizer = None
        
        # Initialize reward calculator
        self.reward_calculator = LogicalReasoningReward()
        
        # Test results storage
        self.test_results = []
        
        print(f"📱 Using device: {self.device}")
    
    def setup_qwen_model(self, model_path=None):
        """Load Qwen model for generation"""
        print("📦 Loading Qwen model for generation...")
        
        if model_path and os.path.exists(model_path):
            print(f"   Loading model from: {model_path}")
            
            # Use your preferred loading approach
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",   # automatically places everything on your 1 GPU
                torch_dtype="auto"   # picks the right dtype (fp16/bf16/fp32 depending on weights)
            )
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
        else:
            # Use base Qwen model as fallback
            print("No model path provided")
        
        # Setup pad token
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token
        self.qwen_tokenizer.pad_token_id = self.qwen_tokenizer.eos_token_id
        
        # Setup chat template
        self.setup_chat_template()
        print("✅ Qwen model loaded successfully")
    
    def setup_llama_model(self):
        """Load Llama model for NL→FOL conversion"""
        print("📦 Loading Llama model for FOL conversion...")
        try:
            self.llama_model, self.llama_tokenizer = setup_llama_lora(device=self.device)
            print("✅ Llama model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Llama model: {e}")
            print("⚠️  FOL conversion will be skipped")
    
    def setup_chat_template(self):
        """Setup chat template following train.py patterns"""
        reasoning_start = "<initial_reasoning>"
        reasoning_end = "</initial_reasoning>"
        steps_start = "<steps>"
        steps_end = "</steps>"
        answer_start = "<answer>"
        answer_end = "</answer>"

        system_prompt = f"""You are an expert in logical reasoning. Analyze the given problem step by step. First, provide your initial reasoning between {reasoning_start} and {reasoning_end}. Then, provide your logical reasoning steps between {steps_start} and {steps_end}. Finally, provide your answer (A, B, or C) between {answer_start} and {answer_end}."""

        chat_template = \
            "{% if messages[0]['role'] == 'system' %}"\
            "{{ messages[0]['content'] + eos_token }}"\
            "{% set loop_messages = messages[1:] %}"\
            "{% else %}"\
            "{{ system_prompt + eos_token }}"\
            "{% set loop_messages = messages %}"\
            "{% endif %}"\
            "{% for message in loop_messages %}"\
            "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
            "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
            "{% endif %}"\
            "{% endfor %}"\
            "{% if add_generation_prompt %}{{ reasoning_start }}"\
            "{% endif %}"

        chat_template = chat_template\
            .replace("system_prompt", f"'{system_prompt}'")\
            .replace("reasoning_start", f"'{reasoning_start}'")
        
        self.qwen_tokenizer.chat_template = chat_template
        self.system_prompt = system_prompt
    
    def parse_qwen_solution(self, qwen_output):
        """Parse Qwen's solution to extract reasoning and answer (from train.py)"""
        
        # Extract answer
        answer_match = re.search(r'<answer>(.*?)</answer>', qwen_output, re.DOTALL | re.IGNORECASE)
        answer = answer_match.group(1).strip() if answer_match else ""
        
        # Extract steps
        steps_match = re.search(r'<steps>(.*?)</steps>', qwen_output, re.DOTALL | re.IGNORECASE)
        reasoning = steps_match.group(1).strip() if steps_match else ""
        
        # Parse reasoning into individual statements
        reasoning_statements = []
        if reasoning:
            sentences = re.split(r'\.(?=\s*[A-Z]|Therefore|So |Thus |Hence )', reasoning)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:
                    sentence = re.sub(r'^\d+\.\s*', '', sentence)
                    sentence = re.sub(r'^-\s*', '', sentence)
                    sentence = sentence.strip(' .,')
                    
                    if sentence:
                        reasoning_statements.append(sentence)
            
            if len(reasoning_statements) < 2:
                lines = reasoning.split('\n')
                reasoning_statements = []
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 10:
                        line = re.sub(r'^\d+\.\s*', '', line)
                        line = re.sub(r'^-\s*', '', line)
                        reasoning_statements.append(line)
        
        return reasoning_statements, answer
    
    def convert_reasoning_to_fol(self, reasoning_statements):
        """Convert reasoning statements to FOL using Llama (from train.py)"""
        if not self.llama_model or not self.llama_tokenizer:
            print("⚠️  Llama model not available, skipping FOL conversion")
            return [(stmt, "FOL conversion skipped") for stmt in reasoning_statements]

        improved_system_prompt = """You are an expert at converting natural language statements into First-Order Logic (FOL). Follow these strict rules:

SYMBOLS: Use only ∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (if and only if)

NAMING RULES:
- All names/entities must be lowercase (natalie, john, stone)
- Use simple, clear predicate names (Weak(x), Resilient(x), Fearless(x))
- Be consistent with predicate names throughout

SIMPLICITY RULES:
- Keep expressions as simple as possible
- Avoid unnecessary quantifiers when dealing with specific individuals
- Use direct predicates: Weak(natalie) instead of exists x (Weakness(x) & Has(natalie, x))

VARIABLE RULES:
- Never use unbound variables
- If you use ∃x or ∀x, make sure x appears in the formula
- For specific people, use their name directly as a constant

Start your answer with '𝜙=' followed by the FOL formula. Do not include any other text."""

        def translate_nl_to_fol(text):
            def formatting_func(text):
                return self.llama_tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": improved_system_prompt
                        },
                        {"role": "user", "content": text},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            
            prompt = formatting_func(text)
            inputs = self.llama_tokenizer(prompt, return_tensors="pt", padding=True)
            
            device = next(self.llama_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.llama_model.generate(
                    **inputs, 
                    max_new_tokens=50,
                    temperature=0.1, 
                    do_sample=True,
                    pad_token_id=self.llama_tokenizer.eos_token_id,
                    eos_token_id=self.llama_tokenizer.eos_token_id,
                    num_beams=1,
                    use_cache=False
                )
            
            result = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result
        
        fol_statements = []
        
        for statement in reasoning_statements:
            try:
                print(f"🔄 Converting to FOL: {statement[:50]}...")
                fol_result = translate_nl_to_fol(statement)
                if "𝜙=" in fol_result:
                    fol_formula = fol_result.split("𝜙=")[-1].strip()
                else:
                    fol_formula = fol_result.strip()
                
                fol_statements.append((statement, fol_formula))
                print(f"   → FOL: {fol_formula}")
                
            except Exception as e:
                print(f"❌ Error converting to FOL: {e}")
                fol_statements.append((statement, f"Error: {e}"))
        
        return fol_statements
    
    def generate_response(self, question):
        """Generate response using Qwen model"""
        if not self.qwen_model or not self.qwen_tokenizer:
            raise Exception("Qwen model not loaded")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]
        
        prompt = self.qwen_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.qwen_tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature,
                do_sample=True,
                pad_token_id=self.qwen_tokenizer.eos_token_id,
                eos_token_id=self.qwen_tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        response = self.qwen_tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):], 
            skip_special_tokens=True
        )
        
        return response
    
    def test_single_example(self, question, expected_answer, example_name="Test Example"):
        """Test reward computation on a single example with detailed output"""
        print("\n" + "="*80)
        print(f"🧪 TESTING: {example_name}")
        print("="*80)
        
        # 1. Generate Response
        print("\n📝 STEP 1: GENERATING RESPONSE")
        print("-" * 40)
        print(f"Question: {question}")
        print(f"Expected Answer: {expected_answer}")
        
        try:
            response = self.generate_response(question)
            print(f"\n✅ Generated Response:")
            print(response)
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return None
        
        # 2. Parse Response
        print("\n🔍 STEP 2: PARSING RESPONSE")
        print("-" * 40)
        
        reasoning_statements, predicted_answer = self.parse_qwen_solution(response)
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Reasoning Statements ({len(reasoning_statements)}):")
        for i, stmt in enumerate(reasoning_statements, 1):
            print(f"  {i}. {stmt}")
        
        # 3. Convert to FOL
        print("\n🧮 STEP 3: CONVERTING TO FOL")
        print("-" * 40)
        
        fol_statements = self.convert_reasoning_to_fol(reasoning_statements)
        print(f"\nFOL Statements ({len(fol_statements)}):")
        for i, (nl_stmt, fol_stmt) in enumerate(fol_statements, 1):
            print(f"  {i}. NL: {nl_stmt[:60]}...")
            print(f"     FOL: {fol_stmt}")
        
        # 4. Verify with Prover9
        print("\n⚖️  STEP 4: PROVER9 VERIFICATION")
        print("-" * 40)
        
        try:
            verification_result = verify_reasoning_with_prover9(
                fol_statements, 
                f"{example_name}_Test"
            )
            print(f"Verification Result: {'✅ VALID' if verification_result['valid'] else '❌ INVALID'}")
            print(f"Details: {verification_result['details']}")
            
            if self.args.show_prover9_details:
                print(f"\nProver9 Input:")
                print(verification_result['prover9_input'])
                print(f"\nProver9 Output:")
                print(verification_result['prover9_output'][:500] + "..." if len(verification_result['prover9_output']) > 500 else verification_result['prover9_output'])
        except Exception as e:
            print(f"❌ Error in Prover9 verification: {e}")
            verification_result = {'valid': False, 'details': f'Verification error: {e}'}
        
        # 5. Calculate Rewards
        print("\n🏆 STEP 5: REWARD CALCULATION")
        print("-" * 40)
        
        # Test both simple and complex rewards
        rewards = {}
        
        # Simple reward (format-based)
        simple_reward = self.calculate_simple_reward(response)
        rewards['simple'] = simple_reward
        print(f"Simple Reward: {simple_reward:.3f}")
        
        # Complex reward (full pipeline)
        complex_reward_components = self.reward_calculator.calculate_composite_reward(
            response=response,
            expected_answer=expected_answer,
            prover9_result=verification_result,
            reasoning_steps=reasoning_statements
        )
        rewards['complex'] = complex_reward_components
        
        print(f"\nComplex Reward Breakdown:")
        print(f"  Answer Correctness: {complex_reward_components.answer_correctness:.3f}")
        print(f"  Logical Validity:   {complex_reward_components.logical_validity:.3f}")
        print(f"  Format Compliance:  {complex_reward_components.format_compliance:.3f}")
        print(f"  Total Reward:       {complex_reward_components.total_reward:.3f}")
        
        # Store results
        test_result = {
            'example_name': example_name,
            'question': question,
            'expected_answer': expected_answer,
            'response': response,
            'predicted_answer': predicted_answer,
            'reasoning_statements': reasoning_statements,
            'fol_statements': fol_statements,
            'verification_result': verification_result,
            'rewards': rewards,
            'timestamp': datetime.now().isoformat()
        }
        
        self.test_results.append(test_result)
        
        return test_result
    
    def calculate_simple_reward(self, completion):
        """Simple reward based on format compliance and length (from train.py)"""
        response = completion
        
        # Basic format check
        has_reasoning = '<initial_reasoning>' in response and '</initial_reasoning>' in response
        has_steps = '<steps>' in response and '</steps>' in response  
        has_answer = '<answer>' in response and '</answer>' in response
        
        format_score = (has_reasoning + has_steps + has_answer) / 3.0
        
        # Length penalty for very short/long responses
        length_score = 1.0
        if len(response) < 100:
            length_score = 0.5
        elif len(response) > 2000:
            length_score = 0.7
        
        # Combined score
        score = (format_score * 0.7 + length_score * 0.3) * 10.0  # Scale to 0-10
        return score
    
    def test_prover9_installation_detailed(self):
        """Test Prover9 installation with detailed output"""
        print("\n🔧 TESTING PROVER9 INSTALLATION")
        print("=" * 50)
        
        success = test_prover9_installation()
        
        if success:
            # Test with a simple example
            print("\n🧪 Testing simple logical inference...")
            test_premises = [("All humans are mortal", "∀x (Human(x) → Mortal(x))")]
            test_premises.append(("Socrates is human", "Human(socrates)"))
            
            try:
                result = verify_reasoning_with_prover9(test_premises, "Prover9_Installation_Test")
                print(f"Test result: {'✅ SUCCESS' if result['valid'] else '❌ FAILED'}")
                if self.args.show_prover9_details:
                    print(f"Details: {result['details']}")
            except Exception as e:
                print(f"❌ Test failed with error: {e}")
        
        return success
    
    def run_comprehensive_test(self):
        """Run comprehensive test on multiple examples"""
        print("🚀 STARTING COMPREHENSIVE REWARD MODEL TEST")
        
        # Test Prover9 first
        if not self.test_prover9_installation_detailed():
            print("❌ Prover9 not working. Some tests will be limited.")
        
        # Load test examples
        test_examples = self.load_test_examples()
        
        print(f"\n📊 Testing {len(test_examples)} examples...")
        
        for i, example in enumerate(test_examples, 1):
            try:
                result = self.test_single_example(
                    example['question'], 
                    example['answer'], 
                    f"Example_{i}"
                )
                
                if self.args.pause_between_examples:
                    input("\nPress Enter to continue to next example...")
                
            except KeyboardInterrupt:
                print("\n⚠️ Test interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error testing example {i}: {e}")
                continue
        
        # Generate summary
        self.generate_test_summary()
    
    def load_test_examples(self):
        """Load test examples from dataset or use built-in examples"""
        # Try user-specified dataset first
        if self.args.test_dataset and os.path.exists(self.args.test_dataset):
            print(f"📂 Loading examples from {self.args.test_dataset}")
            with open(self.args.test_dataset, 'r') as f:
                data = json.load(f)
            return data[:self.args.max_examples] if self.args.max_examples else data
        
        # Try medium dataset as default
        medium_dataset_path = "dataset/dev/medium.json"
        if os.path.exists(medium_dataset_path):
            print(f"📂 Loading examples from {medium_dataset_path} (default)")
            with open(medium_dataset_path, 'r') as f:
                data = json.load(f)
            return data[:self.args.max_examples] if self.args.max_examples else data
        else:
            print("⚠️ Medium dataset not found, using built-in examples")
            # Built-in test examples
            return [
                {
                    "question": "If all birds can fly and penguins are birds, can penguins fly?",
                    "answer": "A"
                },
                {
                    "question": "If it's raining, then the ground is wet. The ground is wet. Is it raining?",
                    "answer": "C"
                },
                {
                    "question": "All cats are mammals. Some mammals are dogs. Are some cats dogs?",
                    "answer": "B"
                }
            ]
    
    def generate_test_summary(self):
        """Generate comprehensive test summary"""
        if not self.test_results:
            print("❌ No test results to summarize")
            return
        
        print("\n" + "="*80)
        print("📋 TEST SUMMARY")
        print("="*80)
        
        total_tests = len(self.test_results)
        simple_rewards = [r['rewards']['simple'] for r in self.test_results if 'simple' in r['rewards']]
        complex_rewards = [r['rewards']['complex'].total_reward for r in self.test_results if 'complex' in r['rewards']]
        
        prover9_valid_count = sum(1 for r in self.test_results if r['verification_result']['valid'])
        correct_answers = sum(1 for r in self.test_results if r['rewards']['complex'].answer_correctness > 0)
        
        print(f"Total Tests: {total_tests}")
        print(f"Correct Answers: {correct_answers}/{total_tests} ({correct_answers/total_tests*100:.1f}%)")
        print(f"Prover9 Valid: {prover9_valid_count}/{total_tests} ({prover9_valid_count/total_tests*100:.1f}%)")
        
        if simple_rewards:
            print(f"\nSimple Rewards - Mean: {sum(simple_rewards)/len(simple_rewards):.3f}, Range: [{min(simple_rewards):.3f}, {max(simple_rewards):.3f}]")
        
        if complex_rewards:
            print(f"Complex Rewards - Mean: {sum(complex_rewards)/len(complex_rewards):.3f}, Range: [{min(complex_rewards):.3f}, {max(complex_rewards):.3f}]")
        
        # Save results if requested
        if self.args.save_results:
            results_file = f"reward_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {results_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Script for Reward Models")
    
    # Model settings
    parser.add_argument("--qwen_model_path", type=str, default=None,
                      help="Path to Qwen model (use base model if not provided)")
    
    # Test settings  
    parser.add_argument("--test_dataset", type=str, default=None,
                      help="Path to test dataset JSON file")
    parser.add_argument("--max_examples", type=int, default=3,
                      help="Maximum number of examples to test")
    parser.add_argument("--pause_between_examples", action="store_true",
                      help="Pause between examples for inspection")
    
    # Generation settings
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                      help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Generation temperature")
    
    # Debug settings
    parser.add_argument("--show_prover9_details", action="store_true",
                      help="Show detailed Prover9 input/output")
    parser.add_argument("--save_results", action="store_true",
                      help="Save test results to JSON file")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = RewardModelTester(args)
    
    # Load models
    tester.setup_qwen_model(args.qwen_model_path)
    tester.setup_llama_model()
    
    # Run tests
    tester.run_comprehensive_test()
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    main()