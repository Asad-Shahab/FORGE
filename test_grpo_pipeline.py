#!/usr/bin/env python3
"""
Test Script for GRPO Pipeline - End-to-End Testing
Tests the logical reasoning pipeline: Generation -> FOL -> Prover9 -> Scoring
"""

import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import pipeline components
from setup.setup_models import setup_qwen3, setup_llama_lora
from reward.reward import LogicalReasoningReward
from verification.prover9_integration import verify_reasoning_with_prover9, test_prover9_installation
from train import parse_qwen_solution, convert_reasoning_to_fol, setup_chat_template

def load_test_problem(dataset_path, problem_index=0):
    """Load a specific problem from the dataset"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if problem_index >= len(data):
        raise ValueError(f"Problem index {problem_index} out of range (max: {len(data)-1})")
    
    return data[problem_index]

def generate_response(model, tokenizer, system_prompt, question):
    """Generate response using the model"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1
        )
    
    # Extract the full response and then get only the generated part
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # The generated part starts after the original prompt
    # Since the chat template adds <initial_reasoning> at the end, we need to include it
    prompt_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    
    if full_response.startswith(prompt_text):
        response = full_response[len(prompt_text):].strip()
        # Ensure we start with <initial_reasoning> tag
        if not response.startswith('<initial_reasoning>'):
            response = '<initial_reasoning>' + response
    else:
        # Fallback: just decode the generated tokens
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if not response.startswith('<initial_reasoning>'):
            response = '<initial_reasoning>' + response
    
    return response

def test_full_pipeline(dataset_path, problem_index=0, model_path=None):
    """Test the complete GRPO pipeline on a single problem"""
    
    print("🧪 GRPO Pipeline Test")
    print("=" * 60)
    
    # Test Prover9 first
    print("1. Testing Prover9 installation...")
    if not test_prover9_installation():
        print("❌ Cannot continue without Prover9")
        return
    print()
    
    # Load test problem
    print("2. Loading test problem...")
    problem = load_test_problem(dataset_path, problem_index)
    print(f"   Problem {problem_index} loaded")
    print(f"   Expected answer: {problem['answer']}")
    print()
    
    # Setup models
    print("3. Loading models...")
    print("   Loading Qwen model for generation...")
    if model_path:
        # Convert relative path to absolute path
        import os
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
        
        print(f"   Using model path: {model_path}")
        
        # Check if this is a LoRA model by looking for adapter_config.json
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            print("   Loading LoRA model...")
            from peft import PeftModel, PeftConfig
            
            # Load the PEFT config to get base model name
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_name = peft_config.base_model_name_or_path
            print(f"   Base model: {base_model_name}")
            
            # Load base model first
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            
            # Load LoRA adapters
            qwen_model = PeftModel.from_pretrained(base_model, model_path)
            qwen_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            qwen_model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
            )
            qwen_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
        
        if qwen_tokenizer.pad_token is None:
            qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
        qwen_tokenizer.pad_token_id = qwen_tokenizer.eos_token_id
    else:
        qwen_model, qwen_tokenizer = setup_qwen3()
    
    print("   Loading Llama model for FOL conversion...")
    llama_model, llama_tokenizer = setup_llama_lora()
    print("   Models loaded successfully")
    print()
    
    # Setup chat template
    qwen_tokenizer, system_prompt = setup_chat_template(qwen_tokenizer)
    
    # Generate response
    print("4. Generating response...")
    response = generate_response(qwen_model, qwen_tokenizer, system_prompt, problem['question'])
    
    print("   Generated Response:")
    print("   " + "─" * 50)
    print(f"   {response}")
    print("   " + "─" * 50)
    print()
    
    # Parse response
    print("5. Parsing response...")
    reasoning_statements, predicted_answer = parse_qwen_solution(response)
    
    print(f"   Predicted Answer: {predicted_answer}")
    print(f"   Reasoning Statements ({len(reasoning_statements)}):")
    for i, stmt in enumerate(reasoning_statements, 1):
        print(f"     {i}. {stmt}")
    print()
    
    if not reasoning_statements:
        print("❌ No reasoning statements found - cannot continue with FOL conversion")
        return
    
    # Convert to FOL
    print("6. Converting to First-Order Logic...")
    fol_statements = convert_reasoning_to_fol(reasoning_statements, llama_model, llama_tokenizer)
    
    print("   FOL Conversion Results:")
    for i, (statement, fol) in enumerate(fol_statements, 1):
        print(f"     {i}. Statement: {statement}")
        print(f"        FOL: {fol}")
    print()
    
    # Verify with Prover9
    print("7. Verifying with Prover9...")
    verification_result = verify_reasoning_with_prover9(fol_statements, f"Test_Problem_{problem_index}")
    
    print("   Prover9 Results:")
    print(f"     Valid: {verification_result['valid']}")
    print(f"     Details: {verification_result['details']}")
    print()
    
    # Calculate reward
    print("8. Calculating rewards...")
    reward_calculator = LogicalReasoningReward()
    reward_components = reward_calculator.calculate_composite_reward(
        response=response,
        expected_answer=problem['answer'],
        prover9_result=verification_result,
        reasoning_steps=reasoning_statements
    )
    
    print("   Reward Breakdown:")
    print(f"     Answer Correctness: {reward_components.answer_correctness:.3f}")
    print(f"     Logical Validity:   {reward_components.logical_validity:.3f}")
    print(f"     Format Compliance:  {reward_components.format_compliance:.3f}")
    print(f"     Total Reward:       {reward_components.total_reward:.3f}")
    print()
    
    # Final summary
    print("9. Final Summary:")
    print("=" * 60)
    print(f"Problem: {problem_index}")
    print(f"Expected Answer: {problem['answer']} | Predicted: {predicted_answer}")
    print(f"Answer Match: {'✅' if predicted_answer.upper() == problem['answer'].upper() else '❌'}")
    print(f"Prover9 Valid: {'✅' if verification_result['valid'] else '❌'}")
    print(f"Final Score: {reward_components.total_reward:.3f}/1.0")
    
    return {
        'problem_index': problem_index,
        'expected_answer': problem['answer'],
        'predicted_answer': predicted_answer,
        'reasoning_statements': reasoning_statements,
        'fol_statements': fol_statements,
        'prover9_result': verification_result,
        'reward_components': reward_components,
        'response': response
    }

def main():
    parser = argparse.ArgumentParser(description="Test GRPO Pipeline on Single Problem")
    parser.add_argument("--dataset", type=str, default="dataset/proverqa_simplified.json",
                       help="Path to dataset")
    parser.add_argument("--problem_index", type=int, default=0,
                       help="Index of problem to test (0-based)")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to fine-tuned model (optional)")
    
    args = parser.parse_args()
    
    try:
        result = test_full_pipeline(args.dataset, args.problem_index, args.model_path)
        print("\n🎉 Pipeline test completed successfully!")
        return result
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()