#!/usr/bin/env python3
"""
Claude → Qwen → FOL → Prover9 Pipeline with Reward Model
Processes logic problems and evaluates responses using the reward system
"""

import re
import torch
from setup.setup_models import setup_qwen3, setup_llama_lora
from verification.prover9_integration import verify_reasoning_with_prover9, test_prover9_installation
from reward.reward import LogicalReasoningReward

def get_claude_generated_problems():
    """Claude-generated logic problems for testing"""
    
    problems = [
        {
            "id": 1,
            "context": "All cats are mammals. All mammals are warm-blooded. Fluffy is a cat.",
            "question": "Is Fluffy warm-blooded?",
            "expected_answer": "A"  # Changed to A/B/C format
        },
        {
            "id": 2, 
            "context": "No reptiles are mammals. All snakes are reptiles. Python is a snake.",
            "question": "Is Python a mammal?",
            "expected_answer": "B"
        },
        {
            "id": 3,
            "context": "All students study hard. Some students are successful. John is a student.",
            "question": "Is John successful?", 
            "expected_answer": "C"
        },
        {
            "id": 4,
            "context": "If it rains, then the ground gets wet. If the ground is wet, then plants grow. It is raining today.",
            "question": "Will plants grow today?",
            "expected_answer": "A"
        },
        {
            "id": 5,
            "context": "All birds can fly. Penguins are birds. Tweety is a penguin.",
            "question": "Can Tweety fly?",
            "expected_answer": "A"
        },
        {
            "id": 6,
            "context": "Every teacher is patient. Patient people are kind. Mrs. Smith is a teacher.",
            "question": "Is Mrs. Smith kind?",
            "expected_answer": "A"
        },
        {
            "id": 7,
            "context": "Some flowers are red. All roses are flowers. This plant is a rose.",
            "question": "Is this plant red?",
            "expected_answer": "C"
        },
        {
            "id": 8,
            "context": "All cars need fuel. Electric vehicles are cars. Tesla is an electric vehicle.",
            "question": "Does Tesla need fuel?",
            "expected_answer": "A"
        }
    ]
    
    return problems

def create_qwen_prompt(context, question):
    """Create a prompt for Qwen to solve the logic problem"""
    
    prompt = f"""Context: {context}

Question: {question}

Please solve this step by step using logical reasoning. Format your response exactly like this:

<initial_reasoning>
[Brief overview of the problem and approach]
</initial_reasoning>

<steps>
[Your step-by-step logical reasoning, one statement per line]
[End with your conclusion]
</steps>

<answer>
[A/B/C] - A for True, B for False, C for Uncertain
</answer>

Solve the problem now:"""
    
    return prompt

def get_qwen_solution(qwen_model, qwen_tokenizer, context, question, max_attempts=3):
    """Get Qwen's solution to the logic problem"""
    
    prompt = create_qwen_prompt(context, question)
    
    for attempt in range(max_attempts):
        try:
            inputs = qwen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
            device = next(qwen_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = qwen_model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=qwen_tokenizer.eos_token_id,
                    eos_token_id=qwen_tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    use_cache=False,
                )
            
            complete_output = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if prompt in complete_output:
                generated_part = complete_output.replace(prompt, "").strip()
            else:
                generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                generated_part = qwen_tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return generated_part
            
        except Exception as e:
            if attempt == max_attempts - 1:
                return create_manual_reasoning(context, question)
    
    return None

def create_manual_reasoning(context, question):
    """Create manual reasoning if Qwen fails"""
    
    return """<initial_reasoning>
Analyzing the given premises to determine logical conclusion.
</initial_reasoning>

<steps>
The context provides logical statements and relationships.
Following logical deduction from the given premises.
Based on the established relationships, making a determination.
</steps>

<answer>
C
</answer>"""

def parse_qwen_solution(qwen_output):
    """Parse Qwen's solution to extract reasoning and answer"""
    
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

def convert_reasoning_to_fol(reasoning_statements, llama_model, llama_tokenizer):
    """Convert Qwen's reasoning statements to FOL using Llama"""
    
    def translate_nl_to_fol(text):
        def formatting_func(text):
            return llama_tokenizer.apply_chat_template(
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
    
    fol_statements = []
    
    for statement in reasoning_statements:
        try:
            fol_result = translate_nl_to_fol(statement)
            if "𝜙=" in fol_result:
                fol_formula = fol_result.split("𝜙=")[-1].strip()
            else:
                fol_formula = fol_result.strip()
            
            fol_statements.append((statement, fol_formula))
            
        except Exception as e:
            fol_statements.append((statement, f"Error: {e}"))
    
    return fol_statements

def run_complete_pipeline(problem):
    """Run the complete pipeline with reward evaluation"""
    
    print(f"\nProblem #{problem['id']}: {problem['question']}")
    print("-" * 60)
    
    # Initialize reward calculator
    reward_calculator = LogicalReasoningReward()
    
    # Load models
    try:
        qwen_model, qwen_tokenizer = setup_qwen3()
    except Exception as e:
        qwen_model, qwen_tokenizer = None, None
    
    llama_model, llama_tokenizer = setup_llama_lora()
    
    # Get Qwen's solution
    if qwen_model and qwen_tokenizer:
        qwen_output = get_qwen_solution(qwen_model, qwen_tokenizer, problem['context'], problem['question'])
    else:
        qwen_output = create_manual_reasoning(problem['context'], problem['question'])
    
    if not qwen_output:
        print("❌ Failed to get solution")
        return None
    
    # Parse solution
    reasoning_statements, qwen_answer = parse_qwen_solution(qwen_output)
    
    if not reasoning_statements:
        print("❌ No reasoning statements found")
        return None
    
    # Convert to FOL
    fol_statements = convert_reasoning_to_fol(reasoning_statements, llama_model, llama_tokenizer)
    
    # Verify with Prover9
    verification_result = verify_reasoning_with_prover9(
        fol_statements, 
        f"Problem {problem['id']}"
    )
    
    # Calculate rewards
    reward_components = reward_calculator.calculate_composite_reward(
        response=qwen_output,
        expected_answer=problem['expected_answer'],
        prover9_result=verification_result,
        reasoning_steps=reasoning_statements
    )
    
    # Display LLM output
    print("\n🤖 LLM Output:")
    print("-" * 40)
    print(qwen_output)
    print("-" * 40)
    
    # Display results
    print(f"Expected: {problem['expected_answer']} | Generated: {qwen_answer}")
    print(f"Prover9 Valid: {verification_result.get('valid', False)}")
    
    print("\n📊 Reward Breakdown:")
    print(f"  Answer Correctness: {reward_components.answer_correctness:.2f} (35%)")
    print(f"  Logical Validity:   {reward_components.logical_validity:.2f} (55%)")
    print(f"  Format Compliance:  {reward_components.format_compliance:.2f} (10%)")
    print(f"  TOTAL REWARD:       {reward_components.total_reward:.3f}")
    
    return {
        'problem': problem,
        'qwen_answer': qwen_answer,
        'reasoning_statements': reasoning_statements,
        'fol_statements': fol_statements,
        'verification': verification_result,
        'reward_components': reward_components,
        'qwen_output': qwen_output
    }

def interactive_problem_selection():
    """Let user select which problem to test"""
    
    problems = get_claude_generated_problems()
    
    print("📋 Available Logic Problems:")
    for problem in problems:
        print(f"{problem['id']}. {problem['question']}")
    
    print(f"{len(problems)+1}. All problems")
    
    while True:
        try:
            choice = input(f"\nChoose problem (1-{len(problems)+1}): ").strip()
            choice = int(choice)
            
            if 1 <= choice <= len(problems):
                return [problems[choice-1]]
            elif choice == len(problems)+1:
                return problems
            else:
                print("❌ Invalid choice")
                
        except ValueError:
            print("❌ Please enter a number")

def main():
    """Main function"""
    
    print("🚀 Pipeline with Reward Evaluation")
    print("=" * 50)
    
    if not test_prover9_installation():
        print("❌ Prover9 not available")
        return
    
    selected_problems = interactive_problem_selection()
    
    results = []
    
    for problem in selected_problems:
        result = run_complete_pipeline(problem)
        if result:
            results.append(result)
    
    # Summary for multiple problems
    if len(results) > 1:
        print(f"\n📊 BATCH SUMMARY ({len(results)} problems)")
        print("-" * 50)
        
        avg_reward = sum(r['reward_components'].total_reward for r in results) / len(results)
        avg_correctness = sum(r['reward_components'].answer_correctness for r in results) / len(results)
        avg_validity = sum(r['reward_components'].logical_validity for r in results) / len(results)
        avg_format = sum(r['reward_components'].format_compliance for r in results) / len(results)
        
        print(f"Average Total Reward:    {avg_reward:.3f}")
        print(f"Average Correctness:     {avg_correctness:.2f}")
        print(f"Average Logical Validity: {avg_validity:.2f}")
        print(f"Average Format Score:    {avg_format:.2f}")

if __name__ == "__main__":
    main()