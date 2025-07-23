#!/usr/bin/env python3
"""
Processes logic problems from ProverQA dataset and evaluates responses using the reward system
"""

import re
import torch
from datasets import load_dataset
from setup.setup_models import setup_qwen3, setup_llama_lora
from verification.prover9_integration import verify_reasoning_with_prover9, test_prover9_installation
from reward.reward import LogicalReasoningReward
import random

import random

def load_proverqa_problems(difficulty="easy", max_problems=None):
    """Load ProverQA problems from the specified difficulty level"""
    
    try:
        if difficulty == "easy":
            dataset = load_dataset("opendatalab/ProverQA", data_files="dev/easy.json")
        elif difficulty == "medium":
            dataset = load_dataset("opendatalab/ProverQA", data_files="dev/medium.json")
        elif difficulty == "hard":
            dataset = load_dataset("opendatalab/ProverQA", data_files="dev/hard.json")
        else:
            raise ValueError("Difficulty must be 'easy', 'medium', or 'hard'")
        
        data = dataset['train']  # The loaded dataset uses 'train' as the split name
        
        # Convert to list for random sampling
        all_items = list(data)
        
        # Randomly sample if max_problems is specified
        if max_problems and max_problems < len(all_items):
            selected_items = random.sample(all_items, max_problems)
        else:
            selected_items = all_items
        
        problems = []
        for item in selected_items:
            problems.append({
                "id": item["id"],
                "context": item["context"],
                "question": item["question"],
                "expected_answer": item["answer"],
                "options": item["options"],
                "reasoning": item.get("reasoning", ""),
                "nl2fol": item.get("nl2fol", {}),
                "conclusion_fol": item.get("conclusion_fol", ""),
                "difficulty": difficulty
            })
        
        return problems
        
    except Exception as e:
        print(f"❌ Error loading ProverQA dataset: {e}")
        return []

def create_qwen_prompt(context, question):
    """Create a chat-formatted prompt for Qwen to solve the logic problem"""
    
    reasoning_start = "<initial_reasoning>"
    reasoning_end = "</initial_reasoning>"
    steps_start = "<steps>"
    steps_end = "</steps>"
    answer_start = "<answer>"
    answer_end = "</answer>"
    
    system_prompt = f"""You are an expert in logical reasoning. Analyze the given problem step by step. First, provide your initial reasoning between {reasoning_start} and {reasoning_end}. Then, provide your logical reasoning steps between {steps_start} and {steps_end}. Finally, provide your answer (A, B, or C) between {answer_start} and {answer_end}.

For the steps section, write simple, clear statements in natural language. Each statement should be on its own line. Do not use "Premise 1:", "Premise 2:" or formal logic notation. Just state the facts and conclusions directly, like:
<Statement 1>
<Statement 2>
...
<Conclusion>"""
    
    user_prompt = f"""Context: {context}

Question: {question}

Please solve this step by step using logical reasoning."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages

def get_qwen_solution(qwen_model, qwen_tokenizer, context, question, max_attempts=3):
    """Get Qwen's solution to the logic problem"""
    
    # Template components for stopping
    reasoning_start = "<initial_reasoning>"
    answer_end = "</answer>"
    
    # Get messages from create_qwen_prompt
    messages = create_qwen_prompt(context, question)
    system_prompt = messages[0]["content"]
    
    # Set up chat template
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
    
    qwen_tokenizer.chat_template = chat_template
    
    for attempt in range(max_attempts):
        try:
            # Apply chat template
            prompt = qwen_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = qwen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
            device = next(qwen_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = qwen_model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=qwen_tokenizer.eos_token_id,
                    eos_token_id=qwen_tokenizer.eos_token_id,
                    stopping_criteria=None,
                    stop_strings=[answer_end],
                    tokenizer=qwen_tokenizer,  
                    repetition_penalty=1.04,
                    use_cache=False,
                )
            
            # Decode only the generated part
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_part = qwen_tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Ensure we stop at </answer> if it wasn't caught by stop_strings
            if answer_end in generated_part:
                generated_part = generated_part.split(answer_end)[0] + answer_end
            
            return generated_part
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt == max_attempts - 1:
                print(f"❌ Failed to get solution after {max_attempts} attempts")
                return None
    
    return None

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

    print(f"\n🔍 Converting {len(reasoning_statements)} statements to FOL:")
    for i, statement in enumerate(reasoning_statements, 1):
        print(f"  {i}. {statement}")
    
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

    EXAMPLES:
    "John is tall" → Tall(john)
    "If someone is weak, they are not resilient" → all x (Weak(x) → ¬Resilient(x))
    "Mary is either smart or funny" → Smart(mary) ∨ Funny(mary)
    "Everyone who studies passes" → all x (Studies(x) → Passes(x))

    Start your answer with '𝜙=' followed by the FOL formula. Do not include any other text."""

    # Update the translate_nl_to_fol function to use this prompt:
    def translate_nl_to_fol(text):
        def formatting_func(text):
            return llama_tokenizer.apply_chat_template(
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
        inputs = llama_tokenizer(prompt, return_tensors="pt", padding=True)
        
        device = next(llama_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = llama_model.generate(
                **inputs, 
                max_new_tokens=150, 
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
    
    print(f"\nProblem #{problem['id']} ({problem['difficulty']})")
    print(f"Question: {problem['question']}")
    print("-" * 80)
    
    # Initialize reward calculator
    reward_calculator = LogicalReasoningReward()
    
    # Load models
    try:
        qwen_model, qwen_tokenizer = setup_qwen3()
    except Exception as e:
        print(f"❌ Failed to load Qwen model: {e}")
        return None
    
    try:
        llama_model, llama_tokenizer = setup_llama_lora()
    except Exception as e:
        print(f"❌ Failed to load Llama model: {e}")
        return None
    
    # Get Qwen's solution
    qwen_output = get_qwen_solution(qwen_model, qwen_tokenizer, problem['context'], problem['question'])
    
    if not qwen_output:
        print("❌ Failed to get solution from Qwen")
        return None
    
    # Display complete Qwen solution
    print("\n🤖 COMPLETE QWEN SOLUTION:")
    print("=" * 60)
    print(qwen_output)
    print("=" * 60)
    
    # Parse solution
    reasoning_statements, qwen_answer = parse_qwen_solution(qwen_output)
    
    if not reasoning_statements:
        print("❌ No reasoning statements found in output")
        return None
    
    # Convert to FOL
    fol_statements = convert_reasoning_to_fol(reasoning_statements, llama_model, llama_tokenizer)
    
    # Verify with Prover9
    verification_result = verify_reasoning_with_prover9(
        fol_statements, 
        f"ProverQA Problem {problem['id']}"
    )
    
    # Calculate rewards
    reward_components = reward_calculator.calculate_composite_reward(
        response=qwen_output,
        expected_answer=problem['expected_answer'],
        prover9_result=verification_result,
        reasoning_steps=reasoning_statements
    )
    
    # Display results
    print(f"\n📝 RESULTS:")
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

def interactive_dataset_selection():
    """Let user select difficulty and number of problems"""
    
    print("📋 ProverQA Dataset Selection:")
    print("1. Easy problems")
    print("2. Medium problems") 
    print("3. Hard problems")
    
    while True:
        try:
            difficulty_choice = input("\nChoose difficulty (1-3): ").strip()
            difficulty_map = {"1": "easy", "2": "medium", "3": "hard"}
            
            if difficulty_choice not in difficulty_map:
                print("❌ Invalid choice")
                continue
                
            difficulty = difficulty_map[difficulty_choice]
            break
            
        except ValueError:
            print("❌ Please enter a number")
    
    while True:
        try:
            num_problems = input("Number of problems (1-50, or 'all'): ").strip()
            
            if num_problems.lower() == 'all':
                max_problems = None
                break
            else:
                max_problems = int(num_problems)
                if 1 <= max_problems <= 50:
                    break
                else:
                    print("❌ Please enter 1-50 or 'all'")
                    
        except ValueError:
            print("❌ Please enter a number or 'all'")
    
    print(f"\n🔄 Loading {difficulty} problems...")
    problems = load_proverqa_problems(difficulty, max_problems)
    
    if not problems:
        print("❌ Failed to load problems")
        return []
    
    print(f"✅ Loaded {len(problems)} problems")
    return problems

def main():
    """Main function"""
    
    print("🚀 ProverQA Pipeline with Reward Evaluation")
    print("=" * 50)
    
    if not test_prover9_installation():
        print("❌ Prover9 not available")
        return
    
    selected_problems = interactive_dataset_selection()
    
    if not selected_problems:
        return
    
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
        
        # Accuracy breakdown
        correct_answers = sum(1 for r in results if r['qwen_answer'] == r['problem']['expected_answer'])
        accuracy = correct_answers / len(results) * 100
        print(f"Answer Accuracy:         {accuracy:.1f}% ({correct_answers}/{len(results)})")

if __name__ == "__main__":
    main()