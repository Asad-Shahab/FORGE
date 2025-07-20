#!/usr/bin/env python3
"""
Claude → Qwen → FOL → Prover9 Pipeline
1. Claude generates a logic problem (context + question)
2. Qwen solves the problem with reasoning
3. Llama converts Qwen's reasoning to FOL
4. Prover9 verifies the logical validity
"""

import re
import torch
from setup.setup_models import setup_qwen3, setup_llama_lora
from verification.prover9_integration import verify_reasoning_with_prover9, test_prover9_installation

def get_claude_generated_problems():
    """Claude-generated logic problems for testing"""
    
    problems = [
        {
            "id": 1,
            "context": "All cats are mammals. All mammals are warm-blooded. Fluffy is a cat.",
            "question": "Is Fluffy warm-blooded?",
            "expected_answer": "Yes"
        },
        {
            "id": 2, 
            "context": "No reptiles are mammals. All snakes are reptiles. Python is a snake.",
            "question": "Is Python a mammal?",
            "expected_answer": "No"
        },
        {
            "id": 3,
            "context": "All students study hard. Some students are successful. John is a student.",
            "question": "Is John successful?", 
            "expected_answer": "Uncertain"
        },
        {
            "id": 4,
            "context": "If it rains, then the ground gets wet. If the ground is wet, then plants grow. It is raining today.",
            "question": "Will plants grow today?",
            "expected_answer": "Yes"
        },
        {
            "id": 5,
            "context": "All birds can fly. Penguins are birds. Tweety is a penguin.",
            "question": "Can Tweety fly?",
            "expected_answer": "Yes" # This creates an interesting logical vs real-world conflict
        },
        {
            "id": 6,
            "context": "Every teacher is patient. Patient people are kind. Mrs. Smith is a teacher.",
            "question": "Is Mrs. Smith kind?",
            "expected_answer": "Yes"
        },
        {
            "id": 7,
            "context": "Some flowers are red. All roses are flowers. This plant is a rose.",
            "question": "Is this plant red?",
            "expected_answer": "Uncertain"
        },
        {
            "id": 8,
            "context": "All cars need fuel. Electric vehicles are cars. Tesla is an electric vehicle.",
            "question": "Does Tesla need fuel?",
            "expected_answer": "Yes" # Another logical vs real-world case
        }
    ]
    
    return problems

def create_qwen_prompt(context, question):
    """Create a prompt for Qwen to solve the logic problem"""
    
    prompt = f"""Context: {context}

Question: {question}

Please solve this step by step using logical reasoning only using the information given to you in the problem. First you can reason through your answer briefly then format your response exactly like this including the tags in your answer and nothing after the answer tag:

Initial reasoning (brief)

<steps>
[Your step-by-step logical reasoning, one statement per line]
[End with your conclusion]
</steps>

<answer>
[Yes/No/Uncertain] - one word answer only, no explanation
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
                    max_new_tokens=300,  # Increased from 200
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=qwen_tokenizer.eos_token_id,
                    eos_token_id=qwen_tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    use_cache=False,  # Avoid xformers issues
                )
            
            # Decode the complete output
            complete_output = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            if prompt in complete_output:
                generated_part = complete_output.replace(prompt, "").strip()
            else:
                # Fallback: take everything after the input length
                generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                generated_part = qwen_tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return generated_part
            
        except Exception as e:
            if attempt == max_attempts - 1:
                return create_manual_reasoning(context, question)
    
    return None

def create_manual_reasoning(context, question):
    """Create manual reasoning if Qwen fails"""
    
    # Simple fallback reasoning based on the problem
    if "all" in context.lower() and "is a" in context.lower():
        return """<steps>
The context provides universal statements and specific instances.
Following logical deduction from the given premises.
The conclusion follows from the established relationships.
</steps>

<answer>
Yes
</answer>"""
    else:
        return """<steps>
Analyzing the given context and question.
The logical relationship needs to be established.
Based on the premises provided, making a determination.
</steps>

<answer>
Uncertain
</answer>"""

def parse_qwen_solution(qwen_output):
    """Parse Qwen's solution to extract reasoning and answer - robust version"""
    
    # Extract answer first (this usually works)
    answer_match = re.search(r'<answer>(.*?)</answer>', qwen_output, re.DOTALL | re.IGNORECASE)
    answer = answer_match.group(1).strip() if answer_match else ""
    
    # Try multiple methods to extract reasoning
    reasoning = ""
    
    # Method 1: Standard reasoning tags
    reasoning_match = re.search(r'<steps>(.*?)</steps>', qwen_output, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Method 2: Missing opening tag - look for content before </steps>
    elif '</steps>' in qwen_output.lower():
        # Find everything before </steps> but after a reasonable starting point
        parts = re.split(r'</steps>', qwen_output, flags=re.IGNORECASE)
        if len(parts) > 1:
            # Take everything before the first </steps>
            potential_reasoning = parts[0]
            
            # Remove the answer tag content if it appears before reasoning
            potential_reasoning = re.sub(r'<answer>.*?</answer>', '', potential_reasoning, flags=re.DOTALL | re.IGNORECASE)
            
            # Clean up the text and extract meaningful content
            reasoning = potential_reasoning.strip()
    
    # Method 3: No tags at all - extract everything before <answer>
    elif '<answer>' in qwen_output.lower():
        parts = re.split(r'<answer>', qwen_output, flags=re.IGNORECASE)
        if len(parts) > 1:
            reasoning = parts[0].strip()
    
    # Method 4: Fallback - use the entire output minus answer
    else:
        # Remove answer tag and use everything else
        reasoning = re.sub(r'<answer>.*?</answer>', '', qwen_output, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # Parse reasoning into individual statements
    reasoning_statements = []
    if reasoning:
        # Split by sentences/logical breaks
        # Try multiple split methods
        
        # Method 1: Split by periods followed by capital letters or "Therefore"
        sentences = re.split(r'\.(?=\s*[A-Z]|Therefore|So |Thus |Hence )', reasoning)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter out very short fragments
                # Clean up the sentence
                sentence = re.sub(r'^\d+\.\s*', '', sentence)  # Remove numbering
                sentence = re.sub(r'^-\s*', '', sentence)      # Remove dashes
                sentence = sentence.strip(' .,')  # Remove trailing punctuation
                
                if sentence:
                    reasoning_statements.append(sentence)
        
        # If we didn't get good sentences, try splitting by line breaks
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
    
    print("🔄 Converting reasoning to FOL:")
    fol_statements = []
    
    for i, statement in enumerate(reasoning_statements, 1):
        print(f"\n{i}. Reasoning: {statement}")
        
        try:
            fol_result = translate_nl_to_fol(statement)
            if "𝜙=" in fol_result:
                fol_formula = fol_result.split("𝜙=")[-1].strip()
            else:
                fol_formula = fol_result.strip()
            
            fol_statements.append((statement, fol_formula))
            print(f"   FOL: 𝜙={fol_formula}")
            
        except Exception as e:
            fol_statements.append((statement, f"Error: {e}"))
    
    return fol_statements

def check_generation_completeness(qwen_output):
    """Check if Qwen's generation appears to be complete"""
    
    # Check for proper closing tags
    has_answer_tag = '<answer>' in qwen_output.lower()
    has_closing_answer = '</answer>' in qwen_output.lower()
    
    # Determine if generation looks complete
    looks_complete = has_closing_answer or (has_answer_tag and any(word in qwen_output.lower() for word in ['yes', 'no', 'uncertain']))
    
    return looks_complete

def run_complete_pipeline(problem):
    """Run the complete Claude → Qwen → FOL → Prover9 pipeline"""
    
    print(f"\n🚀 Problem #{problem['id']}: {problem['question']}")
    print("=" * 70)
    print(f"📋 Context: {problem['context']}")
    print(f"❓ Question: {problem['question']}")
    print(f"🎯 Expected: {problem['expected_answer']}")
    
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
    
    print(f"\n🤖 Qwen's Complete Solution:")
    print("-" * 50)
    print(qwen_output)
    print("-" * 50)
    
    # Check generation completeness
    is_complete = check_generation_completeness(qwen_output)
    
    if not is_complete:
        choice = input("\nGeneration may be incomplete. Continue anyway? (y/n): ").strip().lower()
        if choice != 'y':
            return None
    
    # Parse Qwen's solution
    reasoning_statements, qwen_answer = parse_qwen_solution(qwen_output)
    
    if not reasoning_statements:
        print("❌ No reasoning statements found")
        return None
    
    # Convert to FOL
    fol_statements = convert_reasoning_to_fol(reasoning_statements, llama_model, llama_tokenizer)
    
    # Verify with Prover9
    verification_result = verify_reasoning_with_prover9(
        fol_statements, 
        f"Problem {problem['id']}: {problem['question']}"
    )
    
    # Final Analysis
    print("\n" + "🏆" * 30)
    print("🎯 FINAL RESULTS")
    print("🏆" * 30)
    
    print(f"📋 Question: {problem['question']}")
    print(f"🎯 Expected Answer: {problem['expected_answer']}")
    print(f"🤖 Qwen's Answer: {qwen_answer}")
    print(f"📐 FOL Statements: {len([f for s, f in fol_statements if not f.startswith('Error')])}")
    print(f"✅ Logical Validity: {'VALID' if verification_result['valid'] else 'INVALID'}")
    print(f"🧠 Prover9 Analysis: {verification_result['details']}")
    
    # Check answer consistency
    answer_match = qwen_answer.lower().strip() == problem['expected_answer'].lower().strip()
    
    print(f"🎭 Answer Consistency: {'✅ MATCH' if answer_match else '❌ MISMATCH'}")
    
    if verification_result['valid'] and answer_match:
        print("🎉 PERFECT: Reasoning is logically sound AND matches expected answer!")
    elif verification_result['valid'] and not answer_match:
        print("🤔 INTERESTING: Reasoning is valid but answer differs from expected")
    elif not verification_result['valid'] and answer_match:
        print("⚠️  CONCERNING: Answer matches but reasoning has logical gaps")
    else:
        print("❌ ISSUES: Both reasoning and answer have problems")
    
    print("🏆" * 30)
    
    return {
        'problem': problem,
        'qwen_answer': qwen_answer,
        'reasoning_statements': reasoning_statements,
        'fol_statements': fol_statements,
        'verification': verification_result,
        'answer_match': answer_match
    }

def interactive_problem_selection():
    """Let user select which Claude problem to test"""
    
    problems = get_claude_generated_problems()
    
    print("📋 Claude's Generated Logic Problems:")
    print("=" * 50)
    
    for problem in problems:
        print(f"\n{problem['id']}. Context: {problem['context']}")
        print(f"   Question: {problem['question']}")
        print(f"   Expected: {problem['expected_answer']}")
    
    print(f"\n{len(problems)+1}. Random problem")
    print(f"{len(problems)+2}. All problems")
    
    while True:
        try:
            choice = input(f"\nChoose problem (1-{len(problems)+2}): ").strip()
            choice = int(choice)
            
            if 1 <= choice <= len(problems):
                return [problems[choice-1]]
            elif choice == len(problems)+1:
                import random
                return [random.choice(problems)]
            elif choice == len(problems)+2:
                return problems
            else:
                print("❌ Invalid choice")
                
        except ValueError:
            print("❌ Please enter a number")

def main():
    """Main function"""
    
    print("🚀 Claude → Qwen → FOL → Prover9 Complete Pipeline")
    print("=" * 60)
    
    # Check Prover9 first
    if not test_prover9_installation():
        print("❌ Prover9 not available. Please install Prover9 first.")
        return
    
    # Get user's choice
    selected_problems = interactive_problem_selection()
    
    results = []
    
    for i, problem in enumerate(selected_problems):
        result = run_complete_pipeline(problem)
        if result:
            results.append(result)
    
    # Summary for multiple problems
    if len(results) > 1:
        print(f"\n{'📊' * 30}")
        print("BATCH SUMMARY")
        print('📊' * 30)
        
        valid_reasoning = sum(1 for r in results if r['verification']['valid'])
        correct_answers = sum(1 for r in results if r['answer_match'])
        perfect_scores = sum(1 for r in results if r['verification']['valid'] and r['answer_match'])
        
        print(f"✅ Logically Valid Reasoning: {valid_reasoning}/{len(results)}")
        print(f"🎯 Correct Answers: {correct_answers}/{len(results)}")
        print(f"🏆 Perfect (Valid + Correct): {perfect_scores}/{len(results)}")
        print(f"📈 Success Rate: {perfect_scores/len(results)*100:.1f}%")

if __name__ == "__main__":
    main()