import os
import json
import random
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams

class FoLiOSatoriTester:
    def __init__(self, model_path: str = "Satori-reasoning/Satori-7B-Round2"):
        """
        Initialize the tester with Satori model using vLLM
        
        Args:
            model_path: Hugging Face model path
        """
        print(f"Loading {model_path}...")
        
        self.model_path = model_path
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
        )
        
        self.sampling_params = SamplingParams(
            max_tokens=4096,
            temperature=0.0,
            n=1,
            skip_special_tokens=True
        )
        
        print("Model loaded successfully!")
        self.results = []
    
    def load_folio_data(self, file_path: str) -> List[Dict]:
        """
        Load FoLiO dataset from JSONL file
        
        Args:
            file_path: Path to FoLiO dataset JSONL file
            
        Returns:
            List of FoLiO examples
        """
        data = []
        with open('folio-train.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def create_prompt(self, premises: List[str], conclusion: str, use_cot: bool = True) -> str:
        """
        Create prompt for logical reasoning task using Satori's format
        
        Args:
            premises: List of premise statements
            conclusion: Conclusion to evaluate
            use_cot: Whether to use chain-of-thought prompting
            
        Returns:
            Formatted prompt string
        """
        premises_text = "\n".join([f"{i+1}. {premise}" for i, premise in enumerate(premises)])
        
        if use_cot:
            problem_text = f"""Given the following premises, determine if the conclusion is True, False, or Unknown. Think step by step through the logical reasoning.

Premises:
{premises_text}

Conclusion: {conclusion}

Please analyze this step by step:
1. Identify the key facts from the premises
2. Apply logical reasoning rules
3. Determine if the conclusion follows logically

Your final answer must be exactly one of: True, False, or Unknown"""
        else:
            problem_text = f"""Given the following premises, determine if the conclusion is True, False, or Unknown.

Premises:
{premises_text}

Conclusion: {conclusion}

Your final answer must be exactly one of: True, False, or Unknown"""
        
        prompt = f"<|im_start|>user\nSolve the following logical reasoning problem efficiently and clearly.\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nProblem: {problem_text}<|im_end|>\n<|im_start|>assistant\n"
        return prompt
    
    def extract_answer(self, response: str) -> str:
        """
        Extract the answer (True/False/Unknown) from model response
        
        Args:
            response: Model's text response
            
        Returns:
            Extracted answer
        """
        response_lower = response.lower()
        
        # Look for boxed answers first (Satori format)
        if "\\boxed{" in response_lower:
            start = response_lower.find("\\boxed{") + 7
            end = response_lower.find("}", start)
            if end != -1:
                boxed_content = response[start:end].strip()
                if "true" in boxed_content.lower():
                    return "True"
                elif "false" in boxed_content.lower():
                    return "False"
                elif "unknown" in boxed_content.lower():
                    return "Unknown"
        
        # Look for explicit answer patterns
        if "answer: true" in response_lower or "answer is true" in response_lower:
            return "True"
        elif "answer: false" in response_lower or "answer is false" in response_lower:
            return "False"
        elif "answer: unknown" in response_lower or "answer is unknown" in response_lower:
            return "Unknown"
        
        # Look for conclusion patterns
        if "conclusion is true" in response_lower or "therefore true" in response_lower:
            return "True"
        elif "conclusion is false" in response_lower or "therefore false" in response_lower:
            return "False"
        elif "conclusion is unknown" in response_lower or "cannot be determined" in response_lower:
            return "Unknown"
        
        # Look for final answer patterns
        if "final answer:" in response_lower:
            final_part = response_lower.split("final answer:")[-1][:50]
            if "true" in final_part:
                return "True"
            elif "false" in final_part:
                return "False"
            elif "unknown" in final_part:
                return "Unknown"
        
        # Look for simple patterns in the first 100 characters
        first_part = response_lower[:100]
        if " true" in first_part:
            return "True"
        elif " false" in first_part:
            return "False"
        elif " unknown" in first_part:
            return "Unknown"
        
        # Default fallback
        return "Unknown"
    
    def generate_responses(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts using vLLM
        
        Args:
            prompts: List of prompts
            
        Returns:
            List of generated responses
        """
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=True)
        completions = [output.outputs[0].text for output in outputs]
        return completions
    
    def test_subset(self, data: List[Dict], subset_size: int = 20, use_cot: bool = True, 
                   random_seed: int = 42) -> Dict:
        """
        Test Satori on a subset of FoLiO data
        
        Args:
            data: Full FoLiO dataset
            subset_size: Number of examples to test
            use_cot: Whether to use chain-of-thought prompting
            random_seed: Random seed for reproducibility
            
        Returns:
            Test results summary
        """
        random.seed(random_seed)
        
        # Sample subset
        if len(data) > subset_size:
            test_data = random.sample(data, subset_size)
        else:
            test_data = data
        
        print(f"Testing {len(test_data)} examples with CoT={'enabled' if use_cot else 'disabled'}")
        
        # Prepare all prompts
        prompts = []
        for example in test_data:
            premises = example.get('premises', [])
            conclusion = example.get('conclusion', '')
            prompt = self.create_prompt(premises, conclusion, use_cot)
            prompts.append(prompt)
        
        # Generate all responses at once
        print("Generating responses...")
        responses = self.generate_responses(prompts)
        
        # Process results
        results = []
        for i, (example, response) in enumerate(zip(test_data, responses)):
            premises = example.get('premises', [])
            conclusion = example.get('conclusion', '')
            true_label = example.get('label', 'Unknown')
            story_id = example.get('story_id', 'N/A')
            example_id = example.get('example_id', 'N/A')
            
            predicted_answer = self.extract_answer(response)
            
            result = {
                'story_id': story_id,
                'example_id': example_id,
                'premises': premises,
                'conclusion': conclusion,
                'true_label': true_label,
                'predicted_label': predicted_answer,
                'correct': predicted_answer == true_label,
                'response': response,
                'prompt': prompts[i]
            }
            
            results.append(result)
            print(f"Example {i+1}: {true_label} -> {predicted_answer} ({'✓' if result['correct'] else '✗'})")
        
        # Calculate metrics
        correct_predictions = sum(1 for r in results if r['correct'])
        accuracy = correct_predictions / len(results) if results else 0
        
        # Calculate per-label accuracy
        label_stats = {'True': {'correct': 0, 'total': 0},
                      'False': {'correct': 0, 'total': 0},
                      'Unknown': {'correct': 0, 'total': 0}}
        
        for result in results:
            true_label = result['true_label']
            if true_label in label_stats:
                label_stats[true_label]['total'] += 1
                if result['correct']:
                    label_stats[true_label]['correct'] += 1
        
        summary = {
            'total_examples': len(results),
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'label_accuracies': {
                label: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                for label, stats in label_stats.items()
            },
            'label_counts': {
                label: stats['total'] for label, stats in label_stats.items()
            },
            'use_cot': use_cot,
            'model_path': self.model_path,
            'detailed_results': results
        }
        
        return summary
    
    def save_results(self, results: Dict, output_file: str):
        """
        Save test results to JSON file
        
        Args:
            results: Test results dictionary
            output_file: Output file path
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    
    def print_summary(self, results: Dict):
        """
        Print a summary of test results
        
        Args:
            results: Test results dictionary
        """
        print("\n" + "="*60)
        print("FOLIO DATASET TEST SUMMARY")
        print("="*60)
        print(f"Model: {results['model_path']}")
        print(f"Chain-of-Thought: {'Enabled' if results['use_cot'] else 'Disabled'}")
        print(f"Total Examples: {results['total_examples']}")
        print(f"Overall Accuracy: {results['accuracy']:.3f}")
        print("\nPer-label Performance:")
        
        for label, accuracy in results['label_accuracies'].items():
            count = results['label_counts'][label]
            print(f"  {label}: {accuracy:.3f} ({count} examples)")
        
        print("\nSample Results:")
        for i, result in enumerate(results['detailed_results'][:3]):
            print(f"\nExample {i+1} (Story {result['story_id']}, Example {result['example_id']}):")
            print(f"Conclusion: {result['conclusion'][:100]}...")
            print(f"True Label: {result['true_label']}")
            print(f"Predicted: {result['predicted_label']}")
            print(f"Correct: {result['correct']}")
            print(f"Response preview: {result['response'][:150]}...")

def main():
    """
    Main function to run FoLiO testing with Satori model
    """
    # Configuration
    data_path = "folio-train.jsonl"
    model_path = "Satori-reasoning/Satori-7B-Round2"
    subset_size = 20
    
    # Sample data for testing if file not found
    sample_data = [
        {
            "story_id": 406,
            "example_id": 1131,
            "conclusion": "Rina is a person who jokes about being addicted to caffeine or unaware that caffeine is a drug.",
            "premises": [
                "All people who regularly drink coffee are dependent on caffeine.",
                "People either regularly drink coffee or joke about being addicted to caffeine.",
                "No one who jokes about being addicted to caffeine is unaware that caffeine is a drug.",
                "Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug.",
                "If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student."
            ],
            "label": "True",
            "source": "hyb"
        },
        {
            "story_id": 8,
            "example_id": 20,
            "conclusion": "Miroslav Venhoda loved music.",
            "premises": [
                "Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.",
                "Any choral conductor is a musician.",
                "Some musicians love music.",
                "Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant."
            ],
            "label": "Unknown",
            "source": "wiki"
        }
    ]
    
    # Initialize tester
    try:
        tester = FoLiOSatoriTester(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load data
    try:
        data = tester.load_folio_data(data_path)
        print(f"Loaded {len(data)} examples from {data_path}")
    except FileNotFoundError:
        print(f"Dataset file {data_path} not found, using sample data")
        data = sample_data
    except Exception as e:
        print(f"Error loading data: {e}")
        data = sample_data
    
    # Test with chain-of-thought
    print(f"\nTesting {model_path} on FoLiO dataset...")
    try:
        results = tester.test_subset(data, subset_size=subset_size, use_cot=True)
        tester.print_summary(results)
        tester.save_results(results, "satori_folio_results.json")
    except Exception as e:
        print(f"Error during testing: {e}")
        return
    
    print("\nTesting completed successfully!")

if __name__ == "__main__":
    main()