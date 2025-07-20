#!/usr/bin/env python3
"""
ProverQA Dataset Processor
Loads the ProverQA training dataset and extracts simplified question-answer pairs
"""

import json
import re
from datasets import load_dataset
from typing import List, Dict, Any

def load_proverqa_dataset():
    """Load the ProverQA training dataset from HuggingFace"""
    dataset = load_dataset("opendatalab/ProverQA", data_files="train/provergen-5000.json")
    return dataset['train']

def extract_answer_from_output(output_string: str) -> str:
    """Extract the answer from the JSON output string"""
    try:
        output_json = json.loads(output_string)
        answer = output_json.get('answer', '')
        return answer.strip()
    except json.JSONDecodeError:
        answer_match = re.search(r'"answer":\s*"([ABC])"', output_string)
        if answer_match:
            return answer_match.group(1)
        final_match = re.search(r'([ABC])(?:\s*[}"])*\s*$', output_string)
        if final_match:
            return final_match.group(1)
        return ""

def process_dataset_entry(entry: Dict[str, Any]) -> Dict[str, str]:
    """Process a single dataset entry to extract question and answer"""
    question = entry.get('instruction', '').strip()
    
    # Remove "Context:" from the start if present
    if question.startswith("Context:"):
        question = question[8:].strip()
    
    output_string = entry.get('output', '')
    answer = extract_answer_from_output(output_string)
    
    return {
        'question': question,
        'answer': answer
    }

def create_simplified_dataset(dataset) -> List[Dict[str, str]]:
    """Create simplified dataset with just question and answer"""
    simplified_data = []
    
    for entry in dataset:
        try:
            processed_entry = process_dataset_entry(entry)
            if processed_entry['question'] and processed_entry['answer']:
                simplified_data.append(processed_entry)
        except Exception:
            continue
    
    return simplified_data

def save_simplified_dataset(data: List[Dict[str, str]], filename: str = "proverqa_simplified.json"):
    """Save the simplified dataset to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    """Main function to process ProverQA dataset"""
    dataset = load_proverqa_dataset()
    simplified_data = create_simplified_dataset(dataset)
    save_simplified_dataset(simplified_data)

if __name__ == "__main__":
    main()