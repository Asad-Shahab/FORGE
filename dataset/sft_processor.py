import json
import re
from pathlib import Path

def clean_reasoning_text(reasoning_text):
    """
    Clean the reasoning text by:
    1. Removing fact1:, fact2:, rule:, conclusion: prefixes
    2. Ensuring each line is on a new line
    3. Remove the trailing ‘The correct option is: …’
    """
    lines = reasoning_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove prefixes like fact1:, fact2:, rule:, conclusion:
        # Use regex to match patterns like "fact1:", "fact2:", "rule:", "conclusion:"
        line = re.sub(r'^(fact\d+|rule|conclusion):\s*', '', line.strip())
        
        line = re.sub(r'\s*The\s+correct\s+option\s+is\s*:\s*[A-C]\.?', '', line, flags=re.I)

        # Only add non-empty lines
        if line.strip():
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()

def process_sft_dataset():
    """
    Process SFT dataset by:
    1. Changing 'reasoning' field to 'steps'
    2. Cleaning the reasoning text (removing prefixes)
    3. Ensuring proper line formatting
    """
    
    # Define paths
    sft_dir = Path("dataset/sft")
    input_file = sft_dir / "sft_dataset.json"
    output_file = sft_dir / "sft_cleaned.json"
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        return
    
    # Load the dataset
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    
    print(f"Processing {len(dataset)} instances...")
    
    # Process each instance
    cleaned_dataset = []
    for instance in dataset:
        # Create new instance with cleaned data
        cleaned_instance = {
            "id": instance["id"],
            "question": instance["question"],
            "answer": instance["answer"],
            "steps": clean_reasoning_text(instance["reasoning"])  
        }
        cleaned_dataset.append(cleaned_instance)
    
    # Save cleaned dataset
    with open(output_file, 'w') as f:
        json.dump(cleaned_dataset, f, indent=2)
    
    print(f"Saved {len(cleaned_dataset)} cleaned instances to {output_file}")
    
    # Show example of before/after
    if dataset:
        print("\n" + "="*50)
        print("EXAMPLE TRANSFORMATION")
        print("="*50)
        print("BEFORE (reasoning field):")
        print(dataset[0]["reasoning"][:200] + "...")
        print("\nAFTER (steps field):")
        print(cleaned_dataset[0]["steps"][:200] + "...")

if __name__ == "__main__":
    process_sft_dataset()