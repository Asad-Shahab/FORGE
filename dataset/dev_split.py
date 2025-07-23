import json
import random
from pathlib import Path
from datasets import load_dataset

def clean_instance(instance):
    """Clean instance by combining context, question, and options into one field"""
    # Combine context, question, and options
    context = instance["context"]
    question = instance["question"]
    options = instance["options"]
    
    # Format options
    options_text = "\n".join(options)
    
    # Combine into single question field
    combined_question = f"{context}\n\nQuestion: {question}\n\nOptions:\n{options_text}"
    
    cleaned = {
        "id": instance["id"],
        "question": combined_question,
        "answer": instance["answer"],
        "reasoning": instance["reasoning"]
    }
    return cleaned

def split_proverqa_dev():
    """
    Split ProverQA dev dataset into SFT training set (300 instances) and test set (1200 instances).
    
    SFT: 100 from each difficulty (easy, medium, hard)
    Test: 400 from each difficulty (easy, medium, hard)
    """
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create directories
    dataset_dir = Path("dataset")
    sft_dir = dataset_dir / "sft"
    dev_dir = dataset_dir / "dev"
    
    sft_dir.mkdir(exist_ok=True)
    dev_dir.mkdir(exist_ok=True)
    
    # Process each difficulty level
    difficulties = ["easy", "medium", "hard"]
    sft_combined = []
    
    for difficulty in difficulties:
        print(f"Loading {difficulty} data from HuggingFace...")
        
        # Load data from HuggingFace
        dataset = load_dataset("opendatalab/ProverQA", data_files=f"dev/{difficulty}.json")
        raw_data = list(dataset['train'])  # HuggingFace loads as 'train' split by default
        
        # Clean the data (remove nl2fol field)
        data = [clean_instance(instance) for instance in raw_data]
        
        print(f"Loaded and cleaned {len(data)} instances from {difficulty}")
        
        # Shuffle and split
        random.shuffle(data)
        sft_data = data[:100]
        test_data = data[100:500]  # remaining 400
        
        # Add to SFT combined dataset
        sft_combined.extend(sft_data)
        
        # Save test data
        test_file = dev_dir / f"{difficulty}.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"Saved {len(test_data)} test instances to {test_file}")
    
    # Save combined SFT dataset
    sft_file = sft_dir / "sft_dataset.json"
    with open(sft_file, 'w') as f:
        json.dump(sft_combined, f, indent=2)
    
    print(f"Saved {len(sft_combined)} SFT instances to {sft_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET SPLIT SUMMARY")
    print("="*50)
    print(f"SFT Dataset: {len(sft_combined)} instances")
    print(f"  - Location: {sft_file}")
    print(f"Test Dataset: {len(difficulties) * 400} instances")
    print(f"  - Location: {dev_dir}/")
    print(f"  - Files: easy.json, medium.json, hard.json")

if __name__ == "__main__":
    split_proverqa_dev()