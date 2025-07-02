#!/usr/bin/env python3
"""
Debug script to test ProverQA dataset loading with correct approach.
"""

from datasets import load_dataset
import json

def test_proverqa_loading():
    """Test ProverQA dataset loading with the correct approach from HuggingFace docs."""
    
    print("🔍 Testing ProverQA dataset loading (correct approach)...")
    
    datasets_to_try = [
        ("Training set", "train/provergen-5000.json"),
        ("Easy dev", "dev/easy.json"),
        ("Medium dev", "dev/medium.json"), 
        ("Hard dev", "dev/hard.json"),
        ("All dev", "dev/*.json")
    ]
    
    working_datasets = []
    
    for name, data_files in datasets_to_try:
        try:
            print(f"\n📊 Testing {name}: {data_files}")
            
            # Load using the correct approach
            dataset = load_dataset("opendatalab/ProverQA", data_files=data_files)["train"]
            print(f"✅ Success! Loaded {len(dataset)} examples")
            
            # Check first example
            if len(dataset) > 0:
                first_item = dataset[0]
                print(f"📋 Columns: {list(first_item.keys())}")
                
                # Show sample content
                if 'context' in first_item:
                    context = first_item['context'][:100] + "..." if len(first_item['context']) > 100 else first_item['context']
                    print(f"📝 Sample context: {context}")
                
                if 'question' in first_item:
                    question = first_item['question'][:100] + "..." if len(first_item['question']) > 100 else first_item['question']
                    print(f"❓ Sample question: {question}")
                
                if 'answer' in first_item:
                    print(f"✅ Sample answer: {first_item['answer']}")
            
            working_datasets.append((name, data_files, len(dataset)))
            
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
    
    print(f"\n🎉 Summary: {len(working_datasets)} datasets loaded successfully!")
    for name, files, count in working_datasets:
        print(f"  ✅ {name}: {count} examples")
    
    return len(working_datasets) > 0

def test_example_processing():
    """Test processing examples with the working loader."""
    print("\n🧪 Testing example processing...")
    
    try:
        # Use training set for testing
        dataset = load_dataset("opendatalab/ProverQA", data_files="train/provergen-5000.json")["train"]
        print(f"📊 Loaded {len(dataset)} examples for processing test")
        
        # Process first 3 examples
        for i in range(min(3, len(dataset))):
            item = dataset[i]
            print(f"\n📝 Example {i+1}:")
            
            # Show the structure
            for key in ['context', 'question', 'reasoning', 'answer']:
                if key in item:
                    value = str(item[key])
                    display_value = value[:100] + "..." if len(value) > 100 else value
                    print(f"  {key}: {display_value}")
        
        print("\n✅ Example processing test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Example processing failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 ProverQA Dataset Debug Script")
    
    # Test loading
    loading_success = test_proverqa_loading()
    
    # Test processing
    processing_success = test_example_processing()
    
    if loading_success and processing_success:
        print("\n🎉 All tests passed! You can now run:")
        print("\n📚 Available commands:")
        print("  # Training data (5000 examples)")
        print("  python fol_validator.py --split train --max_examples 5")
        print()
        print("  # Easy development data")
        print("  python fol_validator.py --split easy --max_examples 5")
        print()
        print("  # Medium development data") 
        print("  python fol_validator.py --split medium --max_examples 5")
        print()
        print("  # Hard development data")
        print("  python fol_validator.py --split hard --max_examples 5")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")