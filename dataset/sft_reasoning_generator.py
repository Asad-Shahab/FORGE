#!/usr/bin/env python3
"""
SFT Reasoning Generator using Qwen3-32B
Generates detailed initial reasoning for all logical reasoning problems
Run: python -m dataset.sft_reasoning_generator
"""

# IMPORTANT: Setup cache directories FIRST before importing ML libraries
from setup.setup_cache import setup_cache_directories
print("🗂️  Configuring cache directories...")
cache_dirs = setup_cache_directories()

# IMPORTANT: Setup authentication BEFORE importing transformers
print("🔑 Checking Hugging Face authentication...")
import os
import json
from pathlib import Path

def ensure_hf_auth():
    """Ensure HF authentication is working"""
    # Check for token in default location
    home = Path.home()
    token_file = home / ".cache" / "huggingface" / "token"
    
    token = None
    if token_file.exists():
        try:
            with open(token_file, 'r') as f:
                token = f.read().strip()
        except:
            pass
    
    if not token:
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    
    if token:
        # Set token in environment for all libraries
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
        print(f"✅ HF token configured")
        return True
    else:
        print("❌ No HF token found!")
        print("💡 Run: python quick_fix.py")
        return False

if not ensure_hf_auth():
    print("❌ Authentication required. Exiting.")
    exit(1)

print()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys

def check_gpu():
    """Check GPU availability and specs"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        sys.exit(1)
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"🔧 CUDA Version: {torch.version.cuda}")
    print(f"🔧 PyTorch Version: {torch.__version__}")
    print()

def setup_qwen3():
    """Setup Qwen3-32B using transformers"""
    print("🚀 Setting up Qwen3-32B...")

    model_name = "Qwen/Qwen3-32B"

    bnb_config = None
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )
    
    print("✅ Qwen3-32B loaded successfully!")
    print(f"📍 Model device: {next(model.parameters()).device}")
    print(f"📏 Max sequence length: {model.config.max_position_embeddings}")
    print(f"🔢 Number of parameters: {model.num_parameters():,}")
    print()
    
    return model, tokenizer

def load_all_examples():
    """Load all examples from sft_cleaned.json"""
    sft_file = Path("dataset/sft/sft_cleaned.json")
    
    if not sft_file.exists():
        print(f"❌ {sft_file} not found!")
        print("💡 Make sure you've run the dev_split and sft_processor scripts first")
        sys.exit(1)
    
    with open(sft_file, 'r') as f:
        dataset = json.load(f)
    
    if not dataset:
        print("❌ Dataset is empty!")
        sys.exit(1)
    
    print(f"📋 Loaded dataset with {len(dataset)} instances")
    return dataset

def generate_initial_reasoning(model, tokenizer, example):
    """Generate detailed initial reasoning for a logical reasoning problem"""
    
    question = example["question"]
    steps = example["steps"]
    
    # Create prompt for generating detailed initial reasoning
    prompt = f"""You are an expert in logical reasoning. Given a logical reasoning problem and its solution steps, provide detailed initial reasoning that shows your thought process before working through the formal steps.

Your initial reasoning should include:
1. What the problem is asking
2. What information/statements you have
3. What logical approach you'll take
4. Any key insights or observations
5. How you plan to connect the facts to reach the conclusion

Write 4-6 sentences that capture your detailed analysis and step-by-step approach to the problem. End your reasoning with a conclusion sentence that summarizes your approach. After your reasoning, write "[END]" on a new line.

Problem:
{question}

Solution Steps:
{steps}

Provide only the detailed initial reasoning followed by [END]:"""

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1500,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    # Decode only the new tokens (generated part)
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    reasoning = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Check if [END] token was found (indicating complete reasoning)
    had_end_token = "[END]" in reasoning
    
    # Clean up the output - remove [END] if present
    if had_end_token:
        reasoning = reasoning.split("[END]")[0].strip()
    
    # If no [END] token was found, the reasoning might be incomplete
    if not had_end_token:
        return None
    
    return reasoning

def save_results_incremental(results, output_file):
    """Save results incrementally to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def load_existing_results(output_file):
    """Load existing results if file exists"""
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
            existing_ids = {result["id"] for result in existing_results}
            print(f"📁 Found existing results with {len(existing_results)} instances")
            return existing_results, existing_ids
        except:
            print("⚠️  Could not read existing results file, starting fresh")
            return [], set()
    return [], set()

def process_all_examples(model, tokenizer, dataset):
    """Process all examples and generate reasoning for each with incremental saves"""
    output_file = Path("dataset/sft/sft_with_reasoning.json")
    
    # Load existing results if any
    results, existing_ids = load_existing_results(output_file)
    failed_ids = []
    
    print("🚀 Processing all examples...")
    print(f"📊 Total instances: {len(dataset)}")
    if existing_ids:
        print(f"📋 Skipping {len(existing_ids)} already processed instances")
    print()
    
    # Filter out already processed examples
    remaining_examples = [ex for ex in dataset if ex["id"] not in existing_ids]
    
    if not remaining_examples:
        print("✅ All examples already processed!")
        return results, failed_ids
    
    print(f"🔄 Processing {len(remaining_examples)} remaining examples...")
    
    # Process with progress bar and incremental saves
    batch_count = 0
    for i, example in enumerate(tqdm(remaining_examples, desc="Generating reasoning", unit="example")):
        try:
            # Generate reasoning for this example
            reasoning = generate_initial_reasoning(model, tokenizer, example)
            
            # Check if reasoning generation was successful
            if reasoning is None or not reasoning or len(reasoning.split()) < 10:
                # Reasoning incomplete, too short, or failed
                failed_ids.append(example["id"])
                continue
            
            # Create new entry with reasoning
            new_entry = {
                "id": example["id"],
                "question": example["question"],
                "answer": example["answer"],
                "steps": example["steps"],
                "initial_reasoning": reasoning
            }
            
            results.append(new_entry)
            batch_count += 1
            
            # Save every 20 examples
            if batch_count % 20 == 0:
                save_results_incremental(results, output_file)
                print(f"\n💾 Saved progress: {len(results)} instances processed")
            
        except Exception as e:
            print(f"\n❌ Error processing example {example['id']}: {e}")
            failed_ids.append(example["id"])
    
    # Final save for any remaining results
    save_results_incremental(results, output_file)
    
    return results, failed_ids

def main():
    """Main execution function"""
    print("🚀 Starting SFT Reasoning Generator for All Examples...\n")
    
    # Check GPU
    check_gpu()
    
    # Setup model
    model, tokenizer = setup_qwen3()
    
    # Load all examples
    dataset = load_all_examples()
    
    # Process all examples (saves incrementally)
    results, failed_ids = process_all_examples(model, tokenizer, dataset)
    
    # Final output file path
    output_file = Path("dataset/sft/sft_with_reasoning.json")
    
    # Print summary
    total_attempted = len(dataset)
    successful = len(results)
    failed = len(failed_ids)
    
    print(f"\n🎉 Processing complete!")
    print(f"📊 Total attempted: {total_attempted}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Final results saved to: {output_file}")
    
    if failed_ids:
        print(f"\n⚠️  Failed instance IDs:")
        # Print failed IDs in a nice format (10 per line)
        for i in range(0, len(failed_ids), 10):
            batch = failed_ids[i:i+10]
            print(f"   {', '.join(map(str, batch))}")
    else:
        print(f"\n🎊 All instances processed successfully!")

if __name__ == "__main__":
    main()
