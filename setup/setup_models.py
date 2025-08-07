#!/usr/bin/env python3
"""
Setup script for H100 GPU with Qwen3-8B and Llama NL-to-FOL models
Multi-GPU compatible version
Run: python setup_models.py
"""

# IMPORTANT: Setup cache directories FIRST before importing ML libraries
from .setup_cache import setup_cache_directories
print("🗂️  Configuring cache directories...")
cache_dirs = setup_cache_directories()

# IMPORTANT: Setup authentication BEFORE importing transformers
print("🔑 Checking Hugging Face authentication...")
import os
from pathlib import Path
import torch
import torch.distributed as dist

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

# Only check auth on main process to avoid exit() in distributed training
rank = int(os.environ.get('LOCAL_RANK', 0))
if rank == 0 and not ensure_hf_auth():
    print("❌ Authentication required. Exiting.")
    exit(1)
elif rank > 0:
    # For non-main processes, just set token if available
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        # Check for token file
        from pathlib import Path
        home = Path.home()
        token_file = home / ".cache" / "huggingface" / "token"
        if token_file.exists():
            try:
                with open(token_file, 'r') as f:
                    token = f.read().strip()
                    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
                    os.environ["HF_TOKEN"] = token
            except:
                pass

print()

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

def check_gpu():
    """Check GPU availability and specs"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        sys.exit(1)
    
    num_gpus = torch.cuda.device_count()
    print(f"✅ CUDA available with {num_gpus} GPU(s)")
    
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    print(f"🔧 CUDA Version: {torch.version.cuda}")
    print(f"🔧 PyTorch Version: {torch.__version__}")
    print()
    
    return num_gpus

def setup_qwen3(device=None, use_quantization=True):
    """Load Qwen3-8B with optional 4-bit quantization
    
    Args:
        device: Specific device to load model on (for distributed training)
        use_quantization: Whether to use 4-bit quantization
    """
    print("🚀 Loading Qwen3-8B...")

    model_name = "Qwen/Qwen3-8B"

    bnb_config = None
    if use_quantization:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For distributed training, don't use device_map="auto"
    if device is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )
        model = model.to(device)
    else:
        # Single GPU or CPU mode
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )
    
    print("✅ Qwen3-8B loaded successfully!")
    if device is not None:
        print(f"📍 Model device: {device}")
    else:
        print(f"📍 Model device: {next(model.parameters()).device}")
    print(f"📏 Max sequence length: {model.config.max_position_embeddings}")
    print(f"🔢 Number of parameters: {model.num_parameters():,}")
    print()
    
    return model, tokenizer

def setup_llama_lora(device=None):
    """Setup Llama-3.1-8B with NL-to-FOL LoRA
    
    Args:
        device: Specific device to load model on (for distributed training)
    """
    print("🦙 Setting up Llama-3.1-8B with NL-to-FOL LoRA...")
    
    base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    lora_weights = "fvossel/Llama-3.1-8B-Instruct-nl-to-fol"
    
    # Get token from environment
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    
    # Load tokenizer with explicit token
    tokenizer_lora = AutoTokenizer.from_pretrained(
        base_model_name, 
        trust_remote_code=True,
        token=token
    )
    if tokenizer_lora.pad_token is None:
        tokenizer_lora.pad_token = tokenizer_lora.eos_token
    tokenizer_lora.padding_side = "left"
    
    # Load base model with explicit token
    if device is not None:
        model_lora = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            token=token
        )
        model_lora = model_lora.to(device)
    else:
        model_lora = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            trust_remote_code=True, 
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=token
        )
    
    # Load LoRA adapter
    model_lora = PeftModel.from_pretrained(
        model_lora, 
        lora_weights, 
        device_map=None if device is not None else "auto",
        token=token
    )
    
    print()
    
    return model_lora, tokenizer_lora

def is_distributed():
    """Check if running in distributed mode"""
    return dist.is_available() and dist.is_initialized()

def get_rank():
    """Get current process rank"""
    if is_distributed():
        return dist.get_rank()
    return 0

def get_world_size():
    """Get total number of processes"""
    if is_distributed():
        return dist.get_world_size()
    return 1

def test_qwen3(model, tokenizer):
    """Test Qwen3 model with sample generation"""
    print("🧪 Testing Qwen3 generation...")
    
    def generate_text(prompt, max_length=100):
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    test_prompt = "The future of artificial intelligence is"
    result = generate_text(test_prompt)
    
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {result}")
    print()
    
    return generate_text

def test_llama_lora(model_lora, tokenizer_lora):
    """Test Llama LoRA model with NL-to-FOL translation"""
    print("🧪 Testing NL-to-FOL translation...")
    
    def formatting_func(text):
        return tokenizer_lora.apply_chat_template(
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
    
    def translate_nl_to_fol(input_text, max_tokens=100):
        prompt = formatting_func(input_text)
        inputs = tokenizer_lora(prompt, return_tensors="pt", padding=True)
        
        device = next(model_lora.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model_lora.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                temperature=0.1, 
                do_sample=True
            )
        
        result = tokenizer_lora.decode(outputs[0], skip_special_tokens=True)
        return result
    
    # Test examples
    test_examples = [
        "All dogs are animals.",
        "Some cats are black.",
        "If it rains, then the ground is wet."
    ]
    
    for example in test_examples:
        result = translate_nl_to_fol(example)
        print(f"Input: {example}")
        if "𝜙=" in result:
            fol_part = result.split("𝜙=")[-1].strip()
            print(f"FOL: 𝜙={fol_part}")
        else:
            print(f"Output: {result}")
        print()
    
    return translate_nl_to_fol, formatting_func

def main():
    """Main execution function"""
    print("🚀 Starting H100 GPU setup...\n")
    
    # Check GPU
    num_gpus = check_gpu()
    
    # Determine device for single GPU mode
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    # Setup models
    qwen_model, qwen_tokenizer = setup_qwen3(device=None)  # Use auto device mapping for testing
    llama_model, llama_tokenizer = setup_llama_lora(device=None)
    
    # Test models
    qwen_generate = test_qwen3(qwen_model, qwen_tokenizer)
    llama_translate, llama_format = test_llama_lora(llama_model, llama_tokenizer)
    
    print("🎉 All models loaded and tested successfully!")
    print("\n📝 Available functions:")
    print("- qwen_generate(prompt, max_length=100)")
    print("- llama_translate(text, max_tokens=100)")
    print("- llama_format(text)")
    
    # Return models and functions for interactive use
    return {
        'qwen_model': qwen_model,
        'qwen_tokenizer': qwen_tokenizer,
        'qwen_generate': qwen_generate,
        'llama_model': llama_model, 
        'llama_tokenizer': llama_tokenizer,
        'llama_translate': llama_translate,
        'llama_format': llama_format
    }

if __name__ == "__main__":
    models = main()
