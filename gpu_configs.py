#!/usr/bin/env python3
"""
Recommended configurations for different GPU setups
These are optimized for H100 GPUs with 80GB memory
"""

# Configuration for 2 H100 GPUs
CONFIG_2GPU = {
    "num_gpus": 2,
    "grpo_batch_size": 4,  # Per GPU
    "grpo_gradient_accumulation_steps": 2,
    "grpo_learning_rate": 5e-6,
    "num_generations": 4,
    "temperature": 1.0,
    "max_seq_length": 3000,
    "lora_rank": 32,
    "gradient_checkpointing": False,  # H100 has enough memory
    "use_mixed_precision": True,  # BF16 for faster training
    "use_8bit_optimizer": False,  # Not needed with H100
    "grpo_max_steps": 1250,
    "grpo_logging_steps": 10,
    "grpo_save_steps": 100,
}

# Configuration for 3 H100 GPUs
CONFIG_3GPU = {
    "num_gpus": 3,
    "grpo_batch_size": 3,  # Per GPU (slightly reduced for even distribution)
    "grpo_gradient_accumulation_steps": 2,
    "grpo_learning_rate": 5e-6,
    "num_generations": 4,
    "temperature": 1.0,
    "max_seq_length": 3000,
    "lora_rank": 32,
    "gradient_checkpointing": False,
    "use_mixed_precision": True,
    "use_8bit_optimizer": False,
    "grpo_max_steps": 1000,  # Slightly reduced for 3 GPUs
    "grpo_logging_steps": 10,
    "grpo_save_steps": 100,
}

# Configuration for 4 H100 GPUs
CONFIG_4GPU = {
    "num_gpus": 4,
    "grpo_batch_size": 2,  # Moderate batch size
    "grpo_gradient_accumulation_steps": 2,
    "grpo_learning_rate": 3e-6,
    "num_generations": 8,  # Doubled generations for more signal
    "temperature": 1.2,  # Slightly higher for diversity
    "max_seq_length": 3000,
    "lora_rank": 64,
    "gradient_checkpointing": False,
    "use_mixed_precision": True,
    "use_8bit_optimizer": False,
    "grpo_max_steps": 6000,  # ~10 epochs with 5000 examples
    "grpo_logging_steps": 50,
    "grpo_save_steps": 500,
}

# Memory-optimized configuration (if running into OOM)
CONFIG_MEMORY_OPTIMIZED = {
    "grpo_batch_size": 2,  # Reduced batch size
    "grpo_gradient_accumulation_steps": 4,  # More accumulation
    "max_seq_length": 2048,  # Reduced sequence length
    "lora_rank": 16,  # Lower rank
    "gradient_checkpointing": True,  # Enable checkpointing
    "use_mixed_precision": True,
    "use_8bit_optimizer": True,  # 8-bit optimizer
    "num_generations": 2,  # Fewer generations
}

def get_config(num_gpus):
    """Get recommended configuration for given number of GPUs"""
    configs = {
        2: CONFIG_2GPU,
        3: CONFIG_3GPU,
        4: CONFIG_4GPU,
    }
    return configs.get(num_gpus, CONFIG_2GPU)

def generate_command(config):
    """Generate training command from configuration"""
    cmd = "python train.py"
    
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{key}"
        else:
            cmd += f" --{key} {value}"
    
    return cmd

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        num_gpus = int(sys.argv[1])
        config = get_config(num_gpus)
        
        print(f"Configuration for {num_gpus} GPU(s):")
        print("-" * 40)
        for key, value in config.items():
            print(f"{key:30} : {value}")
        
        print("\n" + "=" * 40)
        print("Generated command:")
        print(generate_command(config))
    else:
        print("Usage: python gpu_configs.py <num_gpus>")
        print("Example: python gpu_configs.py 2")
