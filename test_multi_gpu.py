#!/usr/bin/env python3
"""
Test script to verify multi-GPU setup before training
Run with: torchrun --nproc_per_node=2 test_multi_gpu.py
"""

import os
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import time

def test_distributed_setup():
    """Test basic distributed setup"""
    print(f"🔍 Testing Distributed Setup")
    print("-" * 40)
    
    # Check environment variables
    env_vars = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    for var in env_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"{var:15} : {value}")
    
    # Check CUDA availability
    print(f"\n📊 CUDA Information:")
    print(f"CUDA Available  : {torch.cuda.is_available()}")
    print(f"CUDA Devices    : {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
        print(f"Current Device  : {device}")
        print(f"Device Name     : {torch.cuda.get_device_name(device)}")
        print(f"Device Memory   : {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    return True

def test_model_loading():
    """Test loading a small model on multiple GPUs"""
    print(f"\n🔧 Testing Model Loading")
    print("-" * 40)
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    try:
        # Use a small model for testing
        model_name = "gpt2"  # Small model for testing
        
        print(f"[Rank {local_rank}] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"[Rank {local_rank}] Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
        
        # Move to device
        device = torch.device(f'cuda:{local_rank}')
        model = model.to(device)
        
        print(f"[Rank {local_rank}] Model loaded on {device}")
        
        # Test forward pass
        inputs = tokenizer("Hello world", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.logits.mean()
        
        print(f"[Rank {local_rank}] Forward pass successful, loss shape: {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"[Rank {local_rank}] ❌ Error loading model: {e}")
        return False

def test_distributed_communication():
    """Test distributed communication between GPUs"""
    print(f"\n🌐 Testing Distributed Communication")
    print("-" * 40)
    
    if not dist.is_initialized():
        print("Initializing process group...")
        dist.init_process_group(backend='nccl')
    
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')
    
    print(f"[Rank {local_rank}] Process group initialized")
    print(f"[Rank {local_rank}] World size: {world_size}")
    
    # Test all-reduce operation
    tensor = torch.ones(1).to(device) * (local_rank + 1)
    print(f"[Rank {local_rank}] Local tensor value: {tensor.item()}")
    
    dist.all_reduce(tensor)
    expected = sum(range(1, world_size + 1))
    
    print(f"[Rank {local_rank}] After all-reduce: {tensor.item()} (expected: {expected})")
    
    # Test broadcast
    if local_rank == 0:
        broadcast_tensor = torch.tensor([42.0]).to(device)
    else:
        broadcast_tensor = torch.zeros(1).to(device)
    
    dist.broadcast(broadcast_tensor, src=0)
    print(f"[Rank {local_rank}] Broadcast value: {broadcast_tensor.item()} (expected: 42.0)")
    
    # Synchronize
    dist.barrier()
    print(f"[Rank {local_rank}] ✅ Communication test passed")
    
    return True

def test_accelerate_setup():
    """Test Accelerate setup for distributed training"""
    print(f"\n🚀 Testing Accelerate Setup")
    print("-" * 40)
    
    accelerator = Accelerator()
    
    print(f"Device: {accelerator.device}")
    print(f"Distributed: {accelerator.distributed_type}")
    print(f"Num processes: {accelerator.num_processes}")
    print(f"Process index: {accelerator.process_index}")
    print(f"Local process index: {accelerator.local_process_index}")
    print(f"Is main process: {accelerator.is_main_process}")
    print(f"Is local main process: {accelerator.is_local_main_process}")
    print(f"Mixed precision: {accelerator.mixed_precision}")
    
    # Test data distribution
    test_data = list(range(100))
    
    with accelerator.split_between_processes(test_data) as split_data:
        print(f"[Process {accelerator.process_index}] Data samples: {len(split_data)} items")
        print(f"[Process {accelerator.process_index}] First items: {split_data[:5]}")
    
    return True

def run_all_tests():
    """Run all multi-GPU tests"""
    print("🧪 Multi-GPU Setup Test Suite")
    print("=" * 50)
    
    tests = [
        ("Distributed Setup", test_distributed_setup),
        ("Model Loading", test_model_loading),
        ("Distributed Communication", test_distributed_communication),
        ("Accelerate Setup", test_accelerate_setup),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print("-" * 40)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:25} : {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready for multi-GPU training.")
    else:
        print("\n⚠️ Some tests failed. Please check the setup before training.")
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)