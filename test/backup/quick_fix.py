#!/usr/bin/env python3
"""
Quick fix for Hugging Face authentication issues
Run this before setup_models.py
"""

import os
import sys
from pathlib import Path

def quick_fix():
    """Quick authentication fix"""
    print("🔧 Quick Authentication Fix")
    print("=" * 30)
    
    # Check if we have a token in the default location
    home = Path.home()
    default_token = home / ".cache" / "huggingface" / "token"
    
    token = None
    
    # Method 1: Check default token location
    if default_token.exists():
        try:
            with open(default_token, 'r') as f:
                token = f.read().strip()
                print(f"✅ Found token in: {default_token}")
        except Exception as e:
            print(f"❌ Cannot read token: {e}")
    
    # Method 2: Check environment variables
    if not token:
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if token:
            print("✅ Found token in environment")
    
    # Method 3: Manual entry
    if not token:
        print("❌ No token found. Please enter manually:")
        print("Get your token from: https://huggingface.co/settings/tokens")
        token = input("Enter your HuggingFace token: ").strip()
    
    if not token:
        print("❌ No token provided. Cannot continue.")
        return False
    
    # Set token in ALL possible environment variables
    env_vars = [
        "HUGGINGFACE_HUB_TOKEN",
        "HF_TOKEN", 
        "HF_ACCESS_TOKEN"
    ]
    
    for var in env_vars:
        os.environ[var] = token
        print(f"🔑 Set {var}")
    
    # Test the token
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        
        # Test authentication
        user_info = api.whoami(token=token)
        print(f"✅ Authenticated as: {user_info['name']}")
        
        # Test model access
        try:
            model_info = api.model_info("meta-llama/Llama-3.1-8B-Instruct", token=token)
            print("✅ Can access Llama model")
            return True
        except Exception as e:
            print(f"❌ Cannot access Llama model: {e}")
            print("\n💡 Make sure you've accepted the license at:")
            print("   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
            return False
            
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

if __name__ == "__main__":
    if quick_fix():
        print("\n✅ Authentication successful!")
        print("🚀 You can now run: python setup_models.py")
    else:
        print("\n❌ Authentication failed!")
        print("\n🔄 Manual steps:")
        print("1. Visit: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        print("2. Click 'Agree and access repository'")
        print("3. Get token: https://huggingface.co/settings/tokens")
        print("4. Run this script again")