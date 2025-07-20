# Test Directory - Legacy Files

**Note:** This directory contains backup and experimental files. The main project has been reorganized into a modular structure in the root directory.

## Current Project Structure (in root)
- Main pipeline: `python pipeline_with_rewards.py`
- Modules: `dataset/`, `reward/`, `setup/`, `verification/`, `utils/`

## Contents
- `backup/` - Alternative pipeline implementations and debugging utilities
  - `quick_fix.py` - HuggingFace authentication fix utility
  - `test_prover9_simple.py` - Prover9 testing utilities
  - Various experimental pipeline versions

## Requirements (for main project)
- H100 GPU
- Prover9 installed and in PATH
- Hugging Face authentication for Llama models
