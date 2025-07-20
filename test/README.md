# FOL Reasoning Pipeline

## Quick Setup
1. Setup cache: `python cache_manager.py setup`
2. Install dependencies: `pip install -r requirements.txt`  
3. Run pipeline: `claude_qwen_fol_pipeline`

## Requirements
- H100 GPU
- Prover9 installed and in PATH
- Hugging Face authentication for Llama models

## Files
- `setup_models.py` - Model loading
- `prover9_integration.py` - Logic verification
- `cache_manager.py` - Cache management
- `conda_setup.sh` - Environment setup script
- `logic_problem_20250718_233524.json` - Results file
- `backup/` - Alternative pipeline files and utilities
