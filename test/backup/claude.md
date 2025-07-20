# Directory Cleanup Task for Claude Code

## 🎯 Objective
Clean up the current directory to keep only the essential files needed to run the FOL reasoning pipeline. Move all non-essential files to a backup directory.

## 📋 Essential Files to Keep
Keep ONLY these files in the main directory:

### Core Pipeline Files (KEEP)
- `claude_qwen_fol_pipeline.py` - Main pipeline that runs the complete workflow
- `setup_models.py` - Model setup script for Qwen3-8B and Llama NL-to-FOL
- `prover9_integration.py` - Prover9 integration functions (imported by main pipeline)
- `cache_manager.py` - Cache management utility
- `requirements.txt` - Python dependencies

### Generated/Output Files (KEEP if they exist)
- `conda_setup.sh` - Generated cache environment script
- `cache/` directory - Local cache folder if it exists
- Any `.json` files with results - Logic problem results if they exist

## 🗂️ Files to Move to Backup
Move ALL other `.py` files to a `backup/` subdirectory:

### Alternative Pipeline Files
- `qwen_to_prover9_pipeline.py`
- `complete_pipeline.py` 
- `fol_problem_solver.py`

### Testing & Debug Files
- `interactive_test.py`
- `fol_converter_only.py`
- `simple_example.py`
- `debug_qwen_generation.py`
- `show_raw_qwen.py`
- `test_prover9_simple.py`
- `fix_qwen_generation.py`

### Setup & Auth Files
- `setup_cache.py` (redundant with cache_manager.py)
- `hf_login.py`
- `hf_auth_fix.py` 
- `quick_fix.py`

### Documentation Files
- `README.md` - Old documentation
- `PROJECT_SUMMARY.md` - Project summary
- Any other `.md` files

## 🔧 Detailed Instructions

### Step 1: Analyze Current Directory
```bash
# First, show me what files currently exist
ls -la *.py *.md *.txt *.sh 2>/dev/null || echo "Some file types not found"
```

### Step 2: Create Backup Directory
```bash
mkdir -p backup/
```

### Step 3: Move Non-Essential Files
Move all the backup files listed above to the `backup/` directory. For each file type:

```bash
# Example commands (execute for each file that exists):
mv qwen_to_prover9_pipeline.py backup/ 2>/dev/null || echo "File not found"
mv interactive_test.py backup/ 2>/dev/null || echo "File not found"
# ... continue for all backup files
```

### Step 4: Verify Essential Files Remain
After cleanup, the main directory should contain ONLY:
- `claude_qwen_fol_pipeline.py`
- `setup_models.py` 
- `prover9_integration.py`
- `cache_manager.py`
- `requirements.txt`
- `conda_setup.sh` (if exists)
- `cache/` directory (if exists)
- Any result `.json` files (if exist)

### Step 5: Create New Clean README
Create a new `README.md` file with minimal usage instructions:

```markdown
# FOL Reasoning Pipeline

## Quick Setup
1. Setup cache: `python cache_manager.py setup`
2. Install dependencies: `pip install -r requirements.txt`  
3. Run pipeline: `python claude_qwen_fol_pipeline.py`

## Requirements
- H100 GPU
- Prover9 installed and in PATH
- Hugging Face authentication for Llama models

## Files
- `claude_qwen_fol_pipeline.py` - Main pipeline
- `setup_models.py` - Model loading
- `prover9_integration.py` - Logic verification
- `cache_manager.py` - Cache management
- `backup/` - Non-essential files
```

### Step 6: Final Verification
List the final directory contents to confirm cleanup:
```bash
echo "=== MAIN DIRECTORY ==="
ls -la *.py *.md *.txt *.sh 2>/dev/null
echo ""
echo "=== BACKUP DIRECTORY ==="
ls -la backup/ 2>/dev/null
echo ""
echo "=== CACHE DIRECTORY ==="
ls -la cache/ 2>/dev/null || echo "Cache directory not found"
```

## ⚠️ Important Notes

1. **Double-check before moving** - Ensure files exist before attempting to move them
2. **Preserve permissions** - Use `mv` to maintain file permissions
3. **Handle missing files gracefully** - Use `2>/dev/null || echo "File not found"` for files that might not exist
4. **Keep any user-generated content** - Don't move any `.json` result files or user configurations
5. **Verify imports** - After cleanup, make sure the main pipeline can still import from `prover9_integration.py`

## 🎯 Success Criteria

After completion:
- ✅ Main directory has only 5-8 essential files
- ✅ All non-essential files moved to `backup/`
- ✅ New clean `README.md` created
- ✅ Pipeline can still run: `python claude_qwen_fol_pipeline.py`
- ✅ Directory is clean and organized

## 🔍 Final Test
Run this command to verify the pipeline still works after cleanup:
```bash
python -c "
try:
    from claude_qwen_fol_pipeline import get_claude_generated_problems
    from setup_models import setup_qwen3, setup_llama_lora  
    from prover9_integration import test_prover9_installation
    print('✅ All imports successful - pipeline ready!')
except ImportError as e:
    print(f'❌ Import error: {e}')
"
```