# Logical Reasoning Training Project

## Project Overview

**Project Name:** Logical Reasoning Training with GRPO (Group Relative Policy Optimization)

**Primary Purpose:** A comprehensive machine learning training pipeline designed to improve language models' logical reasoning capabilities through supervised fine-tuning (SFT) and reinforcement learning with human feedback (GRPO). The system combines multiple models to create, solve, verify, and evaluate logical reasoning problems.

**Main Technologies/Frameworks:**
- PyTorch and Unsloth for model training and inference
- Transformers library for model handling
- TRL (Transformers Reinforcement Learning) for GRPO training
- Prover9 for formal logical verification
- W&B (Weights & Biases) for experiment tracking
- HuggingFace Datasets for data management

**Architecture Type:** Multi-stage ML training pipeline with reinforcement learning

**Key Features:**
- End-to-end pipeline from problem generation to logical verification
- Dual training approach: SFT for initial adaptation, GRPO for preference optimization
- Formal logical verification using Prover9 theorem prover
- Comprehensive reward system based on answer correctness, logical validity, and format compliance
- Automated cache management and GPU optimization for H100 systems

## Directory Structure Analysis

### cache/
**Purpose:** Centralized cache management for all ML libraries and frameworks to optimize storage and performance on shared systems.

**Key Files:**
- `datasets/` - HuggingFace datasets cache
- `huggingface/` - HuggingFace Hub model cache
- `transformers/` - Transformers library cache
- `torch/` - PyTorch model cache
- `wandb/` - Weights & Biases experiment logs
- `tensorboard/` - TensorBoard logging data
- `pip/` - Python package cache
- `nltk_data/` - NLTK language data
- `matplotlib/` - Matplotlib configuration

### dataset/
**Purpose:** Contains all datasets used for training and evaluation, including both processed ProverQA data and custom logical reasoning datasets.

**Key Files:**
- `__init__.py` - Dataset module initialization
- `dev_split.py` - Script to split ProverQA dataset into SFT training (300 instances) and test sets (1200 instances)
- `proverqa_processor.py` - Processes ProverQA training data into simplified question-answer pairs
- `proverqa_simplified.json` - Simplified ProverQA dataset for GRPO training (5000 examples)
- `sft_processor.py` - Cleans SFT dataset by removing prefixes and formatting reasoning steps
- `sft_reasoning_generator.py` - Uses Qwen3-32B to generate detailed initial reasoning for SFT examples
- `dev/easy.json`, `dev/medium.json`, `dev/hard.json` - Test datasets split by difficulty level
- `sft/sft_dataset.json` - Raw SFT dataset (300 examples from ProverQA)
- `sft/sft_cleaned.json` - Cleaned SFT dataset with proper formatting
- `sft/sft_with_reasoning.json` - Enhanced SFT dataset with generated initial reasoning

### setup/
**Purpose:** Model setup, configuration, and cache management utilities for efficient GPU training and inference.

**Key Files:**
- `__init__.py` - Setup module initialization  
- `cache_manager.py` - Comprehensive cache management utility with monitoring and cleanup functions
- `setup_cache.py` - Standalone cache directory setup for ML workflows
- `setup_models.py` - Main model setup script for Qwen3-8B and Llama-3.1-8B with proper authentication and cache configuration

### reward/
**Purpose:** Implements the reward calculation system for GRPO training, combining multiple metrics to evaluate logical reasoning quality.

**Key Files:**
- `__init__.py` - Reward module initialization
- `reward.py` - Comprehensive reward system with weighted scoring for answer correctness (35%), logical validity (55%), and format compliance (10%)

### verification/
**Purpose:** Formal logical verification using Prover9 theorem prover to validate the soundness of generated reasoning.

**Key Files:**
- `__init__.py` - Verification module initialization
- `prover9_integration.py` - Converts mathematical FOL notation to Prover9 syntax and verifies logical reasoning chains

### utils/
**Purpose:** General utilities and complete pipeline implementations for end-to-end logical reasoning workflows.

**Key Files:**
- `__init__.py` - Utils module initialization
- `claude_qwen_fol_pipeline.py` - Complete pipeline implementation: problem generation → Qwen solving → Llama FOL conversion → Prover9 verification

### test/
**Purpose:** Contains backup files, testing scripts, and experimental code for pipeline development and validation.

**Key Files:**
- `README.md` - Test directory documentation
- `backup/` - Backup directory with various experimental and development files

## Important Files Deep Dive

### train.py
**Location:** `train.py`
**Role:** Main GRPO training pipeline that integrates SFT pre-training and GRPO training with full pipeline reward functions. This is the primary entry point for training logical reasoning models. It supports both standalone GRPO training and combined SFT+GRPO workflows. The script includes NL→FOL conversion and Prover9 verification during training, making it a comprehensive solution for logical reasoning model development. Key features include configurable model paths, reward function integration, incremental checkpointing, and W&B logging.

### train_sft.py  
**Location:** `train_sft.py`
**Role:** Standalone supervised fine-tuning script following architectural patterns from the main pipeline. This script provides focused SFT training on logical reasoning datasets with proper chat template formatting. It can be used independently or called by the main training pipeline. The implementation includes configurable training parameters, proper tokenization handling, and model persistence with checkpoint management.

### pipeline_dev.py
**Location:** `pipeline_dev.py`
**Role:** Development pipeline for processing ProverQA dataset problems and evaluating responses using the complete reward system. This script serves as the main evaluation and testing interface, providing interactive problem selection, batch processing capabilities, and comprehensive result analysis. It integrates all pipeline components: Qwen for reasoning, Llama for FOL conversion, Prover9 for verification, and the reward system for scoring.

### pipeline_with_rewards.py
**Location:** `pipeline_with_rewards.py`
**Role:** Alternative pipeline implementation focused on reward evaluation with Claude-generated test problems. This script provides a controlled testing environment with predefined logical reasoning problems and serves as a validation tool for the reward system. It includes manual fallback reasoning and robust parsing mechanisms for handling model output variations.

### setup/setup_models.py
**Location:** `setup/setup_models.py`
**Role:** Central model configuration script that handles authentication, cache setup, and model loading for both Qwen3-8B (primary reasoning model) and Llama-3.1-8B with NL-to-FOL LoRA adapter. This script ensures proper GPU utilization, handles HuggingFace authentication, and provides testing functions for model validation. It's essential for establishing the computing environment before training or inference.

### reward/reward.py
**Location:** `reward/reward.py`
**Role:** Implements the sophisticated reward calculation system used in GRPO training. The system uses a weighted combination of three components: answer correctness (35%), logical validity via Prover9 verification (55%), and format compliance (10%). This file defines the RewardComponents dataclass and LogicalReasoningReward class, which are central to the training process. The reward system includes penalty mechanisms for severe violations and supports preference pair creation for advanced training techniques.

### verification/prover9_integration.py
**Location:** `verification/prover9_integration.py`
**Role:** Handles the conversion from mathematical First-Order Logic (FOL) notation to Prover9 syntax and manages the verification process. This module is crucial for the logical validity assessment in the reward system. It includes robust error handling, timeout management, and output parsing to determine proof success. The integration supports complex logical constructs including XOR operations and provides detailed verification results for training feedback.

### dataset/sft_reasoning_generator.py
**Location:** `dataset/sft_reasoning_generator.py`
**Role:** Sophisticated script that uses Qwen3-32B to generate detailed initial reasoning for all logical reasoning problems in the SFT dataset. This script enhances the training data quality by providing comprehensive reasoning explanations that help models learn proper logical thinking patterns. It includes incremental saving, GPU optimization, authentication management, and comprehensive error handling for large-scale data processing.

### utils/claude_qwen_fol_pipeline.py
**Location:** `utils/claude_qwen_fol_pipeline.py`
**Role:** Complete end-to-end pipeline implementation that demonstrates the full logical reasoning workflow. This script processes predefined logical problems through the entire chain: Qwen generates reasoning, Llama converts to FOL, and Prover9 verifies the logic. It serves as both a testing tool and a reference implementation for understanding how all components work together. The script includes interactive problem selection, robust output parsing, and comprehensive result analysis.
