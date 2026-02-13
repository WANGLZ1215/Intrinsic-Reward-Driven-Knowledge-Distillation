# Project Structure Documentation

## Overview

This document provides a comprehensive overview of the project structure for the Intrinsic Reward-based Knowledge Distillation framework. The project implements a novel approach to distilling knowledge from large teacher models to smaller student models using intrinsic rewards derived from the inverse soft Bellman operator, eliminating the need for external reward models.

## Complete Project Structure

```
VAST/
├── README.md                          # Project overview and quick start guide
├── requirements.txt                    # Python dependencies specification
├── run_experiment.sh                   # Automated experiment execution script
├── PROJECT_STRUCTURE.md                # This documentation file
│
├── config/                            # Configuration module
│   ├── __init__.py
│   └── training_config.yaml           # Main training configuration (all module settings)
│
├── data/                              # Data processing module
│   ├── __init__.py
│   └── gsm8k_processor.py             # GSM8K dataset processor and formatter
│
├── models/                            # Model encapsulation module
│   ├── __init__.py
│   ├── teacher_model.py               # Teacher model wrapper (Qwen-32B-Instruct)
│   ├── student_model.py               # Student model wrapper (Qwen-7B-Math) with LoRA support
│   └── cache_manager.py              # Teacher model logits cache manager (LRU/LFU/FIFO)
│
├── rewards/                           # Reward computation module
│   ├── __init__.py
│   ├── intrinsic_reward.py            # Intrinsic reward computation (inverse soft Bellman operator)
│   ├── reward_normalizer.py           # Reward normalization methods (mean-std, min-max, z-score, robust)
│   └── reward_combiner.py             # Reward combination with adaptive weighting
│
├── training/                          # Training module
│   ├── __init__.py
│   ├── sft_trainer.py                 # Supervised fine-tuning trainer
│   ├── rl_trainer.py                  # Reinforcement learning trainer (PPO-based)
│   └── ppo_utils.py                   # PPO utility functions (loss, buffer, scheduler)
│
├── evaluation/                        # Evaluation module
│   ├── __init__.py
│   ├── evaluate_checkpoint.py         # Checkpoint evaluation script
│   ├── reasoning_evaluator.py        # Reasoning quality assessment
│   ├── metrics.py                     # Comprehensive evaluation metrics
│   ├── compare_model_logits.py        # Logits similarity comparison (KL, JS, Cosine)
│   ├── evaluate_all_checkpoints.sh    # Batch evaluation script (Linux)
│   ├── evaluate_all_checkpoints.bat   # Batch evaluation script (Windows)
│   └── SFT_EVALUATION_GUIDE.md        # SFT model evaluation guide
│
├── utils/                             # Utility functions module
│   ├── __init__.py
│   ├── math_utils.py                  # Mathematical utilities (answer extraction, validation)
│   ├── text_utils.py                 # Text processing utilities (cleaning, formatting)
│   └── cache_utils.py                 # Cache utilities for Transformers v4.43+
│
├── scripts/                           # Training and evaluation scripts
│   ├── __init__.py
│   ├── train_sft.py                   # SFT training script
│   ├── train_rl.py                    # RL training script (with checkpoint resumption)
│   ├── prepare_data.py                # Data preparation script
│   ├── check_rl_training.py           # RL training diagnostics script
│   ├── evaluate_sft_model.py          # SFT model evaluation script
│   ├── evaluate_teacher_model.py      # Teacher model evaluation script
│   ├── compare_student_teacher.py     # Student/teacher comparison script
│   ├── compare_sft_rl_results.py      # SFT vs RL results comparison
│   ├── export_gsm8k_answers.py        # Export student model answers (JSONL)
│   ├── export_teacher_gsm8k_answers.py # Export teacher model answers (JSONL)
│   ├── evaluate_sft_vast_ai.sh        # SFT evaluation script for VAST AI
│   ├── re_train.sh                    # Restart training script
│   └── resume_training.sh              # Resume training from checkpoint script
│
├── logs/                              # Log directory (created at runtime)
├── cache/                             # Cache directory (created at runtime)
├── results/                           # Results directory (created at runtime)
└── checkpoints/                       # Model checkpoint directory (created at runtime)
```

## Module Descriptions

### Configuration Module (`config/`)

- **`training_config.yaml`**: Centralized configuration file containing all training parameters, including:
  - Model specifications (teacher, student, LoRA parameters)
  - Training hyperparameters (learning rates, batch sizes, training steps)
  - Reward configuration (weight combinations, normalization methods, temperature parameters)
  - Device configuration (GPU allocation, data types, memory optimization)

### Data Processing Module (`data/`)

- **`gsm8k_processor.py`**: Handles GSM8K mathematical reasoning dataset operations:
  - Dataset loading from HuggingFace
  - Prompt formatting for SFT and RL training
  - Answer extraction and validation
  - Data collation for batch processing

### Model Encapsulation Module (`models/`)

- **`teacher_model.py`**: Wraps the Qwen-32B-Instruct teacher model:
  - Logits extraction for intrinsic reward computation
  - Response generation with caching support
  - Device management and memory optimization
  - Thread-safe tokenization handling

- **`student_model.py`**: Wraps the Qwen-7B-Math student model:
  - LoRA fine-tuning support
  - PPO training integration with value head
  - Safe token generation with boundary checking
  - Token embedding size consistency management

- **`cache_manager.py`**: Manages teacher model logits cache:
  - Multiple eviction policies (LRU, LFU, FIFO)
  - Cache persistence and loading
  - Statistics tracking (hit rate, eviction count)

### Reward Computation Module (`rewards/`)

- **`intrinsic_reward.py`**: Core intrinsic reward computation:
  - Implements inverse soft Bellman operator
  - Soft value function computation: V(s) = α log(∑_a exp(Q(s,a)/α))
  - Token-level intrinsic reward derivation
  - Theoretical foundation based on MaxEnt RL

- **`reward_normalizer.py`**: Reward normalization methods:
  - Mean-std normalization
  - Min-max normalization
  - Z-score normalization
  - Robust normalization with outlier handling
  - Moving average statistics for stability

- **`reward_combiner.py`**: Combines multiple reward signals:
  - Intrinsic reward (from teacher logits)
  - Correctness reward (answer accuracy)
  - Reasoning reward (optional)
  - Format reward (optional)
  - Adaptive weight adjustment based on performance

### Training Module (`training/`)

- **`sft_trainer.py`**: Supervised fine-tuning trainer:
  - Uses HuggingFace Transformers Trainer
  - LoRA configuration and application
  - Automatic checkpoint resumption
  - Evaluation on validation set

- **`rl_trainer.py`**: Reinforcement learning trainer:
  - PPO-based training with intrinsic rewards
  - Policy and value function optimization
  - GAE (Generalized Advantage Estimation) computation
  - Gradient clipping and memory management
  - Comprehensive training diagnostics
  - Checkpoint saving and resumption

- **`ppo_utils.py`**: PPO utility functions:
  - Policy loss computation (clipped surrogate objective)
  - Value loss computation (MSE)
  - Entropy regularization
  - Experience replay buffer
  - Learning rate scheduling

### Evaluation Module (`evaluation/`)

- **`evaluate_checkpoint.py`**: Main checkpoint evaluation script:
  - Loads RL training checkpoints
  - Generates responses on GSM8K test set
  - Computes accuracy and reasoning quality metrics
  - Generates comprehensive evaluation reports

- **`reasoning_evaluator.py`**: Reasoning quality assessment:
  - Reasoning step extraction
  - Step coverage ratio computation
  - Logical consistency scoring
  - Answer correctness evaluation
  - KL divergence between student and teacher logits

- **`metrics.py`**: Comprehensive evaluation metrics:
  - `MathAccuracyMetrics`: Exact match and numerical accuracy
  - `ReasoningQualityMetrics`: Step coverage, logical consistency
  - `DistillationMetrics`: KL divergence, JS divergence, Cosine similarity
  - `TrainingMetrics`: Training stability, convergence rate
  - `ComprehensiveEvaluator`: Aggregates all metrics

- **`compare_model_logits.py`**: Logits similarity analysis:
  - Compares RL, SFT, and teacher model logits
  - Computes KL divergence, JS divergence, Cosine similarity
  - Statistical analysis of distribution differences

### Utility Functions Module (`utils/`)

- **`math_utils.py`**: Mathematical utility functions:
  - Unified answer extraction (`extract_answer_unified`)
  - Mathematical expression validation
  - Arithmetic calculation and simplification
  - Reasoning consistency checks
  - Supports multiple answer formats (####, \\boxed{}, answer:, etc.)

- **`text_utils.py`**: Text processing utilities:
  - Text cleaning and normalization
  - Sentence and paragraph extraction
  - Keyword finding
  - Text chunking for long sequences
  - Mathematical text preprocessing

- **`cache_utils.py`**: Cache utilities for Transformers:
  - Modern cache manager for Transformers v4.43+
  - Handles `past_key_values` deprecation warnings
  - Cache creation and management

### Scripts Module (`scripts/`)

- **`train_sft.py`**: Supervised fine-tuning training script
- **`train_rl.py`**: Reinforcement learning training script with checkpoint support
- **`prepare_data.py`**: Downloads and validates GSM8K dataset
- **`check_rl_training.py`**: Diagnoses RL training progress by analyzing checkpoint statistics
- **`evaluate_sft_model.py`**: Evaluates SFT models on GSM8K
- **`evaluate_teacher_model.py`**: Evaluates teacher model baseline
- **`compare_student_teacher.py`**: Compares student and teacher model performance
- **`compare_sft_rl_results.py`**: Compares SFT and RL training results
- **`export_gsm8k_answers.py`**: Exports student model answers in JSONL format
- **`export_teacher_gsm8k_answers.py`**: Exports teacher model answers in JSONL format
- **`evaluate_sft_vast_ai.sh`**: SFT evaluation script for VAST AI environment
- **`re_train.sh`**: Restart training from scratch
- **`resume_training.sh`**: Resume training from checkpoint

## Workflow

### Training Workflow (SFT + RL)

#### Method 1: Automated Execution (Recommended)
```bash
# Execute complete training pipeline
./run_experiment.sh
```

The script automatically:
- Detects available GPUs and configures environment
- Installs dependencies
- Prepares data (downloads GSM8K dataset)
- Executes SFT training (if not completed)
- Executes RL training (if not completed, automatically resumes from checkpoint)

#### Method 2: Step-by-Step Execution
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data
python scripts/prepare_data.py --show_samples 5

# 3. SFT training
python scripts/train_sft.py --config config/training_config.yaml

# 4. RL training (new training)
python scripts/train_rl.py --config config/training_config.yaml

# Or resume from checkpoint
python scripts/train_rl.py \
    --config config/training_config.yaml \
    --resume_from_checkpoint checkpoints/rl_model/checkpoint-500
```

### Evaluation Workflow

```bash
# Evaluate single checkpoint
python evaluation/evaluate_checkpoint.py \
    --checkpoint_path ./checkpoints/rl_model \
    --eval_samples 500

# Batch evaluate all checkpoints (Linux)
./evaluation/evaluate_all_checkpoints.sh

# Batch evaluate all checkpoints (Windows)
evaluation\evaluate_all_checkpoints.bat

# Export answers for comparison
python scripts/export_gsm8k_answers.py \
    --student_model_path checkpoints/rl_model \
    --out results/student_gsm8k.jsonl

python scripts/export_teacher_gsm8k_answers.py \
    --out results/teacher_gsm8k.jsonl

# Compare student and teacher models
python scripts/compare_student_teacher.py \
    --student_jsonl results/student_gsm8k.jsonl \
    --teacher_jsonl results/teacher_gsm8k.jsonl
```

## Key Features

### Theoretical Foundation

- **Intrinsic Reward Theory**: Based on Maximum Entropy Reinforcement Learning principles
- **Inverse Soft Bellman Operator**: Derives token-level rewards from teacher model logits
- **No External Reward Model**: Directly utilizes teacher model's logit distribution as Q-function
- **Soft Value Function**: V(s) = α log(∑_a exp(Q(s,a)/α)) for reward computation

### Technical Implementation

- **LoRA Fine-tuning**: Reduces computational resource requirements
- **Teacher Logits Caching**: Improves training efficiency with LRU/LFU/FIFO policies
- **Multiple Reward Normalization Methods**: Ensures training stability
- **Complete PPO Training Pipeline**: Supports variable-length sequences
- **Gradient Checkpointing**: Reduces memory usage during training
- **Dynamic Memory Management**: Automatic VRAM cleanup and optimization

### Evaluation Framework

- **Multi-dimensional Reasoning Quality Assessment**: Step coverage, logical consistency
- **Knowledge Distillation Effect Analysis**: KL divergence, JS divergence, Cosine similarity
- **Training Stability Monitoring**: Convergence rate, learning efficiency
- **Comprehensive Metrics**: Accuracy, reasoning quality, distillation metrics, training metrics

## Configuration

Primary configuration parameters are in `config/training_config.yaml`:

- **Model Configuration**: Teacher model, student model, LoRA parameters
- **Training Configuration**: Learning rates, batch sizes, training steps, save/eval intervals
- **Reward Configuration**: Weight combinations, normalization methods, temperature parameters
- **Device Configuration**: GPU allocation, data types, memory optimization settings

## Output Structure

After training completion, the project generates the following files:

### 1. Model Checkpoint Directory (`./checkpoints/`)

#### SFT Model (`./checkpoints/sft_model/`)
- **`adapter_model.safetensors`**: LoRA adapter weights (~50-100MB)
- **`adapter_config.json`**: LoRA configuration parameters (<1MB)
- **`tokenizer.json`, `tokenizer_config.json`**: Tokenizer configuration (~3MB)
- **`config.json`**: Model configuration file (<1MB)

#### RL Model (`./checkpoints/rl_model/`)

**Intermediate Checkpoints** (saved every N steps, configurable):
- **`checkpoint-{step}/`** directories, each containing:
  - `adapter_model.safetensors`: LoRA adapter (~50-100MB)
  - `adapter_config.json`: LoRA configuration (<1MB)
  - Tokenizer files (~3MB)
  - `training_stats.json`: Training statistics (1-10MB)
  - `teacher_cache.pkl`: Teacher cache (if enabled, size varies)

**Adaptive Weight Files** (saved periodically):
- **`adaptive_weights_step_{step}.json`**: Adaptive weight configuration (<1MB)

**Final Model Files**:
- **`adapter_model.safetensors`**: Final LoRA adapter (~50-100MB)
- **`adapter_config.json`**: LoRA configuration (<1MB)
- Tokenizer files (~3MB)
- **`final_training_stats.json`**: Final training statistics (~50MB)
- **`training_config.yaml`**: Complete training configuration (<1MB)
- **`model_info.json`**: Model metadata (<1MB)

### 2. Evaluation Results Directory (`./results/`)

Generated after running evaluation scripts:
- **`evaluation_results.json`**: Complete evaluation report (accuracy, reasoning quality, distillation effects)
- **`evaluation_results.checkpoint.json`**: Evaluation checkpoint (for interrupted evaluation recovery)
- **`{model}_gsm8k.jsonl`**: Exported answers in JSONL format for comparison

### 3. Training Logs Directory (`./logs/`)

- **Training log files**: Detailed training process output (<50MB)
- **TensorBoard logs** (if enabled): Training visualization data (<100MB)

### 4. W&B Logs (if enabled)

- **Weights & Biases tracking data**: Training metrics and visualizations (<50MB)

### Storage Space Estimation

Complete training pipeline requires approximately **2-5 GB**:
- SFT stage: ~110MB
- RL checkpoints: ~2-4GB (N checkpoints × 100-150MB each)
- RL final model: ~160MB
- Evaluation results: ~10-100MB
- Log files: ~150MB

**Notes**:
- Teacher model cache can be disabled to save space (if hit rate is low)
- Old checkpoints can be selectively deleted to reduce storage
- Only the final model is required for inference

## Research Contributions

This project implements a novel approach to knowledge distillation that:

1. **Eliminates External Reward Models**: Uses teacher model logits directly as reward signals
2. **Theoretical Rigor**: Based on established MaxEnt RL and inverse soft Bellman operator theory
3. **Efficient Training**: LoRA fine-tuning and caching mechanisms reduce computational costs
4. **Comprehensive Evaluation**: Multi-dimensional assessment of reasoning quality and distillation effects

## Platform Compatibility

The project is designed to run on:
- **VAST AI**: GPU cloud platform with automatic GPU detection
- **Local GPU Workstations**: Manual GPU configuration
- **Multi-GPU Systems**: Supports 1×, 2×, and 4× GPU configurations

## Dependencies

Key dependencies (see `requirements.txt` for complete list):
- `transformers` (v4.43+): Model loading and training
- `torch`: Deep learning framework
- `peft`: LoRA implementation
- `trl`: PPO training utilities
- `datasets`: Dataset loading
- `sympy`: Mathematical expression processing
- `numpy`: Numerical computations

## Citation

If you use this code in your research, please cite the relevant papers on:
- Maximum Entropy Reinforcement Learning
- Inverse Soft Bellman Operator
- Knowledge Distillation for Large Language Models

---

This project structure is modular, well-organized, and designed for extensibility and maintainability, suitable for execution on GPU cloud platforms such as VAST AI.
