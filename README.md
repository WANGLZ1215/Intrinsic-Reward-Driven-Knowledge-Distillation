# Intrinsic Reward Distillation for Mathematical Reasoning

A novel knowledge distillation framework for mathematical reasoning using intrinsic reward learning (IRL) theory.

## Project Overview

This project implements an innovative teacher-student knowledge distillation framework that enhances mathematical reasoning capabilities through intrinsic reward learning. The framework uses Qwen-32B-Instruct as the teacher model and Qwen-7B-Math as the student model, applying Maximum Entropy Reinforcement Learning principles to distill knowledge without requiring external reward models.

## Core Theoretical Foundation

- **Teacher Model Logits as Q-Function**: The teacher model's logit distribution serves as a Q-function in the reinforcement learning framework, encoding the expected future value of actions
- **Intrinsic Reward Computation**: Recovers reward functions from logits through the inverse soft Bellman operator, enabling reward signal extraction without external reward models
- **Reward Combination**: Combines intrinsic rewards (from teacher logits) with correctness rewards (from answer accuracy) to guide student model learning
- **Soft Value Function**: V(s) = α log(∑_a exp(Q(s,a)/α)) for token-level reward derivation, where α is the temperature parameter controlling the softness of the policy

## Project Structure

```
VAST/
├── config/                    # Configuration module
├── data/                      # Data processing module
├── models/                    # Model encapsulation module
├── rewards/                   # Reward computation module
├── training/                  # Training module
├── evaluation/                # Evaluation module
├── utils/                     # Utility functions module
├── scripts/                   # Training and evaluation scripts
└── results/                   # Results output directory
```

For detailed structure information, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## Quick Start

### Method 1: Automated Execution (Recommended)

```bash
# Execute complete training pipeline (SFT + RL)
./run_experiment.sh
```

The script automatically:
1. Installs dependencies
2. Prepares data (downloads GSM8K dataset)
3. Executes SFT training (if not completed)
4. Executes RL training (if not completed, automatically resumes from checkpoint)

### Method 2: Step-by-Step Execution

#### 1. Environment Setup

```bash
pip install -r requirements.txt
```

#### 2. Data Preparation

```bash
python scripts/prepare_data.py
```

#### 3. Supervised Fine-Tuning (SFT)

```bash
python scripts/train_sft.py --config config/training_config.yaml
```

#### 4. Reinforcement Learning Training (RL)

```bash
# New training
python scripts/train_rl.py --config config/training_config.yaml

# Resume from checkpoint
python scripts/train_rl.py \
    --config config/training_config.yaml \
    --resume_from_checkpoint checkpoints/rl_model/checkpoint-500
```

#### 5. Evaluation (Standalone)

```bash
# Evaluate single checkpoint
python evaluation/evaluate_checkpoint.py \
    --checkpoint_path ./checkpoints/rl_model \
    --eval_samples 500

# Batch evaluate all checkpoints (Linux)
./evaluation/evaluate_all_checkpoints.sh

# Batch evaluate all checkpoints (Windows)
evaluation\evaluate_all_checkpoints.bat
```

## Configuration

Main configuration file:
- **`config/training_config.yaml`**: Centralized training configuration containing all module settings:
  - Model specifications (teacher, student, LoRA parameters)
  - Training hyperparameters (learning rates, batch sizes, training steps)
  - Reward configuration (weight combinations, normalization methods, temperature parameters)
  - Device configuration (GPU allocation, data types, memory optimization)

## Training Output

After training completion, models are saved in:
- **SFT Model**: `./checkpoints/sft_model/`
- **RL Model**: `./checkpoints/rl_model/`
- **RL Checkpoints**: `./checkpoints/rl_model/checkpoint-{step}/`

Each checkpoint contains:
- LoRA adapter weights (`adapter_model.safetensors`)
- Training statistics (`training_stats.json`)
- Model configuration files

## Evaluation

Evaluation scripts are located in the `evaluation/` directory and can be run independently:

- **Single Checkpoint Evaluation**: `evaluation/evaluate_checkpoint.py`
- **Batch Evaluation**: `evaluation/evaluate_all_checkpoints.sh` (Linux) or `evaluation/evaluate_all_checkpoints.bat` (Windows)
- **Answer Export**: `scripts/export_gsm8k_answers.py` (student model) and `scripts/export_teacher_gsm8k_answers.py` (teacher model)
- **Comparison Analysis**: `scripts/compare_student_teacher.py`

## Key Technical Features

### Theoretical Innovation

- **First Application of IRL Theory**: Applies Inverse Reinforcement Learning theory to mathematical reasoning knowledge distillation
- **No External Reward Model**: Directly utilizes teacher model logits as reward signals
- **Theoretical Rigor**: Based on established MaxEnt RL and inverse soft Bellman operator principles

### Computational Optimization

- **Teacher Logits Caching**: Implements LRU/LFU/FIFO cache policies to improve training efficiency
- **LoRA Fine-tuning**: Reduces computational resource requirements significantly
- **Gradient Checkpointing**: Optimizes memory usage during PPO training
- **Dynamic Memory Management**: Automatic VRAM cleanup and optimization

### Training Stability

- **PPO with KL Divergence Penalty**: Ensures stable policy updates
- **Multiple Reward Normalization Methods**: Mean-std, min-max, z-score, and robust normalization
- **Adaptive Reward Weighting**: Dynamically adjusts reward component weights based on performance
- **Comprehensive Diagnostics**: Detailed logging of training metrics and parameter changes

### Comprehensive Evaluation

- **Multi-dimensional Reasoning Quality Assessment**: Step coverage ratio, logical consistency scoring
- **Knowledge Distillation Effect Analysis**: KL divergence, JS divergence, Cosine similarity between student and teacher
- **Training Stability Monitoring**: Convergence rate, learning efficiency metrics
- **Answer Correctness Evaluation**: Numerical and textual answer comparison

## Dependencies

Key dependencies (see `requirements.txt` for complete list):
- `transformers` (v4.43+): Model loading and training
- `torch`: Deep learning framework
- `peft`: LoRA implementation
- `trl`: PPO training utilities
- `datasets`: Dataset loading (GSM8K)
- `sympy`: Mathematical expression processing
- `numpy`: Numerical computations

## Platform Compatibility

The project is designed to run on:
- **VAST AI**: GPU cloud platform with automatic GPU detection
- **Local GPU Workstations**: Manual GPU configuration
- **Multi-GPU Systems**: Supports 1×, 2×, and 4× GPU configurations

## Research Contributions

This project contributes to the field of knowledge distillation by:

1. **Eliminating External Reward Models**: Uses teacher model logits directly as reward signals, reducing training complexity
2. **Theoretical Foundation**: Provides a rigorous theoretical framework based on MaxEnt RL and inverse soft Bellman operator
3. **Efficient Training**: Demonstrates practical training efficiency through LoRA and caching mechanisms
4. **Comprehensive Evaluation**: Establishes a multi-dimensional evaluation framework for reasoning quality and distillation effects

## Citation

If you use this code in your research, please cite the relevant papers on:
- Maximum Entropy Reinforcement Learning
- Inverse Soft Bellman Operator
- Knowledge Distillation for Large Language Models

## Documentation

- **Project Structure**: See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed module descriptions
- **SFT Evaluation Guide**: See [evaluation/SFT_EVALUATION_GUIDE.md](evaluation/SFT_EVALUATION_GUIDE.md) for SFT model evaluation instructions

## License

[Specify your license here]

## Contact

[Specify contact information here]
