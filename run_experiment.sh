#!/bin/bash

# Intrinsic Reward Knowledge Distillation Experiment Script
# Compatible with VAST AI GPU environment
# Supports intelligent skipping of completed steps

set -e

echo "=========================================="
echo "Intrinsic Reward Knowledge Distillation Experiment"
echo "=========================================="

# Set environment variables - automatically detect GPU count
echo "Checking GPU status..."
nvidia-smi || echo "⚠️  nvidia-smi not available"
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
echo "Detected $NUM_GPUS GPU(s)"

# Set CUDA_VISIBLE_DEVICES based on actual GPU count
if [ "$NUM_GPUS" -ge 4 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "✅ Using 4×GPU configuration"
elif [ "$NUM_GPUS" -ge 2 ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    echo "✅ Using 2×GPU configuration"
elif [ "$NUM_GPUS" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "⚠️  Using only 1 GPU"
else
    # If no GPU detected, do not set CUDA_VISIBLE_DEVICES, let Python code handle it
    echo "⚠️  No GPU detected, will use system default configuration"
fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 🔥 Fix memory fragmentation issues

# 🔍 CUDA error diagnostics: enable synchronous execution for more detailed error information
# Note: This will slow down CUDA operations but can accurately locate error positions
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}  # Default off, set to 1 when needed
if [ "$CUDA_LAUNCH_BLOCKING" = "1" ]; then
    echo "⚠️  CUDA_LAUNCH_BLOCKING enabled, this will provide more detailed CUDA error information but reduce performance"
fi

# Create necessary directories
mkdir -p logs
mkdir -p cache
mkdir -p checkpoints

# Function to check step completion status
check_sft_completed() {
    if [ -d "./checkpoints/sft_model" ]; then
        # Check LoRA mode (adapter_model.bin) or full model mode (pytorch_model.bin)
        if [ -f "./checkpoints/sft_model/adapter_model.bin" ] || \
           [ -f "./checkpoints/sft_model/adapter_model.safetensors" ] || \
           [ -f "./checkpoints/sft_model/pytorch_model.bin" ]; then
            return 0  # SFT completed
        fi
    fi
    return 1  # SFT not completed
}

check_rl_completed() {
    if [ -d "./checkpoints/rl_model" ]; then
        # Check LoRA mode (adapter_model.safetensors or adapter_model.bin)
        if [ -f "./checkpoints/rl_model/adapter_model.safetensors" ] || \
           [ -f "./checkpoints/rl_model/adapter_model.bin" ]; then
            return 0  # RL completed (has final model)
        fi
    fi
    return 1  # RL not completed or only has checkpoints
}

find_latest_rl_checkpoint() {
    # Find the latest checkpoint directory
    latest_checkpoint=""
    latest_step=0
    
    if [ -d "./checkpoints/rl_model" ]; then
        for checkpoint_dir in ./checkpoints/rl_model/checkpoint-*; do
            if [ -d "$checkpoint_dir" ]; then
                # Check if adapter files exist (safetensors or bin)
                if [ -f "$checkpoint_dir/adapter_model.safetensors" ] || [ -f "$checkpoint_dir/adapter_model.bin" ]; then
                    # Extract step number (from checkpoint-N directory name)
                    step=$(echo "$checkpoint_dir" | grep -oE 'checkpoint-[0-9]+' | grep -oE '[0-9]+')
                    if [ -n "$step" ] && [ "$step" -gt "$latest_step" ] 2>/dev/null; then
                        latest_step=$step
                        latest_checkpoint="$checkpoint_dir"
                    fi
                fi
            fi
        done
    fi
    
    echo "$latest_checkpoint"
}

echo "1. Installing dependencies..."
pip install -r requirements.txt

echo "2. Preparing data..."
python scripts/prepare_data.py --show_samples 5

echo "3. Supervised Fine-Tuning (SFT)..."
if check_sft_completed; then
    echo "✅ SFT training completed, skipping this step"
    echo "SFT model path: ./checkpoints/sft_model"
else
    echo "🔄 Starting SFT training..."
    python scripts/train_sft.py \
        --config config/training_config.yaml \
        --output_dir ./checkpoints/sft_model \
        --log_level INFO
    echo "✅ SFT training completed"
fi

echo "4. Reinforcement Learning Training (RL)..."
if check_rl_completed; then
    echo "✅ RL training completed, skipping this step"
    echo "RL model path: ./checkpoints/rl_model"
else
    # Check if there are checkpoints to resume from
    latest_checkpoint=$(find_latest_rl_checkpoint)
    if [ -n "$latest_checkpoint" ] && [ -d "$latest_checkpoint" ]; then
        echo "🔄 Checkpoint detected, resuming training from checkpoint..."
        echo "Checkpoint path: $latest_checkpoint"
        python scripts/train_rl.py \
            --config config/training_config.yaml \
            --student_model_path ./checkpoints/sft_model \
            --output_dir ./checkpoints/rl_model \
            --max_steps 1000 \
            --resume_from_checkpoint "$latest_checkpoint" \
            --log_level INFO
    else
        echo "🔄 Starting new RL training..."
        python scripts/train_rl.py \
            --config config/training_config.yaml \
            --student_model_path ./checkpoints/sft_model \
            --output_dir ./checkpoints/rl_model \
            --max_steps 1000 \
            --log_level INFO
    fi
    echo "✅ RL training completed"
fi

echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "📁 Model save locations:"
echo "   - SFT model: ./checkpoints/sft_model"
echo "   - RL model: ./checkpoints/rl_model"
echo ""
echo "📊 Subsequent evaluation:"
echo "   Evaluation scripts are located in the evaluation/ directory and can be run separately"
echo "   - Evaluate single checkpoint: python evaluation/evaluate_checkpoint.py --checkpoint_path <path>"
echo "   - Batch evaluation: ./evaluation/evaluate_all_checkpoints.sh"
echo ""
echo "=========================================="
