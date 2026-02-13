#!/bin/bash

# Restart training script
# Compatible with VAST AI GPU environment

set -e

echo "=========================================="
echo "Restart Training (Completely from Scratch)"
echo "=========================================="

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get parameters
MAX_STEPS=${1:-1000}
RESET_CHECKPOINTS=${2:-false}

echo "Max steps: $MAX_STEPS"
echo "Reset checkpoints: $RESET_CHECKPOINTS"
echo ""

# Set environment variables
echo "Checking GPU status..."
nvidia-smi || echo "⚠️  nvidia-smi not available"
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
echo "Detected $NUM_GPUS GPU(s)"

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
    echo "⚠️  No GPU detected, will use system default configuration"
fi

# If reset checkpoints is requested
if [ "$RESET_CHECKPOINTS" = "true" ]; then
    echo ""
    read -p "⚠️  This will delete all existing checkpoints, continue? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Deleting existing checkpoints..."
        rm -rf ./checkpoints/rl_model/checkpoint-*
        echo "✅ Checkpoints deleted"
    else
        echo "❌ Operation cancelled"
        exit 0
    fi
fi

# Start new training
echo ""
echo "🔄 Starting new training..."
python scripts/train_rl.py \
    --config config/training_config.yaml \
    --student_model_path ./checkpoints/sft_model \
    --output_dir ./checkpoints/rl_model \
    --max_steps $MAX_STEPS \
    --log_level INFO

echo "✅ Training completed"

