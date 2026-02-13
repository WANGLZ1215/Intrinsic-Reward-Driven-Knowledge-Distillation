#!/bin/bash

# Resume training from checkpoint script
# Compatible with VAST AI GPU environment

set -e

echo "=========================================="
echo "Resume Training from Checkpoint"
echo "=========================================="

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check checkpoint parameter
if [ -z "$1" ]; then
    echo "Usage: ./resume_training.sh <checkpoint_dir> [max_steps]"
    echo "Example: ./resume_training.sh checkpoints/rl_model/checkpoint-500 1000"
    exit 1
fi

CHECKPOINT_DIR=$1
MAX_STEPS=${2:-1000}

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint directory does not exist: $CHECKPOINT_DIR"
    exit 1
fi

echo "Checkpoint path: $CHECKPOINT_DIR"
echo "Max steps: $MAX_STEPS"
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

# Resume training from checkpoint
echo ""
echo "🔄 Resuming training from checkpoint..."
python scripts/train_rl.py \
    --config config/training_config.yaml \
    --student_model_path ./checkpoints/sft_model \
    --output_dir ./checkpoints/rl_model \
    --max_steps $MAX_STEPS \
    --resume_from_checkpoint "$CHECKPOINT_DIR" \
    --log_level INFO

echo "✅ Training completed"

