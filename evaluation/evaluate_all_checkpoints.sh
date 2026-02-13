#!/bin/bash
# Batch evaluation of all RL checkpoints

# Configuration
CHECKPOINT_DIR="checkpoints/rl_model"
OUTPUT_DIR="evaluation_results"
EVAL_SAMPLES=100  # Quick test with 100 samples, set to null or comment out for full evaluation

# Create output directory
mkdir -p $OUTPUT_DIR

# Evaluate all checkpoints
echo "Starting batch checkpoint evaluation..."
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Evaluation samples: $EVAL_SAMPLES"
echo ""

# Find all checkpoint directories
for checkpoint in $CHECKPOINT_DIR/checkpoint-*; do
    if [ -d "$checkpoint" ]; then
        checkpoint_name=$(basename $checkpoint)
        echo "=========================================="
        echo "Evaluating checkpoint: $checkpoint_name"
        echo "=========================================="
        
        output_file="$OUTPUT_DIR/evaluation_results_${checkpoint_name}.json"
        
        # Build command
        if [ -z "$EVAL_SAMPLES" ] || [ "$EVAL_SAMPLES" == "null" ]; then
            # Full evaluation (all samples)
            python evaluation/evaluate_checkpoint.py \
                --checkpoint_path "$checkpoint" \
                --output_file "$output_file"
        else
            # Quick evaluation (specified number of samples)
            python evaluation/evaluate_checkpoint.py \
                --checkpoint_path "$checkpoint" \
                --eval_samples $EVAL_SAMPLES \
                --output_file "$output_file"
        fi
        
        if [ $? -eq 0 ]; then
            echo "✅ $checkpoint_name evaluation completed"
        else
            echo "❌ $checkpoint_name evaluation failed"
        fi
        echo ""
    fi
done

echo "=========================================="
echo "Batch evaluation completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "=========================================="

