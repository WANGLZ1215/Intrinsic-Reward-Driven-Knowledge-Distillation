#!/bin/bash
# SFT Model Evaluation Script (VAST AI Version)
# Used to run full GSM8K test of SFT model on VAST AI

set -e  # Exit immediately on error

# Configuration parameters
PROJECT_ROOT="/workspace/Thesis"
SFT_MODEL_PATH="checkpoints/sft_model"
CONFIG_PATH="config/training_config.yaml"
OUTPUT_DIR="results"
LOG_DIR="logs"

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# Navigate to project directory
cd ${PROJECT_ROOT}

# Run evaluation
echo "=========================================="
echo "Starting SFT Model Evaluation"
echo "=========================================="
echo "Project root: ${PROJECT_ROOT}"
echo "SFT model path: ${SFT_MODEL_PATH}"
echo "Config file: ${CONFIG_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "=========================================="

# Run evaluation script
python scripts/evaluate_sft_model.py \
    --sft_model_path ${SFT_MODEL_PATH} \
    --config ${CONFIG_PATH} \
    --eval_samples None \
    --output_file ${OUTPUT_DIR}/sft_evaluation_results.json \
    --output_jsonl ${OUTPUT_DIR}/sft_evaluation_results.jsonl \
    --log_level INFO

# Check results
if [ -f "${OUTPUT_DIR}/sft_evaluation_results.json" ]; then
    echo "=========================================="
    echo "Evaluation Completed!"
    echo "=========================================="
    echo "Result files:"
    echo "  - ${OUTPUT_DIR}/sft_evaluation_results.json"
    echo "  - ${OUTPUT_DIR}/sft_evaluation_results.jsonl"
    echo "=========================================="
    
    # Display results summary
    echo "Results Summary:"
    python -c "
import json
with open('${OUTPUT_DIR}/sft_evaluation_results.json', 'r') as f:
    data = json.load(f)
    print(f'Accuracy: {data[\"accuracy\"]:.4f}')
    print(f'Total samples: {data[\"statistics\"][\"total_samples\"]}')
    print(f'Correct samples: {data[\"statistics\"][\"correct_samples\"]}')
    print(f'Incorrect samples: {data[\"statistics\"][\"incorrect_samples\"]}')
    if 'average_logical_consistency' in data['statistics']:
        print(f'Average logical consistency: {data[\"statistics\"][\"average_logical_consistency\"]:.4f}')
    if 'average_answer_correctness_score' in data['statistics']:
        print(f'Average answer correctness score: {data[\"statistics\"][\"average_answer_correctness_score\"]:.4f}')
"
else
    echo "Error: Evaluation result file not generated!"
    exit 1
fi

echo "=========================================="
echo "Evaluation script execution completed"
echo "=========================================="










