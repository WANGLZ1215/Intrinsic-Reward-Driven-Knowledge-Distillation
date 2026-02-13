#!/usr/bin/env python3
"""
SFT Model GSM8K Full Test Script
Function: Evaluate SFT model on GSM8K test set, generate comprehensive evaluation report for comparison with RL training results

Designed for VAST AI online evaluation environment:
- Supports relative and absolute paths
- Uses project's unified answer extraction function extract_answer_unified
- Enhanced error handling and logging
- Saves detailed results for subsequent comparison and analysis
"""

import argparse
import yaml
import logging
import os
import json
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm
import torch

# Add project root to Python path
# Supports running from scripts directory or project root
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from models.student_model import StudentModel
from evaluation.reasoning_evaluator import ReasoningEvaluator
from evaluation.metrics import ComprehensiveEvaluator
from data.gsm8k_processor import build_prompt
from datasets import load_dataset
from utils.math_utils import extract_answer_unified  # Use project's unified answer extraction function


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging"""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # If no log file is specified, use default name
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"evaluate_sft_model_{timestamp}.log"
    
    # Configure log format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configure logging to output to console and file simultaneously
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    return logger


def load_config(config_path: str) -> dict:
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_sft_model(
    sft_model_path: str,
    config: dict,
    eval_samples: int = None,
    output_file: str = "sft_evaluation_results.json",
    output_jsonl: str = "sft_evaluation_results.jsonl",
    **kwargs
):
    """
    Evaluate SFT model
    
    Args:
        sft_model_path: SFT model path (e.g., checkpoints/sft_model or checkpoints/sft_model/checkpoint-5607)
        config: Configuration dictionary
        eval_samples: Number of samples to evaluate (None for all)
        output_file: Output JSON file path (contains summarized statistics)
        output_jsonl: Output JSONL file path (contains detailed results for each sample)
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Starting SFT Model Evaluation")
    logger.info(f"SFT Model Path: {sft_model_path}")
    logger.info("=" * 80)
    
    # Initialize results dictionary
    results = {
        "model_type": "SFT",
        "model_path": sft_model_path,
        "evaluation_time": datetime.now().isoformat(),
        "accuracy": 0.0,
        "statistics": {
            "total_samples": 0,
            "correct_samples": 0,
            "incorrect_samples": 0,
            "average_response_length": 0.0,
            "average_answer_extraction_success": 0.0
        },
        "answer_extraction_stats": {
            "successful_extractions": 0,
            "failed_extractions": 0,
            "success_rate": 0.0
        },
        "individual_results": []
    }
    
    try:
        # Step 1: Load SFT model
        logger.info("Step 1/4: Loading SFT model...")
        
        # Ensure model path is absolute (VAST AI compatibility)
        sft_model_path_abs = Path(sft_model_path).resolve()
        if not sft_model_path_abs.exists():
            # Try relative to project root
            project_root = Path(__file__).parent.parent
            sft_model_path_abs = (project_root / sft_model_path).resolve()
        
        if not sft_model_path_abs.exists():
            raise FileNotFoundError(f"SFT model path does not exist: {sft_model_path} (tried: {sft_model_path_abs})")
        
        logger.info(f"Using SFT model path: {sft_model_path_abs}")
        
        # Validate model files
        required_files = [
            sft_model_path_abs / "adapter_config.json"
        ]
        weight_files = [
            sft_model_path_abs / "adapter_model.safetensors",
            sft_model_path_abs / "adapter_model.bin"
        ]
        
        if not required_files[0].exists():
            raise FileNotFoundError(f"SFT model missing required file: {required_files[0]}")
        
        if not any(f.exists() for f in weight_files):
            raise FileNotFoundError(f"SFT model missing weight files: {[str(f) for f in weight_files]}")
        
        logger.info(f"SFT model validation passed: {[f.name for f in required_files + weight_files if f.exists()]}")
        
        # Load SFT model (using StudentModel, as SFT model is also a LoRA adapter)
        sft_model = StudentModel(
            model_name=str(sft_model_path_abs),
            lora_config=config["lora"],
            device=config["device"]["device_map"],
            torch_dtype=getattr(torch, config["device"]["torch_dtype"]),
            use_lora=True
        )
        logger.info("✅ SFT model loaded successfully")
        
        # Step 2: Load evaluation data
        logger.info("Step 2/4: Loading evaluation data...")
        try:
            dataset = load_dataset("gsm8k", "main")
            logger.info(f"Dataset loaded successfully: train={len(dataset['train'])}, test={len(dataset['test'])}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        if eval_samples is not None:
            eval_samples = min(eval_samples, len(dataset["test"]))
            eval_dataset = dataset["test"].select(range(eval_samples))
            logger.info(f"Using {eval_samples} test samples (out of {len(dataset['test'])} total)")
        else:
            eval_dataset = dataset["test"]
            logger.info(f"Using all {len(eval_dataset)} test samples")
        
        # Step 3: Initialize evaluators (optional, for reasoning quality evaluation)
        logger.info("Step 3/4: Initializing evaluators...")
        reasoning_evaluator = ReasoningEvaluator()
        logger.info("✅ Evaluators initialized successfully")
        
        # Step 4: Execute evaluation
        logger.info("Step 4/4: Starting evaluation...")
        logger.info("=" * 80)
        
        total_samples = len(eval_dataset)
        correct_count = 0
        response_lengths = []
        extraction_successes = []
        
        # Create JSONL file for detailed results
        jsonl_path = Path(output_jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(jsonl_path, 'w', encoding='utf-8') as jsonl_f:
            # Use tqdm for progress display
            for idx, sample in enumerate(tqdm(eval_dataset, desc="Evaluation Progress", ncols=100)):
                question = sample["question"]
                ground_truth = sample["answer"]
                
                try:
                    # Format question prompt
                    formatted_question = build_prompt(question)
                    
                    # SFT model generation
                    try:
                        sft_response = sft_model.generate(
                            formatted_question,
                            max_length=512,
                            temperature=0.0,
                            do_sample=False
                        )
                        # Ensure it's a string type
                        if not isinstance(sft_response, str):
                            sft_response = str(sft_response) if sft_response else ""
                    except Exception as e:
                        logger.warning(f"Sample {idx+1} SFT model generation failed: {e}")
                        sft_response = ""
                    
                    # Record response length
                    response_lengths.append(len(sft_response))
                    
                    # Extract answers (using project's unified answer extraction function extract_answer_unified)
                    try:
                        ground_truth_text, ground_truth_num = extract_answer_unified(ground_truth)
                    except Exception as e:
                        logger.warning(f"Sample {idx+1} failed to extract ground truth answer: {e}")
                        ground_truth_text, ground_truth_num = "", None
                    
                    try:
                        sft_answer_text, sft_answer_num = extract_answer_unified(sft_response)
                        extraction_success = (sft_answer_num is not None)
                        extraction_successes.append(extraction_success)
                    except Exception as e:
                        logger.warning(f"Sample {idx+1} failed to extract SFT answer: {e}")
                        sft_answer_text, sft_answer_num = "", None
                        extraction_successes.append(False)
                    
                    # Evaluate answer correctness (using numerical comparison, more accurate)
                    is_correct = False
                    if ground_truth_num is not None and sft_answer_num is not None:
                        # Use numerical comparison (tolerating small errors, consistent with utils/math_utils.py)
                        tolerance = 1e-6
                        if abs(ground_truth_num) < 1e-10:
                            # If true value is close to 0, use absolute error
                            is_correct = abs(sft_answer_num - ground_truth_num) < tolerance
                        else:
                            # Use relative error
                            relative_error = abs(sft_answer_num - ground_truth_num) / abs(ground_truth_num)
                            is_correct = relative_error < tolerance
                    elif ground_truth_text and sft_answer_text:
                        # If numbers cannot be extracted, use text comparison
                        is_correct = ground_truth_text.strip().lower() == sft_answer_text.strip().lower()
                    else:
                        # If neither can be extracted, mark as incorrect
                        is_correct = False
                        if idx < 5:  # Log details only for the first few samples
                            logger.debug(f"Sample {idx+1} unable to extract answer: ground_truth_text={ground_truth_text}, sft_answer_text={sft_answer_text}")
                    
                    if is_correct:
                        correct_count += 1
                    
                    # Evaluate reasoning quality (based on response itself, not dependent on teacher model)
                    try:
                        if sft_response:
                            # Extract reasoning steps
                            reasoning_steps = reasoning_evaluator.extract_reasoning_steps(sft_response)
                            
                            # Evaluate logical consistency
                            logical_consistency = reasoning_evaluator.evaluate_logical_consistency(sft_response)
                            
                            # Evaluate answer correctness (if ground_truth_num is available)
                            answer_correctness = None
                            if ground_truth_num is not None:
                                answer_correctness = reasoning_evaluator.evaluate_answer_correctness(
                                    sft_response, ground_truth_num
                                )
                        else:
                            reasoning_steps = []
                            logical_consistency = {"overall_consistency": 0.0}
                            answer_correctness = None
                    except Exception as e:
                        logger.warning(f"Sample {idx+1} reasoning quality evaluation failed: {e}")
                        reasoning_steps = []
                        logical_consistency = {"overall_consistency": 0.0}
                        answer_correctness = None
                    
                    # Save individual results (save full information for later comparison)
                    individual_result = {
                        "index": idx,
                        "question": question,
                        "ground_truth": ground_truth,
                        "ground_truth_text": ground_truth_text if ground_truth_text else "N/A",
                        "ground_truth_num": ground_truth_num if ground_truth_num is not None else "N/A",
                        "sft_response": sft_response,
                        "sft_answer_text": sft_answer_text if sft_answer_text else "N/A",
                        "sft_answer_num": sft_answer_num if sft_answer_num is not None else "N/A",
                        "is_correct": is_correct,
                        "response_length": len(sft_response),
                        "reasoning_steps_count": len(reasoning_steps),
                        "logical_consistency_score": logical_consistency.get("overall_consistency", 0.0),
                        "answer_extraction_success": extraction_successes[-1] if extraction_successes else False
                    }
                    
                    # Add answer correctness score (if available)
                    if answer_correctness is not None:
                        individual_result["answer_correctness_score"] = answer_correctness.get("correctness_score", 0.0)
                        individual_result["answer_relative_error"] = answer_correctness.get("relative_error", float('inf'))
                    
                    results["individual_results"].append(individual_result)
                    
                    # Also write to JSONL file (for line-by-line processing)
                    jsonl_f.write(json.dumps(individual_result, ensure_ascii=False) + "\n")
                    jsonl_f.flush()  # Flush in real-time to prevent data loss
                    
                except Exception as e:
                    logger.error(f"Error evaluating sample {idx+1}: {e}")
                    # Log error but continue evaluation
                    error_result = {
                        "index": idx,
                        "error": str(e),
                        "question": question[:200] if len(question) > 200 else question,
                        "is_correct": False
                    }
                    results["individual_results"].append(error_result)
                    jsonl_f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                    jsonl_f.flush()
        
        # Calculate overall metrics
        logger.info("=" * 80)
        logger.info("Calculating overall metrics...")
        
        results["accuracy"] = correct_count / total_samples if total_samples > 0 else 0.0
        
        # Overall statistics
        results["statistics"]["total_samples"] = total_samples
        results["statistics"]["correct_samples"] = correct_count
        results["statistics"]["incorrect_samples"] = total_samples - correct_count
        results["statistics"]["average_response_length"] = sum(response_lengths) / len(response_lengths) if response_lengths else 0.0
        results["statistics"]["average_answer_extraction_success"] = sum(extraction_successes) / len(extraction_successes) if extraction_successes else 0.0
        
        # Answer extraction statistics
        successful_extractions = sum(extraction_successes)
        failed_extractions = len(extraction_successes) - successful_extractions
        results["answer_extraction_stats"]["successful_extractions"] = successful_extractions
        results["answer_extraction_stats"]["failed_extractions"] = failed_extractions
        results["answer_extraction_stats"]["success_rate"] = successful_extractions / len(extraction_successes) if extraction_successes else 0.0
        
        # Calculate average reasoning quality metrics
        if results["individual_results"]:
            consistency_scores = [
                r.get("logical_consistency_score", 0.0) 
                for r in results["individual_results"] 
                if "logical_consistency_score" in r
            ]
            correctness_scores = [
                r.get("answer_correctness_score", 0.0) 
                for r in results["individual_results"] 
                if "answer_correctness_score" in r
            ]
            reasoning_steps_counts = [
                r.get("reasoning_steps_count", 0) 
                for r in results["individual_results"] 
                if "reasoning_steps_count" in r
            ]
            
            if consistency_scores:
                results["statistics"]["average_logical_consistency"] = sum(consistency_scores) / len(consistency_scores)
            if correctness_scores:
                results["statistics"]["average_answer_correctness_score"] = sum(correctness_scores) / len(correctness_scores)
            if reasoning_steps_counts:
                results["statistics"]["average_reasoning_steps"] = sum(reasoning_steps_counts) / len(reasoning_steps_counts)
        
        # Print results summary
        logger.info("=" * 80)
        logger.info("Evaluation Results Summary")
        logger.info("=" * 80)
        logger.info(f"Accuracy: {results['accuracy']:.4f} ({correct_count}/{total_samples})")
        logger.info(f"Answer Extraction Success Rate: {results['answer_extraction_stats']['success_rate']:.4f} ({successful_extractions}/{len(extraction_successes)})")
        logger.info(f"Average Response Length: {results['statistics']['average_response_length']:.2f} characters")
        if "average_logical_consistency" in results["statistics"]:
            logger.info(f"Average Logical Consistency: {results['statistics']['average_logical_consistency']:.4f}")
        if "average_answer_correctness_score" in results["statistics"]:
            logger.info(f"Average Answer Correctness Score: {results['statistics']['average_answer_correctness_score']:.4f}")
        if "average_reasoning_steps" in results["statistics"]:
            logger.info(f"Average Reasoning Steps: {results['statistics']['average_reasoning_steps']:.2f}")
        logger.info("=" * 80)
        
        # Save results JSON file
        output_path = Path(output_file)
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving evaluation results to: {output_path.resolve()}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Verify file saved
        if output_path.exists():
            file_size = output_path.stat().st_size / 1024 / 1024  # MB
            logger.info(f"✅ Evaluation complete! Result file: {output_path.resolve()} ({file_size:.2f} MB)")
        else:
            logger.error(f"❌ Result file failed to save: {output_path}")
        
        if jsonl_path.exists():
            jsonl_file_size = jsonl_path.stat().st_size / 1024 / 1024  # MB
            logger.info(f"✅ JSONL file: {jsonl_path.resolve()} ({jsonl_file_size:.2f} MB)")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        logger.error(f"Detailed error information:\n{traceback.format_exc()}")
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate SFT Model")
    parser.add_argument("--sft_model_path", type=str, required=True,
                       help="SFT model path (e.g., checkpoints/sft_model or checkpoints/sft_model/checkpoint-5607)")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--eval_samples", type=int, default=None,
                       help="Number of samples to evaluate (None for all test set)")
    parser.add_argument("--output_file", type=str, default="results/sft_evaluation_results.json",
                       help="Output JSON file path (contains summarized statistics)")
    parser.add_argument("--output_jsonl", type=str, default="results/sft_evaluation_results.jsonl",
                       help="Output JSONL file path (contains detailed results for each sample)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Log level")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    # Check model path (supports relative and absolute paths, VAST AI compatible)
    sft_model_path_input = Path(args.sft_model_path)
    project_root = Path(__file__).parent.parent
    current_dir = Path.cwd()
    
    # Try multiple possible paths
    possible_paths = [
        sft_model_path_input,  # Original path
        sft_model_path_input.resolve(),  # Absolute path resolution
        project_root / args.sft_model_path,  # Relative to project root
        current_dir / args.sft_model_path,  # Relative to current working directory
    ]
    
    sft_model_path = None
    for path in possible_paths:
        if path.exists() and path.is_dir():
            sft_model_path = path.resolve()
            logger.info(f"Found SFT model path: {sft_model_path}")
            break
    
    if sft_model_path is None:
        logger.error(f"❌ SFT model path does not exist: {args.sft_model_path}")
        logger.error(f"Attempted paths:")
        for path in possible_paths:
            logger.error(f"  - {path} (exists: {path.exists()}, is_dir: {path.is_dir() if path.exists() else 'N/A'})")
        logger.error(f"Current working directory: {current_dir}")
        logger.error(f"Project root: {project_root}")
        logger.error(f"Script location: {Path(__file__).parent}")
        sys.exit(1)
    
    # Check if adapter files exist in model directory
    adapter_config = sft_model_path / "adapter_config.json"
    adapter_weights = [
        sft_model_path / "adapter_model.safetensors",
        sft_model_path / "adapter_model.bin"
    ]
    
    if not adapter_config.exists():
        logger.error(f"❌ SFT model missing configuration file: {adapter_config}")
        sys.exit(1)
    
    if not any(f.exists() for f in adapter_weights):
        logger.error(f"❌ SFT model missing weight files")
        logger.error(f"   Search paths: {[str(f) for f in adapter_weights]}")
        logger.error(f"   Model directory contents: {list(sft_model_path.iterdir())[:10]}")
        sys.exit(1)
    
    logger.info(f"✅ SFT model validation passed: {sft_model_path}")
    logger.info(f"   Configuration file: ✓ {adapter_config.name}")
    logger.info(f"   Weight files: ✓ {[f.name for f in adapter_weights if f.exists()]}")
    
    try:
        # Load configuration (supports relative and absolute paths)
        config_path = Path(args.config)
        if not config_path.exists():
            project_root = Path(__file__).parent.parent
            config_path = project_root / args.config
        if not config_path.exists():
            logger.error(f"Configuration file does not exist: {args.config}")
            sys.exit(1)
        
        config = load_config(str(config_path))
        logger.info(f"✅ Configuration file loaded successfully: {config_path}")
        
        # Execute evaluation
        # Use absolute path to ensure VAST AI compatibility
        results = evaluate_sft_model(
            sft_model_path=str(sft_model_path.resolve()),
            config=config,
            eval_samples=args.eval_samples,
            output_file=args.output_file,
            output_jsonl=args.output_jsonl
        )
        
        logger.info("=" * 80)
        logger.info("Evaluation task completed!")
        logger.info(f"Result JSON file: {args.output_file}")
        logger.info(f"Result JSONL file: {args.output_jsonl}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()



