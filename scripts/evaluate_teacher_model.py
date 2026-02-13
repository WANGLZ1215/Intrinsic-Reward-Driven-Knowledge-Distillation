#!/usr/bin/env python3
"""
Teacher Model GSM8K Full Test Script
Function: Evaluate teacher model on GSM8K test set, generate complete evaluation report for comparison with SFT/RL training results

Designed for VAST AI online evaluation environment:
- Supports relative and absolute paths
- Uses project's unified answer extraction function extract_answer_unified
- Enhanced error handling and logging
- Saves detailed results for subsequent comparison analysis
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

from models.teacher_model import TeacherModel
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
    
    # If no log file specified, use default name
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"evaluate_teacher_model_{timestamp}.log"
    
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


def evaluate_teacher_model(
    teacher_model_name: str,
    config: dict,
    eval_samples: int = None,
    output_file: str = "results/teacher_evaluation_results.json",
    output_jsonl: str = "results/teacher_evaluation_results.jsonl",
    **kwargs
):
    """
    Evaluate teacher model
    
    Args:
        teacher_model_name: Teacher model name or path (e.g., "Qwen/Qwen2.5-32B-Instruct")
        config: Configuration dictionary
        eval_samples: Number of evaluation samples (None means all)
        output_file: Output result JSON file path (contains summary statistics)
        output_jsonl: Output result JSONL file path (contains detailed results for each sample)
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Starting teacher model evaluation")
    logger.info(f"Teacher model: {teacher_model_name}")
    logger.info("=" * 80)
    
    # Initialize results dictionary
    results = {
        "model_type": "Teacher",
        "model_name": teacher_model_name,
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
        # Step 1: Load teacher model
        logger.info("Step 1/4: Loading teacher model...")
        
        # Load teacher model
        teacher_model = TeacherModel(
            model_name=teacher_model_name,
            cache_size=config["model"].get("cache_size", 500),
            cache_policy=config["model"].get("cache_policy", "LRU"),
            device=config["device"]["device_map"],
            torch_dtype=getattr(torch, config["device"]["torch_dtype"]),
            max_memory=config.get("device", {}).get("max_memory", None)
        )
        logger.info("✅ Teacher model loaded successfully")
        
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
        
        # Step 3: Create evaluators (optional, for reasoning quality evaluation)
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
        
        # Create JSONL file for saving detailed results
        jsonl_path = Path(output_jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(jsonl_path, 'w', encoding='utf-8') as jsonl_f:
            # Use tqdm to display progress
            for idx, sample in enumerate(tqdm(eval_dataset, desc="Evaluation progress", ncols=100)):
                question = sample["question"]
                ground_truth = sample["answer"]
                
                try:
                    # Format question prompt
                    formatted_question = build_prompt(question)
                    
                    # Teacher model generation
                    try:
                        teacher_response = teacher_model.generate_response(
                            formatted_question,
                            max_length=512,
                            temperature=0.0,
                            do_sample=False
                        )
                        # Ensure it's a string type
                        if not isinstance(teacher_response, str):
                            teacher_response = str(teacher_response) if teacher_response else ""
                    except Exception as e:
                        logger.warning(f"Sample {idx+1} teacher model generation failed: {e}")
                        teacher_response = ""
                    
                    # Record response length
                    response_lengths.append(len(teacher_response))
                    
                    # Extract answers (using project's unified answer extraction function extract_answer_unified)
                    try:
                        ground_truth_text, ground_truth_num = extract_answer_unified(ground_truth)
                    except Exception as e:
                        logger.warning(f"Sample {idx+1} failed to extract ground_truth answer: {e}")
                        ground_truth_text, ground_truth_num = "", None
                    
                    try:
                        teacher_answer_text, teacher_answer_num = extract_answer_unified(teacher_response)
                        extraction_success = (teacher_answer_num is not None)
                        extraction_successes.append(extraction_success)
                    except Exception as e:
                        logger.warning(f"Sample {idx+1} failed to extract teacher answer: {e}")
                        teacher_answer_text, teacher_answer_num = "", None
                        extraction_successes.append(False)
                    
                    # Evaluate answer correctness (using numerical comparison, more accurate)
                    is_correct = False
                    if ground_truth_num is not None and teacher_answer_num is not None:
                        # Use numerical comparison (tolerating small errors, consistent with utils/math_utils.py)
                        tolerance = 1e-6
                        if abs(ground_truth_num) < 1e-10:
                            # If true value is close to 0, use absolute error
                            is_correct = abs(teacher_answer_num - ground_truth_num) < tolerance
                        else:
                            # Use relative error
                            relative_error = abs(teacher_answer_num - ground_truth_num) / abs(ground_truth_num)
                            is_correct = relative_error < tolerance
                    elif ground_truth_text and teacher_answer_text:
                        # If numbers cannot be extracted, use text comparison
                        is_correct = ground_truth_text.strip().lower() == teacher_answer_text.strip().lower()
                    else:
                        # If neither can be extracted, mark as incorrect
                        is_correct = False
                        if idx < 5:  # Log details only for first few samples
                            logger.debug(f"Sample {idx+1} unable to extract answer: ground_truth_text={ground_truth_text}, teacher_answer_text={teacher_answer_text}")
                    
                    if is_correct:
                        correct_count += 1
                    
                    # Print result for each question to log
                    status_icon = "✅" if is_correct else "❌"
                    status_text = "Correct" if is_correct else "Incorrect"
                    
                    # Build answer information
                    if ground_truth_num is not None and teacher_answer_num is not None:
                        answer_info = f"Correct answer: {ground_truth_num}, Predicted answer: {teacher_answer_num}"
                    elif ground_truth_text and teacher_answer_text:
                        answer_info = f"Correct answer: {ground_truth_text}, Predicted answer: {teacher_answer_text}"
                    else:
                        answer_info = "Unable to extract answer"
                    
                    # Print question number, status, and answer information
                    logger.info(
                        f"[{idx+1}/{total_samples}] {status_icon} {status_text} - {answer_info}"
                    )
                    
                    # Print cumulative accuracy every 10 questions
                    if (idx + 1) % 10 == 0:
                        current_accuracy = correct_count / (idx + 1)
                        logger.info(
                            f"📊 Progress report: Completed {idx+1}/{total_samples}, "
                            f"Current accuracy: {current_accuracy:.4f} ({correct_count}/{idx+1})"
                        )
                    
                    # Evaluate reasoning quality (based on response itself only)
                    try:
                        if teacher_response:
                            # Extract reasoning steps
                            reasoning_steps = reasoning_evaluator.extract_reasoning_steps(teacher_response)
                            
                            # Evaluate logical consistency
                            logical_consistency = reasoning_evaluator.evaluate_logical_consistency(teacher_response)
                            
                            # Evaluate answer correctness (if ground_truth_num is available)
                            answer_correctness = None
                            if ground_truth_num is not None:
                                answer_correctness = reasoning_evaluator.evaluate_answer_correctness(
                                    teacher_response, ground_truth_num
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
                    
                    # Save individual results (save complete information for subsequent comparison)
                    individual_result = {
                        "index": idx,
                        "question": question,
                        "ground_truth": ground_truth,
                        "ground_truth_text": ground_truth_text if ground_truth_text else "N/A",
                        "ground_truth_num": ground_truth_num if ground_truth_num is not None else "N/A",
                        "teacher_response": teacher_response,
                        "teacher_answer_text": teacher_answer_text if teacher_answer_text else "N/A",
                        "teacher_answer_num": teacher_answer_num if teacher_answer_num is not None else "N/A",
                        "is_correct": is_correct,
                        "response_length": len(teacher_response),
                        "reasoning_steps_count": len(reasoning_steps),
                        "logical_consistency_score": logical_consistency.get("overall_consistency", 0.0),
                        "answer_extraction_success": extraction_successes[-1] if extraction_successes else False
                    }
                    
                    # Add answer correctness score (if available)
                    if answer_correctness is not None:
                        individual_result["answer_correctness_score"] = answer_correctness.get("correctness_score", 0.0)
                        individual_result["answer_relative_error"] = answer_correctness.get("relative_error", float('inf'))
                    
                    results["individual_results"].append(individual_result)
                    
                    # Also write to JSONL file (convenient for line-by-line processing)
                    jsonl_f.write(json.dumps(individual_result, ensure_ascii=False) + "\n")
                    jsonl_f.flush()  # Flush in real-time to avoid data loss
                    
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
        logger.info(f"Answer extraction success rate: {results['answer_extraction_stats']['success_rate']:.4f} ({successful_extractions}/{len(extraction_successes)})")
        logger.info(f"Average response length: {results['statistics']['average_response_length']:.2f} characters")
        if "average_logical_consistency" in results["statistics"]:
            logger.info(f"Average logical consistency: {results['statistics']['average_logical_consistency']:.4f}")
        if "average_answer_correctness_score" in results["statistics"]:
            logger.info(f"Average answer correctness score: {results['statistics']['average_answer_correctness_score']:.4f}")
        if "average_reasoning_steps" in results["statistics"]:
            logger.info(f"Average reasoning steps: {results['statistics']['average_reasoning_steps']:.2f}")
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
            logger.error(f"❌ Result file save failed: {output_path}")
        
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
    parser = argparse.ArgumentParser(description="Evaluate teacher model")
    parser.add_argument("--teacher_model_name", type=str, default=None,
                       help="Teacher model name or path (e.g., 'Qwen/Qwen2.5-32B-Instruct', default read from config file)")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--eval_samples", type=int, default=None,
                       help="Number of evaluation samples (None means entire test set)")
    parser.add_argument("--output_file", type=str, default="results/teacher_evaluation_results.json",
                       help="Output result JSON file path (contains summary statistics)")
    parser.add_argument("--output_jsonl", type=str, default="results/teacher_evaluation_results.jsonl",
                       help="Output result JSONL file path (contains detailed results for each sample)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Log level")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
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
        
        # Determine teacher model name
        if args.teacher_model_name is None:
            teacher_model_name = config["model"]["teacher_model_name"]
            logger.info(f"Read teacher model from config file: {teacher_model_name}")
        else:
            teacher_model_name = args.teacher_model_name
            logger.info(f"Using specified teacher model: {teacher_model_name}")
        
        # Execute evaluation
        results = evaluate_teacher_model(
            teacher_model_name=teacher_model_name,
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

