#!/usr/bin/env python3
"""
RL Model Checkpoint Evaluation Script
Function: Evaluate locally saved RL training checkpoints and generate comprehensive evaluation reports

Designed for VAST AI online evaluation environment:
- Supports relative and absolute paths
- Uses project unified answer extraction function extract_answer_unified
- Enhanced error handling and logging
- Checkpoint validation and path resolution
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

# Add project root directory to Python path
# Supports running from evaluation directory or scripts directory
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from models.student_model import StudentModel
from models.teacher_model import TeacherModel
from evaluation.reasoning_evaluator import ReasoningEvaluator
from evaluation.metrics import ComprehensiveEvaluator
from data.gsm8k_processor import build_prompt
from datasets import load_dataset
from utils.math_utils import extract_answer_unified  # Use project unified answer extraction function


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging"""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # If log file not specified, use default name
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"evaluate_checkpoint_{timestamp}.log"
    
    # Configure log format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configure logging, output to both console and file
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


def evaluate_checkpoint(
    checkpoint_path: str,
    teacher_model_path: str,
    config: dict,
    eval_samples: int = None,
    output_file: str = "evaluation_results.json",
    **kwargs
):
    """
    Evaluate checkpoint model
    
    Args:
        checkpoint_path: Checkpoint path (e.g., checkpoints/rl_model/checkpoint-1000)
        teacher_model_path: Teacher model path
        config: Configuration dictionary
        eval_samples: Number of evaluation samples (None means all)
        output_file: Output result file path
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Starting RL model checkpoint evaluation")
    logger.info(f"Checkpoint path: {checkpoint_path}")
    logger.info(f"Teacher model: {teacher_model_path}")
    logger.info("=" * 80)
    
    # Initialize results dictionary
    results = {
        "checkpoint_path": checkpoint_path,
        "teacher_model": teacher_model_path,
        "evaluation_time": datetime.now().isoformat(),
        "accuracy": 0.0,
        "reasoning_quality": {
            "overall_score": 0.0,
            "step_coverage": 0.0,
            "logical_consistency": 0.0,
            "kl_divergence": 0.0,
            "answer_correctness": 0.0
        },
        "distillation_effect": {
            "overall_score": 0.0,
            "kl_divergence": 0.0,
            "cosine_similarity": 0.0,
            "js_divergence": 0.0
        },
        "statistics": {
            "total_samples": 0,
            "correct_samples": 0,
            "incorrect_samples": 0,
            "average_reasoning_score": 0.0,
            "average_distillation_score": 0.0
        },
        "individual_results": []
    }
    
    try:
        # Step 1: Load student model (from checkpoint)
        logger.info("Step 1/5: Loading student model...")
        
        # Ensure checkpoint path is absolute (VAST AI compatibility)
        checkpoint_path_abs = Path(checkpoint_path).resolve()
        if not checkpoint_path_abs.exists():
            # Try relative to project root
            project_root = Path(__file__).parent.parent
            checkpoint_path_abs = (project_root / checkpoint_path).resolve()
        
        if not checkpoint_path_abs.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path} (tried: {checkpoint_path_abs})")
        
        logger.info(f"Using checkpoint path: {checkpoint_path_abs}")
        
        # Validate checkpoint files
        required_files = [
            checkpoint_path_abs / "adapter_config.json"
        ]
        weight_files = [
            checkpoint_path_abs / "adapter_model.safetensors",
            checkpoint_path_abs / "adapter_model.bin"
        ]
        
        if not required_files[0].exists():
            raise FileNotFoundError(f"Checkpoint missing required file: {required_files[0]}")
        
        if not any(f.exists() for f in weight_files):
            raise FileNotFoundError(f"Checkpoint missing weight files: {[str(f) for f in weight_files]}")
        
        logger.info(f"Checkpoint validation passed: {[f.name for f in required_files + weight_files if f.exists()]}")
        
        student_model = StudentModel(
            model_name=str(checkpoint_path_abs),
            lora_config=config["lora"],
            device=config["device"]["device_map"],
            torch_dtype=getattr(torch, config["device"]["torch_dtype"]),
            use_lora=True
        )
        logger.info("✅ Student model loaded successfully")
        
        # Step 2: Load teacher model (optional)
        teacher_model = None
        skip_teacher = kwargs.get("skip_teacher", False)
        if not skip_teacher:
            logger.info("Step 2/5: Loading teacher model...")
            try:
                teacher_model = TeacherModel(
                    model_name=teacher_model_path,
                    cache_size=config["model"]["cache_size"],
                    cache_policy=config["model"]["cache_policy"],
                    device=config["device"]["device_map"],
                    torch_dtype=getattr(torch, config["device"]["torch_dtype"])
                )
                logger.info("✅ Teacher model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Teacher model loading failed: {e}")
                logger.warning("   Will continue evaluation but skip teacher model related metrics")
                teacher_model = None
        else:
            logger.info("Step 2/5: Skipping teacher model loading (--skip_teacher)")
        
        # Step 3: Load evaluation data
        logger.info("Step 3/5: Loading evaluation data...")
        try:
            dataset = load_dataset("gsm8k", "main")
            logger.info(f"Dataset loaded successfully: train={len(dataset['train'])}, test={len(dataset['test'])}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        if eval_samples is not None:
            eval_samples = min(eval_samples, len(dataset["test"]))
            eval_dataset = dataset["test"].select(range(eval_samples))
            logger.info(f"Using {eval_samples} test samples (total: {len(dataset['test'])})")
        else:
            eval_dataset = dataset["test"]
            logger.info(f"Using all {len(eval_dataset)} test samples")
        
        # Step 4: Create evaluators
        logger.info("Step 4/5: Initializing evaluators...")
        reasoning_evaluator = ReasoningEvaluator()
        comprehensive_evaluator = ComprehensiveEvaluator()
        logger.info("✅ Evaluators initialized successfully")
        
        # Step 5: Execute evaluation
        logger.info("Step 5/5: Starting evaluation...")
        logger.info("=" * 80)
        
        total_samples = len(eval_dataset)
        correct_count = 0
        
        # Accumulate various metrics
        reasoning_scores = []
        distillation_scores = []
        step_coverage_scores = []
        logical_consistency_scores = []
        answer_correctness_scores = []
        kl_divergences = []
        cosine_similarities = []
        
        # Use tqdm to show progress
        for idx, sample in enumerate(tqdm(eval_dataset, desc="Evaluation progress", ncols=100)):
            question = sample["question"]
            ground_truth = sample["answer"]
            
            try:
                # Format question prompt
                formatted_question = build_prompt(question)
                
                # Student model generation
                try:
                    student_response = student_model.generate(
                        formatted_question,
                        max_length=512,
                        temperature=0.0,
                        do_sample=False
                    )
                    # Ensure string type
                    if not isinstance(student_response, str):
                        student_response = str(student_response) if student_response else ""
                except Exception as e:
                    logger.warning(f"Sample {idx+1} student model generation failed: {e}")
                    student_response = ""
                
                # Teacher model generation (optional)
                teacher_response = ""
                if teacher_model is not None:
                    try:
                        teacher_response = teacher_model.generate_response(
                            formatted_question,
                            max_length=512,
                            temperature=0.0,
                            do_sample=False
                        )
                        # Ensure string type
                        if not isinstance(teacher_response, str):
                            teacher_response = str(teacher_response) if teacher_response else ""
                    except Exception as e:
                        logger.warning(f"Sample {idx+1} teacher model generation failed: {e}")
                        teacher_response = ""
                else:
                    teacher_response = ""  # Skip teacher model
                
                # Get logits for distillation evaluation (optional, failure does not affect other evaluations)
                student_logits = None
                teacher_logits = None
                if student_response and teacher_response:
                    try:
                        # 🔍 Safety check: ensure input is not empty to avoid index out of bounds
                        full_text = formatted_question + student_response
                        if len(full_text.strip()) > 0:
                            student_logits = student_model.get_logits(full_text)
                            # 🔍 Validate logits dimensions (prevent empty tensor causing index out of bounds)
                            if student_logits is not None and student_logits.numel() == 0:
                                student_logits = None
                    except (IndexError, RuntimeError) as e:
                        logger.warning(f"Sample {idx+1} failed to get student logits (index out of bounds): {e}")
                        student_logits = None
                    except Exception as e:
                        logger.debug(f"Sample {idx+1} failed to get student logits (skipping): {e}")
                        student_logits = None
                    
                    try:
                        # 🔍 Safety check: ensure input is not empty to avoid index out of bounds
                        full_text = formatted_question + teacher_response
                        if len(full_text.strip()) > 0:
                            teacher_logits = teacher_model.get_logits(full_text)
                            # 🔍 Validate logits dimensions (prevent empty tensor causing index out of bounds)
                            if teacher_logits is not None and teacher_logits.numel() == 0:
                                teacher_logits = None
                    except (IndexError, RuntimeError) as e:
                        logger.warning(f"Sample {idx+1} failed to get teacher logits (index out of bounds): {e}")
                        teacher_logits = None
                    except Exception as e:
                        logger.debug(f"Sample {idx+1} failed to get teacher logits (skipping): {e}")
                        teacher_logits = None
                
                # Extract answers (using project unified answer extraction function extract_answer_unified)
                try:
                    ground_truth_text, ground_truth_num = extract_answer_unified(ground_truth)
                except Exception as e:
                    logger.warning(f"Sample {idx+1} failed to extract ground_truth answer: {e}")
                    ground_truth_text, ground_truth_num = "", None
                
                try:
                    student_answer_text, student_answer_num = extract_answer_unified(student_response)
                except Exception as e:
                    logger.warning(f"Sample {idx+1} failed to extract student answer: {e}")
                    student_answer_text, student_answer_num = "", None
                
                # Evaluate answer correctness (using numerical comparison for accuracy)
                is_correct = False
                if ground_truth_num is not None and student_answer_num is not None:
                    # Use numerical comparison (tolerate small errors, consistent with utils/math_utils.py)
                    tolerance = 1e-6
                    if abs(ground_truth_num) < 1e-10:
                        # Ground truth close to 0, use absolute error
                        is_correct = abs(student_answer_num - ground_truth_num) < tolerance
                    else:
                        # Use relative error
                        relative_error = abs(student_answer_num - ground_truth_num) / abs(ground_truth_num)
                        is_correct = relative_error < tolerance
                elif ground_truth_text and student_answer_text:
                    # If numbers cannot be extracted, use text comparison
                    is_correct = ground_truth_text.strip().lower() == student_answer_text.strip().lower()
                else:
                    # If both cannot be extracted, mark as incorrect
                    is_correct = False
                    if idx < 5:  # Detailed logging only for first few samples
                        logger.debug(f"Sample {idx+1} unable to extract answers: ground_truth_text={ground_truth_text}, student_answer_text={student_answer_text}")
                
                if is_correct:
                    correct_count += 1
                
                # Evaluate reasoning quality (if response is not empty)
                try:
                    if student_response:
                        if teacher_response:
                            reasoning_result = reasoning_evaluator.evaluate_reasoning_quality(
                                student_response=student_response,
                                teacher_response=teacher_response,
                                ground_truth_answer=ground_truth_num,
                                student_logits=student_logits,
                                teacher_logits=teacher_logits
                            )
                        else:
                            # When only student model is available, use logic metrics consistent with SFT evaluation
                            reasoning_steps = reasoning_evaluator.extract_reasoning_steps(student_response)
                            logical_consistency = reasoning_evaluator.evaluate_logical_consistency(student_response)
                            
                            answer_correctness_result = None
                            if ground_truth_num is not None:
                                answer_correctness_result = reasoning_evaluator.evaluate_answer_correctness(
                                    student_response,
                                    ground_truth_num
                                )
                            
                            logical_score = logical_consistency.get("overall_consistency", 0.0)
                            if answer_correctness_result is not None:
                                correctness_score = answer_correctness_result.get("correctness_score", 0.0)
                                overall_score = 0.7 * correctness_score + 0.3 * logical_score
                            else:
                                correctness_score = 0.0
                                overall_score = logical_score
                            
                            reasoning_result = {
                                "overall_score": overall_score,
                                "step_coverage": {"step_coverage": 1.0 if reasoning_steps else 0.0},
                                "logical_consistency": logical_consistency,
                            }
                            if answer_correctness_result is not None:
                                reasoning_result["answer_correctness"] = answer_correctness_result
                    else:
                        reasoning_result = {
                            "overall_score": 0.0,
                            "step_coverage": {"step_coverage": 0.0},
                            "logical_consistency": {"overall_consistency": 0.0}
                        }
                except Exception as e:
                    logger.warning(f"Sample {idx+1} reasoning quality evaluation failed: {e}")
                    reasoning_result = {
                        "overall_score": 0.0,
                        "step_coverage": {"step_coverage": 0.0},
                        "logical_consistency": {"overall_consistency": 0.0}
                    }
                
                # Evaluate distillation effect (if response is not empty and logits are available)
                try:
                    if student_response and ground_truth:
                        # If teacher model exists, try to compute logits for cosine similarity, KL, and other distillation metrics
                        if teacher_model is not None and teacher_response:
                            try:
                                student_logits = student_logits or student_model.get_logits(
                                    formatted_question + student_response
                                )
                                teacher_logits = teacher_logits or teacher_model.get_logits(
                                    formatted_question + teacher_response
                                )
                            except Exception as e:
                                logger.debug(f"Sample {idx+1} failed to get logits: {e}")
                                student_logits = None
                                teacher_logits = None
                        distillation_result = comprehensive_evaluator.evaluate_comprehensive(
                            predictions=[student_response],
                            ground_truths=[ground_truth],
                            student_logits=student_logits,
                            teacher_logits=teacher_logits
                        )
                    else:
                        distillation_result = {"overall_score": 0.0}
                except Exception as e:
                    logger.warning(f"Sample {idx+1} distillation effect evaluation failed: {e}")
                    distillation_result = {"overall_score": 0.0}
                
                # Accumulate metrics
                reasoning_score = reasoning_result["overall_score"]
                reasoning_scores.append(reasoning_score)
                
                distillation_score = distillation_result.get("overall_score", 0.0)
                distillation_scores.append(distillation_score)
                
                step_coverage_scores.append(reasoning_result.get("step_coverage", {}).get("step_coverage", 0.0))
                logical_consistency_scores.append(
                    reasoning_result.get("logical_consistency", {}).get("overall_consistency", 0.0)
                )
                
                if "answer_correctness" in reasoning_result:
                    answer_correctness_scores.append(
                        reasoning_result["answer_correctness"].get("correctness_score", 0.0)
                    )
                
                if student_logits is not None and teacher_logits is not None:
                    kl_divergences.append(distillation_result.get("kl_divergence", 0.0))
                    cosine_similarities.append(distillation_result.get("cosine_similarity", 0.0))
                
                # Save individual results (save complete content for analysis)
                individual_result = {
                    "index": idx + 1,
                    "question": question,  # Save complete question
                    "ground_truth": ground_truth,  # Save complete ground truth
                    "student_response": student_response,  # Save complete student response
                    "teacher_response": teacher_response,  # Save complete teacher response
                    "is_correct": is_correct,
                    "student_answer_text": student_answer_text if student_answer_text else "N/A",
                    "student_answer_num": student_answer_num if student_answer_num is not None else "N/A",
                    "ground_truth_text": ground_truth_text if ground_truth_text else "N/A",
                    "ground_truth_answer_num": ground_truth_num if ground_truth_num is not None else "N/A",
                    "reasoning_score": float(reasoning_score) if isinstance(reasoning_score, (int, float)) else 0.0,
                    "distillation_score": float(distillation_score) if isinstance(distillation_score, (int, float)) else 0.0
                }
                results["individual_results"].append(individual_result)
                
            except Exception as e:
                logger.error(f"Error evaluating sample {idx+1}: {e}")
                # Record error but continue evaluation
                results["individual_results"].append({
                    "index": idx + 1,
                    "error": str(e),
                    "question": question  # Save complete question
                })
        
        # Compute overall metrics
        logger.info("=" * 80)
        logger.info("Computing overall metrics...")
        
        results["accuracy"] = correct_count / total_samples if total_samples > 0 else 0.0
        
        # Reasoning quality statistics
        if reasoning_scores:
            results["reasoning_quality"]["overall_score"] = sum(reasoning_scores) / len(reasoning_scores)
            results["reasoning_quality"]["step_coverage"] = sum(step_coverage_scores) / len(step_coverage_scores) if step_coverage_scores else 0.0
            results["reasoning_quality"]["logical_consistency"] = sum(logical_consistency_scores) / len(logical_consistency_scores) if logical_consistency_scores else 0.0
            results["reasoning_quality"]["answer_correctness"] = sum(answer_correctness_scores) / len(answer_correctness_scores) if answer_correctness_scores else 0.0
            results["reasoning_quality"]["kl_divergence"] = sum(kl_divergences) / len(kl_divergences) if kl_divergences else 0.0
        
        # Distillation effect statistics
        if distillation_scores:
            results["distillation_effect"]["overall_score"] = sum(distillation_scores) / len(distillation_scores)
            results["distillation_effect"]["kl_divergence"] = sum(kl_divergences) / len(kl_divergences) if kl_divergences else 0.0
            results["distillation_effect"]["cosine_similarity"] = sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0.0
        
        # Overall statistics
        results["statistics"]["total_samples"] = total_samples
        results["statistics"]["correct_samples"] = correct_count
        results["statistics"]["incorrect_samples"] = total_samples - correct_count
        results["statistics"]["average_reasoning_score"] = results["reasoning_quality"]["overall_score"]
        results["statistics"]["average_distillation_score"] = results["distillation_effect"]["overall_score"]
        
        # Print results summary
        logger.info("=" * 80)
        logger.info("Evaluation Results Summary")
        logger.info("=" * 80)
        logger.info(f"Accuracy: {results['accuracy']:.4f} ({correct_count}/{total_samples})")
        logger.info(f"Reasoning quality overall score: {results['reasoning_quality']['overall_score']:.4f}")
        logger.info(f"  - Step coverage: {results['reasoning_quality']['step_coverage']:.4f}")
        logger.info(f"  - Logical consistency: {results['reasoning_quality']['logical_consistency']:.4f}")
        logger.info(f"  - Answer correctness: {results['reasoning_quality']['answer_correctness']:.4f}")
        logger.info(f"  - KL divergence: {results['reasoning_quality']['kl_divergence']:.4f}")
        logger.info(f"Distillation effect overall score: {results['distillation_effect']['overall_score']:.4f}")
        logger.info(f"  - Cosine similarity: {results['distillation_effect']['cosine_similarity']:.4f}")
        logger.info("=" * 80)
        
        # Save results
        output_path = Path(output_file)
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving evaluation results to: {output_path.resolve()}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Verify file was saved
        if output_path.exists():
            file_size = output_path.stat().st_size / 1024 / 1024  # MB
            logger.info(f"✅ Evaluation completed! Result file: {output_path.resolve()} ({file_size:.2f} MB)")
        else:
            logger.error(f"❌ Failed to save result file: {output_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in evaluation process: {e}")
        import traceback
        logger.error(f"Detailed error information:\n{traceback.format_exc()}")
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate RL model checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Checkpoint path (e.g., checkpoints/rl_model/checkpoint-1000)")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--teacher_model_path", type=str, default="Qwen/Qwen2.5-32B-Instruct",
                       help="Teacher model path")
    parser.add_argument("--eval_samples", type=int, default=None,
                       help="Number of evaluation samples (None means all)")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                       help="Output result file path")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Log level")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log file path")
    parser.add_argument("--skip_teacher", action="store_true",
                       help="Skip teacher model generation (avoid CUDA errors)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    # Check checkpoint path (supports relative and absolute paths, VAST AI compatible)
    checkpoint_path_input = Path(args.checkpoint_path)
    project_root = Path(__file__).parent.parent
    current_dir = Path.cwd()
    
    # Try multiple possible paths
    possible_paths = [
        checkpoint_path_input,  # Original path
        checkpoint_path_input.resolve(),  # Absolute path resolution
        project_root / args.checkpoint_path,  # Relative to project root
        current_dir / args.checkpoint_path,  # Relative to current working directory
    ]
    
    checkpoint_path = None
    for path in possible_paths:
        if path.exists() and path.is_dir():
            checkpoint_path = path.resolve()
            logger.info(f"Found checkpoint path: {checkpoint_path}")
            break
    
    if checkpoint_path is None:
        logger.error(f"❌ Checkpoint path does not exist: {args.checkpoint_path}")
        logger.error(f"Tried paths:")
        for path in possible_paths:
            logger.error(f"  - {path} (exists: {path.exists()}, is_dir: {path.is_dir() if path.exists() else 'N/A'})")
        logger.error(f"Current working directory: {current_dir}")
        logger.error(f"Project root directory: {project_root}")
        logger.error(f"Script location: {Path(__file__).parent}")
        sys.exit(1)
    
    # Check if checkpoint directory contains adapter files
    adapter_config = checkpoint_path / "adapter_config.json"
    adapter_weights = [
        checkpoint_path / "adapter_model.safetensors",
        checkpoint_path / "adapter_model.bin"
    ]
    
    if not adapter_config.exists():
        logger.error(f"❌ Checkpoint missing configuration file: {adapter_config}")
        sys.exit(1)
    
    if not any(f.exists() for f in adapter_weights):
        logger.error(f"❌ Checkpoint missing weight files")
        logger.error(f"   Searched paths: {[str(f) for f in adapter_weights]}")
        logger.error(f"   Checkpoint directory contents: {list(checkpoint_path.iterdir())[:10]}")
        sys.exit(1)
    
    logger.info(f"✅ Checkpoint validation passed: {checkpoint_path}")
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
        results = evaluate_checkpoint(
            checkpoint_path=str(checkpoint_path.resolve()),
            teacher_model_path=args.teacher_model_path,
            config=config,
            eval_samples=args.eval_samples,
            output_file=args.output_file,
            skip_teacher=args.skip_teacher
        )
        
        logger.info("=" * 80)
        logger.info("Evaluation task completed!")
        logger.info(f"Result file: {args.output_file}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

