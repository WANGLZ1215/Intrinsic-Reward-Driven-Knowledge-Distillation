#!/usr/bin/env python3
"""
Model Logits Comparison Evaluation Script
Function: Compare logits similarity metrics between RL model, SFT model, and teacher model

Metrics computed:
- KL divergence (student vs teacher logits)
- JS divergence (student vs teacher logits)
- Cosine similarity of logits (student vs teacher logits)
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
import torch.nn.functional as F
import numpy as np

# Add project root directory to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from models.student_model import StudentModel
from models.teacher_model import TeacherModel
from evaluation.metrics import DistillationMetrics
from data.gsm8k_processor import build_prompt
from datasets import load_dataset


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"compare_model_logits_{timestamp}.log"
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
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


def align_logits(teacher_logits: torch.Tensor, 
                 student_logits: torch.Tensor) -> tuple:
    """
    Align vocab_size dimension of logits (last dimension)
    This is a 100% accepted practice in academia (commonly used in distillation when teacher/student vocab are not fully consistent)
    
    Args:
        teacher_logits: Teacher model logits
        student_logits: Student model logits
        
    Returns:
        Aligned (teacher_logits, student_logits)
    """
    t_dim = teacher_logits.size(-1)
    s_dim = student_logits.size(-1)
    
    if t_dim == s_dim:
        return teacher_logits, student_logits
    
    # Align to minimum vocab_size
    min_dim = min(t_dim, s_dim)
    teacher_logits = teacher_logits[..., :min_dim]
    student_logits = student_logits[..., :min_dim]
    
    return teacher_logits, student_logits


def align_logits_tensors(student_logits: torch.Tensor, 
                        teacher_logits: torch.Tensor) -> tuple:
    """
    Align sequence length and vocab_size dimensions of logits tensors
    
    Args:
        student_logits: Student model logits [seq_len, vocab_size] or [batch, seq_len, vocab_size]
        teacher_logits: Teacher model logits [seq_len, vocab_size] or [batch, seq_len, vocab_size]
        
    Returns:
        Aligned (student_logits, teacher_logits)
    """
    # Ensure 3D tensors
    if len(student_logits.shape) == 2:
        student_logits = student_logits.unsqueeze(0)
    if len(teacher_logits.shape) == 2:
        teacher_logits = teacher_logits.unsqueeze(0)
    
    # Get sequence length
    student_seq_len = student_logits.shape[1]
    teacher_seq_len = teacher_logits.shape[1]
    
    # Align to minimum sequence length
    min_len = min(student_seq_len, teacher_seq_len)
    
    if min_len > 0:
        student_logits = student_logits[:, :min_len, :]
        teacher_logits = teacher_logits[:, :min_len, :]
    
    # Align vocab_size dimension (last dimension)
    teacher_logits, student_logits = align_logits(teacher_logits, student_logits)
    
    # Remove batch dimension (if only one sample)
    if student_logits.shape[0] == 1:
        student_logits = student_logits[0]
        teacher_logits = teacher_logits[0]
    
    return student_logits, teacher_logits


def compute_metrics(student_logits: torch.Tensor, 
                   teacher_logits: torch.Tensor,
                   distillation_metrics: DistillationMetrics) -> dict:
    """
    Compute logits similarity metrics
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        distillation_metrics: Distillation metrics calculator
        
    Returns:
        Dictionary containing KL divergence, JS divergence, Cosine similarity
    """
    try:
        # Align logits (sequence length and vocab_size)
        student_logits_aligned, teacher_logits_aligned = align_logits_tensors(
            student_logits, teacher_logits
        )
        
        # Ensure vocab_size alignment again (double check)
        teacher_logits_aligned, student_logits_aligned = align_logits(
            teacher_logits_aligned, student_logits_aligned
        )
        
        # Compute metrics
        kl_div = distillation_metrics.kl_divergence(
            student_logits_aligned, teacher_logits_aligned
        )
        js_div = distillation_metrics.js_divergence(
            student_logits_aligned, teacher_logits_aligned
        )
        cos_sim = distillation_metrics.cosine_similarity(
            student_logits_aligned, teacher_logits_aligned
        )
        
        return {
            "kl_divergence": float(kl_div),
            "js_divergence": float(js_div),
            "cosine_similarity": float(cos_sim)
        }
    except Exception as e:
        logging.warning(f"Failed to compute metrics: {e}")
        return {
            "kl_divergence": None,
            "js_divergence": None,
            "cosine_similarity": None
        }


def evaluate_model_logits(
    rl_checkpoint_path: str,
    sft_model_path: str,
    teacher_model_path: str,
    config: dict,
    eval_samples: int = 150,
    output_file: str = "model_logits_comparison.json",
    **kwargs
):
    """
    Evaluate logits similarity of three models
    
    Args:
        rl_checkpoint_path: RL model checkpoint path
        sft_model_path: SFT model path
        teacher_model_path: Teacher model path
        config: Configuration dictionary
        eval_samples: Number of evaluation samples
        output_file: Output result file path
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Starting model logits similarity evaluation")
    logger.info(f"RL model: {rl_checkpoint_path}")
    logger.info(f"SFT model: {sft_model_path}")
    logger.info(f"Teacher model: {teacher_model_path}")
    logger.info(f"Evaluation samples: {eval_samples}")
    logger.info("=" * 80)
    
    # Initialize results dictionary
    results = {
        "rl_checkpoint_path": rl_checkpoint_path,
        "sft_model_path": sft_model_path,
        "teacher_model_path": teacher_model_path,
        "evaluation_time": datetime.now().isoformat(),
        "eval_samples": eval_samples,
        "rl_vs_teacher": {
            "kl_divergence": [],
            "js_divergence": [],
            "cosine_similarity": []
        },
        "sft_vs_teacher": {
            "kl_divergence": [],
            "js_divergence": [],
            "cosine_similarity": []
        },
        "statistics": {
            "rl_vs_teacher": {
                "mean_kl_divergence": 0.0,
                "mean_js_divergence": 0.0,
                "mean_cosine_similarity": 0.0,
                "valid_samples": 0
            },
            "sft_vs_teacher": {
                "mean_kl_divergence": 0.0,
                "mean_js_divergence": 0.0,
                "mean_cosine_similarity": 0.0,
                "valid_samples": 0
            }
        },
        "individual_results": []
    }
    
    try:
        # Step 1: Load RL model
        logger.info("Step 1/4: Loading RL model...")
        rl_checkpoint_path_abs = Path(rl_checkpoint_path).resolve()
        if not rl_checkpoint_path_abs.exists():
            project_root = Path(__file__).parent.parent
            rl_checkpoint_path_abs = (project_root / rl_checkpoint_path).resolve()
        
        if not rl_checkpoint_path_abs.exists():
            raise FileNotFoundError(f"RL checkpoint path does not exist: {rl_checkpoint_path}")
        
        logger.info(f"Using RL checkpoint path: {rl_checkpoint_path_abs}")
        rl_model = StudentModel(
            model_name=str(rl_checkpoint_path_abs),
            lora_config=config["lora"],
            device=config["device"]["device_map"],
            torch_dtype=getattr(torch, config["device"]["torch_dtype"]),
            use_lora=True
        )
        logger.info("✅ RL model loaded successfully")
        
        # Step 2: Load SFT model (optional)
        evaluate_sft = sft_model_path and sft_model_path.lower() not in ["none", "null", ""]
        sft_model = None
        
        if evaluate_sft:
            logger.info("Step 2/4: Loading SFT model...")
            sft_model_path_abs = Path(sft_model_path).resolve()
            if not sft_model_path_abs.exists():
                project_root = Path(__file__).parent.parent
                sft_model_path_abs = (project_root / sft_model_path).resolve()
            
            if not sft_model_path_abs.exists():
                raise FileNotFoundError(f"SFT model path does not exist: {sft_model_path}")
            
            logger.info(f"Using SFT model path: {sft_model_path_abs}")
            sft_model = StudentModel(
                model_name=str(sft_model_path_abs),
                lora_config=config["lora"],
                device=config["device"]["device_map"],
                torch_dtype=getattr(torch, config["device"]["torch_dtype"]),
                use_lora=True
            )
            logger.info("✅ SFT model loaded successfully")
        else:
            logger.info("Step 2/4: Skipping SFT model loading")
        
        # Step 3: Load teacher model
        step_num = "3/3" if not evaluate_sft else "3/4"
        logger.info(f"Step {step_num}: Loading teacher model...")
        teacher_model = TeacherModel(
            model_name=teacher_model_path,
            cache_size=config["model"]["cache_size"],
            cache_policy=config["model"]["cache_policy"],
            device=config["device"]["device_map"],
            torch_dtype=getattr(torch, config["device"]["torch_dtype"])
        )
        logger.info("✅ Teacher model loaded successfully")
        
        # Step 4: Load evaluation data
        step_num = "4/4" if evaluate_sft else "3/3"
        logger.info(f"Step {step_num}: Loading evaluation data...")
        dataset = load_dataset("gsm8k", "main")
        logger.info(f"Dataset loaded successfully: train={len(dataset['train'])}, test={len(dataset['test'])}")
        
        # Random sampling of evaluation samples (to ensure fairer and more representative results)
        eval_samples = min(eval_samples, len(dataset["test"]))
        # Set random seed for reproducibility
        import random
        random_seed = kwargs.get("random_seed", 42)
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Randomly select sample indices
        total_samples = len(dataset["test"])
        sample_indices = random.sample(range(total_samples), eval_samples)
        sample_indices.sort()  # Sort for easier debugging and viewing
        
        eval_dataset = dataset["test"].select(sample_indices)
        logger.info(f"Using {eval_samples} randomly sampled test samples (random seed: {random_seed})")
        logger.info(f"Sample index range: [{min(sample_indices)}, {max(sample_indices)}]")
        
        # Initialize distillation metrics calculator
        distillation_metrics = DistillationMetrics()
        
        # Execute evaluation
        logger.info("=" * 80)
        logger.info("Starting evaluation...")
        logger.info("=" * 80)
        
        rl_kl_values = []
        rl_js_values = []
        rl_cos_values = []
        
        sft_kl_values = []
        sft_js_values = []
        sft_cos_values = []
        
        for idx, sample in enumerate(tqdm(eval_dataset, desc="Evaluation progress", ncols=100)):
            question = sample["question"]
            ground_truth = sample["answer"]
            
            try:
                # Format question prompt
                formatted_question = build_prompt(question)
                
                # Use teacher model to generate standard response as input (ensure all models use the same input)
                teacher_response = teacher_model.generate_response(
                    formatted_question,
                    max_length=512,
                    temperature=0.0,
                    do_sample=False
                )
                if not isinstance(teacher_response, str):
                    teacher_response = str(teacher_response) if teacher_response else ""
                
                # Build unified input text (question + teacher_response)
                # This ensures all models compute logits under the same input, making comparison meaningful
                full_text = formatted_question + teacher_response
                
                if len(full_text.strip()) == 0:
                    logger.debug(f"Sample {idx+1} input text is empty, skipping")
                    continue
                
                # Get logits (all models use the same input)
                rl_logits = None
                sft_logits = None
                teacher_logits = None
                
                # Get RL model logits
                try:
                    rl_logits = rl_model.get_logits(full_text)
                except Exception as e:
                    logger.debug(f"Sample {idx+1} failed to get RL logits: {e}")
                    rl_logits = None
                
                # Get SFT model logits (if SFT model is loaded)
                sft_logits = None
                if sft_model is not None:
                    try:
                        sft_logits = sft_model.get_logits(full_text)
                    except Exception as e:
                        logger.debug(f"Sample {idx+1} failed to get SFT logits: {e}")
                        sft_logits = None
                
                # Get teacher model logits
                try:
                    teacher_logits = teacher_model.get_logits(full_text)
                except Exception as e:
                    logger.debug(f"Sample {idx+1} failed to get teacher logits: {e}")
                    teacher_logits = None
                
                # Compute RL vs Teacher metrics
                rl_metrics = None
                if rl_logits is not None and teacher_logits is not None:
                    rl_metrics = compute_metrics(
                        rl_logits, teacher_logits, distillation_metrics
                    )
                    
                    if rl_metrics["kl_divergence"] is not None:
                        rl_kl_values.append(rl_metrics["kl_divergence"])
                        results["rl_vs_teacher"]["kl_divergence"].append(rl_metrics["kl_divergence"])
                    
                    if rl_metrics["js_divergence"] is not None:
                        rl_js_values.append(rl_metrics["js_divergence"])
                        results["rl_vs_teacher"]["js_divergence"].append(rl_metrics["js_divergence"])
                    
                    if rl_metrics["cosine_similarity"] is not None:
                        rl_cos_values.append(rl_metrics["cosine_similarity"])
                        results["rl_vs_teacher"]["cosine_similarity"].append(rl_metrics["cosine_similarity"])
                
                # Compute SFT vs Teacher metrics
                sft_metrics = None
                if sft_logits is not None and teacher_logits is not None:
                    sft_metrics = compute_metrics(
                        sft_logits, teacher_logits, distillation_metrics
                    )
                    
                    if sft_metrics["kl_divergence"] is not None:
                        sft_kl_values.append(sft_metrics["kl_divergence"])
                        results["sft_vs_teacher"]["kl_divergence"].append(sft_metrics["kl_divergence"])
                    
                    if sft_metrics["js_divergence"] is not None:
                        sft_js_values.append(sft_metrics["js_divergence"])
                        results["sft_vs_teacher"]["js_divergence"].append(sft_metrics["js_divergence"])
                    
                    if sft_metrics["cosine_similarity"] is not None:
                        sft_cos_values.append(sft_metrics["cosine_similarity"])
                        results["sft_vs_teacher"]["cosine_similarity"].append(sft_metrics["cosine_similarity"])
                
                # Save individual results
                individual_result = {
                    "index": idx + 1,
                    "question": question,
                    "rl_vs_teacher": rl_metrics,
                    "sft_vs_teacher": sft_metrics
                }
                results["individual_results"].append(individual_result)
                
            except Exception as e:
                logger.error(f"Error evaluating sample {idx+1}: {e}")
                results["individual_results"].append({
                    "index": idx + 1,
                    "error": str(e),
                    "question": question
                })
        
        # Compute overall statistics
        logger.info("=" * 80)
        logger.info("Computing overall metrics...")
        
        # RL vs Teacher statistics
        if rl_kl_values:
            results["statistics"]["rl_vs_teacher"]["mean_kl_divergence"] = float(np.mean(rl_kl_values))
            results["statistics"]["rl_vs_teacher"]["mean_js_divergence"] = float(np.mean(rl_js_values))
            results["statistics"]["rl_vs_teacher"]["mean_cosine_similarity"] = float(np.mean(rl_cos_values))
            results["statistics"]["rl_vs_teacher"]["valid_samples"] = len(rl_kl_values)
        
        # SFT vs Teacher statistics
        if sft_kl_values:
            results["statistics"]["sft_vs_teacher"]["mean_kl_divergence"] = float(np.mean(sft_kl_values))
            results["statistics"]["sft_vs_teacher"]["mean_js_divergence"] = float(np.mean(sft_js_values))
            results["statistics"]["sft_vs_teacher"]["mean_cosine_similarity"] = float(np.mean(sft_cos_values))
            results["statistics"]["sft_vs_teacher"]["valid_samples"] = len(sft_kl_values)
        
        # Print results summary
        logger.info("=" * 80)
        logger.info("Evaluation Results Summary")
        logger.info("=" * 80)
        logger.info("RL model vs Teacher model:")
        logger.info(f"  Mean KL divergence: {results['statistics']['rl_vs_teacher']['mean_kl_divergence']:.6f}")
        logger.info(f"  Mean JS divergence: {results['statistics']['rl_vs_teacher']['mean_js_divergence']:.6f}")
        logger.info(f"  Mean cosine similarity: {results['statistics']['rl_vs_teacher']['mean_cosine_similarity']:.6f}")
        logger.info(f"  Valid samples: {results['statistics']['rl_vs_teacher']['valid_samples']}")
        logger.info("")
        logger.info("SFT model vs Teacher model:")
        logger.info(f"  Mean KL divergence: {results['statistics']['sft_vs_teacher']['mean_kl_divergence']:.6f}")
        logger.info(f"  Mean JS divergence: {results['statistics']['sft_vs_teacher']['mean_js_divergence']:.6f}")
        logger.info(f"  Mean cosine similarity: {results['statistics']['sft_vs_teacher']['mean_cosine_similarity']:.6f}")
        logger.info(f"  Valid samples: {results['statistics']['sft_vs_teacher']['valid_samples']}")
        logger.info("=" * 80)
        
        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving evaluation results to: {output_path.resolve()}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
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
    parser = argparse.ArgumentParser(description="Compare logits similarity of RL, SFT, and teacher models")
    parser.add_argument("--rl_checkpoint_path", type=str, 
                       default="checkpoints/rl_model/checkpoint-350",
                       help="RL model checkpoint path")
    parser.add_argument("--sft_model_path", type=str,
                       default="checkpoints/sft_model",
                       help="SFT model path")
    parser.add_argument("--teacher_model_path", type=str, 
                       default="Qwen/Qwen2.5-32B-Instruct",
                       help="Teacher model path")
    parser.add_argument("--config", type=str, 
                       default="config/training_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--eval_samples", type=int, 
                       default=150,
                       help="Number of evaluation samples")
    parser.add_argument("--random_seed", type=int,
                       default=42,
                       help="Random seed (for reproducible random sampling)")
    parser.add_argument("--output_file", type=str, 
                       default="model_logits_comparison.json",
                       help="Output result file path")
    parser.add_argument("--log_level", type=str, 
                       default="INFO",
                       help="Log level")
    parser.add_argument("--log_file", type=str, 
                       default=None,
                       help="Log file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    # Load configuration
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
    results = evaluate_model_logits(
        rl_checkpoint_path=args.rl_checkpoint_path,
        sft_model_path=args.sft_model_path,
        teacher_model_path=args.teacher_model_path,
        config=config,
        eval_samples=args.eval_samples,
        output_file=args.output_file,
        random_seed=args.random_seed
    )
    
    logger.info("=" * 80)
    logger.info("Evaluation task completed!")
    logger.info(f"Result file: {args.output_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

