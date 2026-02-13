#!/usr/bin/env python3
"""
Export GSM8K Answers (Teacher Model)

Function:
- Generate answers sample by sample, using project's unified answer extraction (extract_answer_unified, supports ####)
- Optionally export top-k of next token distribution (for subsequent distillation comparison approximate KL/JS/cosine)
- Results saved as JSONL for easy subsequent merging and comparison

Usage example:
  Teacher model
    python scripts/export_teacher_gsm8k_answers.py \
      --config config/training_config.yaml \
      --eval_samples 200 \
      --out results/teacher_gsm8k.jsonl
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm

# Add project root directory
sys.path.append(str(Path(__file__).parent.parent))

from models.teacher_model import TeacherModel
from utils.math_utils import extract_answer_unified
from data.gsm8k_processor import build_prompt


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def softmax_topk_from_logits(logits: torch.Tensor, top_k: int = 50) -> Tuple[List[int], List[float]]:
    """Calculate top-k probability distribution from last step logits (returns ids and probs lists).
    Only take last step (last token of prompt) to reduce volume, for approximate distillation comparison use.
    """
    if logits is None:
        return [], []
    last_logits = logits[-1]  # (vocab_size,)
    probs = torch.softmax(last_logits.float(), dim=-1)
    topk = min(top_k, probs.shape[-1])
    values, indices = torch.topk(probs, k=topk, dim=-1)
    return indices.tolist(), values.tolist()


def main():
    parser = argparse.ArgumentParser(description="Export GSM8K answers (teacher model)")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "test"],
                       help="GSM8K split")
    parser.add_argument("--eval_samples", type=int, default=200,
                       help="Number of evaluation samples (default 200)")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    parser.add_argument("--topk_dist", type=int, default=50,
                       help="Top-k size for next step distribution export (0 means no export)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--out", type=str, required=True,
                       help="Output JSONL file path")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Log level")
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    torch.manual_seed(args.seed)
    
    config = load_config(args.config)
    
    # Load data
    ds = load_dataset("gsm8k", "main")
    split = ds[args.eval_split]
    n = min(args.eval_samples, len(split))
    eval_ds = split.select(range(n))
    logging.info(f"Dataset: GSM8K/{args.eval_split}, samples={n}")
    
    # Load teacher model
    logging.info("Loading teacher model...")
    teacher_model_name = config["model"]["teacher_model_name"]
    teacher = TeacherModel(
        model_name=teacher_model_name,
        cache_size=1000,
        device_map=config["device"]["device_map"] if config.get("device") else "auto",
        torch_dtype=getattr(torch, config["device"]["torch_dtype"]) if config.get("device") else torch.bfloat16
    )
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_ok = 0
    num_failed = 0
    failed_indices = []
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for idx, sample in enumerate(tqdm(eval_ds, desc="Export progress", ncols=100)):
            q = sample["question"]
            gt = sample["answer"]
            prompt = build_prompt(q)
            
            # Generate
            resp = ""
            generation_error = None
            try:
                resp = teacher.generate_response(prompt, max_length=args.max_length,
                                                temperature=args.temperature, do_sample=True)
                if not isinstance(resp, str):
                    resp = str(resp) if resp else ""
            except Exception as e:
                generation_error = str(e)
                logging.warning(f"Sample {idx+1} generation failed: {e}")
                resp = ""
                # If CUDA error, clear cache
                if "CUDA" in str(e) or "device-side assert" in str(e):
                    torch.cuda.empty_cache()
            
            # Extract answers (unified implementation, prioritize ####)
            gt_text, gt_num = extract_answer_unified(gt)
            pred_text, pred_num = extract_answer_unified(resp) if resp else ("", None)
            
            # Optional: Export next step distribution (on prompt, avoid response length differences affecting volume)
            top_ids: List[int] = []
            top_probs: List[float] = []
            logits_error = None
            if args.topk_dist and args.topk_dist > 0 and not generation_error:
                try:
                    logits_tensor = teacher.get_logits(prompt)  # (1, seq_len, vocab_size)
                    if logits_tensor is not None:
                        # Remove batch dimension, take last step
                        if logits_tensor.ndim == 3:
                            logits = logits_tensor[0]  # (seq_len, vocab_size)
                        else:
                            logits = logits_tensor
                        
                        if logits.ndim >= 2:
                            top_ids, top_probs = softmax_topk_from_logits(logits, args.topk_dist)
                except Exception as e:
                    logits_error = str(e)
                    logging.debug(f"Sample {idx+1} failed to extract top-k: {e}")
            
            record = {
                "index": idx,
                "question": q,
                "prompt": prompt,
                "ground_truth": gt,
                "ground_truth_text": gt_text if gt_text else "N/A",
                "ground_truth_num": gt_num if gt_num is not None else "N/A",
                "response": resp if resp else "",
                "answer_text": pred_text if pred_text else "N/A",
                "answer_num": pred_num if pred_num is not None else "N/A",
                "top_ids": top_ids,
                "top_probs": top_probs,
                "error": generation_error if generation_error else (logits_error if logits_error else None)
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            # Flush in real-time to avoid long periods without output
            f.flush()
            os.fsync(f.fileno())
            
            if generation_error:
                num_failed += 1
                failed_indices.append(idx)
            else:
                num_ok += 1
    
    logging.info(f"Export completed: {out_path}")
    logging.info(f"  Success: {num_ok} entries")
    if num_failed > 0:
        logging.warning(f"  Failed: {num_failed} entries")
        logging.warning(f"  Failed sample indices: {failed_indices[:20]}{'...' if len(failed_indices) > 20 else ''}")


if __name__ == "__main__":
    main()

