#!/usr/bin/env python3
"""
SFT vs RL Results Comparison Script
Function: Compare evaluation results of SFT and RL models, generate comparison analysis report

Usage:
    python scripts/compare_sft_rl_results.py \
        --sft_results results/sft_evaluation_results.json \
        --rl_results results/rl_evaluation_results.json \
        --output comparison_report.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import sys

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))


def setup_logging(log_level: str = "INFO"):
    """Setup logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_results(results_file: str) -> Dict[str, Any]:
    """Load evaluation results file"""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_statistics(sft_stats: Dict, rl_stats: Dict) -> Dict[str, Any]:
    """Compare statistical metrics"""
    comparison = {
        "accuracy": {
            "sft": sft_stats.get("accuracy", 0.0),
            "rl": rl_stats.get("accuracy", 0.0),
            "improvement": rl_stats.get("accuracy", 0.0) - sft_stats.get("accuracy", 0.0),
            "improvement_percentage": ((rl_stats.get("accuracy", 0.0) - sft_stats.get("accuracy", 0.0)) 
                                      / max(sft_stats.get("accuracy", 0.0), 1e-6) * 100) if sft_stats.get("accuracy", 0.0) > 0 else 0.0
        },
        "total_samples": {
            "sft": sft_stats.get("statistics", {}).get("total_samples", 0),
            "rl": rl_stats.get("statistics", {}).get("total_samples", 0)
        },
        "correct_samples": {
            "sft": sft_stats.get("statistics", {}).get("correct_samples", 0),
            "rl": rl_stats.get("statistics", {}).get("correct_samples", 0),
            "improvement": (rl_stats.get("statistics", {}).get("correct_samples", 0) - 
                           sft_stats.get("statistics", {}).get("correct_samples", 0))
        }
    }
    
    # Compare average response length
    if "average_response_length" in sft_stats.get("statistics", {}):
        comparison["average_response_length"] = {
            "sft": sft_stats["statistics"]["average_response_length"],
            "rl": rl_stats.get("statistics", {}).get("average_response_length", 0.0),
            "difference": (rl_stats.get("statistics", {}).get("average_response_length", 0.0) - 
                          sft_stats["statistics"]["average_response_length"])
        }
    
    # Compare answer extraction success rate
    if "answer_extraction_stats" in sft_stats:
        comparison["answer_extraction_success_rate"] = {
            "sft": sft_stats["answer_extraction_stats"].get("success_rate", 0.0),
            "rl": rl_stats.get("answer_extraction_stats", {}).get("success_rate", 0.0),
            "difference": (rl_stats.get("answer_extraction_stats", {}).get("success_rate", 0.0) - 
                          sft_stats["answer_extraction_stats"].get("success_rate", 0.0))
        }
    
    # Compare logical consistency (if exists)
    if "average_logical_consistency" in sft_stats.get("statistics", {}):
        comparison["average_logical_consistency"] = {
            "sft": sft_stats["statistics"]["average_logical_consistency"],
            "rl": rl_stats.get("statistics", {}).get("average_logical_consistency", 0.0),
            "difference": (rl_stats.get("statistics", {}).get("average_logical_consistency", 0.0) - 
                          sft_stats["statistics"]["average_logical_consistency"])
        }
    
    # Compare answer correctness score (if exists)
    if "average_answer_correctness_score" in sft_stats.get("statistics", {}):
        comparison["average_answer_correctness_score"] = {
            "sft": sft_stats["statistics"]["average_answer_correctness_score"],
            "rl": rl_stats.get("statistics", {}).get("average_answer_correctness_score", 0.0),
            "difference": (rl_stats.get("statistics", {}).get("average_answer_correctness_score", 0.0) - 
                          sft_stats["statistics"]["average_answer_correctness_score"])
        }
    
    # Compare reasoning steps count (if exists)
    if "average_reasoning_steps" in sft_stats.get("statistics", {}):
        comparison["average_reasoning_steps"] = {
            "sft": sft_stats["statistics"]["average_reasoning_steps"],
            "rl": rl_stats.get("statistics", {}).get("average_reasoning_steps", 0.0),
            "difference": (rl_stats.get("statistics", {}).get("average_reasoning_steps", 0.0) - 
                          sft_stats["statistics"]["average_reasoning_steps"])
        }
    
    return comparison


def compare_individual_samples(sft_results: List[Dict], rl_results: List[Dict]) -> Dict[str, Any]:
    """Compare individual sample results"""
    # Create index mapping
    sft_dict = {r["index"]: r for r in sft_results if "index" in r}
    rl_dict = {r["index"]: r for r in rl_results if "index" in r}
    
    # Find common samples
    common_indices = set(sft_dict.keys()) & set(rl_dict.keys())
    
    comparison = {
        "total_common_samples": len(common_indices),
        "samples_improved": 0,  # Number of samples where RL is correct but SFT is not
        "samples_degraded": 0,   # Number of samples where RL is incorrect but SFT is correct
        "samples_unchanged": 0,  # Number of samples with unchanged results
        "improvement_details": []
    }
    
    for idx in common_indices:
        sft_sample = sft_dict[idx]
        rl_sample = rl_dict[idx]
        
        sft_correct = sft_sample.get("is_correct", False)
        rl_correct = rl_sample.get("is_correct", False)
        
        if not sft_correct and rl_correct:
            comparison["samples_improved"] += 1
            comparison["improvement_details"].append({
                "index": idx,
                "status": "improved",
                "question": sft_sample.get("question", "")[:100] + "..." if len(sft_sample.get("question", "")) > 100 else sft_sample.get("question", "")
            })
        elif sft_correct and not rl_correct:
            comparison["samples_degraded"] += 1
            comparison["improvement_details"].append({
                "index": idx,
                "status": "degraded",
                "question": sft_sample.get("question", "")[:100] + "..." if len(sft_sample.get("question", "")) > 100 else sft_sample.get("question", "")
            })
        else:
            comparison["samples_unchanged"] += 1
    
    return comparison


def generate_comparison_report(sft_results: Dict, rl_results: Dict) -> Dict[str, Any]:
    """Generate comparison report"""
    report = {
        "comparison_time": datetime.now().isoformat(),
        "sft_model_path": sft_results.get("model_path", "N/A"),
        "rl_model_path": rl_results.get("checkpoint_path", "N/A"),
        "statistics_comparison": compare_statistics(sft_results, rl_results),
        "individual_samples_comparison": compare_individual_samples(
            sft_results.get("individual_results", []),
            rl_results.get("individual_results", [])
        )
    }
    
    return report


def print_comparison_summary(report: Dict[str, Any]):
    """Print comparison summary"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("SFT vs RL Comparison Analysis Report")
    logger.info("=" * 80)
    
    # Accuracy comparison
    acc_comp = report["statistics_comparison"]["accuracy"]
    logger.info(f"Accuracy comparison:")
    logger.info(f"  SFT: {acc_comp['sft']:.4f}")
    logger.info(f"  RL:  {acc_comp['rl']:.4f}")
    logger.info(f"  Improvement: {acc_comp['improvement']:.4f} ({acc_comp['improvement_percentage']:.2f}%)")
    
    # Sample comparison
    samples_comp = report["individual_samples_comparison"]
    logger.info(f"\nSample-level comparison:")
    logger.info(f"  Common samples: {samples_comp['total_common_samples']}")
    logger.info(f"  Improved samples: {samples_comp['samples_improved']}")
    logger.info(f"  Degraded samples: {samples_comp['samples_degraded']}")
    logger.info(f"  Unchanged samples: {samples_comp['samples_unchanged']}")
    
    # Other metrics comparison
    if "average_response_length" in report["statistics_comparison"]:
        resp_len = report["statistics_comparison"]["average_response_length"]
        logger.info(f"\nAverage response length:")
        logger.info(f"  SFT: {resp_len['sft']:.2f} characters")
        logger.info(f"  RL:  {resp_len['rl']:.2f} characters")
        logger.info(f"  Difference: {resp_len['difference']:.2f} characters")
    
    if "average_logical_consistency" in report["statistics_comparison"]:
        consistency = report["statistics_comparison"]["average_logical_consistency"]
        logger.info(f"\nAverage logical consistency:")
        logger.info(f"  SFT: {consistency['sft']:.4f}")
        logger.info(f"  RL:  {consistency['rl']:.4f}")
        logger.info(f"  Difference: {consistency['difference']:.4f}")
    
    logger.info("=" * 80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Compare evaluation results of SFT and RL models")
    parser.add_argument("--sft_results", type=str, required=True,
                       help="SFT model evaluation results JSON file path")
    parser.add_argument("--rl_results", type=str, required=True,
                       help="RL model evaluation results JSON file path")
    parser.add_argument("--output", type=str, default="results/comparison_report.json",
                       help="Comparison report output file path")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load results files
        logger.info(f"Loading SFT results: {args.sft_results}")
        sft_results = load_results(args.sft_results)
        
        logger.info(f"Loading RL results: {args.rl_results}")
        rl_results = load_results(args.rl_results)
        
        # Generate comparison report
        logger.info("Generating comparison report...")
        report = generate_comparison_report(sft_results, rl_results)
        
        # Print summary
        print_comparison_summary(report)
        
        # Save report
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Comparison report saved: {output_path}")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()







