#!/usr/bin/env python3
"""
Supervised Fine-tuning Training Script
Function: Supervised fine-tuning of Qwen-7B-math on GSM8K dataset
"""

import argparse
import yaml
import logging
import os
from pathlib import Path
import sys

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from training.sft_trainer import SFTTrainer
from data.gsm8k_processor import GSM8KProcessor
from datasets import load_dataset


def setup_logging(log_level: str = "INFO"):
    """Setup logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> dict:
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict):
    """Prepare data"""
    # Load GSM8K dataset
    dataset = load_dataset("gsm8k", "main")
    
    # Data limits (if configured)
    max_train_samples = config["data"].get("max_train_samples")
    max_eval_samples = config["data"].get("max_eval_samples")
    
    if max_train_samples:
        dataset["train"] = dataset["train"].select(range(min(max_train_samples, len(dataset["train"]))))
    
    if max_eval_samples:
        dataset["test"] = dataset["test"].select(range(min(max_eval_samples, len(dataset["test"]))))
    
    print(f"Training set size: {len(dataset['train'])}")
    print(f"Test set size: {len(dataset['test'])}")
    
    return dataset


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Supervised fine-tuning training")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (overrides config file setting)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Log level")
    # Note: SFT training uses Transformers Trainer, will automatically detect and resume from checkpoints
    # If output directory contains checkpoint-* folders, will automatically resume from latest checkpoint
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override output directory
        if args.output_dir:
            config["sft"]["output_dir"] = args.output_dir
        
        # Create output directory
        output_dir = config["sft"]["output_dir"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting supervised fine-tuning training")
        logger.info(f"Configuration: {config}")
        
        # Prepare data
        logger.info("Preparing dataset...")
        dataset = prepare_data(config)
        
        # Create trainer
        logger.info("Initializing trainer...")
        trainer = SFTTrainer(config)
        
        # Setup model
        logger.info("Setting up model...")
        trainer.setup_model()
        
        # Start training
        logger.info("Starting training...")
        train_dataset = dataset[config["data"]["train_split"]]
        eval_dataset = dataset[config["data"]["eval_split"]]
        
        trainer.train(train_dataset, eval_dataset)
        
        # Final evaluation
        logger.info("Performing final evaluation...")
        eval_results = trainer.evaluate(eval_dataset)
        logger.info(f"Final evaluation results: {eval_results}")
        
        # Save evaluation results
        eval_results_file = os.path.join(output_dir, "eval_results.yaml")
        with open(eval_results_file, 'w', encoding='utf-8') as f:
            yaml.dump(eval_results, f, default_flow_style=False)
        
        logger.info("Supervised fine-tuning training completed!")
        logger.info(f"Model saved at: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()






