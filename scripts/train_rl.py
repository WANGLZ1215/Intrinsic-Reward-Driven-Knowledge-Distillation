#!/usr/bin/env python3
"""
Reinforcement Learning Training Script
Function: PPO training based on intrinsic rewards
"""

import argparse
import yaml
import logging
import os
from pathlib import Path
import sys
from tqdm import tqdm
import time

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from training.rl_trainer import RLTrainer
from data.gsm8k_processor import GSM8KProcessor
from datasets import load_dataset


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging"""
    import os
    from pathlib import Path
    from datetime import datetime
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # If no log file is specified, use default name
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"rl_training_{timestamp}.log"
    
    # Configure log format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configure logging to output to console and file simultaneously
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # File output
            logging.StreamHandler()  # Console output
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")


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
    parser = argparse.ArgumentParser(description="Reinforcement Learning Training")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (overrides config file setting)")
    parser.add_argument("--student_model_path", type=str, default=None,
                       help="Student model path (SFT model)")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Maximum training steps")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Log level")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log file path (default: logs/rl_training_YYYYMMDD_HHMMSS.log)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file if hasattr(args, 'log_file') and args.log_file else None
    setup_logging(args.log_level, log_file)
    logger = logging.getLogger(__name__)
    
    # Overall progress bar (6 steps if resuming from checkpoint, otherwise 5)
    total_steps = 6 if args.resume_from_checkpoint else 5
    main_progress = tqdm(total=total_steps, desc="RL Training Overall Progress", ncols=100, position=0)
    
    try:
        start_time = time.time()
        
        # Step 1: Load configuration
        main_progress.set_description("📋 Loading Configuration")
        config = load_config(args.config)
        
        # Override configuration
        if args.output_dir:
            config["ppo"]["output_dir"] = args.output_dir
        
        if args.student_model_path:
            config["model"]["student_model_name"] = args.student_model_path
        
        if args.max_steps:
            config["training"]["max_steps"] = args.max_steps
        
        # Create output directory
        output_dir = config["ppo"]["output_dir"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        main_progress.update(1)
        main_progress.set_postfix({"status": "Configuration loaded"})
        
        # Step 2: Prepare data
        main_progress.set_description("📊 Preparing Data")
        logger.info("Preparing dataset...")
        dataset = prepare_data(config)
        main_progress.update(1)
        main_progress.set_postfix({"status": "Data prepared"})
        
        # Step 3: Create trainer
        main_progress.set_description("🏗️ Initializing Trainer")
        logger.info("Initializing RL trainer...")
        trainer = RLTrainer(config)
        main_progress.update(1)
        main_progress.set_postfix({"status": "Trainer initialized"})
        
        # Step 4: Setup models and components
        main_progress.set_description("⚙️ Setting Up Models and Components")
        logger.info("Setting up models...")
        trainer.setup_models()
        trainer.setup_components()
        trainer.setup_ppo_trainer()
        main_progress.update(1)
        main_progress.set_postfix({"status": "Models and components set up"})
        
        # Step 5: Resume from checkpoint if needed
        if args.resume_from_checkpoint:
            main_progress.set_description("🔄 Resuming Checkpoint")
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            trainer.load_checkpoint(args.resume_from_checkpoint)
            main_progress.update(1)
            main_progress.set_postfix({"status": "Checkpoint resumed"})
        
        # Step 6: Start training
        main_progress.set_description("🚀 Starting Training")
        logger.info("Starting RL training...")
        train_dataset = dataset[config["data"]["train_split"]]
        
        trainer.train(train_dataset, max_steps=config["training"]["max_steps"])
        main_progress.update(1)
        main_progress.set_postfix({"status": "Training completed"})
        
        # Training completion statistics
        total_time = time.time() - start_time
        main_progress.close()
        
        print(f"\n🎉 Reinforcement learning training completed!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"📁 Model saved at: {output_dir}")
        
        logger.info("Reinforcement learning training completed!")
        logger.info(f"Model saved to: {output_dir}")
        
    except Exception as e:
        main_progress.close()
        logger.error(f"RL training failed: {e}")
        raise
    
    finally:
        # Cleanup resources
        if 'trainer' in locals():
            trainer.cleanup()


if __name__ == "__main__":
    main()






