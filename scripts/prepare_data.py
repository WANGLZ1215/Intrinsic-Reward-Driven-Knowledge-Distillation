#!/usr/bin/env python3
"""
Data Preparation Script
Function: Download GSM8K dataset and display basic information
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
from data.gsm8k_processor import GSM8KProcessor


def setup_logging(log_level: str = "INFO"):
    """Setup logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download GSM8K dataset")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Log level")
    parser.add_argument("--show_samples", type=int, default=3,
                       help="Number of samples to display")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting to download GSM8K dataset...")
        
        # Download dataset
        dataset = load_dataset("gsm8k", "main")
        
        logger.info(f"✅ Dataset downloaded successfully!")
        logger.info(f"📊 Training set size: {len(dataset['train'])} samples")
        logger.info(f"📊 Test set size: {len(dataset['test'])} samples")
        
        # Display samples
        if args.show_samples > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Displaying {args.show_samples} training set samples:")
            logger.info(f"{'='*60}")
            
            for i in range(min(args.show_samples, len(dataset['train']))):
                sample = dataset['train'][i]
                logger.info(f"\nSample {i+1}:")
                logger.info(f"Question: {sample['question'][:100]}...")
                logger.info(f"Answer: {sample['answer'][:100]}...")
        
        logger.info("\n✅ Data preparation completed! Dataset is cached and can be used directly in training scripts.")
        
    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        raise


if __name__ == "__main__":
    main()






