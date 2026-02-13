"""
Supervised Fine-tuning Trainer
Function: Implement supervised fine-tuning of Qwen-7B-math on GSM8K dataset
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import yaml
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import wandb
from torch.utils.data import DataLoader
from utils.cache_utils import suppress_past_key_values_warning, update_model_for_modern_cache
from data.gsm8k_processor import build_prompt


class SFTTrainer:
    """Supervised Fine-tuning Trainer"""
    
    def __init__(self, config: Dict):
        """
        Initialize SFT trainer
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Suppress past_key_values warnings
        suppress_past_key_values_warning()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize wandb (if enabled)
        if config.get("logging", {}).get("use_wandb", False):
            wandb.init(
                project=config["logging"]["wandb_project"],
                config=config
            )
    
    def setup_model(self):
        """Setup model and tokenizer"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model"]["student_model_name"],
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model"]["student_model_name"],
                torch_dtype=getattr(torch, self.config["device"]["torch_dtype"]),
                device_map=self.config["device"]["device_map"],
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 🔥 Critical fix: Check and fix embedding size to match tokenizer (before applying LoRA)
            # This prevents vocab_size mismatch during SFT training and avoids issues when loading checkpoints in RL stage
            tokenizer_vocab_size = len(self.tokenizer)
            try:
                input_emb_size = self.model.get_input_embeddings().weight.size(0)
                output_emb_size = None
                if hasattr(self.model, 'get_output_embeddings') and self.model.get_output_embeddings() is not None:
                    output_emb_size = self.model.get_output_embeddings().weight.size(0)
                
                self.logger.info(f"📊 Embedding size check (before applying LoRA):")
                self.logger.info(f"   tokenizer vocab_size: {tokenizer_vocab_size}")
                self.logger.info(f"   model input_embeddings.size(0): {input_emb_size}")
                if output_emb_size is not None:
                    self.logger.info(f"   model output_embeddings.size(0): {output_emb_size}")
                self.logger.info(f"   model.config.vocab_size: {getattr(self.model.config, 'vocab_size', 'N/A')}")
                
                # If embedding size doesn't match tokenizer, perform resize
                if input_emb_size != tokenizer_vocab_size:
                    self.logger.warning(f"⚠️ Model embedding size ({input_emb_size}) != tokenizer size ({tokenizer_vocab_size})")
                    self.logger.info(f"   Resizing token embeddings to {tokenizer_vocab_size}...")
                    self.model.resize_token_embeddings(tokenizer_vocab_size)
                    self.logger.info(f"✅ resize_token_embeddings completed")
                    
                    # Verify if resize was successful
                    new_input_emb_size = self.model.get_input_embeddings().weight.size(0)
                    if new_input_emb_size != tokenizer_vocab_size:
                        self.logger.error(f"❌ resize_token_embeddings failed! New size: {new_input_emb_size} != {tokenizer_vocab_size}")
                        raise ValueError(f"Resize failed: {new_input_emb_size} != {tokenizer_vocab_size}")
                    else:
                        self.logger.info(f"✅ Resize verification successful: input_embeddings.size(0) = {new_input_emb_size}")
                else:
                    self.logger.info(f"✅ Embedding size matches tokenizer, no resize needed")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Error during resize_token_embeddings (possibly unsupported or quantized): {e}")
                # If ValueError (resize failed), should raise exception
                if isinstance(e, ValueError):
                    raise
                # Other errors (e.g., model doesn't support resize), log warning but continue
            
            # Apply LoRA (now embedding size is matched)
            if self.config["model"].get("use_lora", True):
                lora_config = LoraConfig(**self.config["lora"])
                self.model = get_peft_model(self.model, lora_config)
                self.logger.info("LoRA configuration applied")
                
                # 🔥 Critical: Re-verify embedding size after applying LoRA (LoRA shouldn't change embedding size, but check)
                try:
                    final_input_emb_size = self.model.get_input_embeddings().weight.size(0)
                    if final_input_emb_size != tokenizer_vocab_size:
                        self.logger.warning(f"⚠️ After applying LoRA, input_embeddings size ({final_input_emb_size}) != tokenizer ({tokenizer_vocab_size})")
                        self.logger.info(f"   Re-resizing token embeddings to {tokenizer_vocab_size}...")
                        try:
                            self.model.resize_token_embeddings(tokenizer_vocab_size)
                            new_final_input_emb_size = self.model.get_input_embeddings().weight.size(0)
                            if new_final_input_emb_size == tokenizer_vocab_size:
                                self.logger.info(f"✅ LoRA post-resize successful: input_embeddings = {new_final_input_emb_size}")
                            else:
                                self.logger.error(f"❌ LoRA post-resize failed: {new_final_input_emb_size} != {tokenizer_vocab_size}")
                        except Exception as e2:
                            self.logger.warning(f"⚠️ LoRA post-resize failed: {e2}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Unable to check embedding size after LoRA: {e}")
            
            # Update model to use modern cache
            self.model = update_model_for_modern_cache(self.model)
            
            self.logger.info("Model setup completed")
            
        except Exception as e:
            self.logger.error(f"Model setup failed: {e}")
            raise
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset"""
        def preprocess_function(examples):
            texts = []
            for question, answer in zip(examples["question"], examples["answer"]):
                prompt = build_prompt(question)
                full_text = prompt + answer + self.tokenizer.eos_token
                texts.append(full_text)
            return {"text": texts}
        
        # Preprocess data
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.config["sft"]["max_length"],
                return_tensors="pt"
            )
        
        tokenized_dataset = processed_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        self.logger.info(f"Dataset preparation completed: {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def create_data_collator(self):
        """Create data collator"""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
    
    def setup_training_arguments(self):
        """Setup training arguments"""
        return TrainingArguments(
            output_dir=self.config["sft"]["output_dir"],
            per_device_train_batch_size=self.config["sft"]["per_device_train_batch_size"],
            per_device_eval_batch_size=self.config["sft"]["per_device_eval_batch_size"],
            num_train_epochs=self.config["sft"]["num_train_epochs"],
            learning_rate=float(self.config["sft"]["learning_rate"]),
            save_strategy=self.config["sft"]["save_strategy"],
            eval_strategy=self.config["sft"].get("eval_strategy", self.config["sft"].get("evaluation_strategy", "epoch")),
            logging_steps=self.config["sft"]["logging_steps"],
            save_total_limit=self.config["sft"]["save_total_limit"],
            load_best_model_at_end=self.config["sft"]["load_best_model_at_end"],
            metric_for_best_model=self.config["sft"]["metric_for_best_model"],
            greater_is_better=self.config["sft"]["greater_is_better"],
            warmup_steps=self.config["sft"]["warmup_steps"],
            fp16=self.config["training"]["fp16"],
            bf16=self.config["training"].get("bf16", False),  # Add BF16 support
            dataloader_num_workers=self.config["training"]["dataloader_num_workers"],
            remove_unused_columns=self.config["training"]["remove_unused_columns"],
            report_to="wandb" if self.config.get("logging", {}).get("use_wandb", False) else None,
        )
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Start training"""
        try:
            # Prepare data
            train_dataset = self.prepare_dataset(train_dataset)
            if eval_dataset is not None:
                eval_dataset = self.prepare_dataset(eval_dataset)
            
            # Create data collator
            data_collator = self.create_data_collator()
            
            # Setup training arguments
            training_args = self.setup_training_arguments()
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
            
            # Start training
            self.logger.info("Starting SFT training...")
            self.trainer.train()
            
            # Save final model
            self.save_model(self.config["sft"]["output_dir"])
            
            self.logger.info("SFT training completed")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self, save_path: str):
        """Save model"""
        try:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            if hasattr(self.model, 'save_pretrained'):
                # Save LoRA adapter
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
            else:
                # Save full model
                torch.save(self.model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
                self.tokenizer.save_pretrained(save_path)
            
            self.logger.info(f"Model saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Model save failed: {e}")
            raise
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Evaluate model"""
        if self.trainer is None:
            raise ValueError("Trainer not initialized")
        
        # Prepare evaluation data
        eval_dataset = self.prepare_dataset(eval_dataset)
        
        # Evaluate
        eval_results = self.trainer.evaluate(eval_dataset)
        
        self.logger.info(f"Evaluation results: {eval_results}")
        return eval_results
    
    def generate_sample(self, prompt: str, max_length: int = 256) -> str:
        """Generate sample"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized")
        
        self.model.eval()
        
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text


def load_config(config_path: str) -> Dict:
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main function"""
    # Load configuration
    config = load_config("config/training_config.yaml")
    
    # Create trainer
    trainer = SFTTrainer(config)
    
    # Setup model
    trainer.setup_model()
    
    # Load GSM8K dataset
    from datasets import load_dataset
    from data.gsm8k_processor import GSM8KProcessor
    
    print("Loading GSM8K dataset...")
    try:
        # Load GSM8K dataset
        gsm8k_dataset = load_dataset("gsm8k", "main")
        
        # Create GSM8K processor
        processor = GSM8KProcessor(trainer.tokenizer, max_length=config["sft"]["max_length"])
        
        # Use training set as training data
        train_dataset = gsm8k_dataset["train"]
        
        # Use test set as validation set (full data)
        eval_dataset = gsm8k_dataset["test"]
        
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(eval_dataset)}")
        
        # Validate dataset quality
        processor.validate_data(train_dataset, num_samples=3)
        processor.validate_data(eval_dataset, num_samples=3)
        
    except Exception as e:
        print(f"Failed to load GSM8K dataset: {e}")
        print("Cannot proceed with training, please check network connection and dependencies")
        return
    
    # Train
    trainer.train(train_dataset, eval_dataset)
    
    # Evaluate
    eval_results = trainer.evaluate(eval_dataset)
    print(f"Final evaluation results: {eval_results}")
    
    # Generate samples
    sample_questions = [
        "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many meters does he run a week?",
        "A robe takes 2 bolts of blue fabric and half that much white fabric. How many bolts of fabric does it take?",
        "Josh decides to try flipping a house. He buys it for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?"
    ]
    
    print("\n=== Sample Generation Test ===")
    for i, sample_question in enumerate(sample_questions, 1):
        sample_prompt = build_prompt(sample_question)
        print(f"\nSample {i}:")
        print(sample_prompt)
        sample_response = trainer.generate_sample(sample_prompt, max_length=200)
        print(f"Generated Answer: {sample_response}")
        print("-" * 50)


if __name__ == "__main__":
    main()





