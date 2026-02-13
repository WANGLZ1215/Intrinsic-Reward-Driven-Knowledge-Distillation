"""
Reinforcement Learning Trainer
Function: Implement PPO training based on intrinsic rewards
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import yaml
import logging
import os
import gc
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import wandb
from collections import deque
from tqdm import tqdm
import time

from models.teacher_model import TeacherModel
from models.student_model import StudentModel
from models.cache_manager import CacheManager
from rewards.intrinsic_reward import IntrinsicRewardComputer
from rewards.reward_normalizer import RewardNormalizer
from rewards.reward_combiner import RewardCombiner
from data.gsm8k_processor import GSM8KProcessor
from utils.math_utils import extract_final_answer, is_answer_correct
from training.ppo_utils import (
    ParallelRewardProcessor, ParallelModelInference, 
    AsyncCacheManager, ParallelDataLoader,
    create_parallel_processor, create_parallel_inference, create_async_cache_manager,
    compute_grad_norm  # For checking gradients
)
import functools
from utils.cache_utils import suppress_past_key_values_warning, update_model_for_modern_cache


def handle_errors(func):
    """Error handling decorator"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get logger instance
            if hasattr(args[0], 'logger'):
                logger = args[0].logger
            else:
                logger = logging.getLogger(__name__)
            
            logger.error(f"❌ Function {func.__name__} execution failed: {e}")
            import traceback
            logger.error(f"Detailed error information: {traceback.format_exc()}")
            
            # Decide whether to re-raise based on error type
            if isinstance(e, (ValueError, TypeError, KeyError)):
                raise
            else:
                logger.error(f"❌ Unknown error type, continuing execution")
                return None
    
    return wrapper


def validate_data_batch(batch: Dict) -> bool:
    """Validate data batch"""
    required_keys = ["input_ids", "attention_mask"]
    
    for key in required_keys:
        if key not in batch:
            print(f"❌ Missing required batch key: {key}")
            return False
        
        if not isinstance(batch[key], torch.Tensor):
            print(f"❌ Batch key {key} is not a tensor")
            return False
        
        if batch[key].numel() == 0:
            print(f"❌ Batch key {key} is empty")
            return False
    
    # Check batch size consistency
    batch_size = batch["input_ids"].shape[0]
    for key in required_keys:
        if batch[key].shape[0] != batch_size:
            print(f"❌ Batch size inconsistent: {key}")
            return False
    
    return True


def validate_config(config: Dict) -> Dict:
    """Validate and supplement configuration"""
    # Default configuration
    default_config = {
        "model": {
            "teacher_model_name": "Qwen/Qwen2.5-32B-Instruct",
            "student_model_name": "Qwen/Qwen2.5-7B-Instruct",
            "cache_size": 10000,
            "cache_policy": "LRU",
            "use_lora": True
        },
        "device": {
            "device_map": "auto",
            "torch_dtype": "bfloat16"
        },
        "reward": {
            "temperature": 1.0,
            "normalization": "mean_std",
            "lambda_intrinsic": 0.7,
            "lambda_correctness": 0.3,
            "update_rate": 0.01,
            "clip_min": -5.0,
            "clip_max": 5.0,
            "use_adaptive_weights": True,
            "adaptation_rate": 0.01,
            "reasoning_weight": 0.0,
            "format_weight": 0.0
        },
        "ppo": {
            "learning_rate": 2e-5,
            "batch_size": 4,
            "mini_batch_size": 2,
            "ppo_epochs": 4,
            "cliprange": 0.2,
            "vf_coef": 0.1,
            "entropy_coef": 0.01,
            "init_kl_coef": 0.05,
            "gamma": 0.99,
            "lam": 0.95,
            "max_grad_norm": 1.0,
            "max_length": 480,
            "output_dir": "./checkpoints/rl_model"
        },
        "generation": {
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 0,
            "top_p": 1.0,
            "max_new_tokens": 480
        },
        "training": {
            "max_steps": 1000,
            "save_steps": 50,
            "eval_steps": 100,
            "logging_steps": 10
        },
        "parallel": {
            "enabled": True,
            "num_workers": 4,
            "use_threads": True,
            "inference_batch_size": 16,
            "cache_queue_size": 1000,
            "use_parallel_data_loader": True,
            "data_loader_workers": 4
        },
        "logging": {
            "use_wandb": False,
            "wandb_project": "intrinsic-reward-distillation",
            "use_tensorboard": True,
            "tensorboard_log_dir": "./logs"
        }
    }
    
    # Recursively merge configuration
    def merge_config(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_config(base[key], value)
            else:
                base[key] = value
        return base
    
    return merge_config(default_config, config)


class RLTrainer:
    """Reinforcement Learning Trainer"""
    
    def __init__(self, config: Dict):
        """
        Initialize RL trainer
        
        Args:
            config: Training configuration
        """
        self.config = validate_config(config)  # Validate and supplement configuration
        
        # Suppress past_key_values warnings
        suppress_past_key_values_warning()
        
        self.teacher_model = None
        self.student_model = None
        self.ppo_model = None  # PPO model (with ValueHead)
        self.ppo_trainer = None
        self.cache_manager = None
        
        # Reward computation components
        self.intrinsic_computer = None
        self.reward_normalizer = None
        self.reward_combiner = None
        
        # Data processor
        self.data_processor = None
        
        # Parallel processing components
        self.parallel_processor = None
        self.parallel_inference_student = None
        self.parallel_inference_teacher = None
        self.async_cache_manager = None
        self.parallel_data_loader = None
        
        # Training statistics
        self.training_stats = {
            "step": 0,
            "total_rewards": [],
            "intrinsic_rewards": [],
            "correctness_rewards": [],
            "combined_rewards": [],
            "policy_losses": [],
            "value_losses": [],
            "kl_divergences": []
        }
        
        # Memory management - automatically adjust cleanup frequency based on GPU VRAM
        # For large VRAM GPUs like H200 140GB, can reduce cleanup frequency to improve speed
        # Note: CUDA may not be ready during initialization, set default values first
        self._memory_cleanup_interval = 3
        self._force_cleanup_every_n_steps = 2
        self._last_cleanup_step = 0
        self._vram_detected = False  # Flag whether VRAM has been detected
        
        # Performance optimization
        self._use_mixed_precision = self.config.get("device", {}).get("use_mixed_precision", True)
        self._gradient_accumulation_steps = self.config.get("training", {}).get("gradient_accumulation_steps", 1)
        self._gradient_accumulation_count = 0
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Cache parallel configuration check result
        # Note: Parallel processing disabled by default to avoid tokenizer thread safety issues
        self._use_parallel = self.config.get("parallel", {}).get("enabled", False)

    def _sync_policy_to_student(self):
        """
        Sync PPO policy model weights back to StudentModel (LoRA adapter) for saving/evaluation with latest policy.
        """
        try:
            if self.ppo_trainer is None or self.student_model is None:
                return

            policy_model = getattr(self.ppo_trainer, "model", None)
            if policy_model is None:
                return

            # AutoModelForCausalLMWithValueHead stores base policy model in pretrained_model field
            policy_base = getattr(policy_model, "pretrained_model", policy_model)
            target_model = getattr(self.student_model, "model", None)
            if target_model is None:
                return

            self.logger.debug("Syncing PPO policy weights to student model...")
            with torch.no_grad():
                target_model.load_state_dict(policy_base.state_dict(), strict=False)
        except Exception as sync_error:
            self.logger.error(f"Policy to student model sync failed: {sync_error}")

    def _sync_student_to_policy(self):
        """
        Sync StudentModel latest weights back to PPO policy model (e.g., needed when resuming training after loading from disk).
        """
        try:
            if self.ppo_trainer is None or self.student_model is None:
                return

            policy_model = getattr(self.ppo_trainer, "model", None)
            if policy_model is None:
                return

            policy_base = getattr(policy_model, "pretrained_model", policy_model)
            source_model = getattr(self.student_model, "model", None)
            if source_model is None:
                return

            self.logger.debug("Syncing student model weights to PPO policy...")
            with torch.no_grad():
                policy_base.load_state_dict(source_model.state_dict(), strict=False)
        except Exception as sync_error:
            self.logger.error(f"Student model to policy sync failed: {sync_error}")
    
    def _create_progress_bar(self, iterable, desc: str, total: int = None, unit: str = "sample"):
        """Create standardized progress bar"""
        return tqdm(
            iterable,
            total=total or len(iterable) if hasattr(iterable, '__len__') else None,
            desc=desc,
            unit=unit,
            ncols=80,
            leave=False
        )
    
    def _cleanup_memory(self, step: int, force: bool = False):
        """Clean up memory"""
        should_cleanup = force or (step - self._last_cleanup_step >= self._memory_cleanup_interval)
        
        if should_cleanup:
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear Python garbage collection
            gc.collect()
            
            # Clean training statistics (keep recent data)
            for key in ["total_rewards", "intrinsic_rewards", "correctness_rewards", 
                       "combined_rewards", "policy_losses", "value_losses", "kl_divergences"]:
                if len(self.training_stats[key]) > 500:
                    self.training_stats[key] = self.training_stats[key][-500:]
            
            # Log cache statistics
            if self.cache_manager:
                cache_stats = self.cache_manager.get_stats()
                self.logger.info(f"📊 Cache statistics (CacheManager): hit_rate={cache_stats['hit_rate']:.3f}, "
                               f"size={cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
            
            # ✅ Fix: Teacher model also has internal cache, log its statistics
            if self.teacher_model and hasattr(self.teacher_model, 'get_cache_stats'):
                teacher_cache_stats = self.teacher_model.get_cache_stats()
                self.logger.info(f"📊 Cache statistics (Teacher internal): hit_rate={teacher_cache_stats['hit_rate']:.3f}, "
                               f"size={teacher_cache_stats['cache_size']}/{teacher_cache_stats['max_cache_size']}")
            
            self._last_cleanup_step = step
            if not force:  # Don't log during forced cleanup to avoid excessive logging
                self.logger.info(f"🧹 Memory cleanup completed (step {step})")
    
    def cleanup_resources(self):
        """Clean up all resources"""
        try:
            # Clean models
            if self.teacher_model:
                del self.teacher_model
                self.teacher_model = None
            
            if self.student_model:
                del self.student_model
                self.student_model = None
            
            if self.ppo_trainer:
                del self.ppo_trainer
                self.ppo_trainer = None
            
            # Clean cache
            if self.cache_manager:
                self.cache_manager.clear()  # Use correct method name
                del self.cache_manager
                self.cache_manager = None
            
            # Clean parallel components
            if self.parallel_processor:
                del self.parallel_processor
                self.parallel_processor = None
            
            if self.parallel_inference_student:
                del self.parallel_inference_student
                self.parallel_inference_student = None
            
            if self.parallel_inference_teacher:
                del self.parallel_inference_teacher
                self.parallel_inference_teacher = None
            
            if self.async_cache_manager:
                del self.async_cache_manager
                self.async_cache_manager = None
            
            if self.parallel_data_loader:
                del self.parallel_data_loader
                self.parallel_data_loader = None
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("🧹 All resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"❌ Resource cleanup failed: {e}")
            # Log detailed error information
            import traceback
            self.logger.error(f"Detailed error information: {traceback.format_exc()}")
        
        # Initialize wandb (if enabled)
        if self.config.get("logging", {}).get("use_wandb", False):
            wandb.init(
                project=self.config["logging"]["wandb_project"],
                config=self.config
            )
    
    def setup_models(self):
        """Setup teacher and student models"""
        # Note: Don't use @handle_errors, need to ensure exceptions are raised on failure, not return None
        try:
            self.logger.info("🚀 Starting model setup...")
            
            # Check GPU count and decide model allocation strategy
            num_gpus = torch.cuda.device_count()
            self.logger.info(f"📊 Detected {num_gpus} GPU devices")
            
            # 🎯 Detect GPU VRAM size and adjust cleanup frequency (H200 optimization)
            if num_gpus >= 1 and not self._vram_detected:
                try:
                    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    self.logger.info(f"📊 GPU 0 VRAM: {total_vram_gb:.1f}GB")
                    if total_vram_gb >= 120:  # H200 140GB or similar large VRAM GPU
                        self._memory_cleanup_interval = 5  # 🔥 Urgent fix: reduced from 50 to 5, more frequent cleanup
                        self._force_cleanup_every_n_steps = 3  # 🔥 Urgent fix: reduced from 50 to 3
                        self.logger.info("⚠️ Detected H200 or similar large VRAM GPU, but using conservative cleanup strategy (cleanup every 5 steps) to avoid OOM")
                    else:  # A100 80GB or smaller VRAM
                        self._memory_cleanup_interval = 3
                        self._force_cleanup_every_n_steps = 2
                        self.logger.info(f"📊 Using conservative cleanup strategy (cleanup every 3 steps)")
                    self._vram_detected = True
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to detect GPU VRAM, using default strategy: {e}")
            
            # Choose optimal allocation strategy based on GPU count
            if num_gpus >= 4:
                # 4 GPUs or more: Teacher distributed across GPU 0,1 automatically, Student on GPU 2, GPU 3 reserved/cache
                # Use max_memory to limit Teacher model allocation to GPU 0 and 1 only
                # This ensures Student model can safely use GPU 2
                import os
                # Set max_memory limit, only allow model allocation on GPU 0 and 1
                max_memory = {
                    0: "75GB",  # GPU 0: Reserve 5GB system VRAM
                    1: "75GB",  # GPU 1: Reserve 5GB system VRAM
                }
                teacher_device_map = "auto"  # Use with max_memory, automatically allocate to GPU 0 and 1
                # Temporarily set max_memory environment variable (if HuggingFace supports)
                # Note: In actual use, need to pass max_memory parameter in from_pretrained
                student_device_map = "cuda:2"  # Student model (7B) on GPU 2
                self.logger.info("✅ 4-GPU configuration: Teacher→GPU 0,1 (auto-balanced distribution), Student→GPU 2, GPU 3 reserved/cache")
                self.logger.info("   VRAM allocation: Teacher(32B) ~70GB, Student(7B+PPO) ~40-50GB, remaining ~200GB safety margin")
                # Store max_memory for later use
                self._teacher_max_memory = max_memory
            elif num_gpus >= 2:
                # 2 GPUs: Teacher single GPU 1, Student single GPU 0
                # 🎯 Note: Model parallelism (across GPUs) increases communication overhead and latency
                # Single 140GB H200 is sufficient for Teacher 32B (64GB) + Student 7B (14GB)
                teacher_device_map = "cuda:1"  # Teacher 32B single GPU 1
                student_device_map = "cuda:0"  # Student 7B + PPO single GPU 0
                self.logger.info("✅ 2-GPU configuration: Student+PPO→GPU 0, Teacher→GPU 1")
                self.logger.info("   VRAM: GPU 0 ~50GB, GPU 1 ~70GB, both sufficient")
                self.logger.info("   💡 If OOM occurs, may be due to batch_size too large, not insufficient single GPU capacity")
            else:
                # Single GPU: Explicitly specify cuda:0 to avoid HF default to CPU
                if num_gpus >= 1:
                    teacher_device_map = "cuda:0"
                    student_device_map = "cuda:0"
                    self.logger.info("✅ Single GPU configuration: Both Teacher and Student on cuda:0")
                else:
                    teacher_device_map = None
                    student_device_map = None
                    self.logger.warning("⚠️ No GPU detected, using CPU device")
            
            # Load teacher model
            self.logger.info("📚 Loading teacher model...")
            with tqdm(total=3, desc="Teacher Model Loading", ncols=80) as pbar:
                from models.teacher_model import TeacherModel
                # Prepare Teacher model initialization parameters
                teacher_kwargs = {
                    "model_name": self.config["model"]["teacher_model_name"],
                    "cache_size": self.config["model"]["cache_size"],
                    "cache_policy": self.config["model"]["cache_policy"],
                    "device": teacher_device_map,  # Use explicitly allocated GPU
                    "torch_dtype": getattr(torch, self.config["device"]["torch_dtype"])
                }
                # If 4-GPU configuration, pass max_memory limit
                if num_gpus >= 4 and hasattr(self, '_teacher_max_memory'):
                    teacher_kwargs["max_memory"] = self._teacher_max_memory
                
                self.teacher_model = TeacherModel(**teacher_kwargs)
                pbar.update(1)
                pbar.set_postfix({"status": "Teacher model loaded"})
                
                # Check Teacher model actual distribution
                if hasattr(self.teacher_model.model, 'hf_device_map'):
                    device_map = self.teacher_model.model.hf_device_map
                    self.logger.info(f"📊 Teacher model device distribution: {device_map}")
                    if isinstance(teacher_device_map, str) and teacher_device_map.startswith("cuda"):
                        try:
                            target_device = torch.device(teacher_device_map)
                            self.teacher_model.model = self.teacher_model.model.to(target_device)
                            self.logger.info(f"✅ Teacher model forced to device: {target_device}")
                        except Exception as move_err:
                            self.logger.warning(f"⚠️ Failed to move Teacher model to {teacher_device_map}: {move_err}")

                    # If CPU remnants exist, force model to specified device
                    if isinstance(teacher_device_map, str) and teacher_device_map.startswith("cuda"):
                        try:
                            target_device = torch.device(teacher_device_map)
                            self.teacher_model.model = self.teacher_model.model.to(target_device)
                            self.logger.info(f"✅ Teacher model forced to device: {target_device}")
                        except Exception as move_err:
                            self.logger.warning(f"⚠️ Failed to move Teacher model to {teacher_device_map}: {move_err}")
                    
                    # ⚠️ Check device allocation balance (only for 4-GPU configuration)
                    if num_gpus >= 4 and isinstance(device_map, dict):
                        gpu_0_layers = sum(1 for v in device_map.values() if v == 0 or (isinstance(v, (list, tuple)) and 0 in v))
                        gpu_1_layers = sum(1 for v in device_map.values() if v == 1 or (isinstance(v, (list, tuple)) and 1 in v))
                        total_layers = gpu_0_layers + gpu_1_layers
                        if total_layers > 0:
                            balance_ratio = min(gpu_0_layers, gpu_1_layers) / max(gpu_0_layers, gpu_1_layers)
                            if balance_ratio < 0.7:  # If allocation imbalance exceeds 30%
                                self.logger.warning(f"⚠️ Teacher model device allocation unbalanced: GPU 0 has {gpu_0_layers} layers, GPU 1 has {gpu_1_layers} layers (balance {balance_ratio:.2%})")
                                self.logger.warning("   Recommend checking VRAM usage, may need manual device_map adjustment")
                
                # Load student model
                self.logger.info("🎓 Loading student model...")
                from models.student_model import StudentModel
                self.student_model = StudentModel(
                    model_name=self.config["model"]["student_model_name"],
                    lora_config=self.config["lora"],
                    device=student_device_map,  # Use explicitly allocated GPU
                    torch_dtype=getattr(torch, self.config["device"]["torch_dtype"]),
                    use_lora=True
                )
                pbar.update(1)
                pbar.set_postfix({"status": "Student model loaded"})
                
                # 🔥 Critical validation: Ensure teacher and student use same tokenizer or same size
                teacher_tok_size = len(self.teacher_model.tokenizer)
                student_tok_size = len(self.student_model.tokenizer)
                if teacher_tok_size != student_tok_size:
                    self.logger.warning(f"⚠️ Teacher tokenizer size ({teacher_tok_size}) != Student tokenizer size ({student_tok_size})")
                    self.logger.warning(f"   This may cause vocab_size mismatch issues, enabled 'range limit + clamp' strategy will protect")
                else:
                    self.logger.info(f"✅ Teacher and Student tokenizer sizes match: {teacher_tok_size}")
                
                # 🔥 Critical validation: Check actual embedding sizes
                try:
                    teacher_input_emb = self.teacher_model.model.get_input_embeddings().weight.size(0)
                    student_input_emb = self.student_model.model.get_input_embeddings().weight.size(0)
                    self.logger.info(f"📊 Actual embedding sizes:")
                    self.logger.info(f"   Teacher input_embeddings: {teacher_input_emb}")
                    self.logger.info(f"   Student input_embeddings: {student_input_emb}")
                    self.logger.info(f"   Teacher tokenizer: {teacher_tok_size}")
                    self.logger.info(f"   Student tokenizer: {student_tok_size}")
                    
                    if teacher_input_emb != teacher_tok_size:
                        self.logger.warning(f"⚠️ Teacher embedding ({teacher_input_emb}) != tokenizer ({teacher_tok_size})")
                    if student_input_emb != student_tok_size:
                        self.logger.warning(f"⚠️ Student embedding ({student_input_emb}) != tokenizer ({student_tok_size})")
                except Exception as e:
                    self.logger.warning(f"⚠️ Unable to check embedding sizes: {e}")
                
                # Check Student model actual distribution
                if hasattr(self.student_model.model, 'hf_device_map'):
                    self.logger.info(f"📊 Student model device distribution: {self.student_model.model.hf_device_map}")
                    if isinstance(student_device_map, str) and student_device_map.startswith("cuda"):
                        try:
                            target_device = torch.device(student_device_map)
                            self.student_model.model = self.student_model.model.to(target_device)
                            self.logger.info(f"✅ Student model forced to device: {target_device}")
                        except Exception as move_err:
                            self.logger.warning(f"⚠️ Failed to move Student model to {student_device_map}: {move_err}")
                
                # Setup PPO model
                self.logger.info("⚙️ Setting up PPO model...")
                self.ppo_model = self.student_model.setup_for_ppo()
                pbar.update(1)
                pbar.set_postfix({"status": "PPO model setup completed"})
            
            # Update models to use modern cache
            self.teacher_model.model = update_model_for_modern_cache(self.teacher_model.model)
            self.student_model.model = update_model_for_modern_cache(self.student_model.model)
            
            self.logger.info("✅ Model setup completed")
            self.logger.info("Model setup completed")
            
        except Exception as e:
            self.logger.error(f"Model setup failed: {e}")
            raise
    
    def setup_components(self):
        """Setup reward computation components"""
        try:
            print("🔧 Starting component setup...")
            
            with tqdm(total=6, desc="Component Setup", ncols=80) as pbar:
                # Cache manager
                print("💾 Setting up cache manager...")
                from models.cache_manager import CacheManager
                self.cache_manager = CacheManager(
                    max_cache_size=self.config["model"]["cache_size"],
                    eviction_policy=self.config["model"]["cache_policy"]
                )
                pbar.update(1)
                pbar.set_postfix({"status": "Cache manager setup completed"})
                
                # Intrinsic reward computer
                print("🧠 Setting up intrinsic reward computer...")
                self.intrinsic_computer = IntrinsicRewardComputer(
                    temperature=self.config["reward"]["temperature"],
                    normalization_method=self.config["reward"]["normalization"],
                    update_rate=self.config["reward"].get("update_rate", 0.01)
                )
                pbar.update(1)
                pbar.set_postfix({"status": "Intrinsic reward computer setup completed"})
                
                # Reward normalizer
                print("📊 Setting up reward normalizer...")
                self.reward_normalizer = RewardNormalizer(
                    method=self.config["reward"]["normalization"],
                    clip_min=self.config["reward"].get("clip_min", -5.0),
                    clip_max=self.config["reward"].get("clip_max", 5.0)
                )
                pbar.update(1)
                pbar.set_postfix({"status": "Reward normalizer setup completed"})
                
                # Reward combiner
                print("🔗 Setting up reward combiner...")
                self.reward_combiner = RewardCombiner(
                    lambda_intrinsic=self.config["reward"]["lambda_intrinsic"],
                    lambda_correctness=self.config["reward"]["lambda_correctness"],
                    lambda_reasoning=self.config["reward"].get("reasoning_weight", 0.0),
                    lambda_format=self.config["reward"].get("format_weight", 0.0),
                    use_adaptive_weights=self.config["reward"].get("use_adaptive_weights", True),
                    adaptation_rate=self.config["reward"].get("adaptation_rate", 0.01),
                    reward_scale=self.config["reward"].get("reward_scale", 1.0)
                )
                pbar.update(1)
                pbar.set_postfix({"status": "Reward combiner setup completed"})
                
                # Log adaptive weight status
                if self.config["reward"].get("use_adaptive_weights", True):
                    self.logger.info("Adaptive weight feature enabled")
                    self.logger.info(f"Initial weights - intrinsic: {self.config['reward']['lambda_intrinsic']}, "
                                   f"correctness: {self.config['reward']['lambda_correctness']}, "
                                   f"reasoning: {self.config['reward'].get('reasoning_weight', 0.0)}, "
                                   f"format: {self.config['reward'].get('format_weight', 0.0)}")
                    self.logger.info(f"Weight adaptation rate: {self.config['reward'].get('adaptation_rate', 0.01)}")
                else:
                    self.logger.info("Adaptive weight feature disabled, using fixed weights")
                
                # Data processor
                print("📝 Setting up data processor...")
                if self.student_model is None or self.student_model.tokenizer is None:
                    raise ValueError("student_model or tokenizer not set. Please call setup_models() first")
                
                self.data_processor = GSM8KProcessor(
                    tokenizer=self.student_model.tokenizer,
                    max_length=self.config["ppo"]["max_length"]
                )
                pbar.update(1)
                pbar.set_postfix({"status": "Data processor setup completed"})
                
                # Initialize parallel processing components
                print("⚡ Setting up parallel processing components...")
                self._setup_parallel_components()
                pbar.update(1)
                pbar.set_postfix({"status": "Parallel processing components setup completed"})
            
            print("✅ Component setup completed")
            self.logger.info("Component setup completed")
            
        except Exception as e:
            self.logger.error(f"Component setup failed: {e}")
            raise
    
    def _setup_parallel_components(self):
        """Setup parallel processing components"""
        try:
            # Check if parallel processing is enabled
            use_parallel = self._use_parallel
            if not use_parallel:
                self.logger.info("Parallel processing disabled")
                return
            
            # Parallel reward processor
            self.parallel_processor = create_parallel_processor(self.config)
            
            # Parallel model inference
            self.parallel_inference_student = create_parallel_inference(
                self.student_model, self.config
            )
            self.parallel_inference_teacher = create_parallel_inference(
                self.teacher_model, self.config
            )
            
            # Async cache manager
            self.async_cache_manager = create_async_cache_manager(
                self.cache_manager, self.config
            )
            
            # Start async cache worker thread
            self.async_cache_manager.start_async_worker()
            
            self.logger.info("Parallel processing components setup completed")
            
        except Exception as e:
            self.logger.error(f"Parallel processing components setup failed: {e}")
            # If parallel processing setup fails, fall back to serial processing
            self.logger.warning("Falling back to serial processing mode")
    
    def setup_ppo_trainer(self):
        """Setup PPO trainer"""
        try:
            # Check if ppo_model is set
            if self.ppo_model is None:
                raise ValueError("ppo_model not set. Please call setup_models() first")
            
            # Check if student_model and tokenizer are set
            if self.student_model is None or self.student_model.tokenizer is None:
                raise ValueError("student_model or tokenizer not set. Please call setup_models() first")
            
            from inspect import signature

            raw = dict(
                model_name=self.config["model"]["student_model_name"],
                learning_rate=float(self.config["ppo"]["learning_rate"]),
                batch_size=self.config["ppo"]["batch_size"],
                mini_batch_size=self.config["ppo"]["mini_batch_size"],
                ppo_epochs=self.config["ppo"]["ppo_epochs"],
                # Compatible with clip_range / clip_ratio / cliprange
                cliprange=self.config["ppo"].get(
                    "clip_range",
                    self.config["ppo"].get("cliprange", self.config["ppo"].get("clip_ratio", 0.2))
                ),
                vf_coef=self.config["ppo"].get("vf_coef", self.config["ppo"].get("value_loss_coef", 0.5)),
                ent_coef=self.config["ppo"].get("entropy_coef", 0.0),  # Compatible with entropy_coef
                init_kl_coef=self.config["ppo"].get("init_kl_coef", self.config["ppo"].get("kl_penalty_coef", self.config["ppo"].get("kl_coef", 0.01))),
                gamma=self.config["ppo"].get("gamma", 0.99),
                lam=self.config["ppo"].get("lam", self.config["ppo"].get("lambda_gae", 0.95)),
                max_grad_norm=self.config["ppo"].get("max_grad_norm", 1.0),
                # ✅ Add generation parameters to ensure KL divergence calculation is correct
                temperature=self.config.get("generation", {}).get("temperature", 0.7),
                top_k=self.config.get("generation", {}).get("top_k", 50),
                top_p=self.config.get("generation", {}).get("top_p", 1.0),
                log_with="wandb" if self.config.get("logging", {}).get("use_wandb", False) else None,
                tracker_project_name=self.config.get("logging", {}).get("wandb_project", "intrinsic-reward-distillation"),
            )

            # Only keep keys actually supported by PPOConfig.__init__
            allowed = set(signature(PPOConfig.__init__).parameters.keys())
            filtered_raw = {k: v for k, v in raw.items() if k in allowed}
            
            # Debug: print filtered out keys
            filtered_out = {k: v for k, v in raw.items() if k not in allowed}
            if filtered_out:
                self.logger.warning(f"⚠️ PPOConfig unsupported parameters (will be ignored): {filtered_out.keys()}")
            
            ppo_config = PPOConfig(**filtered_raw)

            # 📋 Print key hyperparameters to confirm they take effect
            cliprange_val = getattr(ppo_config, "cliprange", None)
            kl_coef_val = getattr(ppo_config, "init_kl_coef", None)
            forward_bs_val = getattr(ppo_config, "forward_batch_size", None)
            self.logger.info(
                "PPO configuration confirmed: "
                f"cliprange={cliprange_val}, "
                f"init_kl_coef={kl_coef_val}, "
                f"learning_rate={ppo_config.learning_rate}, "
                f"batch/mini/forward={ppo_config.batch_size}/{ppo_config.mini_batch_size}/{forward_bs_val}, "
                f"ppo_epochs={ppo_config.ppo_epochs}"
            )
            
            # Note: We don't pass dataset here because data is created dynamically in training loop
            # Explicitly set dataset=None to avoid warnings
            self.ppo_trainer = PPOTrainer(
                config=ppo_config,
                model=self.ppo_model,
                tokenizer=self.student_model.tokenizer,
                dataset=None  # Use custom data loading logic, don't pass dataset
            )
            
            # 🔍 Diagnosis: Check ref_model setup
            if hasattr(self.ppo_trainer, 'ref_model'):
                self.logger.info("✅ PPO trainer has ref_model")
                # Check if ref_model parameters are frozen
                if hasattr(self.ppo_trainer.ref_model, 'parameters'):
                    frozen_params = sum(1 for p in self.ppo_trainer.ref_model.parameters() if not p.requires_grad)
                    total_params = sum(1 for _ in self.ppo_trainer.ref_model.parameters())
                    self.logger.info(f"   Ref model frozen parameters: {frozen_params}/{total_params}")
            else:
                self.logger.error("❌ PPO trainer has no ref_model! This is the root cause of KL=0!")
            
            # ✅ Apply gradient checkpointing configuration (if enabled): Enable after PPO trainer initialization
            # Note: gradient_checkpointing is already enabled in setup_for_ppo, but ensure PPO trainer also applies it
            if self.config["ppo"].get("gradient_checkpointing", False):
                try:
                    # Ensure PPO trainer's model also has gradient checkpointing enabled
                    if hasattr(self.ppo_trainer, 'model') and hasattr(self.ppo_trainer.model, 'gradient_checkpointing_enable'):
                        self.ppo_trainer.model.gradient_checkpointing_enable()
                        self.logger.info("✅ PPO Trainer model gradient checkpointing enabled")
                    elif hasattr(self.ppo_trainer, 'ref_model') and hasattr(self.ppo_trainer.ref_model, 'gradient_checkpointing_enable'):
                        # Ensure ref_model is also enabled (for KL divergence calculation)
                        self.ppo_trainer.ref_model.gradient_checkpointing_enable()
                        self.logger.info("✅ PPO Trainer ref_model gradient checkpointing enabled")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to enable gradient checkpointing in PPO trainer: {e}")
            
            # 📊 GPU allocation notes:
            # - Teacher model: Automatically distributed across GPU 0 and 1 (max_memory already set)
            # - Student base model: On GPU 2 (device="cuda:2" already set)
            # - PPO model (policy + ref): Automatically managed by Accelerator, usually on GPU 0 (main device)
            #   Reason: PPO training requires policy and ref on same device for KL divergence calculation and gradient synchronization
            # 
            # ⚠️ Important: PPO model will not automatically distribute across different GPUs
            # - This is a design limitation of PPO trainer, not a bug
            # - Avoid OOM through gradient_checkpointing, forward_batch_size=1, etc.
            # - GPU 0 total usage ~75GB < 80GB, sufficient safety margin
            
            num_gpus = torch.cuda.device_count()
            # Print GPU allocation (applicable to all GPU configurations)
            try:
                self.logger.info("📊 Model device allocation:")
                
                def get_model_device(model_obj):
                    """Safely get model device"""
                    if model_obj is None:
                        return None
                    try:
                        # Try to get first parameter's device
                        for param in model_obj.parameters():
                            return param.device
                    except:
                        pass
                    # Try to get from pretrained_model
                    if hasattr(model_obj, 'pretrained_model'):
                        try:
                            for param in model_obj.pretrained_model.parameters():
                                return param.device
                        except:
                            pass
                    return None
                
                # Check Teacher model distribution
                if self.teacher_model and hasattr(self.teacher_model, 'model'):
                    try:
                        teacher_params = list(self.teacher_model.model.parameters())[:5]  # Check first 5 parameters
                        devices = [p.device for p in teacher_params]
                        unique_devices = set(str(d) for d in devices)
                        self.logger.info(f"   Teacher model devices: {', '.join(unique_devices)}")
                    except:
                        self.logger.info(f"   Teacher model device: unknown")
                
                # Check Student base model
                if self.student_model and hasattr(self.student_model, 'model'):
                    student_device = get_model_device(self.student_model.model)
                    if student_device:
                        self.logger.info(f"   Student base model device: {student_device}")
                
                # Check PPO model
                if hasattr(self.ppo_trainer, 'model') and self.ppo_trainer.model is not None:
                    policy_device = get_model_device(self.ppo_trainer.model)
                    if policy_device:
                        self.logger.info(f"   PPO Policy model device: {policy_device} (managed by Accelerator)")
                    else:
                        self.logger.warning("   ⚠️ Unable to get Policy model device")
                
                if hasattr(self.ppo_trainer, 'ref_model') and self.ppo_trainer.ref_model is not None:
                    ref_device = get_model_device(self.ppo_trainer.ref_model)
                    if ref_device:
                        self.logger.info(f"   PPO Ref model device: {ref_device} (managed by Accelerator)")
                    else:
                        self.logger.warning("   ⚠️ Unable to get Ref model device")
                
                # Print VRAM usage
                self.logger.info("📊 VRAM usage per GPU:")
                for gpu_id in range(num_gpus):
                    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    max_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                    usage_pct = (allocated / max_memory * 100) if max_memory > 0 else 0
                    self.logger.info(f"   GPU {gpu_id}: {allocated:.2f}GB / {max_memory:.2f}GB ({usage_pct:.1f}%)")
                    
                # Provide configuration notes based on GPU count
                if num_gpus >= 4:
                    self.logger.info("💡 4-GPU configuration notes:")
                    self.logger.info("   - Teacher model distributed across GPU 0,1 (via device_map='auto')")
                    self.logger.info("   - PPO model (policy+ref) on GPU 0 (automatically managed by Accelerator, this is normal)")
                elif num_gpus >= 2:
                    self.logger.info("💡 2-GPU configuration notes:")
                    self.logger.info("   - Teacher model on GPU 1")
                    self.logger.info("   - Student+PPO model on GPU 0")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Error checking model device allocation: {e}")
            
            self.logger.info("PPO trainer setup completed")
            
        except Exception as e:
            self.logger.error(f"PPO trainer setup failed: {e}")
            raise
    
    def compute_intrinsic_rewards(self, questions: List[str], 
                                 student_responses: List[str]) -> torch.Tensor:
        """Compute intrinsic rewards (supports parallel processing)"""
        use_parallel = self._use_parallel
        
        if use_parallel and self.parallel_processor:
            return self._compute_intrinsic_rewards_parallel(questions, student_responses)
        else:
            return self._compute_intrinsic_rewards_sequential(questions, student_responses)
    
    def _compute_intrinsic_rewards_sequential(self, questions: List[str], 
                                            student_responses: List[str]) -> torch.Tensor:
        """Compute intrinsic rewards sequentially (original implementation)"""
        # Check necessary components
        if self.teacher_model is None:
            raise ValueError("teacher_model not set. Please call setup_models() first")
        if self.student_model is None:
            raise ValueError("student_model not set. Please call setup_models() first")
        if self.cache_manager is None:
            raise ValueError("cache_manager not set. Please call setup_components() first")
        if self.intrinsic_computer is None:
            raise ValueError("intrinsic_computer not set. Please call setup_components() first")
        if self.reward_normalizer is None:
            raise ValueError("reward_normalizer not set. Please call setup_components() first")
        
        intrinsic_rewards = []
        
        # Create progress bar
        progress_bar = self._create_progress_bar(
            zip(questions, student_responses), 
            desc="Computing Intrinsic Rewards"
        )
        
        for question, response in progress_bar:
            # Build full sequence
            full_sequence = question + response
            
            # 🔥 Disable cache: directly compute teacher logits (hit rate is always 0%)
            teacher_logits = self.teacher_model.get_logits(full_sequence, use_cache=False)
            
            # Get student tokens (called in loop, but tokenizer itself is fast)
            student_tokens = self.student_model.tokenizer.encode(response, add_special_tokens=False)
            student_tokens = torch.tensor(student_tokens).unsqueeze(0)
            
            # Calculate question part length
            question_tokens = self.student_model.tokenizer.encode(question, add_special_tokens=False)
            question_length = len(question_tokens)
            
            # Compute intrinsic reward
            intrinsic_reward = self.intrinsic_computer.compute_intrinsic_reward(
                teacher_logits, student_tokens, question_length
            )
            
            # Normalize
            normalized_intrinsic = self.reward_normalizer.normalize_intrinsic_rewards(
                intrinsic_reward
            )
            
            # Compute trajectory-level reward
            trajectory_reward = self.intrinsic_computer.compute_trajectory_reward(
                normalized_intrinsic
            )
            
            intrinsic_rewards.append(trajectory_reward)
        
        return torch.tensor(intrinsic_rewards)
    
    def _compute_intrinsic_rewards_parallel(self, questions: List[str], 
                                          student_responses: List[str]) -> torch.Tensor:
        """Compute intrinsic rewards in parallel (using teacher model parallel inference)"""
        # Build full sequence list
        full_sequences = [question + response for question, response in zip(questions, student_responses)]
        
        # Use teacher model parallel inference to get logits
        teacher_logits_list = []
        if self.parallel_inference_teacher:
            # Use parallel inference to get teacher logits
            self.logger.info("Using teacher model parallel inference to compute logits")
            teacher_logits_list = self.parallel_inference_teacher.get_logits_batch_parallel(full_sequences)
        else:
            # Fall back to serial inference
            self.logger.info("Using teacher model serial inference to compute logits")
            for full_sequence in full_sequences:
                # 🔥 Disable cache: directly compute teacher logits (hit rate is always 0%)
                teacher_logits = self.teacher_model.get_logits(full_sequence, use_cache=False)
                teacher_logits_list.append(teacher_logits)
        
        # Compute intrinsic rewards in parallel
        with self.parallel_processor as processor:
            def compute_single_intrinsic_reward(question: str, response: str, teacher_logits: torch.Tensor) -> float:
                try:
                    # Get student tokens
                    student_tokens = self.student_model.tokenizer.encode(response, add_special_tokens=False)
                    student_tokens = torch.tensor(student_tokens).unsqueeze(0)
                    
                    # Calculate question part length
                    question_tokens = self.student_model.tokenizer.encode(question, add_special_tokens=False)
                    question_length = len(question_tokens)
                    
                    # Compute intrinsic reward
                    intrinsic_reward = self.intrinsic_computer.compute_intrinsic_reward(
                        teacher_logits, student_tokens, question_length
                    )
                    
                    # Normalize
                    normalized_intrinsic = self.reward_normalizer.normalize_intrinsic_rewards(
                        intrinsic_reward
                    )
                    
                    # Compute trajectory-level reward
                    trajectory_reward = self.intrinsic_computer.compute_trajectory_reward(
                        normalized_intrinsic
                    )
                    
                    return trajectory_reward
                    
                except Exception as e:
                    self.logger.error(f"Parallel intrinsic reward computation failed: {e}")
                    return 0.0
            
            # Compute rewards in parallel
            intrinsic_rewards = []
            for i, (question, response) in enumerate(zip(questions, student_responses)):
                if i < len(teacher_logits_list) and teacher_logits_list[i] is not None:
                    reward = compute_single_intrinsic_reward(question, response, teacher_logits_list[i])
                    intrinsic_rewards.append(reward)
                else:
                    self.logger.warning(f"Teacher logits missing, using default reward: {i}")
                    intrinsic_rewards.append(0.0)
        
        return torch.tensor(intrinsic_rewards)
    
    def compute_correctness_rewards(self, questions: List[str], 
                                   student_responses: List[str]) -> torch.Tensor:
        """Compute answer correctness rewards (supports parallel processing)"""
        use_parallel = self._use_parallel
        
        if use_parallel and self.parallel_processor:
            return self._compute_correctness_rewards_parallel(questions, student_responses)
        else:
            return self._compute_correctness_rewards_sequential(questions, student_responses)
    
    def _compute_correctness_rewards_sequential(self, questions: List[str], 
                                              student_responses: List[str]) -> torch.Tensor:
        """Compute answer correctness rewards sequentially (original implementation)"""
        # Check necessary components
        if self.teacher_model is None:
            raise ValueError("teacher_model not set. Please call setup_models() first")
        
        correctness_rewards = []
        
        # Create progress bar
        progress_bar = self._create_progress_bar(
            zip(questions, student_responses), 
            desc="Computing Correctness Rewards"
        )
        
        for question, response in progress_bar:
            # Extract student answer
            student_answer = extract_final_answer(response)
            
            # Extract correct answer (from question or using teacher model generation)
            # Simplified processing here, should actually get correct answer from dataset
            teacher_response = self.teacher_model.generate_response(question, max_length=256)
            correct_answer = extract_final_answer(teacher_response)
            
            # Determine if answer is correct
            is_correct = is_answer_correct(student_answer, correct_answer)
            
            # Correct answer: +1, incorrect answer: -0.5
            correctness_rewards.append(1.0 if is_correct else -0.5)
        
        return torch.tensor(correctness_rewards)
    
    def _compute_correctness_rewards_parallel(self, questions: List[str], 
                                            student_responses: List[str]) -> torch.Tensor:
        """Compute answer correctness rewards in parallel"""
        with self.parallel_processor as processor:
            # Define reward computation function
            def compute_single_correctness_reward(question: str, response: str) -> float:
                try:
                    # Extract student answer
                    student_answer = extract_final_answer(response)
                    
                    # Extract correct answer (from question or using teacher model generation)
                    # Simplified processing here, should actually get correct answer from dataset
                    teacher_response = self.teacher_model.generate_response(question, max_length=256)
                    correct_answer = extract_final_answer(teacher_response)
                    
                    # Determine if answer is correct
                    is_correct = is_answer_correct(student_answer, correct_answer)
                    
                    # Correct answer: +1, incorrect answer: -0.5
                    return 1.0 if is_correct else -0.5
                    
                except Exception as e:
                    self.logger.error(f"Parallel correctness reward computation failed: {e}")
                    return 0.0
            
            # Compute rewards in parallel
            correctness_rewards = processor.compute_rewards_parallel(
                questions, student_responses, compute_single_correctness_reward
            )
            
            return torch.tensor(correctness_rewards)
    
    def compute_combined_rewards(self, questions: List[str], 
                                 student_responses: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined rewards (single-threaded serial processing to avoid tokenizer thread safety issues), and return component rewards"""
        # Check if reward combiner is set
        if self.reward_combiner is None:
            raise ValueError("reward_combiner not set. Please call setup_components() first")
        
        # Force serial computation to avoid tokenizer thread safety issues from multithreading
        # Serial computation
        intrinsic_rewards = self.compute_intrinsic_rewards(questions, student_responses)
        correctness_rewards = self.compute_correctness_rewards(questions, student_responses)
        
        # Combine rewards
        combined_rewards = self.reward_combiner.combine_rewards(
            intrinsic_rewards, correctness_rewards
        )
        
        return combined_rewards, intrinsic_rewards, correctness_rewards
    
    def train_step(self, batch: Dict[str, List[str]]) -> Dict[str, float]:
        """Executes one training step (supports parallel processing)"""
        # Note: Do not use @handle_errors, as we need to ensure exceptions are raised on failure, not return None
        # This allows the caller to decide how to handle errors
        try:
            # Check necessary components
            if self.student_model is None:
                raise ValueError("student_model is not set. Please call setup_models() first")
            if self.ppo_trainer is None:
                raise ValueError("ppo_trainer is not set. Please call setup_ppo_trainer() first")
            
            # Validate data batch
            if "questions" not in batch or not batch["questions"]:
                raise ValueError("❌ Missing question data")
            
            if not isinstance(batch["questions"], list):
                raise ValueError("❌ Question data is not a list")
            
            questions = batch["questions"]
            
            # Check question count
            if len(questions) == 0:
                raise ValueError("❌ Question list is empty")
            
            if len(questions) > 100:  # Prevent batch size from being too large
                self.logger.warning(f"⚠️ Batch size too large: {len(questions)}, truncating to 100")
                questions = questions[:100]
            
            # Student model generates responses (supports parallel processing)
            use_parallel = self._use_parallel
            generation_cfg = self.config.get("generation", {})
            gen_temperature = generation_cfg.get("temperature", 0.7)
            gen_do_sample = generation_cfg.get("do_sample", True)
            gen_top_k = generation_cfg.get("top_k", 0)
            gen_top_p = generation_cfg.get("top_p", 1.0)
            gen_max_new_tokens = generation_cfg.get("max_new_tokens", self.config["ppo"].get("max_length", 256))
            
            # Use no_grad for inference to save memory
            with torch.no_grad():
                if use_parallel and self.parallel_inference_student:
                    # ✅ Use max_length from config uniformly
                    student_responses = self.parallel_inference_student.generate_batch_parallel(
                        questions,
                        max_length=self.config["ppo"]["max_length"],  # ✅ Use config max_length uniformly
                        temperature=gen_temperature,
                        do_sample=gen_do_sample,
                        top_k=gen_top_k,
                        top_p=gen_top_p,
                        max_new_tokens=gen_max_new_tokens
                    )
                else:
                    # ✅ Use max_length from config uniformly to ensure consistency
                    # Note: max_length during generation is the number of new tokens to generate, actual total length = query_length + max_length
                    # But here we use the config max_length as a reference, actual generation will be shorter (limited during generation)
                    student_responses = self.student_model.generate(
                        questions,
                        max_length=self.config["ppo"]["max_length"],  # ✅ Use config max_length uniformly
                        temperature=gen_temperature,
                        do_sample=gen_do_sample,
                        top_k=gen_top_k,
                        top_p=gen_top_p
                    )
                    # Ensure return value is a list type (student_model.generate may return a single string for a single prompt)
                    if isinstance(student_responses, str):
                        student_responses = [student_responses]
                    if not isinstance(student_responses, list):
                        raise TypeError(f"student_model.generate returned unexpected type: {type(student_responses)}")
            
            # Compute rewards
            combined_rewards, intrinsic_rewards, correctness_rewards = self.compute_combined_rewards(
                questions, student_responses
            )
            
            # Print reward mean every 10 steps to diagnose if reward signal disappears
            if self.training_stats["step"] % 10 == 0:
                self.logger.info(
                    f"[dbg] Step {self.training_stats['step'] + 1} reward mean - "
                    f"intrinsic={float(intrinsic_rewards.mean()):.4f}, "
                    f"correctness={float(correctness_rewards.mean()):.4f}, "
                    f"combined={float(combined_rewards.mean()):.4f}"
                )
            
            # 🔍 Diagnostics: Print reward statistics (first few steps)
            if self.training_stats["step"] < 3:
                self.logger.info(f"🎁 Step {self.training_stats['step'] + 1} - Reward diagnostics:")
                self.logger.info(f"   Combined rewards: shape={combined_rewards.shape}, "
                               f"mean={combined_rewards.mean():.4f}, std={combined_rewards.std():.4f}, "
                               f"min={combined_rewards.min():.4f}, max={combined_rewards.max():.4f}")
                self.logger.info(f"   Reward value distribution: {combined_rewards.tolist()}")
            
            # 🔥 Critical optimization: Clear VRAM immediately after Teacher inference (Teacher model occupies large VRAM)
            # After reward computation is complete, Teacher's intermediate activations are no longer needed, release immediately
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Validate length matching
            if len(combined_rewards) != len(questions):
                raise ValueError(f"Reward count {len(combined_rewards)} does not match question count {len(questions)}")
            if len(student_responses) != len(questions):
                raise ValueError(f"Response count {len(student_responses)} does not match question count {len(questions)}")
            
            # Convert questions to tokenized tensor list
            # Use batch tokenization to improve efficiency and avoid thread safety issues
            # Note: tokenizer is thread-safe, but use batch processing for extra safety
            try:
                tokenized_queries_batch = self.student_model.tokenizer(
                    questions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config["ppo"]["max_length"]
                )
                tokenized_queries = [tokenized_queries_batch["input_ids"][i] for i in range(len(questions))]
            except RuntimeError as e:
                if "Already borrowed" in str(e):
                    # If thread safety issue is encountered, use single-threaded processing
                    self.logger.warning("Detected tokenizer thread safety issue, using single-threaded processing")
                    tokenized_queries = []
                    for question in questions:
                        tokenized = self.student_model.tokenizer(
                            question,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=self.config["ppo"]["max_length"]
                        )
                        tokenized_queries.append(tokenized["input_ids"])
                else:
                    raise
            
            # Convert responses to tokenized tensor list
            try:
                tokenized_responses_batch = self.student_model.tokenizer(
                    student_responses,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config["ppo"]["max_length"]
                )
                tokenized_responses = [tokenized_responses_batch["input_ids"][i] for i in range(len(student_responses))]
            except RuntimeError as e:
                if "Already borrowed" in str(e):
                    # If thread safety issue is encountered, use single-threaded processing
                    self.logger.warning("Detected tokenizer thread safety issue, using single-threaded processing")
                    tokenized_responses = []
                    for response in student_responses:
                        tokenized = self.student_model.tokenizer(
                            response,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=self.config["ppo"]["max_length"]
                        )
                        tokenized_responses.append(tokenized["input_ids"])
                else:
                    raise
            
            # PPO update
            # Ensure scores are tensors, not floats
            device = getattr(self.ppo_trainer.accelerator, "device", "cuda")
            
            # Ensure combined_rewards is 1D tensor, then convert to tensor list
            if combined_rewards.dim() > 1:
                combined_rewards = combined_rewards.squeeze()
            
            # Validate tokenized list length matching
            if len(tokenized_queries) != len(tokenized_responses):
                raise ValueError(f"Question token count {len(tokenized_queries)} does not match response token count {len(tokenized_responses)}")
            if len(tokenized_queries) != len(combined_rewards):
                raise ValueError(f"Question token count {len(tokenized_queries)} does not match reward count {len(combined_rewards)}")
            
            scores = []
            for i in range(len(combined_rewards)):
                reward = combined_rewards[i]
                if torch.is_tensor(reward):
                    scores.append(reward.to(device=device, dtype=torch.float32))
                else:
                    scores.append(torch.tensor(reward, dtype=torch.float32, device=device))
            
            # Final validation of scores length
            if len(scores) != len(tokenized_queries):
                raise ValueError(f"Score count {len(scores)} does not match query count {len(tokenized_queries)}")
            
            # ✅ Fix: Save statistics before deleting combined_rewards (for subsequent logging and statistics updates)
            # Save reward statistics (move to CPU and convert to Python scalars, release GPU VRAM)
            mean_reward_for_stats = torch.mean(combined_rewards).cpu().item()
            intrinsic_reward_mean = intrinsic_rewards.mean().cpu().item() if torch.is_tensor(intrinsic_rewards) else float(intrinsic_rewards)
            correctness_reward_mean = correctness_rewards.mean().cpu().item() if torch.is_tensor(correctness_rewards) else float(correctness_rewards)
            # Save CPU copy of combined_rewards for statistics update (keep as tensor format, as _update_training_stats requires it)
            combined_rewards_cpu = combined_rewards.cpu().clone()
            
            # 🔥 Critical optimization: Immediately clean up memory and release unnecessary variables before PPO step
            # Release original string data (already tokenized, no longer needed)
            if 'questions' in locals():
                del questions
            if 'student_responses' in locals():
                del student_responses
            # Release batch tokenized tensors (already extracted as lists)
            if 'tokenized_queries_batch' in locals():
                del tokenized_queries_batch
            if 'tokenized_responses_batch' in locals():
                del tokenized_responses_batch
            # Release combined_rewards (already converted to scores, CPU copy saved)
            del combined_rewards
            
            # 🔥 Extreme VRAM cleanup: Before PPO step (log_softmax is VRAM peak)
            # Under 4×GPU configuration, log_softmax needs to compute both policy and ref models simultaneously, causing extreme VRAM pressure
            if torch.cuda.is_available():
                # Clear VRAM on all GPUs
                for gpu_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()  # Python garbage collection
            
            # 🔥 Core issue: PPO trainer expands full vocabulary matrix (B×T×V) when computing log_softmax in batched_forward_pass
            # Even with batch_size=1, seq_len=192, vocab_size=100k, this matrix requires ~38MB in bfloat16
            # But with policy and ref models computing simultaneously, plus gradients, VRAM peak can surge to hundreds of MB or even GB
            # Key: Even with 4 GPUs in data parallelism, each GPU still needs to compute full (seq_len × vocab_size) on its own micro-batch
            
            # ✅ Optimization strategy: Check and truncate over-long sequences before calling PPO trainer
            max_allowed_length = self.config["ppo"]["max_length"]
            
            # Truncate over-long sequences (avoid log_softmax OOM)
            truncated_queries = []
            truncated_responses = []
            token_length_stats = {}
            for q, r in zip(tokenized_queries, tokenized_responses):
                # Calculate total length (query + response)
                total_len = len(q) + len(r)
                
                if total_len > max_allowed_length:
                    # If over-long, prioritize keeping query, truncate response
                    query_len = len(q)
                    max_response_len = max(0, max_allowed_length - query_len)
                    
                    if max_response_len > 0:
                        # Truncate response to allowed maximum length
                        truncated_r = r[:max_response_len]
                        truncated_queries.append(q)
                        truncated_responses.append(truncated_r)
                        if self.training_stats["step"] < 3:  # Only warn in first few steps
                            self.logger.warning(f"⚠️ Sequence over-long ({total_len} > {max_allowed_length}), truncated response to {max_response_len}. Consider reducing max_length to avoid log_softmax OOM.")
                    else:
                        # Query itself is too long, skip this sample
                        self.logger.warning(f"⚠️ Query too long ({query_len} > {max_allowed_length}), skipping this sample.")
                        continue
                else:
                    truncated_queries.append(q)
                    truncated_responses.append(r)
            
            # If all samples are truncated, skip this step
            if len(truncated_queries) == 0:
                self.logger.error("❌ All sequences truncated, skipping this training step")
                return None
            
            # 📊 Statistics on sequence length (for logging and diagnostics)
            try:
                query_lengths_tensor = torch.tensor([len(q) for q in truncated_queries], dtype=torch.float32)
                response_lengths_tensor = torch.tensor([len(r) for r in truncated_responses], dtype=torch.float32)
                total_lengths_tensor = torch.tensor(
                    [len(q) + len(r) for q, r in zip(truncated_queries, truncated_responses)],
                    dtype=torch.float32
                )
                
                def calc_stats(length_tensor: torch.Tensor):
                    if length_tensor.numel() == 0:
                        return 0.0, 0.0, 0.0, 0.0
                    mean_val = float(length_tensor.mean().item())
                    std_val = float(length_tensor.std(unbiased=False).item()) if length_tensor.numel() > 1 else 0.0
                    min_val = float(length_tensor.min().item())
                    max_val = float(length_tensor.max().item())
                    return mean_val, std_val, min_val, max_val
                
                q_mean, q_std, q_min, q_max = calc_stats(query_lengths_tensor)
                r_mean, r_std, r_min, r_max = calc_stats(response_lengths_tensor)
                t_mean, t_std, t_min, t_max = calc_stats(total_lengths_tensor)
                
                token_length_stats = {
                    "tokens/queries_len_mean": q_mean,
                    "tokens/queries_len_std": q_std,
                    "tokens/queries_len_min": q_min,
                    "tokens/queries_len_max": q_max,
                    "tokens/responses_len_mean": r_mean,
                    "tokens/responses_len_std": r_std,
                    "tokens/responses_len_min": r_min,
                    "tokens/responses_len_max": r_max,
                    "tokens/total_len_mean": t_mean,
                    "tokens/total_len_std": t_std,
                    "tokens/total_len_min": t_min,
                    "tokens/total_len_max": t_max,
                }
                
                # If total length is below 420, provide a hint
                if t_mean < 420 and self.training_stats["step"] % 10 == 0:
                    self.logger.warning(
                        f"⚠️ Current batch average total token count only {t_mean:.1f} (<420), "
                        f"consider increasing ppo.max_length or check generation length settings."
                    )
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to compute token length statistics: {e}")
                token_length_stats = {}
            
            # 🔍 Set up gradient hook to capture gradient norm (before PPO step)
            grad_norm_from_hook = None
            grad_hook_handles = []
            grad_norms = []  # Defined externally to ensure visibility throughout try-finally block
            
            # Register hook for trainable parameters (only when needed)
            try:
                if hasattr(self.ppo_trainer, 'model') and self.ppo_trainer.model is not None:
                    # Store gradients for norm computation
                    for name, param in self.ppo_trainer.model.named_parameters():
                        if param.requires_grad:
                            # Register hook to record during gradient computation
                            def make_hook(n=name):
                                def hook(grad):
                                    if grad is not None:
                                        grad_norms.append((n, grad.norm().item()))
                                    return grad  # Keep gradient unchanged, only for monitoring
                                return hook
                            handle = param.register_hook(make_hook(name))
                            grad_hook_handles.append(handle)
            except Exception as e:
                # Hook registration failure does not affect training
                pass
            
            # Execute PPO step (log_softmax computation happens here)
            try:
                stats = self.ppo_trainer.step(
                    queries=truncated_queries,
                    responses=truncated_responses,
                    scores=scores[:len(truncated_queries)]  # Adjust scores length to match truncated sequences
                )
                
                # Get gradient norm from hook (if available)
                if grad_norms:
                    total_grad_norm_sq = sum(norm**2 for _, norm in grad_norms)
                    grad_norm_from_hook = (total_grad_norm_sq ** 0.5)
            except torch.cuda.OutOfMemoryError as e:
                # 🔥 If OOM during log_softmax phase, perform extreme cleanup and provide detailed diagnostics
                self.logger.error("❌ OOM during log_softmax phase in PPO step, performing extreme VRAM cleanup...")
                for gpu_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                gc.collect()
                
                # Detailed diagnostic information
                max_seq_len = max((len(q) + len(r) for q, r in zip(truncated_queries, truncated_responses)), default=0)
                vocab_size = len(self.student_model.tokenizer) if hasattr(self.student_model, 'tokenizer') else "unknown"
                estimated_memory_mb = (max_seq_len * vocab_size * 2) / (1024 * 1024) if isinstance(vocab_size, int) else "unknown"
                
                error_msg = (
                    f"PPO step log_softmax OOM (even with batch_size=1, max_length={max_allowed_length}).\n"
                    f"Diagnostic information:\n"
                    f"  - Maximum sequence length: {max_seq_len}\n"
                    f"  - Vocabulary size: {vocab_size}\n"
                    f"  - Estimated log_softmax VRAM: ~{estimated_memory_mb}MB (single model, excluding gradients)\n"
                    f"  - Actual VRAM requirement: estimated value × 2 (policy+ref) × 2 (gradients) ≈ {estimated_memory_mb * 4 if isinstance(estimated_memory_mb, (int, float)) else 'unknown'}MB\n"
                    f"Solutions:\n"
                    f"  1) Reduce max_length to 128 or smaller (most effective)\n"
                    f"  2) Check if GPU allocation is uniform (Teacher should not be concentrated on one card)\n"
                    f"  3) Consider using 2×GPU instead of 4×GPU (reduce data parallelism overhead)\n"
                    f"  4) Use larger GPUs (H100 120GB)\n"
                    f"Original error: {e}"
                )
                raise RuntimeError(error_msg)
            finally:
                # Clean up hooks
                for handle in grad_hook_handles:
                    handle.remove()
                grad_hook_handles.clear()
                grad_norms.clear()  # Clear gradient norm list
            
            # 🔥 Extreme VRAM cleanup: Immediately after PPO step
            # VRAM fragmentation after log_softmax computation needs immediate cleanup
            if torch.cuda.is_available():
                for gpu_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
            
            # 🔍 Diagnostics: Print detailed stats information (print first 3 steps every step, then every 10 steps)
            should_log_verbose = self.training_stats["step"] < 3 or self.training_stats["step"] % 10 == 0
            if should_log_verbose and stats is not None and isinstance(stats, dict):
                available_keys = list(stats.keys())
                self.logger.info(f"📊 Step {self.training_stats['step'] + 1} - PPO stats available keys: {available_keys}")
                
                # Print all stats values for debugging
                for key in available_keys:
                    try:
                        value = stats[key]
                        # Try to convert to scalar
                        if isinstance(value, (torch.Tensor, np.ndarray)):
                            value = value.item() if hasattr(value, 'item') else float(value)
                        self.logger.info(f"  {key} = {value}")
                    except Exception as e:
                        self.logger.warning(f"  Unable to print {key}: {e}")
                
                # 🔍 Additional diagnostics: Check key metrics
                self.logger.info(f"🔍 Key diagnostics:")
                self.logger.info(f"  Reward: mean={mean_reward_for_stats:.4f}")
                self.logger.info(f"  stats type: {type(stats)}")
                self.logger.info(f"  stats key count: {len(stats)}")
            
            # 🔍 Additional diagnostics: If stats is empty or contains very few keys, issue warning
            if stats is not None and isinstance(stats, dict) and len(stats) < 3:
                self.logger.error(f"❌ PPO stats dictionary abnormal: only {len(stats)} keys, PPOTrainer may not have computed correctly!")
                self.logger.error(f"   Available keys: {list(stats.keys())}")
                self.logger.error(f"   stats content: {stats}")
            
            # 🔍 Diagnostics: Check if policy is actually updating (based on log analysis, found serious issue with KL=0)
            if stats is not None and isinstance(stats, dict):
                # Define to_scalar helper function (for diagnostics)
                def _to_scalar(value, default=0):
                    if value is None:
                        return default
                    if isinstance(value, (np.ndarray, np.generic)):
                        return float(value.item() if hasattr(value, 'item') else float(value))
                    if isinstance(value, torch.Tensor):
                        return float(value.item() if hasattr(value, 'item') else float(value))
                    return float(value)
                
                # Check KL divergence
                approx_kl = _to_scalar(stats.get("ppo/policy/approxkl") or stats.get("ppo/policy/policykl") or 0)
                advantages_mean = _to_scalar(stats.get("ppo/policy/advantages_mean") or 0)
                clipfrac = _to_scalar(stats.get("ppo/policy/clipfrac") or 0)
                policy_loss_val = _to_scalar(stats.get("ppo/loss/policy") or stats.get("ppo/policy/loss") or 0)
                
                # If KL is 0 and advantages are close to 0, policy is barely updating
                # Note: This may be normal in early training (first few steps), only warn after persisting for multiple steps
                if abs(approx_kl) < 1e-6 and abs(advantages_mean) < 1e-6:
                    # Only warn after step 5, to avoid normal fluctuations in early training
                    if self.training_stats["step"] >= 5 and (self.training_stats["step"] % 50 == 0 or self.training_stats["step"] < 20):
                        # Check if recent steps are all like this (exclude single fluctuation)
                        if len(self.training_stats["kl_divergences"]) >= 5:
                            recent_kls = self.training_stats["kl_divergences"][-5:]
                            all_kl_zero = all(abs(k) < 1e-6 for k in recent_kls if k is not None)
                            if all_kl_zero:
                                self.logger.warning(f"⚠️ Warning: Policy may not be updating! (persisted for at least 5 steps)")
                                self.logger.warning(f"   KL divergence: {approx_kl:.10f} (close to 0)")
                                self.logger.warning(f"   Advantages mean: {advantages_mean:.10f} (close to 0)")
                                self.logger.warning(f"   Policy loss: {policy_loss_val:.10f}")
                                self.logger.warning(f"   Clip fraction: {clipfrac:.4f}")
                                self.logger.warning(f"   Possible causes:")
                                self.logger.warning(f"     1. Reward scale issue (rewards too small or too large)")
                                self.logger.warning(f"     2. Advantage normalization issue")
                                self.logger.warning(f"     3. Learning rate too small")
                                self.logger.warning(f"     4. policy and ref_model are identical (should be different)")
                        elif self.training_stats["step"] < 10:
                            # First few steps only record info, no warning
                            self.logger.info(f"ℹ️ Step {self.training_stats['step'] + 1}: KL={approx_kl:.10f}, advantages={advantages_mean:.10f} (early training, continue observing)")
                
                # Check value function training
                val_var_explained = _to_scalar(stats.get("ppo/val/var_explained") or 0)
                if val_var_explained < 0:
                    # Only warn after step 5 and persisting for multiple steps
                    if self.training_stats["step"] >= 5 and (self.training_stats["step"] % 50 == 0 or self.training_stats["step"] < 20):
                        self.logger.warning(f"⚠️ Value function training abnormal: var_explained = {val_var_explained:.4f} (negative value indicates value function is worse than simple mean prediction)")
            
            # 🔍 Check if model parameters are updating (for diagnosing if policy is actually training)
            # Note: PPOTrainer completes gradient computation and optimization inside step(), gradients are cleared after step() returns
            # So we verify if updates occur by checking model parameter changes
            
            # Initialize: Save initial parameter state (only on first step)
            if not hasattr(self, '_prev_model_params'):
                if hasattr(self.ppo_trainer, 'model') and self.ppo_trainer.model is not None:
                    self._prev_model_params = {}
                    for name, param in self.ppo_trainer.model.named_parameters():
                        if param.requires_grad:
                            self._prev_model_params[name] = param.data.clone()
                    if self.training_stats["step"] == 0:
                        self.logger.info(f"📊 Saved initial model parameter state ({len(self._prev_model_params)} trainable parameters)")
            
            # Check parameter changes (computed every time, for wandb logging)
            param_change_info = {}
            if hasattr(self, '_prev_model_params'):
                try:
                    if hasattr(self.ppo_trainer, 'model') and self.ppo_trainer.model is not None:
                        max_change = 0.0
                        total_change = 0.0
                        changed_params = 0
                        total_params = 0
                        
                        for name, param in self.ppo_trainer.model.named_parameters():
                            if param.requires_grad and name in self._prev_model_params:
                                total_params += 1
                                prev_param = self._prev_model_params[name]
                                # Compute parameter change
                                param_diff = (param.data - prev_param).abs()
                                max_param_change = param_diff.max().item()
                                mean_param_change = param_diff.mean().item()
                                
                                if max_param_change > 1e-8:  # Significant change
                                    changed_params += 1
                                    max_change = max(max_change, max_param_change)
                                    total_change += mean_param_change
                                
                                # Update saved parameters
                                self._prev_model_params[name] = param.data.clone()
                        
                        if total_params > 0:
                            param_change_info = {
                                "total_params": total_params,
                                "changed_params": changed_params,
                                "max_change": max_change,
                                "avg_change": total_change / changed_params if changed_params > 0 else 0.0,
                                "change_ratio": changed_params / total_params
                            }
                except Exception as e:
                    param_change_info = {"error": str(e)}
            
            # Print parameter change information (every 5 or 50 steps)
            if param_change_info and "error" not in param_change_info:
                if self.training_stats["step"] < 10 or self.training_stats["step"] % 50 == 0:
                    self.logger.info(
                        f"📊 Step {self.training_stats['step'] + 1} - Parameter updates: "
                        f"{param_change_info['changed_params']}/{param_change_info['total_params']} parameters changed, "
                        f"max_change={param_change_info['max_change']:.8f}, "
                        f"avg_change={param_change_info['avg_change']:.8f}"
                    )
                    
                    # If parameters have no changes at all, issue warning
                    if param_change_info['changed_params'] == 0 and self.training_stats["step"] >= 5:
                        self.logger.warning(
                            f"⚠️ Step {self.training_stats['step'] + 1} - Model parameters have no updates at all! "
                            f"This may indicate issues with policy training (gradients are 0 or optimizer not executing)."
                        )
            
            # 🔍 Use gradient norm from hook (preferred)
            grad_norm = grad_norm_from_hook
            
            # Update statistics (using CPU copy)
            self._update_training_stats(stats, combined_rewards_cpu)
            
            # Log to wandb
            if self.config.get("logging", {}).get("use_wandb", False):
                # Ensure all values are converted to Python scalars
                def to_scalar(value, default=0):
                    if value is None:
                        return default
                    if isinstance(value, (np.ndarray, np.generic)):
                        return float(value.item() if hasattr(value, 'item') else float(value))
                    if isinstance(value, torch.Tensor):
                        return float(value.item() if hasattr(value, 'item') else float(value))
                    return float(value)
                
                # 🔍 Correctly get loss value: do not use clipfrac to impersonate loss
                # First try to get the true loss value
                policy_loss_key = None
                value_loss_key = None
                for key in stats.keys():
                    if 'policy' in key and 'loss' in key and 'clip' not in key:
                        policy_loss_key = key
                    if 'value' in key and 'loss' in key and 'clip' not in key:
                        value_loss_key = key
                
                log_data = {
                    "step": self.training_stats["step"],
                    "mean_reward": mean_reward_for_stats,  # ✅ Use saved mean
                    "reward/intrinsic_mean": intrinsic_reward_mean,
                    "reward/correctness_mean": correctness_reward_mean,
                    # 🔍 Use correct key names to get loss (found from logs that actual key names are ppo/loss/policy and ppo/loss/value)
                    "policy_loss": to_scalar(stats.get(policy_loss_key) if policy_loss_key else stats.get("ppo/loss/policy") or stats.get("ppo/policy/loss", 0)),
                    "value_loss": to_scalar(stats.get(value_loss_key) if value_loss_key else stats.get("ppo/loss/value") or stats.get("ppo/val/loss", 0)),
                    "kl_divergence": to_scalar(stats.get("ppo/policy/kl") or stats.get("objective/kl") or 0),
                    "policy_clipfrac": to_scalar(stats.get("ppo/policy/clipfrac") or 0),  # ✅ Clip rate
                    "value_clipfrac": to_scalar(stats.get("ppo/val/clipfrac") or 0),  # ✅ Value function clip rate
                    "objective/clipfrac": to_scalar(stats.get("objective/clipfrac") or 0),
                    "objective/entropy": to_scalar(stats.get("objective/entropy") or stats.get("ppo/policy/entropy") or 0),
                }
                
                # Add sequence length statistics
                if token_length_stats:
                    log_data.update(token_length_stats)
                
                # Add gradient information (if available)
                if grad_norm is not None:
                    log_data["grad_norm"] = grad_norm
                else:
                    # Try to get from stats (some PPO implementations may record gradient norm)
                    if isinstance(stats, dict):
                        grad_norm_from_stats = stats.get("ppo/grad_norm") or stats.get("train/grad_norm") or stats.get("grad_norm")
                        if grad_norm_from_stats is not None:
                            log_data["grad_norm"] = to_scalar(grad_norm_from_stats)
                
                # Add parameter change information (if available)
                if param_change_info and "error" not in param_change_info:
                    log_data.update({
                        "param_change/ratio": param_change_info.get("change_ratio", 0.0),
                        "param_change/changed_count": param_change_info.get("changed_params", 0),
                        "param_change/total_count": param_change_info.get("total_params", 0),
                        "param_change/max": param_change_info.get("max_change", 0.0),
                        "param_change/avg": param_change_info.get("avg_change", 0.0),
                    })
                
                # 🔍 Diagnostics: If loss is 0, print all stats keys to help debugging
                if log_data["policy_loss"] == 0 and log_data["value_loss"] == 0:
                    self.logger.warning(f"⚠️ PPO losses are zero! Available stats keys: {list(stats.keys())}")
                
                # Add adaptive weight information
                if self.config["reward"].get("use_adaptive_weights", True):
                    weight_stats = self.reward_combiner.get_statistics()
                    if "adaptive_weights" in weight_stats:
                        log_data.update({
                            "adaptive_weight_intrinsic": float(weight_stats["adaptive_weights"].get("intrinsic", 0.0)),
                            "adaptive_weight_correctness": float(weight_stats["adaptive_weights"].get("correctness", 0.0))
                        })
                    if "weight_performance" in weight_stats:
                        log_data.update({
                            "weight_performance_intrinsic": float(weight_stats["weight_performance"].get("intrinsic", 0.0)),
                            "weight_performance_correctness": float(weight_stats["weight_performance"].get("correctness", 0.0))
                        })
                    
                    # Add weight change trends
                    if "weight_trend" in weight_stats and weight_stats["weight_trend"]:
                        log_data.update({
                            "weight_trend_intrinsic": float(weight_stats["weight_trend"].get("intrinsic", 0.0)),
                            "weight_trend_correctness": float(weight_stats["weight_trend"].get("correctness", 0.0))
                        })
                
                wandb.log(log_data)
            
            self.training_stats["step"] += 1
            
            # 🔥 Clean up VRAM before end of each step to avoid accumulation causing OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Training step failed: {e}")
            raise
    
    def _update_training_stats(self, stats: Dict, rewards: torch.Tensor):
        """Updates training statistics"""
        # Check if stats is None
        if stats is None:
            self.logger.warning("stats is None, updating training statistics with default values")
            stats = {}
        
        # 🔍 Ensure rewards is a tensor, then convert to Python scalar
        if isinstance(rewards, torch.Tensor):
            mean_reward = torch.mean(rewards).item()
        elif isinstance(rewards, (list, tuple)):
            # If it's a list, convert to tensor then compute
            mean_reward = float(np.mean([float(r.item() if hasattr(r, 'item') else float(r)) for r in rewards]))
        else:
            mean_reward = float(rewards) if isinstance(rewards, (int, float)) else 0.0
        
        self.training_stats["total_rewards"].append(mean_reward)
        
        # Ensure values retrieved from stats are converted to Python scalars
        def to_scalar(value, default=0):
            if value is None:
                return default
            if isinstance(value, (np.ndarray, np.generic)):
                return float(value.item() if hasattr(value, 'item') else float(value))
            if isinstance(value, torch.Tensor):
                return float(value.item() if hasattr(value, 'item') else float(value))
            if isinstance(value, (np.int64, np.int32)):
                return int(value)
            if isinstance(value, (np.float64, np.float32)):
                return float(value)
            return float(value)
        
        # 🔍 Fix: Use correct metric names, do not use clipfrac to impersonate loss
        # TRL may use different key names: ppo/policy/loss, objective/policy_loss, etc.
        # First find the true loss keys
        policy_loss_key = None
        value_loss_key = None
        if isinstance(stats, dict):
            # 🔍 Diagnostics: Print all available keys (only on first step)
            if len(self.training_stats["policy_losses"]) == 0:
                self.logger.info(f"🔍 Step 1 stats dictionary all keys: {list(stats.keys())}")
            
            # Try multiple possible key names (based on actual logs, TRL uses ppo/loss/policy format)
            possible_policy_keys = [
                'ppo/loss/policy',  # ✅ Actual key name (found from logs)
                'ppo/policy/loss', 'objective/policy_loss', 'policy_loss',
                'ppo/policy/clipped_objective', 'objective/clipped_surrogate',
                'train/policy/loss', 'loss/policy'
            ]
            possible_value_keys = [
                'ppo/loss/value',  # ✅ Actual key name (found from logs)
                'ppo/value/loss', 'ppo/val/loss', 'value_loss',
                'objective/value_loss', 'train/value/loss', 'loss/value'
            ]
            
            # First try exact match
            for key in stats.keys():
                key_lower = key.lower()
                if 'policy' in key_lower and 'loss' in key_lower and 'clip' not in key_lower:
                    policy_loss_key = key
                if 'value' in key_lower and 'loss' in key_lower and 'clip' not in key_lower:
                    value_loss_key = key
            
            # If exact match fails, try possible key names
            if policy_loss_key is None:
                for possible_key in possible_policy_keys:
                    if possible_key in stats:
                        policy_loss_key = possible_key
                        break
            
            if value_loss_key is None:
                for possible_key in possible_value_keys:
                    if possible_key in stats:
                        value_loss_key = possible_key
                        break
        
        # 🔍 Diagnostics: If policy loss key cannot be found, log warning and print all keys
        if policy_loss_key is None and isinstance(stats, dict):
            if len(self.training_stats["policy_losses"]) < 10:  # Warn in first 10 steps
                self.logger.warning(f"⚠️ Unable to find policy_loss key!")
                self.logger.warning(f"   Available keys: {list(stats.keys())}")
                self.logger.warning(f"   Tried key names: {possible_policy_keys}")
                # Try to find any key containing 'loss'
                loss_keys = [k for k in stats.keys() if 'loss' in k.lower()]
                if loss_keys:
                    self.logger.warning(f"   Keys containing 'loss': {loss_keys}")
        
        if value_loss_key is None and isinstance(stats, dict):
            if len(self.training_stats["value_losses"]) < 10:  # Warn in first 10 steps
                self.logger.warning(f"⚠️ Unable to find value_loss key!")
                self.logger.warning(f"   Available keys: {list(stats.keys())}")
        
        # Try to get loss values, use default 0 if not found, but log warning
        if isinstance(stats, dict):
            if policy_loss_key:
                policy_loss_value = to_scalar(stats.get(policy_loss_key), 0)
            else:
                # Try last fallback: directly search all possible values
                policy_loss_value = to_scalar(
                    stats.get("ppo/loss/policy") or  # ✅ Actual key name (found from logs)
                    stats.get("ppo/policy/loss") or 
                    stats.get("objective/policy_loss") or 
                    stats.get("policy_loss") or 0
                )
                if policy_loss_value == 0 and len(self.training_stats["policy_losses"]) < 5:
                    # Check if all loss-related keys are 0 or non-existent
                    all_loss_zero = True
                    for key in stats.keys():
                        if 'loss' in key.lower() and to_scalar(stats.get(key), -1) != 0:
                            all_loss_zero = False
                            break
                    if all_loss_zero:
                        self.logger.error(f"❌ All loss-related keys are 0 or non-existent!")
                        self.logger.error(f"   This may indicate PPO training was not executed correctly!")
            
            if value_loss_key:
                value_loss_value = to_scalar(stats.get(value_loss_key), 0)
            else:
                value_loss_value = to_scalar(
                    stats.get("ppo/loss/value") or  # ✅ Actual key name (found from logs)
                    stats.get("ppo/value/loss") or 
                    stats.get("ppo/val/loss") or 
                    stats.get("value_loss") or 0
                )
        else:
            policy_loss_value = 0
            value_loss_value = 0
        
        # 🔍 Ensure all values are Python scalars (prevent JSON serialization errors)
        self.training_stats["policy_losses"].append(float(policy_loss_value))
        self.training_stats["value_losses"].append(float(value_loss_value))
        kl_value = to_scalar(stats.get("ppo/policy/kl") or stats.get("objective/kl") or 0 if isinstance(stats, dict) else 0)
        self.training_stats["kl_divergences"].append(float(kl_value))
        
        # Keep statistics for recent 1000 steps
        max_history = 1000
        for key in self.training_stats:
            if isinstance(self.training_stats[key], list) and len(self.training_stats[key]) > max_history:
                self.training_stats[key] = self.training_stats[key][-max_history:]
        
        # Periodically clean up memory
        current_step = len(self.training_stats["total_rewards"])
        self._cleanup_memory(current_step)
    
    def train(self, train_dataset, max_steps: Optional[int] = None):
        """Starts training (supports parallel data loading)"""
        try:
            max_steps = max_steps or self.config["training"]["max_steps"]

            # ✅ Compatible with HF datasets: Convert arrow dataset to raw dict list, only take question field
            if train_dataset is None:
                raise ValueError("Training dataset is empty, cannot start training")
            if hasattr(train_dataset, "with_format"):
                try:
                    # Try to extract only question field
                    questions = train_dataset.with_format(type=None)["question"]
                    train_dataset = [{"question": q, "answer": train_dataset[i].get("answer", "")}
                                     for i, q in enumerate(questions)]
                except Exception:
                    train_dataset = list(train_dataset)
            elif not isinstance(train_dataset, (list, tuple)):
                train_dataset = list(train_dataset)
            
            # Check if resuming from checkpoint
            current_step = self.training_stats.get("step", 0)
            start_step = current_step
            remaining_steps = max_steps - current_step
            
            if current_step > 0:
                self.logger.info(f"Resuming training from checkpoint, current step: {current_step}")
                self.logger.info(f"Remaining training steps: {remaining_steps}")
            else:
                self.logger.info(f"Starting new RL training, max steps: {max_steps}")
            
            # Set up parallel data loader
            use_parallel = self._use_parallel
            if use_parallel and self.config.get("parallel", {}).get("use_parallel_data_loader", True):
                batch_size = self.config["ppo"]["batch_size"]
                num_workers = self.config.get("parallel", {}).get("data_loader_workers", 4)
                self.parallel_data_loader = ParallelDataLoader(
                    train_dataset, batch_size, num_workers, shuffle=True
                )
                self.logger.info("Parallel data loader enabled")
            
            # Create progress bar
            progress_bar = tqdm(range(current_step, max_steps),
                                desc="RL Training Progress",
                                unit="step",
                                ncols=100,
                                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            
            # Training statistics
            start_time = time.time()
            step_times = []
            
            for step in range(current_step, max_steps):
                progress_bar.set_description(f"RL Training Progress (Step {step+1}/{max_steps})")
                step_start_time = time.time()
                
                # ⭐⭐⭐⭐☆ Force memory cleanup every 2 steps (more frequent)
                if step > 0 and step % self._force_cleanup_every_n_steps == 0:
                    self._cleanup_memory(step, force=True)
                
                # Create batch data
                batch = None  # ✅ Initialize batch variable to avoid UnboundLocalError
                try:
                    batch = self._create_batch(train_dataset)
                    
                    # Execute training step
                    stats = self.train_step(batch)
                except torch.cuda.OutOfMemoryError as e:
                    # 🔥 OOM error handling: Clean up memory
                    self.logger.error(f"❌ Step {step + 1} OOM error, cleaning up memory...")
                    self._cleanup_memory(step, force=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    raise RuntimeError(f"OOM error, consider further reducing batch_size or max_length: {e}")
                except RuntimeError as e:
                    error_str = str(e)
                    # 🔍 CUDA device-side assert error handling
                    if "device-side assert" in error_str or "CUDA error" in error_str:
                        self.logger.error(f"❌ Step {step + 1} CUDA device-side assert error")
                        self.logger.error(f"   Error message: {error_str[:200]}")
                        # Clear CUDA cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        # 🔍 Critical: CUDA errors may corrupt model state, need to skip this step
                        self.logger.warning(f"   Skipping this training step, continuing training...")
                        stats = None  # Mark as failed
                    else:
                        # Other RuntimeErrors raise directly
                        raise
                finally:
                    # ✅ Fix: Safely clean up batch, check if exists
                    if batch is not None:
                        try:
                            del batch
                        except:
                            pass
                    
                    # Periodically clean up memory
                    if step % self._memory_cleanup_interval == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Check if train_step succeeded (may return None)
                if stats is None:
                    self.logger.warning(f"Training step {step + 1} failed, skipping this update")
                    # Calculate step time
                    step_time = time.time() - step_start_time
                    step_times.append(step_time)
                    # Continue with empty stats dict
                    stats = {}
                    continue
                
                # Calculate step time
                step_time = time.time() - step_start_time
                step_times.append(step_time)
                
                # Update progress bar information
                avg_reward = float(np.mean(self.training_stats["total_rewards"][-10:])) if self.training_stats["total_rewards"] else 0.0
                avg_step_time = float(np.mean(step_times[-10:])) if step_times and len(step_times) > 0 else 0.0
                progress_bar.set_postfix({
                    'avg_reward': f'{avg_reward:.4f}',
                    'step_time': f'{step_time:.2f}s',
                    'avg_step_time': f'{avg_step_time:.2f}s'
                })
                
                # Periodic saving and evaluation
                if (step + 1) % self.config["training"]["save_steps"] == 0:
                    try:
                        self.save_checkpoint(step + 1)
                        progress_bar.write(f"✅ Checkpoint saved: step {step + 1}")
                    except Exception as e:
                        # 🔍 Critical fix: Checkpoint save failure should not interrupt training
                        self.logger.error(f"❌ Step {step + 1} checkpoint save failed: {e}")
                        self.logger.error(f"   Training will continue, but this checkpoint may be incomplete or lost")
                        import traceback
                        self.logger.error(f"   Detailed error: {traceback.format_exc()}")
                        # Do not raise exception, continue training
                    
                    # Save adaptive weights
                    if self.config["reward"].get("use_adaptive_weights", True):
                        weight_file = f"{self.config['ppo']['output_dir']}/adaptive_weights_step_{step + 1}.json"
                        self.reward_combiner.save_weights(weight_file)
                
                if (step + 1) % self.config["training"]["eval_steps"] == 0:
                    self.evaluate_model()
                    progress_bar.write(f"📊 Model evaluation completed: step {step + 1}")
                
                # Log output
                if (step + 1) % self.config["training"]["logging_steps"] == 0:
                    # Ensure stats is not None
                    if stats is not None:
                        self._log_training_progress(step + 1, stats)
                    else:
                        self.logger.warning(f"Step {step + 1} stats is None, skipping log recording")
                    progress_bar.write(f"📝 Training log: step {step + 1}")
                    
                    # Output adaptive weight status
                    if self.config["reward"].get("use_adaptive_weights", True):
                        weight_stats = self.reward_combiner.get_statistics()
                        if "adaptive_weights" in weight_stats:
                            intrinsic_weight = float(weight_stats['adaptive_weights'].get('intrinsic', 0.0))
                            correctness_weight = float(weight_stats['adaptive_weights'].get('correctness', 0.0))
                            progress_bar.write(f"🎯 Adaptive weights: intrinsic={intrinsic_weight:.4f}, "
                                             f"correctness={correctness_weight:.4f}")
            
            # Close progress bar
            progress_bar.close()
            
            # Save final model
            final_model_dir = self.config['ppo']['output_dir']
            self.save_final_model(final_model_dir)
            
            # Training completion statistics
            total_time = time.time() - start_time
            avg_step_time = float(np.mean(step_times)) if step_times and len(step_times) > 0 else 0.0
            
            self.logger.info("🎉 RL training completed!")
            self.logger.info(f"⏱️  Total training time: {total_time:.2f} seconds")
            self.logger.info(f"⚡ Average step time: {float(avg_step_time):.2f} seconds")
            final_avg_reward = float(np.mean(self.training_stats['total_rewards'][-10:])) if self.training_stats['total_rewards'] else 0.0
            self.logger.info(f"📈 Final average reward: {final_avg_reward:.4f}")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def _create_batch(self, dataset) -> Dict[str, List[str]]:
        """Creates batch data (supports parallel data loading)"""
        batch_size = self.config["ppo"]["batch_size"]
        
        # Check if dataset is empty
        if len(dataset) == 0:
            raise ValueError("Dataset is empty, cannot create batch")
        
        # If batch size is larger than dataset size, use dataset size and allow repeated sampling
        actual_batch_size = min(batch_size, len(dataset))
        replace = actual_batch_size > len(dataset)
        
        # Use parallel data loader or directly random selection
        if self.parallel_data_loader:
            # Use parallel data loader
            try:
                batch_data = next(iter(self.parallel_data_loader))
                questions = [item["question"] for item in batch_data]
            except StopIteration:
                # If data loader is empty, fall back to random selection
                self.logger.warning("Parallel data loader is empty, falling back to random selection")
                indices = np.random.choice(len(dataset), size=actual_batch_size, replace=replace)
                questions = [dataset[int(i)]["question"] for i in indices]
        else:
            # Randomly select samples
            indices = np.random.choice(len(dataset), size=actual_batch_size, replace=replace)
            # Convert numpy.int64 to Python int to avoid TypeError
            questions = [dataset[int(i)]["question"] for i in indices]
        
        return {"questions": questions}
    
    def save_checkpoint(self, step: int):
        """Saves checkpoint"""
        checkpoint_dir = f"{self.config['ppo']['output_dir']}/checkpoint-{step}"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        save_success = True
        failed_parts = []
        
        # Ensure student model holds latest policy weights before saving
        self._sync_policy_to_student()
        
        # Save student model
        try:
            self.student_model.save_model(checkpoint_dir)
            self.logger.debug("✓ Student model saved")
        except Exception as e:
            self.logger.error(f"❌ Student model save failed: {e}")
            save_success = False
            failed_parts.append("student_model")
        
        # Save cache
        try:
            cache_file = os.path.join(checkpoint_dir, "teacher_cache.pkl")
            self.cache_manager.save_cache(cache_file)
            self.logger.debug("✓ Teacher cache saved")
        except Exception as e:
            self.logger.error(f"❌ Teacher cache save failed: {e}")
            # Cache failure is not considered a critical error, only log warning
            failed_parts.append("teacher_cache")
        
        # Save training statistics
        try:
            stats_file = os.path.join(checkpoint_dir, "training_stats.json")
            import json
            
            # 🔍 Ensure all values are JSON serializable
            def make_json_serializable(obj):
                """Recursively converts objects to JSON serializable format"""
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, (np.ndarray, np.generic)):
                    return float(obj.item() if hasattr(obj, 'item') else float(obj))
                elif isinstance(obj, torch.Tensor):
                    return float(obj.item() if hasattr(obj, 'item') else float(obj))
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, (int, float, str, bool)) or obj is None:
                    return obj
                else:
                    # Try to convert to string (last fallback)
                    try:
                        return str(obj)
                    except:
                        return None
            
            # Create serializable copy of training statistics
            serializable_stats = make_json_serializable(self.training_stats)
            
            # 🔍 Validate data integrity before saving
            if not isinstance(serializable_stats.get("step"), (int, float)):
                self.logger.warning(f"⚠️ 'step' type abnormal: {type(serializable_stats.get('step'))}, fixing to int")
                serializable_stats["step"] = int(self.training_stats.get("step", 0))
            
            # Validate list length consistency (for debugging)
            list_keys = ["total_rewards", "policy_losses", "value_losses", "kl_divergences"]
            list_lengths = {key: len(serializable_stats.get(key, [])) for key in list_keys}
            if len(set(list_lengths.values())) > 1:
                self.logger.warning(f"⚠️ Training statistics list lengths inconsistent: {list_lengths}")
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
            
            # Verify file was saved
            if os.path.exists(stats_file):
                file_size = os.path.getsize(stats_file) / 1024  # KB
                self.logger.debug(f"✓ Training statistics saved ({file_size:.2f} KB)")
            else:
                raise FileNotFoundError(f"Training statistics file does not exist after saving: {stats_file}")
        except TypeError as e:
            self.logger.error(f"❌ Training statistics JSON serialization failed (type error): {e}")
            self.logger.error(f"   This may be due to non-serializable types in training statistics (e.g., torch.Tensor)")
            # Try to save a simplified version
            try:
                simple_stats = {
                    "step": int(self.training_stats.get("step", 0)),
                    "total_rewards_count": len(self.training_stats.get("total_rewards", [])),
                    "policy_losses_count": len(self.training_stats.get("policy_losses", [])),
                    "value_losses_count": len(self.training_stats.get("value_losses", [])),
                    "kl_divergences_count": len(self.training_stats.get("kl_divergences", [])),
                    "last_10_rewards": [float(r) if isinstance(r, (int, float)) else 0.0 
                                      for r in self.training_stats.get("total_rewards", [])[-10:]],
                    "last_10_policy_losses": [float(l) if isinstance(l, (int, float)) else 0.0 
                                            for l in self.training_stats.get("policy_losses", [])[-10:]],
                    "note": "Full statistics serialization failed, only saving summary information"
                }
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(simple_stats, f, indent=2, ensure_ascii=False)
                self.logger.warning(f"⚠️ Saved simplified training statistics (summary information)")
            except Exception as e2:
                self.logger.error(f"❌ Saving simplified training statistics also failed: {e2}")
                save_success = False
                failed_parts.append("training_stats")
        except Exception as e:
            self.logger.error(f"❌ Training statistics save failed: {e}")
            import traceback
            self.logger.error(f"Detailed error: {traceback.format_exc()}")
            save_success = False
            failed_parts.append("training_stats")
        
        # Summarize save results
        if save_success:
            if failed_parts:
                self.logger.warning(f"⚠️ Checkpoint partially saved: {checkpoint_dir} (failed parts: {failed_parts})")
            else:
                self.logger.info(f"✓ Checkpoint saved: {checkpoint_dir}")
        else:
            self.logger.error(f"❌ Checkpoint save failed: {checkpoint_dir} (failed parts: {failed_parts})")
            raise RuntimeError(f"Checkpoint save failed, critical components not saved: {failed_parts}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Resumes training from checkpoint"""
        try:
            import json
            import pickle
            
            if not Path(checkpoint_dir).exists():
                raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
            
            self.logger.info(f"Resuming from checkpoint: {checkpoint_dir}")
            
            # Load model (need to initialize model first)
            if self.student_model is None:
                raise ValueError("Model not initialized, cannot load checkpoint. Please call setup_models() first")
            
            # Load student model weights
            self.logger.info("Loading student model weights...")
            self.student_model.load_model(checkpoint_dir, load_adapter=True)
            
            # Re-setup PPO model (because model weights have been updated)
            self.logger.info("Re-setting up PPO model...")
            self.ppo_model = self.student_model.setup_for_ppo()
            
            # Re-setup PPO trainer
            self.logger.info("Re-setting up PPO trainer...")
            self.setup_ppo_trainer()

            # After loading latest LoRA, sync weights back to policy model
            self._sync_student_to_policy()
            
            # Load cache
            cache_file = os.path.join(checkpoint_dir, "teacher_cache.pkl")
            if os.path.exists(cache_file):
                self.logger.info("Loading cache...")
                self.cache_manager.load_cache(cache_file)
            
            # Load training statistics
            stats_file = os.path.join(checkpoint_dir, "training_stats.json")
            if os.path.exists(stats_file):
                self.logger.info("Loading training statistics...")
                try:
                    with open(stats_file, 'r', encoding='utf-8') as f:
                        loaded_stats = json.load(f)
                    
                    # 🔍 Validate integrity of loaded statistics
                    required_keys = ["step", "total_rewards", "policy_losses", "value_losses", "kl_divergences"]
                    missing_keys = [key for key in required_keys if key not in loaded_stats]
                    if missing_keys:
                        self.logger.warning(f"⚠️ Loaded training statistics missing keys: {missing_keys}, will use default values")
                        # Supplement missing keys
                        for key in missing_keys:
                            if key == "step":
                                loaded_stats[key] = 0
                            else:
                                loaded_stats[key] = []
                    
                    # Validate data types
                    if not isinstance(loaded_stats.get("step"), int):
                        self.logger.warning(f"⚠️ 'step' type incorrect: {type(loaded_stats.get('step'))}, converting to int")
                        loaded_stats["step"] = int(loaded_stats.get("step", 0))
                    
                    for key in ["total_rewards", "policy_losses", "value_losses", "kl_divergences"]:
                        if key in loaded_stats and not isinstance(loaded_stats[key], list):
                            self.logger.warning(f"⚠️ '{key}' type incorrect: {type(loaded_stats[key])}, converting to list")
                            loaded_stats[key] = []
                    
                    self.training_stats = loaded_stats
                    self.logger.info(f"✓ Training statistics loaded successfully")
                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ Training statistics JSON parsing failed: {e}")
                    self.logger.warning("Will use empty training statistics, starting from step 0")
                    # Do not raise exception, continue training
                except Exception as e:
                    self.logger.error(f"❌ Loading training statistics failed: {e}")
                    import traceback
                    self.logger.error(f"Detailed error: {traceback.format_exc()}")
                    self.logger.warning("Will use empty training statistics, starting from step 0")
                    # Do not raise exception, continue training
            
            self.logger.info(f"✅ Successfully resumed from checkpoint")
            # 🔍 Ensure step is int type
            current_step = int(self.training_stats.get('step', 0))
            self.training_stats['step'] = current_step
            self.logger.info(f"   Current step: {current_step}")
            
            # Safely calculate average reward (avoid empty list or type errors)
            rewards_list = self.training_stats.get('total_rewards', [])
            if rewards_list and len(rewards_list) > 0:
                try:
                    # Ensure all values are numeric types
                    numeric_rewards = [float(r) for r in rewards_list[-100:] if isinstance(r, (int, float))]
                    if numeric_rewards:
                        avg_reward = np.mean(numeric_rewards)
                        self.logger.info(f"   Average reward: {avg_reward:.4f}")
                    else:
                        self.logger.warning("   Average reward: No valid reward data")
                except Exception as e:
                    self.logger.warning(f"   Failed to calculate average reward: {e}")
            else:
                self.logger.warning("   Average reward: Reward list is empty")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Checkpoint loading failed: {e}")
            import traceback
            self.logger.error(f"Detailed error: {traceback.format_exc()}")
            raise
    
    def save_final_model(self, save_path: str):
        """Saves the final trained model"""
        try:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            # Sync latest policy weights before saving final model
            self._sync_policy_to_student()
            
            # Save student model
            self.student_model.save_model(save_path)
            
            # Save training configuration
            config_file = os.path.join(save_path, "training_config.yaml")
            import yaml
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            # Save final training statistics
            final_stats_file = os.path.join(save_path, "final_training_stats.json")
            import json
            
            # 🔍 Ensure all values are JSON serializable (reuse same function)
            def make_json_serializable(obj):
                """Recursively converts objects to JSON serializable format"""
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, (np.ndarray, np.generic)):
                    return float(obj.item() if hasattr(obj, 'item') else float(obj))
                elif isinstance(obj, torch.Tensor):
                    return float(obj.item() if hasattr(obj, 'item') else float(obj))
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, (int, float, str, bool)) or obj is None:
                    return obj
                else:
                    try:
                        return str(obj)
                    except:
                        return None
            
            # Create serializable copy of training statistics
            serializable_stats = make_json_serializable(self.training_stats)
            
            # 🔍 Validate data integrity before saving
            if not isinstance(serializable_stats.get("step"), (int, float)):
                self.logger.warning(f"⚠️ 'step' type abnormal: {type(serializable_stats.get('step'))}, fixing to int")
                serializable_stats["step"] = int(self.training_stats.get("step", 0))
            
            with open(final_stats_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
            
            # Verify file was saved
            if os.path.exists(final_stats_file):
                file_size = os.path.getsize(final_stats_file) / 1024  # KB
                self.logger.info(f"✓ Final training statistics saved ({file_size:.2f} KB)")
            else:
                raise FileNotFoundError(f"Final training statistics file does not exist after saving: {final_stats_file}")
            
            # Save model information
            model_info = {
                "model_type": "RL_trained_student_model",
                "base_model": self.config["model"]["student_model_name"],
                "teacher_model": self.config["model"]["teacher_model_name"],
                "training_steps": self.training_stats["step"],
                "final_reward": self.training_stats["total_rewards"][-1] if self.training_stats["total_rewards"] else 0.0,
                "training_date": str(Path().cwd()),
                "config_summary": {
                    "ppo_learning_rate": self.config["ppo"]["learning_rate"],
                    "ppo_epochs": self.config["ppo"]["ppo_epochs"],
                    "reward_lambda_intrinsic": self.config["reward"]["lambda_intrinsic"],
                    "reward_lambda_correctness": self.config["reward"]["lambda_correctness"]
                }
            }
            
            model_info_file = os.path.join(save_path, "model_info.json")
            with open(model_info_file, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            self.logger.info(f"Final model saved to: {save_path}")
            self.logger.info(f"Model information: {model_info}")
            
        except Exception as e:
            self.logger.error(f"Final model save failed: {e}")
            raise
    
    def evaluate_model(self):
        """Evaluates the model"""
        try:
            # Ensure student model holds latest weights before evaluation
            self._sync_policy_to_student()
            # Here can implement specific evaluation logic
            # For example, test accuracy on validation set, etc.
            
            avg_reward = np.mean(self.training_stats["total_rewards"][-100:]) if self.training_stats["total_rewards"] else 0.0
            
            self.logger.info(f"Model evaluation - Average reward: {avg_reward:.4f}")
            
            if self.config.get("logging", {}).get("use_wandb", False):
                wandb.log({
                    "eval/avg_reward": avg_reward,
                    "eval/step": self.training_stats["step"]
                })
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
    
    def _log_training_progress(self, step: int, stats: Dict):
        """Logs training progress"""
        # Check if stats is None
        if stats is None:
            self.logger.warning(f"Step {step} stats is None, cannot log training progress")
            return
        
        # Ensure conversion to Python scalars to avoid numpy array formatting errors
        avg_reward = float(np.mean(self.training_stats["total_rewards"][-100:])) if self.training_stats["total_rewards"] else 0.0
        avg_kl = float(np.mean(self.training_stats["kl_divergences"][-100:])) if self.training_stats["kl_divergences"] else 0.0
        
        # Get values from stats and convert to Python scalars
        # 🔍 Correctly get loss value: do not use clipfrac to impersonate loss
        policy_loss = 0.0
        if isinstance(stats, dict):
            # First try to find the true loss key
            for key in stats.keys():
                if 'policy' in key and 'loss' in key and 'clip' not in key:
                    policy_loss = stats[key]
                    break
            else:
                # If not found, try standard key name
                policy_loss = stats.get('ppo/policy/loss', 0)
            
            if isinstance(policy_loss, (np.ndarray, np.generic)):
                policy_loss = float(policy_loss.item() if hasattr(policy_loss, 'item') else float(policy_loss))
            else:
                policy_loss = float(policy_loss)
        
        # Basic training information
        log_message = (
            f"Step {step}: "
            f"Avg Reward: {avg_reward:.4f}, "
            f"Avg KL: {avg_kl:.4f}, "
            f"Policy Loss: {policy_loss:.4f}"
        )
        
        # If adaptive weights are enabled, add weight information
        if self.config["reward"].get("use_adaptive_weights", False):
            current_weights = self.reward_combiner.get_current_weights()
            log_message += (
                f" | Weights - Intrinsic: {current_weights['intrinsic']:.3f}, "
                f"Correctness: {current_weights['correctness']:.3f}"
            )
            if current_weights.get('reasoning', 0) > 0:
                log_message += f", Reasoning: {current_weights['reasoning']:.3f}"
            if current_weights.get('format', 0) > 0:
                log_message += f", Format: {current_weights['format']:.3f}"
        
        self.logger.info(log_message)
    
    def cleanup(self):
        """Cleans up resources"""
        try:
            # Stop async cache worker thread
            if self.async_cache_manager:
                self.async_cache_manager.stop_async_worker()
            
            # Clean up cache manager
            if self.cache_manager:
                self.cache_manager.cleanup()
            
            # Clean up parallel processor
            if self.parallel_processor:
                self.parallel_processor = None
            
            # Clean up parallel inference engines
            if self.parallel_inference_student:
                self.parallel_inference_student = None
            if self.parallel_inference_teacher:
                self.parallel_inference_teacher = None
            
            # Clean up parallel data loader
            if self.parallel_data_loader:
                self.parallel_data_loader = None
            
            if self.config.get("logging", {}).get("use_wandb", False):
                wandb.finish()
            
            self.logger.info("Resource cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {e}")


def load_config(config_path: str) -> Dict:
    """Loads configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main function"""
    # Load configuration
    config = load_config("config/training_config.yaml")
    
    # Create trainer
    trainer = RLTrainer(config)
    
    # Set up models and components
    trainer.setup_models()
    trainer.setup_components()
    trainer.setup_ppo_trainer()
    
    # Load GSM8K dataset
    from datasets import load_dataset
    from data.gsm8k_processor import GSM8KProcessor
    
    print("Loading GSM8K dataset...")
    try:
        # Load GSM8K dataset
        gsm8k_dataset = load_dataset("gsm8k", "main")
        
        # Create GSM8K processor
        processor = GSM8KProcessor(trainer.student_model.tokenizer, max_length=config["ppo"]["max_length"])
        
        # Use training set as training data
        dataset = gsm8k_dataset["train"]
        
        print(f"Training set size: {len(dataset)}")
        
        # Validate dataset quality
        processor.validate_data(dataset, num_samples=3)
        
        # Train
        trainer.train(dataset, max_steps=100)
        
    except Exception as e:
        print(f"Failed to load GSM8K dataset: {e}")
        print("Cannot proceed with training, please check network connection and dependencies")
        return
        
    finally:
        # Clean up resources
        trainer.cleanup()


if __name__ == "__main__":
    main()



