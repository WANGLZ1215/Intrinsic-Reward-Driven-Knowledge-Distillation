"""
Student Model Wrapper Module
Function: Wrap Qwen-7B-math student model, support LoRA fine-tuning and PPO training
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import AutoModelForCausalLMWithValueHead
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
from utils.cache_utils import suppress_past_key_values_warning, update_model_for_modern_cache


class StudentModel:
    """Student Model Wrapper Class"""
    
    def __init__(self, model_name: str = "Qwen/Qwen-7B-Math",
                 lora_config: Optional[Dict] = None,
                 device: str = "auto", 
                 torch_dtype: torch.dtype = torch.bfloat16,
                 use_lora: bool = True):
        """
        Initialize student model
        
        Args:
            model_name: Model name
            lora_config: LoRA configuration
            device: Device
            torch_dtype: Data type
            use_lora: Whether to use LoRA
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_lora = use_lora
        self.lora_config = lora_config or self._default_lora_config()
        
        # Suppress past_key_values warnings
        suppress_past_key_values_warning()
        
        # Load model and tokenizer
        self._load_model()
        
        # Update model to use modern cache
        self.model = update_model_for_modern_cache(self.model)
        
        logging.info(f"Student model {model_name} loaded successfully")
        logging.info(f"LoRA configuration: {self.lora_config}")
    
    def _default_lora_config(self) -> Dict:
        """Default LoRA configuration"""
        return {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM
        }

    def _ensure_inputs_on_model_device(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Ensure all input tensors are on the same device as model parameters
        """
        try:
            target_device = next(self.model.parameters()).device
        except StopIteration:
            target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.device != target_device:
                inputs[key] = value.to(target_device)
        return inputs
    
    def _ensure_lora_trainable(self, model=None, context: str = ""):
        """
        Ensure LoRA weights are in trainable state and log statistics
        
        Args:
            model: Model to check (default: current model)
            context: Log context hint
        """
        if model is None:
            model = self.model
        if not self.use_lora or model is None:
            return
        
        lora_tensor_count = 0
        newly_enabled = 0
        for name, param in model.named_parameters():
            if "lora_" in name:
                lora_tensor_count += 1
                if not param.requires_grad:
                    param.requires_grad_(True)
                    newly_enabled += 1
        
        if lora_tensor_count == 0:
            logging.warning(f"⚠️ {context}No LoRA parameters detected, LoRA configuration may not be applied correctly")
        else:
            logging.info(
                f"✅ {context}LoRA parameter check: {lora_tensor_count} LoRA tensors total, {newly_enabled} newly enabled for training"
            )
    
    def _log_trainable_parameter_summary(self, model=None, context: str = ""):
        """
        Print trainable parameter statistics of current model for diagnosing if LoRA is effective
        """
        if model is None:
            model = self.model
        if model is None:
            logging.warning(f"⚠️ {context}Model is None, cannot count trainable parameters")
            return
        
        total_params = 0
        trainable_params = 0
        lora_params = 0
        other_trainable_names = []
        
        for name, param in model.named_parameters():
            numel = param.numel()
            total_params += numel
            if param.requires_grad:
                trainable_params += numel
                if "lora_" in name:
                    lora_params += numel
                elif "v_head" not in name:
                    other_trainable_names.append(name)
        
        trainable_ratio = (trainable_params / total_params * 100) if total_params > 0 else 0.0
        lora_ratio = (lora_params / trainable_params * 100) if trainable_params > 0 else 0.0
        
        logging.info(
            f"📊 {context}Trainable parameter statistics: total={total_params:,}, trainable={trainable_params:,} "
            f"({trainable_ratio:.2f}%), LoRA parameters={lora_params:,} ({lora_ratio:.2f}% of trainable parameters)"
        )
        if other_trainable_names:
            logging.info(
                f"ℹ️ {context}Other trainable weights besides LoRA: {other_trainable_names[:10]}"
                + ("..." if len(other_trainable_names) > 10 else "")
            )
        if lora_params == 0 and self.use_lora:
            logging.warning(f"⚠️ {context}LoRA weights are not set to trainable, please check LoRA configuration and loading process")
    
    def _load_model(self):
        """Load model and tokenizer"""
        try:
            import os
            from pathlib import Path
            
            # Check if model_name is a local path or HuggingFace model name
            model_path = Path(self.model_name)
            is_local_path = model_path.exists() and model_path.is_dir()
            
            # Check if it's a trained model (contains adapter files)
            is_trained_model = False
            if is_local_path:
                adapter_files = [
                    model_path / "adapter_model.bin",
                    model_path / "adapter_model.safetensors",
                    model_path / "adapter_config.json"
                ]
                is_trained_model = any(f.exists() for f in adapter_files[:2])  # Check weight files
                
                if is_trained_model:
                    logging.info(f"Detected trained model directory: {self.model_name}")
                    logging.info("Will load trained LoRA adapter")
            
            if is_trained_model and self.use_lora:
                # Case 1: Load trained LoRA adapter
                # First need to load base model (from config or parent directory of model directory)
                base_model_name = None
                
                # Check if adapter_config.json specifies base model
                config_path = model_path / "adapter_config.json"
                if config_path.exists():
                    import json
                    with open(config_path, 'r') as f:
                        adapter_config = json.load(f)
                        # PEFT adapter config contains base model path
                        base_model_name = adapter_config.get("base_model_name_or_path", None)
                
                # If cannot get base model from adapter config, use default base model
                if base_model_name is None:
                    # Use base model name from config file, or default value
                    default_base = "Qwen/Qwen2.5-7B-Instruct"
                    logging.info(f"Base model path not found in adapter config, using default: {default_base}")
                    base_model_name = default_base
                else:
                    # Check if base_model_name is a local path and exists
                    if Path(base_model_name).exists():
                        logging.info(f"Found base model path from adapter config: {base_model_name}")
                    else:
                        # May be HuggingFace model name, use directly
                        logging.info(f"Found base model from adapter config: {base_model_name}")
                
                # ✅ Fix: Load tokenizer first (from base model or local directory)
                # If local directory has tokenizer.json, prefer local
                if (model_path / "tokenizer.json").exists() or (model_path / "tokenizer_config.json").exists():
                    logging.info(f"Loading tokenizer from local directory: {self.model_name}")
                    tokenizer_path = self.model_name
                else:
                    logging.info(f"Loading tokenizer from base model: {base_model_name}")
                    tokenizer_path = base_model_name
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=True,
                    padding_side="left"
                )
                
                # Set pad token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load base model
                # Note: If using device_map, must set low_cpu_mem_usage=True
                # RL training requires model fully on GPU, cannot use offload
                device_map_for_load = None if self.device == "auto" else self.device
                # ✅ Fix: If using device_map, must set low_cpu_mem_usage=True
                low_cpu_mem_usage = True if device_map_for_load is not None else False
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=self.torch_dtype,
                    device_map=device_map_for_load,  # If None, do not use device_map
                    trust_remote_code=True,
                    low_cpu_mem_usage=low_cpu_mem_usage  # ✅ Fix: Must be True when using device_map
                )
                
                # If single CUDA device specified during loading, ensure all parameters moved to that device
                if isinstance(device_map_for_load, str) and device_map_for_load.startswith("cuda"):
                    try:
                        target_device = torch.device(device_map_for_load)
                        self.base_model = self.base_model.to(target_device)
                        logging.info(f"✅ Base model moved to device: {target_device}")
                    except Exception as move_err:
                        logging.warning(f"⚠️ Failed to move base model to device {device_map_for_load}: {move_err}")
                
                # If device_map is None, need to manually move to device
                if device_map_for_load is None and torch.cuda.is_available():
                    self.base_model = self.base_model.to(torch.device("cuda:0"))
                
                # 🔥 Critical fix: Base first, then LoRA - resize_token_embeddings must be executed before loading LoRA
                tokenizer_vocab_size = len(self.tokenizer)
                try:
                    # Get actual embedding size of base model
                    input_emb_size = self.base_model.get_input_embeddings().weight.size(0)
                    output_emb_size = None
                    if hasattr(self.base_model, 'get_output_embeddings') and self.base_model.get_output_embeddings() is not None:
                        output_emb_size = self.base_model.get_output_embeddings().weight.size(0)
                    
                    logging.info(f"📊 Embedding size check (before loading LoRA):")
                    logging.info(f"   tokenizer vocab_size: {tokenizer_vocab_size}")
                    logging.info(f"   model input_embeddings.size(0): {input_emb_size}")
                    if output_emb_size is not None:
                        logging.info(f"   model output_embeddings.size(0): {output_emb_size}")
                    logging.info(f"   model.config.vocab_size: {getattr(self.base_model.config, 'vocab_size', 'N/A')}")
                    
                    # If embedding size doesn't match tokenizer, perform resize
                    if input_emb_size != tokenizer_vocab_size:
                        logging.warning(f"⚠️ Model embedding size ({input_emb_size}) != tokenizer size ({tokenizer_vocab_size})")
                        logging.info(f"   Resizing token_embeddings to {tokenizer_vocab_size}...")
                        self.base_model.resize_token_embeddings(tokenizer_vocab_size)
                        logging.info(f"✅ resize_token_embeddings completed")
                        
                        # Verify if resize was successful
                        new_input_emb_size = self.base_model.get_input_embeddings().weight.size(0)
                        if new_input_emb_size != tokenizer_vocab_size:
                            logging.error(f"❌ resize_token_embeddings failed! New size: {new_input_emb_size} != {tokenizer_vocab_size}")
                            logging.warning(f"   Will keep 'limit range + clamp' safety strategy")
                        else:
                            logging.info(f"✅ Resize verification successful: input_embeddings.size(0) = {new_input_emb_size}")
                    else:
                        logging.info(f"✅ Embedding size matches tokenizer, no resize needed")
                    
                    # 🔥 Critical fix: Ensure vocab_size and pad/eos_token_id in model.config match tokenizer
                    self.base_model.config.vocab_size = tokenizer_vocab_size
                    self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
                    self.base_model.config.eos_token_id = self.tokenizer.eos_token_id
                    logging.info(f"✅ Updated base_model.config: vocab_size={tokenizer_vocab_size}, pad_token_id={self.tokenizer.pad_token_id}, eos_token_id={self.tokenizer.eos_token_id}")
                        
                except Exception as e:
                    logging.warning(f"⚠️ Error during resize_token_embeddings (may not be supported or already quantized): {e}")
                    logging.warning(f"   Will keep 'limit range + clamp' safety strategy")
                
                # Load trained LoRA adapter
                self.model = PeftModel.from_pretrained(
                    self.base_model,
                    self.model_name,
                    torch_dtype=self.torch_dtype
                )
                logging.info(f"Successfully loaded trained LoRA adapter from: {self.model_name}")
                self._ensure_lora_trainable(self.model, context="After loading trained LoRA: ")
                self._log_trainable_parameter_summary(self.model, context="After loading trained LoRA: ")

                # Ensure LoRA adapter is on same device as base model
                if isinstance(device_map_for_load, str) and device_map_for_load.startswith("cuda"):
                    try:
                        target_device = torch.device(device_map_for_load)
                        self.model = self.model.to(target_device)
                        logging.info(f"✅ Student model moved to device: {target_device}")
                    except Exception as move_err:
                        logging.warning(f"⚠️ Failed to move student model to device {device_map_for_load}: {move_err}")
                
                # 🔥 Critical: Verify embedding size again after loading LoRA, resize again if still not matching
                try:
                    final_input_emb_size = self.model.get_input_embeddings().weight.size(0)
                    final_output_emb_size = None
                    if hasattr(self.model, 'get_output_embeddings') and self.model.get_output_embeddings() is not None:
                        final_output_emb_size = self.model.get_output_embeddings().weight.size(0)
                    logging.info(f"📊 Embedding size check (after loading LoRA):")
                    logging.info(f"   tokenizer vocab_size: {tokenizer_vocab_size}")
                    logging.info(f"   model input_embeddings.size(0): {final_input_emb_size}")
                    if final_output_emb_size is not None:
                        logging.info(f"   model output_embeddings.size(0): {final_output_emb_size}")
                    
                    # 🔥 Critical fix: If embedding size doesn't match after loading LoRA, resize again
                    if final_input_emb_size != tokenizer_vocab_size:
                        logging.warning(f"⚠️ After loading LoRA, input_embeddings size ({final_input_emb_size}) != tokenizer ({tokenizer_vocab_size})")
                        logging.info(f"   Resizing token_embeddings again to {tokenizer_vocab_size}...")
                        try:
                            self.model.resize_token_embeddings(tokenizer_vocab_size)
                            # Verify if resize was successful
                            new_final_input_emb_size = self.model.get_input_embeddings().weight.size(0)
                            if new_final_input_emb_size == tokenizer_vocab_size:
                                logging.info(f"✅ Resize after LoRA successful: input_embeddings = {new_final_input_emb_size}")
                            else:
                                logging.error(f"❌ Resize after LoRA failed: {new_final_input_emb_size} != {tokenizer_vocab_size}")
                        except Exception as e:
                            logging.warning(f"⚠️ Resize after LoRA failed: {e}")
                            logging.warning(f"   Will use 'limit range + clamp' strategy")
                    
                    # 🔥 Critical: Check and resize output embeddings (lm_head)
                    if final_output_emb_size is not None and final_output_emb_size != tokenizer_vocab_size:
                        logging.warning(f"⚠️ output_embeddings size ({final_output_emb_size}) != tokenizer ({tokenizer_vocab_size})")
                        # resize_token_embeddings should resize both input and output, but if it fails, check manually
                        try:
                            # Check if fixed after resize
                            check_output_emb_size = self.model.get_output_embeddings().weight.size(0)
                            if check_output_emb_size != tokenizer_vocab_size:
                                logging.warning(f"   output_embeddings still not matching, will use 'limit range + clamp' strategy")
                        except:
                            pass
                except Exception as e:
                    logging.warning(f"⚠️ Unable to check embedding size after LoRA: {e}")
                
            else:
                # Case 2: Load base model and apply new LoRA configuration
                # Load tokenizer first
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    padding_side="left"
                )
                
                # Set pad token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Note: If using device_map, must set low_cpu_mem_usage=True
                # RL training requires model fully on GPU, cannot use offload
                device_map_for_load = None if self.device == "auto" else self.device
                # ✅ Fix: If using device_map, must set low_cpu_mem_usage=True
                low_cpu_mem_usage = True if device_map_for_load is not None else False
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    device_map=device_map_for_load,  # If None, do not use device_map
                    trust_remote_code=True,
                    low_cpu_mem_usage=low_cpu_mem_usage  # ✅ Fix: Must be True when using device_map
                )
                
                # If device_map is None, need to manually move to device
                if device_map_for_load is None and torch.cuda.is_available():
                    self.base_model = self.base_model.to(torch.device("cuda:0"))
                
                # 🔥 Critical fix: Base first, then LoRA - resize_token_embeddings must be executed before loading LoRA
                tokenizer_vocab_size = len(self.tokenizer)
                try:
                    # Get actual embedding size of base model
                    input_emb_size = self.base_model.get_input_embeddings().weight.size(0)
                    output_emb_size = None
                    if hasattr(self.base_model, 'get_output_embeddings') and self.base_model.get_output_embeddings() is not None:
                        output_emb_size = self.base_model.get_output_embeddings().weight.size(0)
                    
                    logging.info(f"📊 Embedding size check (before loading LoRA):")
                    logging.info(f"   tokenizer vocab_size: {tokenizer_vocab_size}")
                    logging.info(f"   model input_embeddings.size(0): {input_emb_size}")
                    if output_emb_size is not None:
                        logging.info(f"   model output_embeddings.size(0): {output_emb_size}")
                    logging.info(f"   model.config.vocab_size: {getattr(self.base_model.config, 'vocab_size', 'N/A')}")
                    
                    # If embedding size doesn't match tokenizer, perform resize
                    if input_emb_size != tokenizer_vocab_size:
                        logging.warning(f"⚠️ Model embedding size ({input_emb_size}) != tokenizer size ({tokenizer_vocab_size})")
                        logging.info(f"   Resizing token_embeddings to {tokenizer_vocab_size}...")
                        self.base_model.resize_token_embeddings(tokenizer_vocab_size)
                        logging.info(f"✅ resize_token_embeddings completed")
                        
                        # Verify if resize was successful
                        new_input_emb_size = self.base_model.get_input_embeddings().weight.size(0)
                        if new_input_emb_size != tokenizer_vocab_size:
                            logging.error(f"❌ resize_token_embeddings failed! New size: {new_input_emb_size} != {tokenizer_vocab_size}")
                            logging.warning(f"   Will keep 'limit range + clamp' safety strategy")
                        else:
                            logging.info(f"✅ Resize verification successful: input_embeddings.size(0) = {new_input_emb_size}")
                    else:
                        logging.info(f"✅ Embedding size matches tokenizer, no resize needed")
                    
                    # 🔥 Critical fix: Ensure vocab_size and pad/eos_token_id in model.config match tokenizer
                    self.base_model.config.vocab_size = tokenizer_vocab_size
                    self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
                    self.base_model.config.eos_token_id = self.tokenizer.eos_token_id
                    logging.info(f"✅ Updated base_model.config: vocab_size={tokenizer_vocab_size}, pad_token_id={self.tokenizer.pad_token_id}, eos_token_id={self.tokenizer.eos_token_id}")
                        
                except Exception as e:
                    logging.warning(f"⚠️ Error during resize_token_embeddings (may not be supported or already quantized): {e}")
                    logging.warning(f"   Will keep 'limit range + clamp' safety strategy")
                
                # Apply LoRA
                if self.use_lora:
                    peft_config = LoraConfig(**self.lora_config)
                    self.model = get_peft_model(self.base_model, peft_config)
                    logging.info("Applied new LoRA configuration")
                    self._ensure_lora_trainable(self.model, context="After applying new LoRA: ")
                    self._log_trainable_parameter_summary(self.model, context="After applying new LoRA: ")
                else:
                    self.model = self.base_model

                # Ensure model is on specified device
                if isinstance(device_map_for_load, str) and device_map_for_load.startswith("cuda"):
                    try:
                        target_device = torch.device(device_map_for_load)
                        self.model = self.model.to(target_device)
                        logging.info(f"✅ Student model moved to device: {target_device}")
                    except Exception as move_err:
                        logging.warning(f"⚠️ Failed to move student model to device {device_map_for_load}: {move_err}")
                
                # 🔥 Critical: Verify embedding size again after applying LoRA, resize again if still not matching
                try:
                    final_input_emb_size = self.model.get_input_embeddings().weight.size(0)
                    final_output_emb_size = None
                    if hasattr(self.model, 'get_output_embeddings') and self.model.get_output_embeddings() is not None:
                        final_output_emb_size = self.model.get_output_embeddings().weight.size(0)
                    logging.info(f"📊 Embedding size check (after loading LoRA):")
                    logging.info(f"   tokenizer vocab_size: {tokenizer_vocab_size}")
                    logging.info(f"   model input_embeddings.size(0): {final_input_emb_size}")
                    if final_output_emb_size is not None:
                        logging.info(f"   model output_embeddings.size(0): {final_output_emb_size}")
                    
                    # 🔥 Critical fix: If embedding size doesn't match after applying LoRA, resize again
                    if final_input_emb_size != tokenizer_vocab_size:
                        logging.warning(f"⚠️ After applying LoRA, input_embeddings size ({final_input_emb_size}) != tokenizer ({tokenizer_vocab_size})")
                        logging.info(f"   Resizing token_embeddings again to {tokenizer_vocab_size}...")
                        try:
                            self.model.resize_token_embeddings(tokenizer_vocab_size)
                            # Verify if resize was successful
                            new_final_input_emb_size = self.model.get_input_embeddings().weight.size(0)
                            if new_final_input_emb_size == tokenizer_vocab_size:
                                logging.info(f"✅ Resize after LoRA successful: input_embeddings = {new_final_input_emb_size}")
                            else:
                                logging.error(f"❌ Resize after LoRA failed: {new_final_input_emb_size} != {tokenizer_vocab_size}")
                        except Exception as e:
                            logging.warning(f"⚠️ Resize after LoRA failed: {e}")
                            logging.warning(f"   Will use 'limit range + clamp' strategy")
                    
                    # 🔥 Critical: Check and resize output embeddings (lm_head)
                    if final_output_emb_size is not None and final_output_emb_size != tokenizer_vocab_size:
                        logging.warning(f"⚠️ output_embeddings size ({final_output_emb_size}) != tokenizer ({tokenizer_vocab_size})")
                        # resize_token_embeddings should resize both input and output, but if it fails, check manually
                        try:
                            # Check if fixed after resize
                            check_output_emb_size = self.model.get_output_embeddings().weight.size(0)
                            if check_output_emb_size != tokenizer_vocab_size:
                                logging.warning(f"   output_embeddings still not matching, will use 'limit range + clamp' strategy")
                        except:
                            pass
                except Exception as e:
                    logging.warning(f"⚠️ Unable to check embedding size after LoRA: {e}")
            
            logging.info("Student model loaded successfully")
            
        except Exception as e:
            logging.error(f"Student model loading failed: {e}")
            raise
    
    def setup_for_ppo(self) -> AutoModelForCausalLMWithValueHead:
        """
        Setup model for PPO training
        
        Returns:
            Model with value head
        """
        try:
            # ValueHead model does not support CPU/disk offloading, need to ensure model is fully on GPU
            # Check if base model uses device_map="auto" (may have some layers offloaded)
            model_to_check = self.base_model if hasattr(self, 'base_model') else self.model
            has_device_map = False
            
            # Check device mapping (may be in different locations)
            if hasattr(model_to_check, 'hf_device_map') and model_to_check.hf_device_map:
                has_device_map = True
            elif hasattr(model_to_check, 'device_map') and model_to_check.device_map:
                has_device_map = True
            
            if has_device_map:
                logging.warning("Detected model uses device mapping, need to move all layers to single device to avoid ValueHead not supporting offloading")
                # Find first GPU device
                target_device = None
                if torch.cuda.is_available():
                    target_device = torch.device("cuda:0")
                    logging.info(f"Moving all model parameters to device: {target_device}")
                else:
                    target_device = torch.device("cpu")
                    logging.warning("CUDA not detected, using CPU (may affect performance)")
                
                # Check if model is on meta device
                # If first parameter's device type is meta, need to use to_empty instead of to
                try:
                    first_param = next(model_to_check.parameters())
                    is_meta_device = first_param.device.type == 'meta'
                except StopIteration:
                    is_meta_device = False
                
                # Move model to single device
                # For PEFT models, need to move both base model and PEFT model
                if hasattr(self, 'base_model'):
                    # Move base model
                    if is_meta_device:
                        logging.warning("Detected model on meta device, need to load weights first")
                        # For meta device, should reload instead of moving
                        # This should not normally happen, as we already disabled low_cpu_mem_usage during loading
                        raise RuntimeError("Model is on meta device. Please ensure low_cpu_mem_usage=False when loading model")
                    else:
                        if hasattr(self.base_model, 'to'):
                            self.base_model = self.base_model.to(target_device)
                    
                    # Move PEFT model
                    if hasattr(self.model, 'to'):
                        self.model = self.model.to(target_device)
                else:
                    # Regular model
                    if is_meta_device:
                        logging.warning("Detected model on meta device, need to load weights first")
                        raise RuntimeError("Model is on meta device. Please ensure low_cpu_mem_usage=False when loading model")
                    else:
                        if hasattr(self.model, 'to'):
                            self.model = self.model.to(target_device)
                
                # ✅ Fix: Cannot set hf_device_map to None, because TRL library expects it to be a dict
                # Should set it to dict format representing all layers on same device
                if hasattr(model_to_check, 'hf_device_map'):
                    # Get string representation of target device
                    if isinstance(target_device, torch.device):
                        device_str = str(target_device)
                    else:
                        device_str = target_device
                    
                    # If hf_device_map exists and is not None, update it to single device mapping
                    if model_to_check.hf_device_map is not None and isinstance(model_to_check.hf_device_map, dict):
                        # Unify all device mappings to target device
                        model_to_check.hf_device_map = {name: device_str for name in model_to_check.hf_device_map.keys()}
                    else:
                        # If hf_device_map is None or not a dict, create a basic mapping
                        # TRL library needs it to be a dict, so provide at least one key-value pair
                        model_to_check.hf_device_map = {"model": device_str}
                
                # device_map can be set to None, as it's not used by TRL library
                if hasattr(model_to_check, 'device_map'):
                    model_to_check.device_map = None
            
            # Create model with value head
            # Note: Do not use device_map="auto", because ValueHead does not support offloading
            # Ensure model is already on correct device (not meta device)
            
            # Check model's current device
            try:
                current_device = next(self.model.parameters()).device
                if current_device.type == 'meta':
                    raise RuntimeError("Model is still on meta device. Please ensure low_cpu_mem_usage=False when loading model")
            except StopIteration:
                # Model has no parameters, this is abnormal
                raise RuntimeError("Model has no parameters, cannot determine device location")
            
            # If model is on CPU, move to original GPU device (if available)
            if torch.cuda.is_available() and current_device.type == 'cpu':
                # Try to keep model on original device, not force move to cuda:0
                # Check actual device of model parameters
                try:
                    # Get first parameter's device as target device
                    for param in self.model.parameters():
                        if param.device.type == 'cuda':
                            target_device = param.device
                            break
                    else:
                        # If no GPU parameters found, use cuda:0
                        target_device = torch.device("cuda:0")
                except:
                    target_device = torch.device("cuda:0")
                
                logging.info(f"Moving model to GPU device: {target_device} (keeping original device)")
                self.model = self.model.to(target_device)
            
            # ✅ Fix: Ensure hf_device_map exists and is dict format before creating ValueHead
            # TRL library's post_init checks hf_device_map.values(), cannot be None
            if hasattr(self.model, 'hf_device_map'):
                if self.model.hf_device_map is None or not isinstance(self.model.hf_device_map, dict):
                    # Get current device
                    try:
                        current_device_str = str(next(self.model.parameters()).device)
                    except:
                        current_device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
                    # Create a valid device mapping
                    self.model.hf_device_map = {"model": current_device_str}
            
            # Create ValueHead model
            # Note: from_pretrained's first parameter can be model instance or path
            # Here pass model instance to ensure using already loaded weights
            ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.model,  # Pass model instance, not path
                torch_dtype=self.torch_dtype,
                device_map=None  # Do not use automatic device mapping, avoid offloading
            )
            
            # Ensure LoRA parameters remain trainable on PPO model again, and output statistics
            if hasattr(ppo_model, "pretrained_model"):
                self._ensure_lora_trainable(ppo_model.pretrained_model, context="PPO model (pretrained_model) stage: ")
                self._log_trainable_parameter_summary(ppo_model.pretrained_model, context="PPO model (pretrained_model) stage: ")
            self._log_trainable_parameter_summary(ppo_model, context="PPO model overall: ")
            
            # ✅ Enable gradient checkpointing (always enabled to maximize memory savings): sacrifice training speed for memory
            # Gradient checkpointing can save 30-40% activation memory, especially effective for log_softmax OOM
            # Note: Configuration will be checked again in rl_trainer, here always enabled to maximize memory savings
            try:
                # Try to enable on ppo_model
                if hasattr(ppo_model, 'gradient_checkpointing_enable'):
                    ppo_model.gradient_checkpointing_enable()
                    logging.info("✅ Gradient checkpointing enabled (saves activation memory)")
                # Try to enable on pretrained_model (AutoModelForCausalLMWithValueHead structure)
                elif hasattr(ppo_model, 'pretrained_model') and hasattr(ppo_model.pretrained_model, 'gradient_checkpointing_enable'):
                    ppo_model.pretrained_model.gradient_checkpointing_enable()
                    logging.info("✅ Gradient checkpointing enabled on pretrained_model (saves activation memory)")
                # Try to enable on base model
                elif hasattr(ppo_model, 'base_model') and hasattr(ppo_model.base_model, 'gradient_checkpointing_enable'):
                    ppo_model.base_model.gradient_checkpointing_enable()
                    logging.info("✅ Gradient checkpointing enabled on base_model (saves activation memory)")
            except Exception as e:
                logging.warning(f"⚠️ Failed to enable gradient checkpointing (may not be supported): {e}")
            
            logging.info("PPO model setup completed")
            return ppo_model
            
        except Exception as e:
            logging.error(f"PPO model setup failed: {e}")
            raise
    
    def generate(self, prompts: Union[str, List[str]], 
                max_length: int = 256,
                temperature: float = 0.7,
                do_sample: bool = True,
                top_p: float = 0.9,
                top_k: int = 50) -> Union[str, List[str]]:
        """
        Generate text
        
        Args:
            prompts: Prompt text or list
            max_length: Maximum length
            temperature: Temperature parameter
            do_sample: Whether to sample
            top_p: top_p parameter
            top_k: top_k parameter
            
        Returns:
            Generated text
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        self.model.eval()
        
        max_retries = 2
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with torch.no_grad():
                    # Tokenize
                    inputs = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True
                    )
                    
                    # 🔍 Detailed diagnosis: Validate token ID range to prevent CUDA index out of bounds
                    # ⚠️ Critical fix: Use tokenizer's vocab_size as limit (stricter)
                    # Because need to decode with tokenizer after generation, must ensure generated token IDs are within tokenizer range
                    model_vocab_size = getattr(self.model.config, 'vocab_size', None)
                    tokenizer_vocab_size = len(self.tokenizer)
                    
                    # 🔥 Use smaller vocab_size (tokenizer's actual range) to prevent generating invalid tokens
                    if model_vocab_size is not None and tokenizer_vocab_size is not None:
                        if model_vocab_size > tokenizer_vocab_size:
                            # 🔍 Critical issue: Model and tokenizer vocab mismatch
                            vocab_diff = model_vocab_size - tokenizer_vocab_size
                            if retry_count == 0:  # Only log on first time
                                logging.warning(f"⚠️ Model vocab_size ({model_vocab_size}) > tokenizer vocab_size ({tokenizer_vocab_size})")
                                logging.warning(f"   Difference: {vocab_diff} tokens, this may cause invalid token generation")
                                logging.warning(f"   Will use tokenizer range ({tokenizer_vocab_size}) as limit")
                            vocab_size = tokenizer_vocab_size  # Use stricter limit
                        else:
                            vocab_size = min(model_vocab_size, tokenizer_vocab_size)
                    else:
                        vocab_size = tokenizer_vocab_size if tokenizer_vocab_size is not None else (model_vocab_size if model_vocab_size is not None else 50000)
                    
                    # 🔍 Log final vocab_size used
                    if retry_count == 0 and model_vocab_size is not None and tokenizer_vocab_size is not None:
                        if model_vocab_size != tokenizer_vocab_size:
                            logging.debug(f"📊 Vocab size check: model={model_vocab_size}, tokenizer={tokenizer_vocab_size}, using={vocab_size}")
                    
                    if 'input_ids' in inputs:
                        input_ids = inputs['input_ids']
                        # 🔍 Detailed check: Print input information
                        max_token_id = input_ids.max().item()
                        min_token_id = input_ids.min().item()
                        input_len = input_ids.shape[1]
                        
                        # Check if all token IDs are within valid range
                        invalid_mask = (input_ids >= vocab_size) | (input_ids < 0)
                        if torch.any(invalid_mask):
                            invalid_ids = input_ids[invalid_mask].unique().tolist()
                            logging.error(f"❌ Invalid token IDs detected!")
                            logging.error(f"   Invalid ID list: {invalid_ids[:10]}")  # Only show first 10
                            logging.error(f"   Input sequence length: {input_len}")
                            logging.error(f"   Max token ID: {max_token_id}, min: {min_token_id}")
                            logging.error(f"   Model vocab_size: {model_vocab_size}")
                            logging.error(f"   Tokenizer vocab_size: {tokenizer_vocab_size}")
                            logging.error(f"   Using vocab_size: {vocab_size}")
                            logging.error(f"   Will clamp to valid range [0, {vocab_size-1}]")
                            inputs['input_ids'] = torch.clamp(input_ids, 0, vocab_size - 1)
                        
                        # 🔍 Check if input length exceeds model limit
                        max_position_embeddings = getattr(self.model.config, 'max_position_embeddings', None)
                        if max_position_embeddings and input_len > max_position_embeddings:
                            logging.error(f"❌ Input sequence length {input_len} exceeds model max position {max_position_embeddings}!")
                            # Truncate to max length
                            inputs['input_ids'] = input_ids[:, :max_position_embeddings]
                            if 'attention_mask' in inputs:
                                inputs['attention_mask'] = inputs['attention_mask'][:, :max_position_embeddings]
                            logging.warning(f"   Truncated to {max_position_embeddings}")
                    
                    # 🔥 Ensure all input tensors are on model device
                    inputs = self._ensure_inputs_on_model_device(inputs)
                    
                    # 🔍 Set valid pad_token_id and eos_token_id
                    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                    eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else pad_token_id
                    
                    # Ensure token IDs are within valid range
                    if pad_token_id is not None:
                        pad_token_id = min(pad_token_id, vocab_size - 1)
                    if eos_token_id is not None:
                        eos_token_id = min(eos_token_id, vocab_size - 1)
                    
                    # 🔍 Pre-generation: Ensure all parameters are correct
                    max_positions = getattr(self.model.config, 'max_position_embeddings', None)
                    max_total_len = max_positions if max_positions else 2048
                    current_len = inputs['input_ids'].shape[1]
                    max_allowed_new_tokens = min(max_length, max_total_len - current_len - 10)  # Leave 10 token safety margin
                    
                    if max_allowed_new_tokens <= 0:
                        logging.error(f"❌ Cannot generate new tokens: current length {current_len} + reserved {10} >= max length {max_total_len}")
                        return [""] * len(prompts) if len(prompts) > 1 else ""
                    
                    # 🔍 Detailed generation parameter logs (only on errors)
                    if retry_count > 0:
                        logging.info(f"🔍 Generation parameters: max_new_tokens={max_allowed_new_tokens}, current_len={current_len}, max_total={max_total_len}")
                        logging.info(f"   pad_token_id={pad_token_id}, eos_token_id={eos_token_id}, vocab_size={vocab_size}")
                    
                    # 🔥 Critical fix: Add LogitsProcessor to force model to only generate tokens within tokenizer range
                    from transformers import LogitsProcessorList
                    
                    class TokenRangeLogitsProcessor:
                        """Force model to only generate tokens within valid range (strictly limit to tokenizer range)"""
                        def __init__(self, max_valid_token_id: int):
                            self.max_valid_token_id = max_valid_token_id
                            self.vocab_end_idx = max_valid_token_id + 1
                        
                        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                            # 🔥 Critical fix: Only mask, do not slice (keep shape unchanged)
                            # Transformers convention: last dimension shape of logits must remain unchanged
                            if scores.shape[-1] > self.vocab_end_idx:
                                scores[..., self.vocab_end_idx:] = float('-inf')
                            return scores
                    
                    # 🔥 Critical: Use tokenizer's actual max token ID (vocab_size - 1)
                    # This is the only safe range, because tokenizer cannot decode tokens beyond this range
                    max_valid_token_id = vocab_size - 1
                    logits_processor = LogitsProcessorList([
                        TokenRangeLogitsProcessor(max_valid_token_id)
                    ])
                    
                    # 🔍 Verify LogitsProcessor settings
                    if retry_count > 0:
                        logging.info(f"   LogitsProcessor: max valid token ID = {max_valid_token_id} (vocab_size={vocab_size})")
                    
                    # 🔥 Critical fix: Wrap model's forward method to ensure all input_ids are within valid range
                    # This prevents model's internal operations from accessing embeddings beyond tokenizer range during generation
                    original_forward = None
                    original_model = self.model
                    
                    # Get actual model (may be base model wrapped by PeftModel)
                    model_to_wrap = self.model
                    if hasattr(self.model, 'get_base_model'):
                        model_to_wrap = self.model.get_base_model()
                    elif hasattr(self.model, 'base_model'):
                        model_to_wrap = self.model.base_model.model if hasattr(self.model.base_model, 'model') else self.model.base_model
                    
                    # Wrap embedding layer to ensure all token IDs are within valid range
                    def create_safe_embedding_wrapper(embedding_layer, max_valid_token_id, layer_name="embedding"):
                        """Create safe embedding wrapper"""
                        original_embed = embedding_layer.forward
                        
                        # Get actual size of embedding layer
                        try:
                            actual_emb_size = embedding_layer.weight.size(0)
                        except:
                            actual_emb_size = None
                        
                        def safe_forward(input_ids, *args, **kwargs):
                            # 🔥 Critical: Before embedding lookup, limit all token IDs to valid range
                            if input_ids is not None and isinstance(input_ids, torch.Tensor):
                                # Check if there are tokens beyond actual embedding size
                                if actual_emb_size is not None:
                                    max_id_in_input = input_ids.max().item() if input_ids.numel() > 0 else -1
                                    if max_id_in_input >= actual_emb_size:
                                        if not hasattr(safe_forward, '_warned'):
                                            logging.error(f"❌ {layer_name}: input_ids contains tokens beyond embedding size! max={max_id_in_input}, embedding_size={actual_emb_size}, limit to={max_valid_token_id}")
                                            safe_forward._warned = True
                                        # Use stricter limit: min(actual embedding size, tokenizer size)
                                        safe_max = min(actual_emb_size - 1, max_valid_token_id)
                                        input_ids = torch.clamp(input_ids, 0, safe_max)
                                    else:
                                        # Even if not exceeded, limit to tokenizer range
                                        input_ids = torch.clamp(input_ids, 0, max_valid_token_id)
                                else:
                                    # If cannot get actual size, use tokenizer range
                                    input_ids = torch.clamp(input_ids, 0, max_valid_token_id)
                                
                                # 🔥 Additional protection: Ensure all input tensors are on same device as embedding weights
                                embed_device = embedding_layer.weight.device
                                if input_ids.device != embed_device:
                                    input_ids = input_ids.to(embed_device)
                            return original_embed(input_ids, *args, **kwargs)
                        
                        embedding_layer.forward = safe_forward
                        return original_embed
                    
                    # Wrap embedding layers (if exist)
                    # Try multiple possible paths to find embedding layer
                    restored_embeddings = []
                    embedding_layers_to_wrap = []
                    
                    # Check multiple possible embedding layer paths
                    if hasattr(model_to_wrap, 'embed_tokens'):
                        embedding_layers_to_wrap.append(('embed_tokens', model_to_wrap.embed_tokens))
                    elif hasattr(model_to_wrap, 'model') and hasattr(model_to_wrap.model, 'embed_tokens'):
                        embedding_layers_to_wrap.append(('model.embed_tokens', model_to_wrap.model.embed_tokens))
                    elif hasattr(model_to_wrap, 'wte'):  # GPT-style models
                        embedding_layers_to_wrap.append(('wte', model_to_wrap.wte))
                    
                    for layer_name, embedding_layer in embedding_layers_to_wrap:
                        original_embed = create_safe_embedding_wrapper(
                            embedding_layer, 
                            vocab_size - 1,
                            layer_name
                        )
                        restored_embeddings.append((embedding_layer, original_embed, layer_name))
                        if retry_count == 0:
                            logging.debug(f"✅ Wrapped embedding layer: {layer_name}, limit range: [0, {vocab_size - 1}]")
                    
                    try:
                        # 🔥 Critical: Ensure model uses tokenizer's vocab_size, not model's own vocab_size
                        # By limiting logits dimension and explicitly specifying vocab_size in generation parameters
                        generate_kwargs = {
                            **inputs,
                            "max_new_tokens": max_allowed_new_tokens,
                            "temperature": temperature,
                            "do_sample": do_sample,
                            "top_p": top_p,
                            "top_k": min(top_k, vocab_size - 1),  # Ensure top_k does not exceed tokenizer range
                            "pad_token_id": pad_token_id,
                            "eos_token_id": eos_token_id,
                            "repetition_penalty": 1.1,
                            "logits_processor": logits_processor,  # 🔥 Critical: Force use tokenizer range
                            "use_cache": True
                        }
                        
                        # Generate
                        outputs = self.model.generate(**generate_kwargs)
                    finally:
                        # Restore original embedding forward methods
                        for embedding_layer, original_embed, layer_name in restored_embeddings:
                            embedding_layer.forward = original_embed
                    
                    # 🔍 Detailed validation of generated token IDs
                    invalid_mask = (outputs >= vocab_size) | (outputs < 0)
                    if torch.any(invalid_mask):
                        invalid_ids = outputs[invalid_mask].unique().tolist()
                        output_max = outputs.max().item()
                        output_min = outputs.min().item()
                        logging.error(f"❌ Generated token IDs out of range!")
                        logging.error(f"   Invalid ID list: {invalid_ids[:10]}")
                        logging.error(f"   Output token range: [{output_min}, {output_max}], vocab_size={vocab_size}")
                        logging.error(f"   Output shape: {outputs.shape}")
                        logging.error(f"   Will clamp to valid range")
                        outputs = torch.clamp(outputs, 0, vocab_size - 1)
                    
                    # Decode
                    generated_texts = []
                    for i, output in enumerate(outputs):
                        try:
                            input_length = inputs['input_ids'][i].shape[0]
                            if len(output) > input_length:
                                generated_text = self.tokenizer.decode(
                                    output[input_length:],
                                    skip_special_tokens=True
                                )
                                # 🔥 Critical fix: Truncate to answer end marker to avoid "cross-talk"
                                # If dialogue start markers appear (e.g., "Human:"), truncate first to prevent mixing other questions
                                # Then find first complete "#### <number>" answer marker, extract to that position
                                import re
                                
                                # Method 1: Prioritize checking dialogue start markers (these pollute text)
                                # If "Human:", "Assistant:" etc. appear, indicates multi-turn dialogue content generated, need to truncate
                                dialogue_markers = ['Human:', 'Assistant:', 'User:', 'Question:']
                                # Also check dialogue markers after newline (more common)
                                dialogue_markers_with_newline = ['\n\nHuman:', '\nHuman:', '\nAssistant:', '\n\nAssistant:', '\nUser:', '\n\nUser:']
                                
                                # Check all dialogue markers first, find earliest position
                                earliest_dialogue_pos = len(generated_text)
                                for marker in dialogue_markers + dialogue_markers_with_newline:
                                    marker_pos = generated_text.find(marker)
                                    if marker_pos > 0 and marker_pos < earliest_dialogue_pos:
                                        earliest_dialogue_pos = marker_pos
                                
                                # If dialogue marker found, truncate to that position first
                                if earliest_dialogue_pos < len(generated_text):
                                    generated_text = generated_text[:earliest_dialogue_pos].strip()
                                
                                # Method 2: Find first complete "#### <number>" answer marker
                                # This is GSM8K format's standard answer end marker
                                answer_pattern = r'####\s*(\d+(?:\.\d+)?)'
                                match = re.search(answer_pattern, generated_text)
                                if match:
                                    # Find first answer marker, extract to that position (including answer)
                                    answer_end_pos = match.end()
                                    generated_text = generated_text[:answer_end_pos].strip()
                            else:
                                generated_text = ""
                            generated_texts.append(generated_text)
                        except Exception as e:
                            logging.warning(f"Decoding failed: {e}, using empty string")
                            generated_texts.append("")
                    
                return generated_texts if len(generated_texts) > 1 else generated_texts[0]
                
            except RuntimeError as e:
                error_str = str(e)
                if "device-side assert" in error_str or "CUDA error" in error_str:
                    retry_count += 1
                    # 🔍 Detailed error diagnosis
                    logging.error(f"❌ CUDA device-side assert error (retry {retry_count}/{max_retries})")
                    logging.error(f"   Error message: {error_str[:500]}")  # Only show first 500 characters
                    
                    # 🔍 Diagnostic information: Check model and input state
                    try:
                        if 'inputs' in locals():
                            logging.error(f"   Input shape: {inputs.get('input_ids', 'N/A').shape if 'input_ids' in inputs else 'N/A'}")
                            if 'input_ids' in inputs:
                                input_ids = inputs['input_ids']
                                logging.error(f"   Input token range: [{input_ids.min().item()}, {input_ids.max().item()}]")
                                logging.error(f"   Input sequence length: {input_ids.shape[1]}")
                        
                        model_vocab = getattr(self.model.config, 'vocab_size', None)
                        max_pos = getattr(self.model.config, 'max_position_embeddings', None)
                        logging.error(f"   Model vocab_size: {model_vocab}")
                        logging.error(f"   Model max_position_embeddings: {max_pos}")
                        logging.error(f"   Tokenizer vocab_size: {len(self.tokenizer)}")
                        logging.error(f"   Number of prompts: {len(prompts)}")
                    except Exception as diag_e:
                        logging.error(f"   Failed to get diagnostic information: {diag_e}")
                    
                    if retry_count < max_retries:
                        logging.warning(f"   Clearing CUDA cache and retrying...")
                        # Clear CUDA cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            import gc
                            gc.collect()
                        # Reset model state
                        self.model.eval()
                        import time
                        time.sleep(0.1 * retry_count)
                    else:
                        logging.error(f"❌ CUDA error, reached maximum retries")
                        logging.error(f"   Suggestion: Check if model weights are corrupted, or try reloading the model")
                        # Return empty string instead of raising exception
                        return [""] * len(prompts) if len(prompts) > 1 else ""
                else:
                    # Other errors raise directly
                    raise
            except Exception as e:
                logging.error(f"Text generation failed: {e}")
                # Return empty string instead of raising exception
                return [""] * len(prompts) if len(prompts) > 1 else ""
    
    def get_logits(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Get logits for text
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Logits tensor
        """
        if isinstance(text, str):
            text = [text]
        
        self.model.eval()
        
        # 🔥 Critical fix: Get tokenizer's actual vocab_size as limit
        model_vocab_size = getattr(self.model.config, 'vocab_size', None)
        tokenizer_vocab_size = len(self.tokenizer)
        vocab_size = tokenizer_vocab_size if tokenizer_vocab_size is not None else (model_vocab_size if model_vocab_size is not None else 50000)
        if model_vocab_size is not None and tokenizer_vocab_size is not None:
            if model_vocab_size > tokenizer_vocab_size:
                vocab_size = tokenizer_vocab_size  # Use stricter limit
        
        # 🔥 Critical fix: Wrap embedding layer to ensure token IDs are within valid range
        restored_embeddings = []
        model_to_wrap = self.model
        if hasattr(self.model, 'get_base_model'):
            model_to_wrap = self.model.get_base_model()
        elif hasattr(self.model, 'base_model'):
            if hasattr(self.model.base_model, 'model'):
                model_to_wrap = self.model.base_model.model
            else:
                model_to_wrap = self.model.base_model
        elif hasattr(self.model, 'pretrained_model'):
            model_to_wrap = self.model.pretrained_model
        
        def create_safe_embedding_wrapper(embedding_layer, max_valid_token_id):
            """Create safe embedding wrapper"""
            original_embed = embedding_layer.forward
            
            def safe_forward(input_ids, *args, **kwargs):
                if input_ids is not None and isinstance(input_ids, torch.Tensor):
                    input_ids = torch.clamp(input_ids, 0, max_valid_token_id)
                return original_embed(input_ids, *args, **kwargs)
            
            embedding_layer.forward = safe_forward
            return original_embed
        
        embedding_layers_to_wrap = []
        if hasattr(model_to_wrap, 'embed_tokens'):
            embedding_layers_to_wrap.append(('embed_tokens', model_to_wrap.embed_tokens))
        elif hasattr(model_to_wrap, 'model') and hasattr(model_to_wrap.model, 'embed_tokens'):
            embedding_layers_to_wrap.append(('model.embed_tokens', model_to_wrap.model.embed_tokens))
        elif hasattr(model_to_wrap, 'wte'):
            embedding_layers_to_wrap.append(('wte', model_to_wrap.wte))
        
        for layer_name, embedding_layer in embedding_layers_to_wrap:
            original_embed = create_safe_embedding_wrapper(embedding_layer, vocab_size - 1)
            restored_embeddings.append((embedding_layer, original_embed, layer_name))
        
        try:
            with torch.no_grad():
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                # 🔥 Ensure input tensors are on model device
                inputs = self._ensure_inputs_on_model_device(inputs)
                
                # 🔥 Critical: Ensure input_ids are within valid range
                if 'input_ids' in inputs:
                    input_ids = inputs['input_ids']
                    invalid_mask = (input_ids >= vocab_size) | (input_ids < 0)
                    if torch.any(invalid_mask):
                        inputs['input_ids'] = torch.clamp(input_ids, 0, vocab_size - 1)
                
                # Forward pass
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # 🔥 Note: No longer slice logits, because vocab_size already matches (if there are still issues, should error during loading)
                
                # If single text, remove batch dimension (consistent with TeacherModel)
                if logits.shape[0] == 1:
                    return logits[0]
                
                return logits
        finally:
            # Restore original embedding forward methods
            for embedding_layer, original_embed, layer_name in restored_embeddings:
                embedding_layer.forward = original_embed
    
    def compute_log_probs(self, text: str) -> torch.Tensor:
        """
        Compute log probabilities for text
        
        Args:
            text: Input text
            
        Returns:
            Log probabilities tensor
        """
        # 🔥 Critical fix: Get tokenizer's actual vocab_size as limit
        model_vocab_size = getattr(self.model.config, 'vocab_size', None)
        tokenizer_vocab_size = len(self.tokenizer)
        vocab_size = tokenizer_vocab_size if tokenizer_vocab_size is not None else (model_vocab_size if model_vocab_size is not None else 50000)
        if model_vocab_size is not None and tokenizer_vocab_size is not None:
            if model_vocab_size > tokenizer_vocab_size:
                vocab_size = tokenizer_vocab_size  # Use stricter limit
        
        # 🔥 Critical fix: Wrap embedding layer to ensure token IDs are within valid range
        restored_embeddings = []
        model_to_wrap = self.model
        if hasattr(self.model, 'get_base_model'):
            model_to_wrap = self.model.get_base_model()
        elif hasattr(self.model, 'base_model'):
            if hasattr(self.model.base_model, 'model'):
                model_to_wrap = self.model.base_model.model
            else:
                model_to_wrap = self.model.base_model
        elif hasattr(self.model, 'pretrained_model'):
            model_to_wrap = self.model.pretrained_model
        
        def create_safe_embedding_wrapper(embedding_layer, max_valid_token_id):
            """Create safe embedding wrapper"""
            original_embed = embedding_layer.forward
            
            def safe_forward(input_ids, *args, **kwargs):
                if input_ids is not None and isinstance(input_ids, torch.Tensor):
                    input_ids = torch.clamp(input_ids, 0, max_valid_token_id)
                return original_embed(input_ids, *args, **kwargs)
            
            embedding_layer.forward = safe_forward
            return original_embed
        
        embedding_layers_to_wrap = []
        if hasattr(model_to_wrap, 'embed_tokens'):
            embedding_layers_to_wrap.append(('embed_tokens', model_to_wrap.embed_tokens))
        elif hasattr(model_to_wrap, 'model') and hasattr(model_to_wrap.model, 'embed_tokens'):
            embedding_layers_to_wrap.append(('model.embed_tokens', model_to_wrap.model.embed_tokens))
        elif hasattr(model_to_wrap, 'wte'):
            embedding_layers_to_wrap.append(('wte', model_to_wrap.wte))
        
        for layer_name, embedding_layer in embedding_layers_to_wrap:
            original_embed = create_safe_embedding_wrapper(embedding_layer, vocab_size - 1)
            restored_embeddings.append((embedding_layer, original_embed, layer_name))
        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                # 🔥 Ensure input tensors are on model device
                inputs = self._ensure_inputs_on_model_device(inputs)
                
                # 🔥 Critical: Ensure input_ids are within valid range
                if 'input_ids' in inputs:
                    input_ids = inputs['input_ids']
                    invalid_mask = (input_ids >= vocab_size) | (input_ids < 0)
                    if torch.any(invalid_mask):
                        inputs['input_ids'] = torch.clamp(input_ids, 0, vocab_size - 1)
                
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Compute log probabilities
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                
                return log_probs
        finally:
            # Restore original embedding forward methods
            for embedding_layer, original_embed, layer_name in restored_embeddings:
                embedding_layer.forward = original_embed
    
    def save_model(self, save_path: str, save_adapter: bool = True):
        """
        Save model
        
        Args:
            save_path: Save path
            save_adapter: Whether to save only adapter
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        if save_adapter and self.use_lora:
            # Save only LoRA adapter
            self.model.save_pretrained(save_path)
            # ✅ Fix: Save tokenizer, needed for evaluation
            self.tokenizer.save_pretrained(save_path)
            logging.info(f"LoRA adapter and tokenizer saved to: {save_path}")
        else:
            # Save complete model
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logging.info(f"Complete model saved to: {save_path}")
    
    def load_model(self, load_path: str, load_adapter: bool = True):
        """
        Load model
        
        Args:
            load_path: Load path
            load_adapter: Whether to load only adapter
        """
        try:
            if load_adapter and self.use_lora:
                # Load LoRA adapter
                self.model = PeftModel.from_pretrained(self.base_model, load_path)
                logging.info(f"LoRA adapter loaded from {load_path}")
            else:
                # Load complete model
                self.model = AutoModelForCausalLM.from_pretrained(load_path)
                self.tokenizer = AutoTokenizer.from_pretrained(load_path)
                logging.info(f"Complete model loaded from {load_path}")
                
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            "model_name": self.model_name,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "use_lora": self.use_lora
        }
        
        if self.use_lora:
            info["lora_config"] = self.lora_config
        
        return info
    
    def freeze_base_model(self):
        """Freeze base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        logging.info("Base model parameters frozen")
    
    def unfreeze_base_model(self):
        """Unfreeze base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = True
        logging.info("Base model parameters unfrozen")
    
    def print_trainable_parameters(self):
        """Print trainable parameter information"""
        trainable_params = 0
        all_param = 0
        
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}")


class StudentModelManager:
    """Student Model Manager"""
    
    def __init__(self, config: Dict):
        """
        Initialize manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.student_model = None
        self.ppo_model = None
        
    def initialize_student(self) -> StudentModel:
        """Initialize student model"""
        if self.student_model is None:
            self.student_model = StudentModel(
                model_name=self.config["student_model"]["base_model_name"],
                lora_config=self.config["student_model"]["lora_config"],
                device=self.config["device"]["device_map"],
                torch_dtype=getattr(torch, self.config["device"]["torch_dtype"]),
                use_lora=self.config["student_model"]["use_lora"]
            )
        
        return self.student_model
    
    def get_student(self) -> StudentModel:
        """Get student model instance"""
        if self.student_model is None:
            return self.initialize_student()
        return self.student_model
    
    def setup_ppo_model(self) -> AutoModelForCausalLMWithValueHead:
        """Setup PPO model"""
        if self.ppo_model is None:
            student = self.get_student()
            self.ppo_model = student.setup_for_ppo()
        
        return self.ppo_model
    
    def cleanup(self):
        """Cleanup resources"""
        if self.ppo_model is not None:
            del self.ppo_model
        if self.student_model is not None:
            del self.student_model
        torch.cuda.empty_cache()


def create_student_model(config: Dict) -> StudentModel:
    """
    Convenience function to create student model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Student model instance
    """
    return StudentModel(
        model_name=config["student_model"]["base_model_name"],
        lora_config=config["student_model"]["lora_config"],
        device=config["device"]["device_map"],
        torch_dtype=getattr(torch, config["device"]["torch_dtype"]),
        use_lora=config["student_model"]["use_lora"]
    )

