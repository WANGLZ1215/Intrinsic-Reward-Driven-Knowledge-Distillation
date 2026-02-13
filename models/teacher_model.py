"""
Teacher Model Wrapper Module
Function: Wrap Qwen-32B-instruct teacher model, provide logits computation and caching functionality
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple, Union
import hashlib
from collections import OrderedDict
import logging
from pathlib import Path
import threading
from utils.cache_utils import suppress_past_key_values_warning, update_model_for_modern_cache


class TeacherModel:
    """Teacher Model Wrapper Class"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-32B-Instruct", 
                 cache_size: int = 10000, cache_policy: str = "LRU",
                 device: str = "auto", torch_dtype: torch.dtype = torch.bfloat16,
                 max_memory: Optional[Dict[int, str]] = None):
        """
        Initialize teacher model
        
        Args:
            model_name: Model name
            cache_size: Cache size
            cache_policy: Cache policy
            device: Device
            torch_dtype: Data type
            max_memory: Maximum memory limit per GPU (dict, e.g., {0: "75GB", 1: "75GB"})
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_memory = max_memory
        
        # Initialize cache
        self.cache_size = cache_size
        self.cache_policy = cache_policy
        self.cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Thread lock for tokenizer thread safety
        self._tokenizer_lock = threading.Lock()
        
        # Suppress past_key_values warnings
        suppress_past_key_values_warning()
        
        # Load model and tokenizer
        self._load_model()
        
        # Update model to use modern cache
        self.model = update_model_for_modern_cache(self.model)
        
        logging.info(f"Teacher model {model_name} loaded successfully")
        logging.info(f"Cache configuration: size={cache_size}, policy={cache_policy}")
    
    def _load_model(self):
        """Load model and tokenizer"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            # ✅ Ensure: If using device_map, must set low_cpu_mem_usage=True
            # If device is None or empty string, do not use device_map
            device_map_value = self.device if (self.device and self.device.lower() != 'none') else None
            low_cpu_mem_usage = True if device_map_value is not None else False
            
            load_kwargs = {
                "torch_dtype": self.torch_dtype,
                "trust_remote_code": True,
                "low_cpu_mem_usage": low_cpu_mem_usage  # ✅ Conditional setting: Must be True when using device_map
            }
            
            # Only add if device_map is not None
            if device_map_value is not None:
                load_kwargs["device_map"] = device_map_value
            
            # If max_memory is specified, add this parameter (for limiting memory usage of specific GPUs)
            if self.max_memory is not None:
                load_kwargs["max_memory"] = self.max_memory
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )

            # If device_map specifies a single CUDA device, ensure all weights and cache are moved to that device
            if isinstance(device_map_value, str) and device_map_value.startswith("cuda"):
                try:
                    target_device = torch.device(device_map_value)
                    self.model = self.model.to(target_device)
                    logging.info(f"✅ Teacher model weights moved to device: {target_device}")
                except Exception as move_err:
                    logging.warning(f"⚠️ Failed to move teacher model to device {device_map_value}: {move_err}")
            elif device_map_value is None and torch.cuda.is_available():
                try:
                    target_device = torch.device("cuda:0")
                    self.model = self.model.to(target_device)
                    logging.info(f"✅ Teacher model weights moved to default device: {target_device}")
                except Exception as move_err:
                    logging.warning(f"⚠️ Failed to move teacher model to default CUDA device: {move_err}")
            
            # ✅ Method 1: Do not resize, directly use model's original weights (most stable method)
            tokenizer_vocab_size = len(self.tokenizer)
            model_emb_size = self.model.get_input_embeddings().weight.size(0)
            
            logging.info(f"📊 Vocab size check:")
            logging.info(f"   tokenizer vocab_size: {tokenizer_vocab_size}")
            logging.info(f"   model embedding size: {model_emb_size}")
            logging.info(f"   model.config.vocab_size: {getattr(self.model.config, 'vocab_size', 'N/A')}")
            
            if model_emb_size != tokenizer_vocab_size:
                logging.warning(f"⚠️  Vocab size mismatch ({model_emb_size} vs {tokenizer_vocab_size})")
                logging.info(f"   But using Method 1: Do not resize, directly use model's original weights")
                logging.info(f"   This is the most stable method, even with differences it won't trigger CUDA errors")
            else:
                logging.info(f"✅ Vocab size matches: {model_emb_size}")
            
            self.model.eval()  # Set to evaluation mode
            
            logging.info("Model loaded successfully")
            
        except FileNotFoundError as e:
            logging.error(f"Model file not found: {e}")
            raise
        except RuntimeError as e:
            logging.error(f"Model loading runtime error: {e}")
            raise
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _update_cache(self, key: str, value: torch.Tensor):
        """Update cache"""
        # 🔥 Fix memory leak: Move logits to CPU
        self.cache[key] = value.clone().detach().cpu()
        
        # If cache is full, remove oldest item
        if len(self.cache) > self.cache_size:
            if self.cache_policy == "LRU":
                self.cache.popitem(last=False)
            else:
                # Random removal
                import random
                random_key = random.choice(list(self.cache.keys()))
                del self.cache[random_key]
    
    def get_logits(self, text: Union[str, List[str]], 
                   use_cache: bool = True) -> torch.Tensor:
        """
        Get logits for text
        
        Args:
            text: Input text or list of texts
            use_cache: Whether to use cache
            
        Returns:
            Logits tensor
        """
        if isinstance(text, str):
            text = [text]
        
        batch_logits = []
        
        for single_text in text:
            cache_key = None
            if use_cache:
                cache_key = self._get_cache_key(single_text)
                
            # Check cache
            if cache_key in self.cache:
                self.cache_hits += 1
                # Move to end (LRU)
                self.cache.move_to_end(cache_key)
                # 🔥 Fix: Do not move cache to device, let model handle automatically
                cached_logits = self.cache[cache_key]
                batch_logits.append(cached_logits)
                continue
            
            # Cache miss, compute logits
            self.cache_misses += 1
            
            # 🔥 Get vocab_size (should already match, checked in _load_model)
            vocab_size = len(self.tokenizer)
            
            # 🔥 Critical fix: Only clamp input_ids during validation, do not modify embedding layer
            # Remove monkey patch, because vocab_size already matches, no need to wrap embedding
            
            # Use thread lock to protect tokenizer calls, avoid "Already borrowed" error
            max_retries = 3
            retry_count = 0
            logits = None
            
            while retry_count < max_retries and logits is None:
                try:
                    with self._tokenizer_lock:
                        with torch.no_grad():
                            # Tokenize
                            inputs = self.tokenizer(
                                single_text,
                                return_tensors="pt",
                                truncation=True,
                                max_length=512,
                                padding=True
                            )
                        # 🔥 Move inputs to model's device, avoid CPU/CUDA mismatch
                        target_device = next(self.model.parameters()).device
                        for key, value in inputs.items():
                            if isinstance(value, torch.Tensor):
                                inputs[key] = value.to(target_device)
                            
                            # 🔥 Critical: Ensure input_ids are within valid range (validate before sending to model)
                            if 'input_ids' in inputs:
                                input_ids = inputs['input_ids']
                                # Check if there are out-of-range tokens (should not happen, vocab already matches)
                                if input_ids.numel() > 0:
                                    max_id = input_ids.max().item()
                                    min_id = input_ids.min().item()
                                    if max_id >= vocab_size or min_id < 0:
                                        logging.warning(f"⚠️ input_ids out of range: [{min_id}, {max_id}], vocab_size={vocab_size}, auto clamp")
                                        inputs['input_ids'] = torch.clamp(input_ids, 0, vocab_size - 1)
                            
                            # Forward pass (do not move inputs to device, let HF handle device_map automatically)
                            outputs = self.model(**inputs)
                            logits = outputs.logits
                            
                            # Store to cache
                            if use_cache and cache_key is not None:
                                self._update_cache(cache_key, logits)
                            
                            batch_logits.append(logits)
                            break  # Successfully got logits, exit retry loop
                except RuntimeError as e:
                    if "Already borrowed" in str(e) and retry_count < max_retries - 1:
                        retry_count += 1
                        logging.warning(f"Tokenizer thread safety issue, retry {retry_count}/{max_retries}: {e}")
                        import time
                        time.sleep(0.01 * retry_count)  # Incremental wait time
                    else:
                        logging.error(f"Tokenizer call failed, reached maximum retries: {e}")
                        raise
        
        # If single text, return single logits
        if len(batch_logits) == 1:
            return batch_logits[0]
        
        # For multiple texts, return list (because seq_len may differ, cat will fail)
        return batch_logits
    
    def generate_response(self, prompt: str, max_length: int = 256,
                         temperature: float = 0.7, do_sample: bool = True) -> str:
        """
        Generate response
        
        Args:
            prompt: Prompt text
            max_length: Maximum length
            temperature: Temperature parameter
            do_sample: Whether to sample
            
        Returns:
            Generated response
        """
        # Use thread lock to protect tokenizer calls, avoid "Already borrowed" error
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with self._tokenizer_lock:
                    with torch.no_grad():
                        inputs = self.tokenizer(
                            prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512
                        )

                        # 🔥 Move inputs to model's device, avoid CPU/CUDA mismatch
                        model_device = next(self.model.parameters()).device
                        for key, value in inputs.items():
                            if isinstance(value, torch.Tensor):
                                inputs[key] = value.to(model_device)
                        
                        # 🔥 Get vocab_size (should already match, checked in _load_model)
                        vocab_size = len(self.tokenizer)
                        
                        # 🔥 Critical fix: Ensure input_ids are within valid range
                        if 'input_ids' in inputs:
                            input_ids = inputs['input_ids']
                            if input_ids.numel() > 0:
                                max_token_id = input_ids.max().item()
                                min_token_id = input_ids.min().item()
                                input_len = input_ids.shape[1]
                                
                                if max_token_id >= vocab_size or min_token_id < 0:
                                    logging.warning(f"⚠️ input_ids out of range: [{min_token_id}, {max_token_id}], vocab_size={vocab_size}, auto clamp")
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
                            
                            # Log input information (only for first few samples or when problems occur)
                            if retry_count > 0 or max_token_id >= vocab_size * 0.9:
                                logging.debug(f"📊 Pre-generation check: input_len={input_len}, token_range=[{min_token_id}, {max_token_id}], vocab_size={vocab_size}")
                        
                        # 🔥 Fix: Do not move inputs to device, let HF handle device_map automatically
                        
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
                            return ""
                        
                        # 🔍 Detailed generation parameter logs (only on errors)
                        if retry_count > 0:
                            logging.info(f"🔍 Generation parameters: max_new_tokens={max_allowed_new_tokens}, current_len={current_len}, max_total={max_total_len}")
                            logging.info(f"   pad_token_id={pad_token_id}, eos_token_id={eos_token_id}, vocab_size={vocab_size}")
                        
                        # 🔥 Critical fix: Add LogitsProcessor to mask out-of-range tokens (without changing shape)
                        from transformers import LogitsProcessorList
                        
                        class TokenRangeLogitsProcessor:
                            """Mask logits beyond tokenizer range (without changing shape)"""
                            def __init__(self, max_valid_token_id: int):
                                self.max_valid_token_id = max_valid_token_id
                                self.vocab_end_idx = max_valid_token_id + 1
                            
                            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                                # 🔥 Critical: Only mask, do not slice (keep shape unchanged)
                                if scores.shape[-1] > self.vocab_end_idx:
                                    scores[..., self.vocab_end_idx:] = float('-inf')
                                return scores
                        
                        max_valid_token_id = vocab_size - 1
                        logits_processor = LogitsProcessorList([
                            TokenRangeLogitsProcessor(max_valid_token_id)
                        ])
                        
                        # 🔥 Critical: Only validate input_ids at input stage, do not modify embedding layer
                        if 'input_ids' in inputs and inputs['input_ids'].numel() > 0:
                            input_ids = inputs['input_ids']
                            if input_ids.max().item() >= vocab_size or input_ids.min().item() < 0:
                                logging.warning(f"⚠️ input_ids out of range, clamp to [0, {vocab_size-1}]")
                                inputs['input_ids'] = torch.clamp(input_ids, 0, vocab_size - 1)
                        
                        # 🔥 Fix: Do not move inputs to device, let HF handle device_map automatically
                        # Generate (inputs stay on CPU, model.generate will handle device allocation automatically)
                        generate_kwargs = {
                            **inputs,
                            "max_new_tokens": max_allowed_new_tokens,
                            "temperature": temperature,
                            "do_sample": do_sample,
                            "pad_token_id": pad_token_id,
                            "eos_token_id": eos_token_id,
                            "repetition_penalty": 1.1,
                            "logits_processor": logits_processor,
                            "use_cache": True
                        }
                        
                        outputs = self.model.generate(**generate_kwargs)
                        
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
                            logging.error(f"   Will truncate to valid range")
                            outputs = torch.clamp(outputs, 0, vocab_size - 1)
                        
                        # Decode
                        input_length = inputs['input_ids'].shape[1]
                        if len(outputs[0]) > input_length:
                            generated_text = self.tokenizer.decode(
                                outputs[0][input_length:],
                                skip_special_tokens=True
                            )
                        else:
                            generated_text = ""
                        
                        return generated_text
            except RuntimeError as e:
                error_str = str(e)
                if "device-side assert" in error_str or "CUDA error" in error_str:
                    retry_count += 1
                    # 🔍 Detailed error diagnosis
                    logging.error(f"❌ CUDA device-side assert error (retry {retry_count}/{max_retries})")
                    logging.error(f"   Error message: {error_str[:500]}")  # Only show first 500 characters
                    
                    # 🔍 Diagnostic information: Check model and input state
                    try:
                        if 'input_ids' in locals():
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
                        return ""
                elif "Already borrowed" in error_str and retry_count < max_retries - 1:
                    retry_count += 1
                    logging.warning(f"Tokenizer thread safety issue, retry {retry_count}/{max_retries}: {e}")
                    import time
                    time.sleep(0.01 * retry_count)  # Incremental wait time
                else:
                    logging.error(f"Failed to generate response, reached maximum retries: {e}")
                    return ""  # Return empty string instead of raising exception
            except Exception as e:
                logging.error(f"Unknown error occurred while generating response: {e}")
                return ""  # Return empty string instead of raising exception
    
    def compute_log_probs(self, text: str) -> torch.Tensor:
        """
        Compute log probabilities for text
        
        Args:
            text: Input text
            
        Returns:
            Log probabilities tensor
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # 🔥 Fix: Do not move inputs to device, let HF handle device_map automatically
            
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Compute log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            return log_probs
    
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size
        }
    
    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logging.info("Cache cleared")
    
    def save_cache(self, filepath: str):
        """Save cache to file"""
        # 🔥 Fix: Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            "cache": dict(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses
        }
        
        torch.save(cache_data, filepath)
        logging.info(f"Cache saved to: {filepath}")
    
    def load_cache(self, filepath: str):
        """Load cache from file"""
        if Path(filepath).exists():
            cache_data = torch.load(filepath, map_location='cpu')
            self.cache = OrderedDict(cache_data["cache"])
            self.cache_hits = cache_data["cache_hits"]
            self.cache_misses = cache_data["cache_misses"]
            logging.info(f"Cache loaded from {filepath}")
        else:
            logging.warning(f"Cache file does not exist: {filepath}")
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """Get model information"""
        # 🔥 Fix: When device_map="auto", model does not have a single device
        try:
            if hasattr(self.model, 'hf_device_map'):
                device_info = "sharded"  # Distributed model
            elif hasattr(self.model, 'device'):
                device_info = str(self.model.device)
            else:
                device_info = "unknown"
        except:
            device_info = "unknown"
        
        return {
            "model_name": self.model_name,
            "device": device_info,
            "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown",
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "cache_size": len(self.cache),
            "cache_policy": self.cache_policy
        }


class TeacherModelManager:
    """Teacher Model Manager"""
    
    def __init__(self, config: Dict):
        """
        Initialize manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.teacher_model = None
        
    def initialize_teacher(self) -> TeacherModel:
        """Initialize teacher model"""
        if self.teacher_model is None:
            self.teacher_model = TeacherModel(
                model_name=self.config["teacher_model"]["model_name"],
                cache_size=self.config["teacher_model"]["cache_size"],
                cache_policy=self.config["teacher_model"]["cache_policy"],
                device=self.config["device"]["device_map"],
                torch_dtype=getattr(torch, self.config["device"]["torch_dtype"])
            )
        
        return self.teacher_model
    
    def get_teacher(self) -> TeacherModel:
        """Get teacher model instance"""
        if self.teacher_model is None:
            return self.initialize_teacher()
        return self.teacher_model
    
    def cleanup(self):
        """Cleanup resources"""
        if self.teacher_model is not None:
            # Save cache
            cache_file = "./cache/teacher_cache.pkl"
            self.teacher_model.save_cache(cache_file)
            
            # Clear GPU memory
            del self.teacher_model
            torch.cuda.empty_cache()


def create_teacher_model(config: Dict) -> TeacherModel:
    """
    Convenience function to create teacher model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Teacher model instance
    """
    return TeacherModel(
        model_name=config["teacher_model"]["model_name"],
        cache_size=config["teacher_model"]["cache_size"],
        cache_policy=config["teacher_model"]["cache_policy"],
        device=config["device"]["device_map"],
        torch_dtype=getattr(torch, config["device"]["torch_dtype"])
    )

