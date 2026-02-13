"""
PPO Utility Functions
Function: Provides PPO training-related utility functions and helper classes, supports parallel processing
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from functools import partial


class PPOMetrics:
    """PPO Metrics Calculator"""
    
    @staticmethod
    def compute_kl_divergence(old_log_probs: torch.Tensor, 
                            new_log_probs: torch.Tensor,
                            attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute KL divergence
        
        Args:
            old_log_probs: Old policy log probabilities
            new_log_probs: New policy log probabilities
            attention_mask: Attention mask
            
        Returns:
            KL divergence
        """
        kl_div = old_log_probs - new_log_probs
        
        if attention_mask is not None:
            kl_div = kl_div * attention_mask.float()
            mask_sum = attention_mask.float().sum(dim=-1)
            # Avoid division by zero
            mask_sum = torch.clamp(mask_sum, min=1e-8)
            kl_div = kl_div.sum(dim=-1) / mask_sum
        else:
            kl_div = kl_div.mean(dim=-1)
        
        return kl_div
    
    @staticmethod
    def compute_entropy(log_probs: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute entropy
        
        Args:
            log_probs: Log probabilities
            attention_mask: Attention mask
            
        Returns:
            Entropy
        """
        entropy = -log_probs
        
        if attention_mask is not None:
            entropy = entropy * attention_mask.float()
            mask_sum = attention_mask.float().sum(dim=-1)
            # Avoid division by zero
            mask_sum = torch.clamp(mask_sum, min=1e-8)
            entropy = entropy.sum(dim=-1) / mask_sum
        else:
            entropy = entropy.mean(dim=-1)
        
        return entropy
    
    @staticmethod
    def compute_advantages(rewards: torch.Tensor,
                          values: torch.Tensor,
                          gamma: float = 0.99,
                          lambda_gae: float = 0.95) -> torch.Tensor:
        """
        Compute GAE (Generalized Advantage Estimation) advantages
        
        Args:
            rewards: Rewards
            values: Value function
            gamma: Discount factor
            lambda_gae: GAE parameter
            
        Returns:
            Advantages
        """
        batch_size, seq_len = rewards.shape
        
        # Compute TD errors
        td_errors = torch.zeros_like(rewards)
        for t in range(seq_len - 1):
            td_errors[:, t] = rewards[:, t] + gamma * values[:, t + 1] - values[:, t]
        
        # Compute GAE advantages
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(batch_size)
        
        for t in reversed(range(seq_len)):
            delta = td_errors[:, t]
            gae = delta + gamma * lambda_gae * gae
            advantages[:, t] = gae
        
        return advantages


class PPOLossCalculator:
    """PPO Loss Calculator"""
    
    def __init__(self, clip_ratio: float = 0.2,
                 value_loss_coef: float = 0.1,
                 entropy_coef: float = 0.01,
                 kl_coef: float = 0.05):
        """
        Initialize loss calculator
        
        Args:
            clip_ratio: Clipping ratio
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            kl_coef: KL divergence coefficient
        """
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
    
    def compute_policy_loss(self, advantages: torch.Tensor,
                           old_log_probs: torch.Tensor,
                           new_log_probs: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute policy loss
        
        Args:
            advantages: Advantages
            old_log_probs: Old policy log probabilities
            new_log_probs: New policy log probabilities
            attention_mask: Attention mask
            
        Returns:
            Policy loss
        """
        # Compute probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clip probability ratio
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        # Compute loss
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        
        policy_loss = -torch.min(surr1, surr2)
        
        if attention_mask is not None:
            policy_loss = policy_loss * attention_mask.float()
            mask_sum = attention_mask.float().sum(dim=-1)
            # Avoid division by zero
            mask_sum = torch.clamp(mask_sum, min=1e-8)
            policy_loss = policy_loss.sum(dim=-1) / mask_sum
        else:
            policy_loss = policy_loss.mean(dim=-1)
        
        return policy_loss.mean()
    
    def compute_value_loss(self, returns: torch.Tensor,
                          values: torch.Tensor,
                          old_values: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute value loss
        
        Args:
            returns: Returns
            values: Current value function
            old_values: Old value function
            attention_mask: Attention mask
            
        Returns:
            Value loss
        """
        # Compute value loss
        value_loss = F.mse_loss(values, returns, reduction='none')
        
        if attention_mask is not None:
            value_loss = value_loss * attention_mask.float()
            mask_sum = attention_mask.float().sum(dim=-1)
            # Avoid division by zero
            mask_sum = torch.clamp(mask_sum, min=1e-8)
            value_loss = value_loss.sum(dim=-1) / mask_sum
        else:
            value_loss = value_loss.mean(dim=-1)
        
        return value_loss.mean()
    
    def compute_entropy_loss(self, log_probs: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute entropy loss
        
        Args:
            log_probs: Log probabilities
            attention_mask: Attention mask
            
        Returns:
            Entropy loss
        """
        entropy = -log_probs
        
        if attention_mask is not None:
            entropy = entropy * attention_mask.float()
            mask_sum = attention_mask.float().sum(dim=-1)
            # Avoid division by zero
            mask_sum = torch.clamp(mask_sum, min=1e-8)
            entropy = entropy.sum(dim=-1) / mask_sum
        else:
            entropy = entropy.mean(dim=-1)
        
        return -entropy.mean()  # Negative entropy to encourage exploration
    
    def compute_total_loss(self, policy_loss: torch.Tensor,
                          value_loss: torch.Tensor,
                          entropy_loss: torch.Tensor,
                          kl_div: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute total loss
        
        Args:
            policy_loss: Policy loss
            value_loss: Value loss
            entropy_loss: Entropy loss
            kl_div: KL divergence (optional)
            
        Returns:
            Total loss
        """
        total_loss = (policy_loss + 
                     self.value_loss_coef * value_loss + 
                     self.entropy_coef * entropy_loss)
        
        if kl_div is not None:
            total_loss += self.kl_coef * kl_div.mean()
        
        return total_loss


class PPOBuffer:
    """PPO Experience Buffer"""
    
    def __init__(self, buffer_size: int = 10000):
        """
        Initialize buffer
        
        Args:
            buffer_size: Buffer size
        """
        self.buffer_size = buffer_size
        self.reset()
    
    def reset(self):
        """Reset buffer"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.advantages = []
        self.returns = []
        self.attention_masks = []
        
        self.ptr = 0
        self.size = 0
    
    def add(self, observation: torch.Tensor,
            action: torch.Tensor,
            reward: torch.Tensor,
            value: torch.Tensor,
            log_prob: torch.Tensor,
            advantage: torch.Tensor,
            return_val: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None):
        """
        Add experience to buffer
        
        Args:
            observation: Observation
            action: Action
            reward: Reward
            value: Value
            log_prob: Log probability
            advantage: Advantage
            return_val: Return
            attention_mask: Attention mask
        """
        if self.size < self.buffer_size:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.log_probs.append(log_prob)
            self.advantages.append(advantage)
            self.returns.append(return_val)
            self.attention_masks.append(attention_mask)
            self.size += 1
        else:
            self.observations[self.ptr] = observation
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.values[self.ptr] = value
            self.log_probs[self.ptr] = log_prob
            self.advantages[self.ptr] = advantage
            self.returns[self.ptr] = return_val
            self.attention_masks[self.ptr] = attention_mask
            self.ptr = (self.ptr + 1) % self.buffer_size
    
    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch from buffer
        
        Args:
            batch_size: Batch size
            
        Returns:
            Batch data
        """
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        
        batch = {
            "observations": torch.stack([self.observations[i] for i in indices]),
            "actions": torch.stack([self.actions[i] for i in indices]),
            "rewards": torch.stack([self.rewards[i] for i in indices]),
            "values": torch.stack([self.values[i] for i in indices]),
            "log_probs": torch.stack([self.log_probs[i] for i in indices]),
            "advantages": torch.stack([self.advantages[i] for i in indices]),
            "returns": torch.stack([self.returns[i] for i in indices]),
        }
        
        if self.attention_masks[0] is not None:
            batch["attention_masks"] = torch.stack([self.attention_masks[i] for i in indices])
        
        return batch
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self.size >= self.buffer_size
    
    def get_size(self) -> int:
        """Get buffer size"""
        return self.size


class PPOScheduler:
    """PPO Learning Rate Scheduler"""
    
    def __init__(self, initial_lr: float = 1e-5,
                 final_lr: float = 1e-6,
                 total_steps: int = 1000,
                 warmup_steps: int = 100):
        """
        Initialize scheduler
        
        Args:
            initial_lr: Initial learning rate
            final_lr: Final learning rate
            total_steps: Total steps
            warmup_steps: Warmup steps
        """
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
    
    def get_lr(self, step: int) -> float:
        """
        Get learning rate for current step
        
        Args:
            step: Current step
            
        Returns:
            Learning rate
        """
        if step < self.warmup_steps:
            # Warmup phase
            return self.initial_lr * (step / self.warmup_steps)
        else:
            # Decay phase
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return self.initial_lr * (1 - progress) + self.final_lr * progress


def create_ppo_optimizer(model: torch.nn.Module, 
                        learning_rate: float = 1e-5,
                        weight_decay: float = 0.01) -> torch.optim.Optimizer:
    """
    Create PPO optimizer
    
    Args:
        model: Model
        learning_rate: Learning rate
        weight_decay: Weight decay
        
    Returns:
        Optimizer
    """
    # Separate trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8
    )
    
    return optimizer


def compute_grad_norm(model: torch.nn.Module) -> float:
    """
    Compute gradient norm
    
    Args:
        model: Model
        
    Returns:
        Gradient norm
    """
    total_norm = 0.0
    param_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    return total_norm


def clip_grad_norm(model: torch.nn.Module, max_norm: float = 1.0):
    """
    Clip gradient norm
    
    Args:
        model: Model
        max_norm: Maximum norm
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


class ParallelRewardProcessor:
    """Parallel Reward Processor"""
    
    def __init__(self, num_workers: int = 4, use_threads: bool = True):
        """
        Initialize parallel processor
        
        Args:
            num_workers: Number of worker processes/threads
            use_threads: Whether to use thread pool (True) or process pool (False)
        """
        self.num_workers = num_workers
        self.use_threads = use_threads
        self.executor = None
        
    def __enter__(self):
        """Context manager entry"""
        if self.use_threads:
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        else:
            self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def compute_rewards_parallel(self, questions: List[str], 
                               student_responses: List[str],
                               reward_func,
                               **kwargs) -> List[float]:
        """
        Compute rewards in parallel
        
        Args:
            questions: List of questions
            student_responses: List of student responses
            reward_func: Reward computation function
            **kwargs: Additional parameters passed to reward function
            
        Returns:
            List of rewards
        """
        if not self.executor:
            raise RuntimeError("ParallelRewardProcessor must be used as context manager")
        
        # Create tasks
        tasks = []
        for question, response in zip(questions, student_responses):
            task = self.executor.submit(reward_func, question, response, **kwargs)
            tasks.append(task)
        
        # Collect results
        rewards = []
        for task in as_completed(tasks):
            try:
                reward = task.result()
                rewards.append(reward)
            except Exception as e:
                logging.error(f"Reward computation failed: {e}")
                rewards.append(0.0)
        
        return rewards


class ParallelModelInference:
    """Parallel Model Inference"""
    
    def __init__(self, model, batch_size: int = 8, num_workers: int = 4):
        """
        Initialize parallel inference
        
        Args:
            model: Model instance
            batch_size: Batch size
            num_workers: Number of worker threads
        """
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._lock = threading.Lock()
    
    def generate_batch_parallel(self, prompts: List[str], 
                              max_length: int = 256,
                              temperature: float = 0.7,
                              **kwargs) -> List[str]:
        """
        Generate batch text in parallel
        
        Args:
            prompts: List of prompt texts
            max_length: Maximum length
            temperature: Temperature parameter
            **kwargs: Other generation parameters
            
        Returns:
            List of generated texts
        """
        # Split input into batches
        batches = [prompts[i:i + self.batch_size] for i in range(0, len(prompts), self.batch_size)]
        
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit batch tasks
            futures = []
            for batch in batches:
                future = executor.submit(self._generate_single_batch, batch, max_length, temperature, **kwargs)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    logging.error(f"Batch generation failed: {e}")
                    # Add empty results as placeholders
                    results.extend([""] * len(batch))
        
        return results
    
    def _generate_single_batch(self, batch_prompts: List[str], 
                             max_length: int, temperature: float, **kwargs) -> List[str]:
        """
        Generate single batch
        
        Args:
            batch_prompts: Batch prompts
            max_length: Maximum length
            temperature: Temperature parameter
            **kwargs: Other parameters
            
        Returns:
            Batch results
        """
        with self._lock:  # Ensure thread-safe model access
            try:
                return self.model.generate(
                    batch_prompts,
                    max_length=max_length,
                    temperature=temperature,
                    **kwargs
                )
            except Exception as e:
                logging.error(f"Single batch generation failed: {e}")
                return [""] * len(batch_prompts)
    
    def get_logits_batch_parallel(self, sequences: List[str]) -> List[torch.Tensor]:
        """
        Get batch logits in parallel
        
        Args:
            sequences: List of input sequences
            
        Returns:
            List of logits tensors
        """
        # Split input into batches
        batches = [sequences[i:i + self.batch_size] for i in range(0, len(sequences), self.batch_size)]
        
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit batch tasks
            futures = []
            for batch in batches:
                future = executor.submit(self._get_logits_single_batch, batch)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    logging.error(f"Batch logits retrieval failed: {e}")
                    # Add None as placeholders
                    results.extend([None] * len(batch))
        
        return results
    
    def _get_logits_single_batch(self, batch_sequences: List[str]) -> List[torch.Tensor]:
        """
        Get logits for single batch
        
        Args:
            batch_sequences: Batch sequences
            
        Returns:
            List of batch logits
        """
        with self._lock:  # Ensure thread-safe model access
            try:
                # Teacher model's get_logits method returns a single tensor, needs to be split
                batch_logits = self.model.get_logits(batch_sequences)
                # If a single tensor is returned, split by batch
                if isinstance(batch_logits, torch.Tensor) and batch_logits.dim() == 3:
                    # Split by batch size
                    batch_size = len(batch_sequences)
                    return [batch_logits[i:i+1] for i in range(batch_size)]
                else:
                    # If already a list, return directly
                    return batch_logits if isinstance(batch_logits, list) else [batch_logits]
            except Exception as e:
                logging.error(f"Single batch logits retrieval failed: {e}")
                return [None] * len(batch_sequences)


class AsyncCacheManager:
    """Async Cache Manager"""
    
    def __init__(self, cache_manager, max_queue_size: int = 1000):
        """
        Initialize async cache manager
        
        Args:
            cache_manager: Original cache manager
            max_queue_size: Maximum queue size
        """
        self.cache_manager = cache_manager
        self.max_queue_size = max_queue_size
        self._queue = []
        self._lock = threading.Lock()
        self._worker_thread = None
        self._stop_event = threading.Event()
        
    def start_async_worker(self):
        """Start async worker thread"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(target=self._async_worker, daemon=True)
            self._worker_thread.start()
    
    def stop_async_worker(self):
        """Stop async worker thread"""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join()
    
    def _async_worker(self):
        """Async worker thread"""
        while not self._stop_event.is_set():
            # Get items to process
            items_to_process = []
            with self._lock:
                if self._queue:
                    # Batch processing for efficiency
                    batch_size = min(10, len(self._queue))
                    items_to_process = self._queue[:batch_size]
                    self._queue = self._queue[batch_size:]
            
            # Process outside lock to avoid holding lock for long time
            for key, value in items_to_process:
                try:
                    self.cache_manager.put(key, value)
                except Exception as e:
                    logging.error(f"Async cache update failed: {e}")
            
            # Brief sleep to avoid excessive CPU usage
            threading.Event().wait(0.01)
    
    def put_async(self, key: str, value: torch.Tensor):
        """
        Put into cache asynchronously
        
        Args:
            key: Cache key
            value: Cache value
        """
        with self._lock:
            if len(self._queue) < self.max_queue_size:
                self._queue.append((key, value))
            else:
                # When queue is full, remove oldest item
                self._queue.pop(0)
                self._queue.append((key, value))
        
        # Ensure worker thread is running
        if not self._worker_thread or not self._worker_thread.is_alive():
            self.start_async_worker()
    
    def get(self, key: str):
        """Get cache value"""
        return self.cache_manager.get(key)


class ParallelDataLoader:
    """Parallel Data Loader"""
    
    def __init__(self, dataset, batch_size: int = 8, num_workers: int = 4, shuffle: bool = True):
        """
        Initialize parallel data loader
        
        Args:
            dataset: Dataset
            batch_size: Batch size
            num_workers: Number of worker processes
            shuffle: Whether to shuffle data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        """Iterator"""
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Process in batches
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            yield batch_data
    
    def __len__(self):
        """Return number of batches"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_parallel_processor(config: Dict) -> ParallelRewardProcessor:
    """
    Convenience function to create parallel processor
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Parallel processor instance
    """
    num_workers = config.get("parallel", {}).get("num_workers", 4)
    use_threads = config.get("parallel", {}).get("use_threads", True)
    
    return ParallelRewardProcessor(
        num_workers=num_workers,
        use_threads=use_threads
    )


def create_parallel_inference(model, config: Dict) -> ParallelModelInference:
    """
    Convenience function to create parallel inference
    
    Args:
        model: Model instance
        config: Configuration dictionary
        
    Returns:
        Parallel inference instance
    """
    batch_size = config.get("parallel", {}).get("inference_batch_size", 8)
    num_workers = config.get("parallel", {}).get("num_workers", 4)
    
    return ParallelModelInference(
        model=model,
        batch_size=batch_size,
        num_workers=num_workers
    )


def create_async_cache_manager(cache_manager, config: Dict) -> AsyncCacheManager:
    """
    Convenience function to create async cache manager
    
    Args:
        cache_manager: Original cache manager
        config: Configuration dictionary
        
    Returns:
        Async cache manager instance
    """
    max_queue_size = config.get("parallel", {}).get("cache_queue_size", 1000)
    
    return AsyncCacheManager(
        cache_manager=cache_manager,
        max_queue_size=max_queue_size
    )


