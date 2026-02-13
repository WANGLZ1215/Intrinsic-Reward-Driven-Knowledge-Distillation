"""
Reward Normalization Module
Function: Normalize intrinsic and external rewards to ensure training stability

This module implements various normalization methods to scale rewards to appropriate ranges,
preventing extreme reward values that could destabilize policy gradient training.
"""

import torch
import numpy as np
from typing import Optional, Dict


class RewardNormalizer:
    """Reward Normalizer"""
    
    def __init__(self, 
                 method: str = "mean_std",
                 clip_min: float = -5.0,
                 clip_max: float = 5.0,
                 epsilon: float = 1e-8,
                 momentum: float = 0.99):
        """
        Initialize reward normalizer
        
        Args:
            method: Normalization method ('mean_std', 'min_max', 'z_score', 'robust', 'none')
            clip_min: Clipping minimum value
            clip_max: Clipping maximum value
            epsilon: Numerical stability parameter
            momentum: Moving average momentum for running statistics
        """
        self.method = method
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Statistics (using moving average)
        self.running_mean = 0.0
        self.running_std = 1.0
        self.running_min = 0.0
        self.running_max = 1.0
        
        # Whether statistics are initialized
        self.initialized = False
    
    def normalize_intrinsic_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize intrinsic rewards
        
        Args:
            rewards: Reward tensor [batch_size] or [batch_size, seq_len]
            
        Returns:
            Normalized rewards
        """
        if self.method in (None, "none"):
            # Only perform optional clipping
            if self.clip_min is None and self.clip_max is None:
                return rewards
            return torch.clamp(rewards, min=self.clip_min, max=self.clip_max)
        if self.method == "mean_std":
            return self._normalize_mean_std(rewards)
        elif self.method == "min_max":
            return self._normalize_min_max(rewards)
        elif self.method == "z_score":
            return self._normalize_z_score(rewards)
        elif self.method == "robust":
            return self._normalize_robust(rewards)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def normalize_external_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize external rewards
        
        External rewards are typically already in a reasonable range (e.g., 0-1),
        so only simple clipping is applied.
        
        Args:
            rewards: Reward tensor
            
        Returns:
            Normalized rewards
        """
        if self.method in (None, "none"):
            return torch.clamp(rewards, min=self.clip_min, max=self.clip_max)
        # External rewards are usually already in reasonable range (e.g., 0-1), only simple clipping
        return torch.clamp(rewards, self.clip_min, self.clip_max)
    
    def _normalize_mean_std(self, rewards: torch.Tensor) -> torch.Tensor:
        """Mean-standard deviation normalization (Z-score with running statistics)"""
        # Compute current batch statistics
        batch_mean = rewards.mean().item()
        batch_std = rewards.std().item()
        
        # Update moving average
        if not self.initialized:
            self.running_mean = batch_mean
            self.running_std = max(batch_std, self.epsilon)
            self.initialized = True
        else:
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_std = self.momentum * self.running_std + (1 - self.momentum) * batch_std
        
        # Normalize: (x - mean) / std
        normalized = (rewards - self.running_mean) / (self.running_std + self.epsilon)
        
        # Clip to specified range
        normalized = torch.clamp(normalized, self.clip_min, self.clip_max)
        
        return normalized
    
    def _normalize_min_max(self, rewards: torch.Tensor) -> torch.Tensor:
        """Min-max normalization"""
        # Compute current batch min and max
        batch_min = rewards.min().item()
        batch_max = rewards.max().item()
        
        # Update moving average
        if not self.initialized:
            self.running_min = batch_min
            self.running_max = batch_max
            self.initialized = True
        else:
            self.running_min = self.momentum * self.running_min + (1 - self.momentum) * batch_min
            self.running_max = self.momentum * self.running_max + (1 - self.momentum) * batch_max
        
        # Normalize to [0, 1]
        range_val = self.running_max - self.running_min
        if abs(range_val) < self.epsilon:
            normalized = torch.zeros_like(rewards)
        else:
            normalized = (rewards - self.running_min) / (range_val + self.epsilon)
        
        # Scale to [clip_min, clip_max]
        normalized = normalized * (self.clip_max - self.clip_min) + self.clip_min
        
        return normalized
    
    def _normalize_z_score(self, rewards: torch.Tensor) -> torch.Tensor:
        """Z-score standardization (similar to mean_std but without moving average)"""
        mean = rewards.mean()
        std = rewards.std()
        
        normalized = (rewards - mean) / (std + self.epsilon)
        normalized = torch.clamp(normalized, self.clip_min, self.clip_max)
        
        return normalized
    
    def _normalize_robust(self, rewards: torch.Tensor) -> torch.Tensor:
        """Robust normalization (using median and interquartile range)"""
        # Convert to numpy for computation
        rewards_np = rewards.detach().cpu().numpy().flatten()
        
        median = np.median(rewards_np)
        q75 = np.percentile(rewards_np, 75)
        q25 = np.percentile(rewards_np, 25)
        iqr = q75 - q25
        
        # Normalize: (x - median) / IQR
        if iqr < self.epsilon:
            normalized = torch.zeros_like(rewards)
        else:
            normalized = (rewards - median) / (iqr + self.epsilon)
        
        # Clip
        normalized = torch.clamp(normalized, self.clip_min, self.clip_max)
        
        return normalized
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get current normalization statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            "running_mean": self.running_mean,
            "running_std": self.running_std,
            "running_min": self.running_min,
            "running_max": self.running_max,
            "initialized": self.initialized
        }
    
    def reset_stats(self):
        """Reset normalization statistics"""
        self.running_mean = 0.0
        self.running_std = 1.0
        self.running_min = 0.0
        self.running_max = 1.0
        self.initialized = False
    
    def load_stats(self, stats: Dict[str, float]):
        """
        Load normalization statistics
        
        Args:
            stats: Statistics dictionary
        """
        self.running_mean = stats.get("running_mean", 0.0)
        self.running_std = stats.get("running_std", 1.0)
        self.running_min = stats.get("running_min", 0.0)
        self.running_max = stats.get("running_max", 1.0)
        self.initialized = stats.get("initialized", False)

