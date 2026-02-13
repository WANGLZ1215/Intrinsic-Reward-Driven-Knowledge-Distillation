"""
Reward Combination Module
Function: Combine intrinsic and external rewards, implement multi-signal fusion

This module implements the combination of multiple reward signals, including intrinsic rewards
derived from teacher model logits and external rewards such as correctness-based rewards.
The combination can use fixed weights or adaptive weights that adjust during training.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
from collections import deque


class RewardCombiner:
    """Reward Combiner"""
    
    def __init__(self, lambda_intrinsic: float = 0.7,
                 lambda_correctness: float = 0.3,
                 lambda_reasoning: float = 0.0,
                 lambda_format: float = 0.0,
                 use_adaptive_weights: bool = True,
                 adaptation_rate: float = 0.01,
                 reward_scale: float = 1.0):
        """
        Initialize reward combiner
        
        Args:
            lambda_intrinsic: Intrinsic reward weight (from teacher logits via inverse soft Bellman operator)
            lambda_correctness: Answer correctness reward weight (external reward signal)
            lambda_reasoning: Reasoning process reward weight
            lambda_format: Format constraint reward weight
            use_adaptive_weights: Whether to use adaptive weights that adjust during training
            adaptation_rate: Weight adaptation rate for adaptive weight updates
        """
        self.lambda_intrinsic = lambda_intrinsic
        self.lambda_correctness = lambda_correctness
        self.lambda_reasoning = lambda_reasoning
        self.lambda_format = lambda_format
        self.use_adaptive_weights = use_adaptive_weights
        self.adaptation_rate = adaptation_rate
        self.reward_scale = reward_scale
        
        # Statistics (initialize first to avoid accessing uninitialized attributes in save_weights, etc.)
        self.combination_stats = {
            "total_combinations": 0,
            "intrinsic_mean": 0.0,
            "correctness_mean": 0.0,
            "combined_mean": 0.0,
            "weight_history": []
        }
        
        # Adaptive weights
        if use_adaptive_weights:
            self.adaptive_weights = {
                "intrinsic": lambda_intrinsic,
                "correctness": lambda_correctness,
                "reasoning": lambda_reasoning,
                "format": lambda_format
            }
            
            # Weight performance tracking
            self.weight_performance = {key: 0.0 for key in self.adaptive_weights.keys()}
            self.performance_window = deque(maxlen=100)
    
    def save_weights(self, filepath: str):
        """Save adaptive weights to file"""
        if self.use_adaptive_weights:
            import json
            weight_data = {
                "adaptive_weights": self.adaptive_weights,
                "weight_performance": self.weight_performance,
                "combination_stats": self.combination_stats
            }
            with open(filepath, 'w') as f:
                json.dump(weight_data, f, indent=2)
            logging.info(f"Adaptive weights saved to: {filepath}")
    
    def load_weights(self, filepath: str):
        """Load adaptive weights from file"""
        if self.use_adaptive_weights:
            import json
            try:
                with open(filepath, 'r') as f:
                    weight_data = json.load(f)
                
                self.adaptive_weights = weight_data["adaptive_weights"]
                self.weight_performance = weight_data["weight_performance"]
                self.combination_stats = weight_data["combination_stats"]
                
                logging.info(f"Adaptive weights loaded from {filepath}")
            except Exception as e:
                logging.warning(f"Failed to load adaptive weights: {e}")
        
        # Statistics already initialized above, no need to re-initialize here
        
        logging.info(f"Reward combiner initialized: λ_intrinsic={self.lambda_intrinsic}, λ_correctness={self.lambda_correctness}")
    
    def combine_rewards(self, intrinsic_rewards: torch.Tensor,
                       correctness_rewards: torch.Tensor,
                       reasoning_rewards: Optional[torch.Tensor] = None,
                       format_rewards: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Combine multiple reward signals
        
        This function combines intrinsic rewards (derived from teacher logits) with external
        rewards (e.g., correctness-based rewards) using weighted linear combination.
        
        Args:
            intrinsic_rewards: Intrinsic rewards from teacher model logits (token-level)
            correctness_rewards: Answer correctness rewards (external signal)
            reasoning_rewards: Reasoning process rewards (optional)
            format_rewards: Format constraint rewards (optional)
            
        Returns:
            Combined rewards
        """
        # Get current weights
        if self.use_adaptive_weights:
            weights = self.adaptive_weights.copy()
        else:
            weights = {
                "intrinsic": self.lambda_intrinsic,
                "correctness": self.lambda_correctness,
                "reasoning": self.lambda_reasoning,
                "format": self.lambda_format
            }
        
        # Combine rewards using weighted linear combination
        combined_rewards = (
            self.reward_scale * weights["intrinsic"] * intrinsic_rewards +
            weights["correctness"] * correctness_rewards
        )
        
        if reasoning_rewards is not None:
            combined_rewards += weights["reasoning"] * reasoning_rewards
        
        if format_rewards is not None:
            combined_rewards += weights["format"] * format_rewards
        
        # if self.reward_scale != 1.0:  # 0.02
        #     combined_rewards = combined_rewards * self.reward_scale
        
        # Update statistics
        self._update_statistics(intrinsic_rewards, correctness_rewards, combined_rewards, weights)
        
        # Adaptive weight updates
        if self.use_adaptive_weights:
            self._adapt_weights(intrinsic_rewards, correctness_rewards, combined_rewards)
        
        return combined_rewards
    
    def combine_batch_rewards(self, batch_rewards: Dict[str, List[torch.Tensor]]) -> List[torch.Tensor]:
        """
        Combine rewards for a batch of sequences
        
        Args:
            batch_rewards: Dictionary of batch reward tensors
            
        Returns:
            List of combined reward tensors
        """
        batch_size = len(batch_rewards["intrinsic"])
        combined_batch = []
        
        for i in range(batch_size):
            intrinsic = batch_rewards["intrinsic"][i]
            correctness = batch_rewards["correctness"][i]
            reasoning = batch_rewards.get("reasoning", [None] * batch_size)[i]
            format_reward = batch_rewards.get("format", [None] * batch_size)[i]
            
            combined = self.combine_rewards(intrinsic, correctness, reasoning, format_reward)
            combined_batch.append(combined)
        
        return combined_batch
    
    def _update_statistics(self, intrinsic_rewards: torch.Tensor,
                          correctness_rewards: torch.Tensor,
                          combined_rewards: torch.Tensor,
                          weights: Dict[str, float]):
        """Update combination statistics"""
        # Ensure combination_stats is initialized
        if not hasattr(self, 'combination_stats'):
            self.combination_stats = {
                "total_combinations": 0,
                "intrinsic_mean": 0.0,
                "correctness_mean": 0.0,
                "combined_mean": 0.0,
                "weight_history": []
            }
        
        self.combination_stats["total_combinations"] += 1
        
        # Compute means
        intrinsic_mean = torch.mean(intrinsic_rewards).item()
        correctness_mean = torch.mean(correctness_rewards).item()
        combined_mean = torch.mean(combined_rewards).item()
        
        # Update statistics (exponential moving average)
        alpha = 0.01
        self.combination_stats["intrinsic_mean"] = (
            (1 - alpha) * self.combination_stats["intrinsic_mean"] + 
            alpha * intrinsic_mean
        )
        self.combination_stats["correctness_mean"] = (
            (1 - alpha) * self.combination_stats["correctness_mean"] + 
            alpha * correctness_mean
        )
        self.combination_stats["combined_mean"] = (
            (1 - alpha) * self.combination_stats["combined_mean"] + 
            alpha * combined_mean
        )
        
        # Record weight history
        self.combination_stats["weight_history"].append(weights.copy())
        if len(self.combination_stats["weight_history"]) > 1000:
            self.combination_stats["weight_history"] = self.combination_stats["weight_history"][-500:]
    
    def _adapt_weights(self, intrinsic_rewards: torch.Tensor,
                      correctness_rewards: torch.Tensor,
                      combined_rewards: torch.Tensor):
        """Adaptively adjust weights based on reward contributions"""
        # Calculate contribution of each reward signal
        intrinsic_contribution = self._calculate_contribution(intrinsic_rewards, combined_rewards)
        correctness_contribution = self._calculate_contribution(correctness_rewards, combined_rewards)
        
        # Update performance scores
        self.weight_performance["intrinsic"] = intrinsic_contribution
        self.weight_performance["correctness"] = correctness_contribution
        
        # Record weight history
        self.combination_stats["weight_history"].append(self.adaptive_weights.copy())
        
        # Adaptively adjust weights
        self._update_adaptive_weights()
        
        # Log weight changes
        logging.info(f"Adaptive weights updated: intrinsic={self.adaptive_weights['intrinsic']:.4f}, "
                    f"correctness={self.adaptive_weights['correctness']:.4f}")
    
    def _calculate_contribution(self, individual_rewards: torch.Tensor,
                              combined_rewards: torch.Tensor) -> float:
        """Calculate contribution of individual reward signal to combined reward"""
        # When sample count is too small or variance is too small, corrcoef becomes unstable,
        # fall back to mean magnitude as a degradation metric
        individual_flat = individual_rewards.flatten()
        combined_flat = combined_rewards.flatten()
        
        if individual_flat.numel() < 2 or combined_flat.numel() < 2:
            return float(torch.mean(torch.abs(individual_flat)).item())
        
        # Calculate contribution based on correlation
        covariance_matrix = torch.corrcoef(torch.stack([individual_flat, combined_flat]))
        if torch.isnan(covariance_matrix).any():
            return float(torch.mean(torch.abs(individual_flat)).item())
        
        correlation = covariance_matrix[0, 1].item()
        if np.isnan(correlation):
            return float(torch.mean(torch.abs(individual_flat)).item())
        
        return abs(correlation)
    
    def _update_adaptive_weights(self):
        """Update adaptive weights based on performance scores"""
        # Adjust weights based on performance scores
        total_performance = sum(self.weight_performance.values())
        
        if total_performance > 0:
            for key in self.adaptive_weights:
                # Calculate new weight ratio
                # Avoid division by zero
                if total_performance < 1e-8:
                    new_ratio = 1.0 / len(self.weight_performance)  # Equal distribution
                else:
                    new_ratio = self.weight_performance[key] / total_performance
                
                # Smooth weight update
                self.adaptive_weights[key] = (
                    (1 - self.adaptation_rate) * self.adaptive_weights[key] +
                    self.adaptation_rate * new_ratio
                )
            
            # Normalize weights, avoid division by zero
            total_weight = sum(self.adaptive_weights.values())
            if total_weight > 1e-8:
                for key in self.adaptive_weights:
                    self.adaptive_weights[key] /= total_weight
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current reward weights"""
        if self.use_adaptive_weights:
            return self.adaptive_weights.copy()
        else:
            return {
                "intrinsic": self.lambda_intrinsic,
                "correctness": self.lambda_correctness,
                "reasoning": self.lambda_reasoning,
                "format": self.lambda_format
            }
    
    def get_statistics(self) -> Dict:
        """Get combination statistics"""
        # Ensure combination_stats is initialized
        if not hasattr(self, 'combination_stats'):
            self.combination_stats = {
                "total_combinations": 0,
                "intrinsic_mean": 0.0,
                "correctness_mean": 0.0,
                "combined_mean": 0.0,
                "weight_history": []
            }
        
        stats = self.combination_stats.copy()
        stats["current_weights"] = self.get_current_weights()
        
        if self.use_adaptive_weights:
            stats["adaptive_weights"] = self.adaptive_weights.copy()
            stats["weight_performance"] = self.weight_performance.copy()
            
            # Calculate weight change statistics
            weight_history = self.combination_stats["weight_history"]
            if len(weight_history) > 1:
                recent_weights = weight_history[-10:]  # Take last 10
                stats["weight_variance"] = {
                    "intrinsic": float(np.var([w["intrinsic"] for w in recent_weights])),
                    "correctness": float(np.var([w["correctness"] for w in recent_weights]))
                }
                
                # Calculate weight trend (need at least 5 history records)
                if len(recent_weights) >= 5:
                    first_half = recent_weights[:len(recent_weights)//2] if len(recent_weights) >= 10 else []
                    second_half = recent_weights[-len(recent_weights)//2:] if len(recent_weights) >= 10 else recent_weights[-5:]
                    
                    if len(first_half) > 0 and len(second_half) > 0:
                        stats["weight_trend"] = {
                            "intrinsic": float(np.mean([w["intrinsic"] for w in second_half]) - 
                                               np.mean([w["intrinsic"] for w in first_half])),
                            "correctness": float(np.mean([w["correctness"] for w in second_half]) - 
                                                np.mean([w["correctness"] for w in first_half]))
                        }
                    else:
                        # If records are too few, use simple difference
                        stats["weight_trend"] = {
                            "intrinsic": float(recent_weights[-1]["intrinsic"] - recent_weights[0]["intrinsic"]),
                            "correctness": float(recent_weights[-1]["correctness"] - recent_weights[0]["correctness"])
                        }
                else:
                    # Records too few, cannot calculate trend
                    stats["weight_trend"] = {
                        "intrinsic": 0.0,
                        "correctness": 0.0
                    }
        
        return stats
    
    def reset_statistics(self):
        """Reset combination statistics"""
        self.combination_stats = {
            "total_combinations": 0,
            "intrinsic_mean": 0.0,
            "correctness_mean": 0.0,
            "combined_mean": 0.0,
            "weight_history": []
        }
        
        if self.use_adaptive_weights:
            self.weight_performance = {key: 0.0 for key in self.adaptive_weights.keys()}
            self.performance_window.clear()


class RewardBalancer:
    """Reward Balancer"""
    
    def __init__(self, target_balance: float = 0.5,
                 balance_threshold: float = 0.1,
                 adjustment_rate: float = 0.01):
        """
        Initialize reward balancer
        
        Args:
            target_balance: Target balance ratio between intrinsic and correctness rewards
            balance_threshold: Balance threshold for triggering adjustments
            adjustment_rate: Rate of adjustment when rebalancing
        """
        self.target_balance = target_balance
        self.balance_threshold = balance_threshold
        self.adjustment_rate = adjustment_rate
        
        # Balance statistics
        self.balance_stats = {
            "intrinsic_ratio": 0.5,
            "correctness_ratio": 0.5,
            "balance_score": 0.0
        }
    
    def balance_rewards(self, intrinsic_rewards: torch.Tensor,
                       correctness_rewards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Balance two reward signals
        
        Args:
            intrinsic_rewards: Intrinsic rewards
            correctness_rewards: Answer correctness rewards
            
        Returns:
            Balanced reward pair
        """
        # Calculate current ratio
        intrinsic_mean = torch.mean(torch.abs(intrinsic_rewards)).item()
        correctness_mean = torch.mean(torch.abs(correctness_rewards)).item()
        
        total_mean = intrinsic_mean + correctness_mean
        
        # Avoid division by zero
        if abs(total_mean) > 1e-8:
            current_intrinsic_ratio = intrinsic_mean / total_mean
            current_correctness_ratio = correctness_mean / total_mean
        else:
            current_intrinsic_ratio = 0.5  # Default equal distribution
            current_correctness_ratio = 0.5
        
        # Calculate balance score
        balance_score = 1.0 - abs(current_intrinsic_ratio - self.target_balance)
        self.balance_stats["balance_score"] = balance_score
        
        # If unbalanced, perform adjustment
        if balance_score < (1.0 - self.balance_threshold):
            intrinsic_rewards, correctness_rewards = self._adjust_rewards(
                intrinsic_rewards, correctness_rewards,
                current_intrinsic_ratio, current_correctness_ratio
            )
        
        # Update statistics
        self.balance_stats["intrinsic_ratio"] = current_intrinsic_ratio
        self.balance_stats["correctness_ratio"] = current_correctness_ratio
        
        return intrinsic_rewards, correctness_rewards
    
    def _adjust_rewards(self, intrinsic_rewards: torch.Tensor,
                       correctness_rewards: torch.Tensor,
                       current_intrinsic_ratio: float,
                       current_correctness_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adjust rewards to improve balance"""
        # Calculate adjustment factors
        if current_intrinsic_ratio > self.target_balance:
            # Intrinsic reward too strong, need to weaken
            intrinsic_factor = 1.0 - self.adjustment_rate
            correctness_factor = 1.0 + self.adjustment_rate
        else:
            # Correctness reward too strong, need to weaken
            intrinsic_factor = 1.0 + self.adjustment_rate
            correctness_factor = 1.0 - self.adjustment_rate
        
        # Apply adjustments
        adjusted_intrinsic = intrinsic_rewards * intrinsic_factor
        adjusted_correctness = correctness_rewards * correctness_factor
        
        return adjusted_intrinsic, adjusted_correctness
    
    def get_balance_statistics(self) -> Dict[str, float]:
        """Get balance statistics"""
        return self.balance_stats.copy()


def create_reward_combiner(config: Dict) -> RewardCombiner:
    """
    Convenience function to create a reward combiner from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RewardCombiner instance
    """
    return RewardCombiner(
        lambda_intrinsic=config["reward_combination"]["lambda_intrinsic"],
        lambda_correctness=config["reward_combination"]["lambda_correctness"],
        lambda_reasoning=config.get("lambda_reasoning", 0.0),
        lambda_format=config.get("lambda_format", 0.0),
        use_adaptive_weights=config.get("use_adaptive_weights", False),
        adaptation_rate=config.get("adaptation_rate", 0.01)
    )





