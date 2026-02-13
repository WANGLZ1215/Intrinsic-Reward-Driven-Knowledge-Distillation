"""
Intrinsic Reward Computation Module
Function: Compute intrinsic rewards based on paper theory, implement inverse soft Bellman operator

This module implements the intrinsic reward mechanism for knowledge distillation from teacher to student
models using Maximum Entropy (MaxEnt) Reinforcement Learning principles. The core idea is to derive
token-level rewards from teacher model logits, which are interpreted as soft Q-values in the MaxEnt RL
framework. These rewards guide the student model to learn the teacher's reasoning patterns at a
fine-grained level, beyond simple correctness-based supervision.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path


class IntrinsicRewardComputer:
    """
    Intrinsic Reward Computer
    
    This class implements the computation of intrinsic rewards based on the inverse soft Bellman operator.
    The fundamental principle is that teacher model logits represent soft Q-values Q̂(s, a) in the MaxEnt RL
    framework, where states s correspond to token sequences and actions a correspond to vocabulary tokens.
    """
    
    def __init__(self, temperature: float = 1.0, 
                 normalization_method: str = "mean_std",
                 update_rate: float = 0.01):
        """
        Initialize the intrinsic reward computer
        
        The temperature parameter α plays a crucial role in the soft value function computation:
        V(s) = α log(∑_a exp(Q(s,a)/α)). A lower temperature makes the distribution more peaked,
        emphasizing high-Q actions, while a higher temperature encourages exploration.
        
        Args:
            temperature: Temperature parameter α for soft value function computation. Controls the
                        entropy regularization strength in MaxEnt RL. Lower values (e.g., 0.8) make
                        the policy more deterministic, while higher values encourage exploration.
            normalization_method: Normalization method for reward scaling ("mean_std", "min_max", etc.)
            update_rate: Update rate for running statistics (exponential moving average coefficient)
        """
        self.temperature = temperature
        self.normalization_method = normalization_method
        self.update_rate = update_rate
        
        # Statistics for reward normalization and monitoring
        self.intrinsic_stats = {
            "mean": 0.0,
            "std": 1.0,
            "min": -float('inf'),
            "max": float('inf'),
            "count": 0
        }
        
        logging.info(f"Intrinsic reward computer initialized: temperature={temperature}, normalization={normalization_method}")
    
    def compute_intrinsic_reward(self, teacher_logits: torch.Tensor, 
                               student_tokens: torch.Tensor,
                               question_length: int = 0) -> torch.Tensor:
        """
        Compute intrinsic rewards based on Equation (10) in the paper.
        
        Theoretical Foundation:
        -----------------------
        This function implements the inverse soft Bellman operator to derive token-level intrinsic
        rewards from teacher model logits. The key insight is that teacher logits represent soft
        Q-values Q̂(s, a) in the MaxEnt RL framework:
        
        - States s_h: Token sequences up to position h (context)
        - Actions a: Vocabulary tokens (next token to generate)
        - Q̂(s_h, a): Teacher logit for token a at state s_h (unnormalized log-probability)
        
        The soft value function V(s) = α log(∑_a exp(Q(s,a)/α)) represents the expected future
        return under the optimal soft policy. It aggregates over all possible actions, weighted by
        their soft Q-values, with temperature α controlling the softness.
        
        The intrinsic reward r_h = Q̂(s_h, a_h) - V(s_{h+1}) measures the advantage of taking action
        a_h at state s_h, relative to the expected value of the next state. This reward captures
        the teacher's preference for specific reasoning patterns at each token position, beyond
        simple correctness.
        
        Difference from Correctness Reward:
        -----------------------------------
        Traditional correctness rewards only provide feedback at the end of generation (binary:
        correct/incorrect). In contrast, intrinsic rewards provide dense, token-level supervision
        that guides the student to learn the teacher's reasoning process, not just the final answer.
        This enables the student to learn intermediate reasoning steps, mathematical operations,
        and logical structures that lead to correct solutions.
        
        Args:
            teacher_logits: Teacher model logits [batch_size, full_seq_len, vocab_size]
                          These represent soft Q-values Q̂(s, a) for the full sequence.
            student_tokens: Student model generated tokens [batch_size, response_len]
                          The actual actions taken by the student model.
            question_length: Length of the question part, used to extract logits corresponding
                           to the student's response portion.
            
        Returns:
            Intrinsic rewards [batch_size, response_len]
            Each element r_h represents the advantage of the chosen token at position h.
        """
        alpha = max(self.temperature, 1e-8)  # Numerical stability: prevent division by zero
        
        # Ensure student_tokens and teacher_logits are on the same device
        student_tokens = student_tokens.to(teacher_logits.device)
        
        batch_size, response_len = student_tokens.shape
        full_seq_len = teacher_logits.shape[1]
        
        # Ensure question_length does not exceed teacher_logits length
        if question_length >= full_seq_len:
            question_length = 0
        
        # Calculate the starting position of student-generated portion in the full sequence
        start_pos = question_length
        end_pos = min(start_pos + response_len, full_seq_len)
        
        # Extract teacher logits corresponding to the student's response portion
        # These logits represent Q̂(s_h, a) for each position h in the response
        if end_pos > start_pos and start_pos < full_seq_len:
            student_logits = teacher_logits[:, start_pos:end_pos, :]
            actual_response_len = end_pos - start_pos
        else:
            # If extraction fails, return zero rewards
            logging.warning(f"Cannot extract student response logits: start_pos={start_pos}, end_pos={end_pos}, full_seq_len={full_seq_len}")
            return torch.zeros_like(student_tokens, dtype=torch.float32)
        
        # Compute soft value function V_Q̂(s_h) = α log(∑_a exp(Q̂(s_h, a)/α))
        # This represents the expected future return from state s_h under the optimal soft policy.
        # The logsumexp operation is numerically stable and computes the log of the sum of exponentials.
        # Lower temperature α makes the distribution more peaked (more deterministic), while higher
        # temperature encourages exploration by flattening the distribution.
        value_function = alpha * torch.logsumexp(student_logits / alpha, dim=-1)
        
        # Initialize reward tensor
        intrinsic_rewards = torch.zeros_like(student_tokens, dtype=torch.float32)
        
        # Compute token-level intrinsic rewards using the inverse soft Bellman operator
        # For each token position h, the reward is: r_h = Q̂(s_h, a_h) - V(s_{h+1})
        # This measures the advantage of action a_h at state s_h relative to the next state's value.
        for h in range(actual_response_len - 1):  # Last token has no next state
            # Current state's Q-values: Q̂(s_h, a) for all possible actions a
            current_q = student_logits[:, h, :]  # [batch_size, vocab_size]
            
            # Q-value of the token actually chosen by the student: Q̂(s_h, a_h)
            # This represents the teacher's evaluation of the student's action choice.
            selected_token_q = current_q.gather(dim=-1, index=student_tokens[:, h:h+1]).squeeze(-1)
            
            # Soft value of the next state: V(s_{h+1}) = α log(∑_a exp(Q̂(s_{h+1}, a)/α))
            # This represents the expected future return from the next state.
            next_value = value_function[:, h + 1]
            
            # Intrinsic reward = Q-value of chosen action - value of next state
            # This is the advantage: how much better (or worse) is this action compared to
            # the expected value of continuing from the next state.
            intrinsic_rewards[:, h] = selected_token_q - next_value
        
        # Reward for the last token: only Q-value, no next state value
        # The last token's reward is simply Q̂(s_H, a_H), as there is no future state to compare.
        if actual_response_len > 0:
            last_q = student_logits[:, actual_response_len - 1, :]
            last_selected_q = last_q.gather(dim=-1, index=student_tokens[:, actual_response_len - 1:actual_response_len]).squeeze(-1)
            intrinsic_rewards[:, actual_response_len - 1] = last_selected_q
        
        # Clip rewards to improve training stability
        # Prevents extreme reward values that could destabilize policy gradient updates.
        intrinsic_rewards = torch.clamp(intrinsic_rewards, -10.0, 10.0)
        
        return intrinsic_rewards
    
    def compute_outcome_reward(self, teacher_model, question: str, 
                             student_response: str) -> float:
        """
        Compute outcome reward based on Equation (12) in the paper.
        
        This function computes a trajectory-level reward for the complete response, as opposed to
        token-level intrinsic rewards. The outcome reward represents the teacher's overall evaluation
        of the entire response trajectory τ, given the initial state s₁ (the question).
        
        Theoretical Foundation:
        -----------------------
        The outcome reward is defined as: R_outcome = α * log(π̂(τ|s₁))
        where π̂(τ|s₁) is the teacher's probability of generating trajectory τ starting from state s₁.
        This is computed as the product of token-level probabilities under the teacher's policy.
        
        This reward provides a global signal about the quality of the entire response, complementing
        the fine-grained token-level intrinsic rewards. It can be used in combination with intrinsic
        rewards for more stable training.
        
        Args:
            teacher_model: Teacher model instance
            question: Input question (initial state s₁)
            student_response: Complete response generated by the student model (trajectory τ)
            
        Returns:
            Outcome reward value (scalar)
        """
        alpha = self.temperature
        
        # Construct full sequence: question + response
        full_sequence = question + student_response
        
        with torch.no_grad():
            # Get teacher model logits for the full sequence
            inputs = teacher_model.tokenizer(
                full_sequence,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            inputs = {k: v.to(teacher_model.model.device) for k, v in inputs.items()}
            outputs = teacher_model.model(**inputs)
            logits = outputs.logits
            
            # Compute log probability of the complete response under teacher's policy
            # This is the sum of log probabilities for each token in the response
            response_tokens = teacher_model.tokenizer.encode(student_response, add_special_tokens=False)
            
            total_log_prob = 0.0
            for i, token_id in enumerate(response_tokens):
                if i < logits.shape[1]:
                    token_log_prob = F.log_softmax(logits[0, i], dim=-1)[token_id]
                    total_log_prob += token_log_prob.item()
            
            # Outcome reward = α * log(π̂(τ|s₁))
            # This scales the log-probability by temperature α for consistency with intrinsic rewards
            outcome_reward = alpha * total_log_prob
            
            return outcome_reward
    
    def compute_token_level_rewards(self, teacher_logits: torch.Tensor,
                                  student_tokens: torch.Tensor,
                                  attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute token-level rewards (more stable method)
        
        This is a wrapper function that computes intrinsic rewards and applies attention masking.
        Token-level rewards provide dense supervision signal at every generation step, enabling
        the student to learn fine-grained reasoning patterns from the teacher.
        
        Args:
            teacher_logits: Teacher model logits [batch_size, seq_len, vocab_size]
            student_tokens: Student model tokens [batch_size, response_len]
            attention_mask: Attention mask to zero out padding positions [batch_size, response_len]
            
        Returns:
            Token-level intrinsic rewards [batch_size, response_len]
        """
        # Compute intrinsic rewards using the inverse soft Bellman operator
        intrinsic_rewards = self.compute_intrinsic_reward(teacher_logits, student_tokens)
        
        # If attention mask is provided, zero out rewards at padding positions
        # This ensures that padding tokens do not contribute to the reward signal
        if attention_mask is not None:
            intrinsic_rewards = intrinsic_rewards * attention_mask.float()
        
        return intrinsic_rewards
    
    def compute_trajectory_reward(self, token_rewards: torch.Tensor,
                                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aggregate token-level rewards into trajectory-level rewards
        
        This function sums token-level intrinsic rewards to obtain a single scalar reward for
        the entire response trajectory. The trajectory reward R(τ) = Σ_h r_h represents the
        cumulative advantage of all actions taken in the trajectory.
        
        Args:
            token_rewards: Token-level intrinsic rewards [batch_size, seq_len]
            attention_mask: Attention mask to exclude padding tokens [batch_size, seq_len]
            
        Returns:
            Trajectory-level rewards [batch_size]
        """
        if attention_mask is not None:
            # Sum only over valid (non-padding) tokens
            trajectory_rewards = (token_rewards * attention_mask.float()).sum(dim=-1)
        else:
            trajectory_rewards = token_rewards.sum(dim=-1)
        
        return trajectory_rewards
    
    def update_statistics(self, rewards: torch.Tensor):
        """
        Update reward statistics for normalization
        
        This function maintains running statistics (mean, std, min, max) of intrinsic rewards
        using exponential moving average. These statistics are used for reward normalization
        to stabilize training.
        
        Args:
            rewards: Reward tensor [batch_size, seq_len] or any shape
        """
        if rewards.numel() == 0:
            return
        
        # Flatten rewards to compute statistics
        flat_rewards = rewards.flatten()
        
        # Compute current batch statistics
        batch_mean = torch.mean(flat_rewards).item()
        batch_std = torch.std(flat_rewards).item()
        batch_min = torch.min(flat_rewards).item()
        batch_max = torch.max(flat_rewards).item()
        batch_count = flat_rewards.numel()
        
        # Update global statistics using exponential moving average
        # This provides a smooth estimate of reward distribution over time
        alpha = self.update_rate
        
        if self.intrinsic_stats["count"] == 0:
            # First update: initialize with batch statistics
            self.intrinsic_stats["mean"] = batch_mean
            self.intrinsic_stats["std"] = batch_std
            self.intrinsic_stats["min"] = batch_min
            self.intrinsic_stats["max"] = batch_max
        else:
            # Exponential moving average: E[x]_t = (1-α) * E[x]_{t-1} + α * x_t
            self.intrinsic_stats["mean"] = (1 - alpha) * self.intrinsic_stats["mean"] + alpha * batch_mean
            self.intrinsic_stats["std"] = (1 - alpha) * self.intrinsic_stats["std"] + alpha * batch_std
            self.intrinsic_stats["min"] = min(self.intrinsic_stats["min"], batch_min)
            self.intrinsic_stats["max"] = max(self.intrinsic_stats["max"], batch_max)
        
        self.intrinsic_stats["count"] += batch_count
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current reward statistics"""
        return self.intrinsic_stats.copy()
    
    def reset_statistics(self):
        """Reset reward statistics to initial values"""
        self.intrinsic_stats = {
            "mean": 0.0,
            "std": 1.0,
            "min": -float('inf'),
            "max": float('inf'),
            "count": 0
        }


class IntrinsicRewardBatchProcessor:
    """
    Intrinsic Reward Batch Processor
    
    This class processes batches of sequences to compute intrinsic rewards. It handles
    variable-length sequences by processing them individually and collecting results.
    """
    
    def __init__(self, intrinsic_computer: IntrinsicRewardComputer):
        """
        Initialize batch processor
        
        Args:
            intrinsic_computer: Intrinsic reward computer instance
        """
        self.intrinsic_computer = intrinsic_computer
    
    def process_batch(self, teacher_logits_list: List[torch.Tensor],
                     student_tokens_list: List[torch.Tensor],
                     attention_masks: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        """
        Process batch of sequences to compute intrinsic rewards
        
        This function processes each sequence in the batch individually, as sequences may
        have different lengths. The results are collected into a list.
        
        Args:
            teacher_logits_list: List of teacher logits tensors, one per sequence
            student_tokens_list: List of student token sequences
            attention_masks: Optional list of attention masks
            
        Returns:
            List of reward tensors, one per sequence
        """
        rewards_list = []
        
        for i, (teacher_logits, student_tokens) in enumerate(zip(teacher_logits_list, student_tokens_list)):
            attention_mask = attention_masks[i] if attention_masks else None
            
            # Compute token-level rewards for this sequence
            token_rewards = self.intrinsic_computer.compute_token_level_rewards(
                teacher_logits.unsqueeze(0), 
                student_tokens.unsqueeze(0),
                attention_mask.unsqueeze(0) if attention_mask is not None else None
            )
            
            rewards_list.append(token_rewards.squeeze(0))
        
        return rewards_list


def create_intrinsic_reward_computer(config: Dict) -> IntrinsicRewardComputer:
    """
    Convenience function to create an intrinsic reward computer from configuration
    
    Args:
        config: Configuration dictionary containing intrinsic_reward settings
        
    Returns:
        IntrinsicRewardComputer instance
    """
    return IntrinsicRewardComputer(
        temperature=config["intrinsic_reward"]["temperature"],
        normalization_method=config["intrinsic_reward"]["normalization_method"],
        update_rate=config["intrinsic_reward"]["update_rate"]
    )


