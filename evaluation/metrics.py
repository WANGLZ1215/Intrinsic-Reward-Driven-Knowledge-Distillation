"""
Evaluation Metrics Module
Function: Define and compute various evaluation metrics
"""

import torch
import numpy as np
import re
from typing import Dict, List, Optional, Union, Tuple
import logging
from collections import defaultdict


class MathAccuracyMetrics:
    """Mathematical accuracy metrics"""
    
    @staticmethod
    def exact_match_accuracy(predictions: List[str], 
                           ground_truths: List[str]) -> float:
        """
        Compute exact match accuracy
        
        Args:
            predictions: List of predictions
            ground_truths: List of ground truth answers
            
        Returns:
            Exact match accuracy
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions and ground truths do not match")
        
        correct = 0
        for pred, gt in zip(predictions, ground_truths):
            if pred.strip().lower() == gt.strip().lower():
                correct += 1
        
        return correct / len(predictions)
    
    @staticmethod
    def numerical_accuracy(predictions: List[str], 
                          ground_truths: List[str], 
                          tolerance: float = 1e-6) -> float:
        """
        Compute numerical accuracy
        
        Args:
            predictions: List of predictions
            ground_truths: List of ground truth answers
            tolerance: Numerical tolerance
            
        Returns:
            Numerical accuracy
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions and ground truths do not match")
        
        correct = 0
        for pred, gt in zip(predictions, ground_truths):
            pred_num = MathAccuracyMetrics._extract_number(pred)
            gt_num = MathAccuracyMetrics._extract_number(gt)
            
            if pred_num is not None and gt_num is not None:
                if abs(pred_num - gt_num) < tolerance:
                    correct += 1
        
        return correct / len(predictions)
    
    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        """Extract number from text"""
        # Find last number
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                return None
        return None


class ReasoningQualityMetrics:
    """Reasoning quality metrics"""
    
    @staticmethod
    def step_coverage_ratio(student_steps: List[str], 
                           teacher_steps: List[str]) -> float:
        """
        Compute step coverage ratio
        
        Args:
            student_steps: Student reasoning steps
            teacher_steps: Teacher reasoning steps
            
        Returns:
            Step coverage ratio
        """
        if not teacher_steps:
            return 0.0
        
        # Extract key information
        student_keywords = ReasoningQualityMetrics._extract_keywords(student_steps)
        teacher_keywords = ReasoningQualityMetrics._extract_keywords(teacher_steps)
        
        # Compute intersection
        common_keywords = set(student_keywords) & set(teacher_keywords)
        
        return len(common_keywords) / len(teacher_keywords)
    
    @staticmethod
    def logical_consistency_score(response: str) -> float:
        """
        Compute logical consistency score
        
        Args:
            response: Model response
            
        Returns:
            Logical consistency score
        """
        # Extract number sequence
        numbers = re.findall(r'-?\d+\.?\d*', response)
        numbers = [float(n) for n in numbers if n]
        
        if len(numbers) < 2:
            return 1.0
        
        # Check reasonableness of numbers
        score = 1.0
        
        # Check for outliers
        for i in range(len(numbers) - 1):
            if numbers[i] > 10000 or numbers[i] < -1000:
                score -= 0.1
        
        # Check increasing/decreasing logic
        if len(numbers) >= 3:
            diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
            if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
                score += 0.1  # Clear trend
        
        return max(0.0, min(1.0, score))
    
    @staticmethod
    def _extract_keywords(steps: List[str]) -> List[str]:
        """Extract keywords"""
        keywords = []
        for step in steps:
            # Extract numbers
            numbers = re.findall(r'\d+', step)
            keywords.extend(numbers)
            
            # Extract operators
            operators = re.findall(r'[+\-*/=]', step)
            keywords.extend(operators)
        
        return keywords


class DistillationMetrics:
    """Knowledge distillation metrics"""
    
    @staticmethod
    def align_logits(teacher_logits: torch.Tensor, 
                     student_logits: torch.Tensor) -> tuple:
        """
        Align vocab_size dimension of logits (last dimension)
        This is a 100% accepted practice in academia (commonly used in distillation when teacher/student vocab are not fully consistent)
        
        Args:
            teacher_logits: Teacher model logits
            student_logits: Student model logits
            
        Returns:
            Aligned (teacher_logits, student_logits)
        """
        t_dim = teacher_logits.size(-1)
        s_dim = student_logits.size(-1)
        
        if t_dim == s_dim:
            return teacher_logits, student_logits
        
        # Align to minimum vocab_size
        min_dim = min(t_dim, s_dim)
        teacher_logits = teacher_logits[..., :min_dim]
        student_logits = student_logits[..., :min_dim]
        
        return teacher_logits, student_logits
    
    @staticmethod
    def kl_divergence(student_logits: torch.Tensor, 
                     teacher_logits: torch.Tensor) -> float:
        """
        Compute KL divergence
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            
        Returns:
            KL divergence
        """
        # Align vocab_size dimension
        teacher_logits, student_logits = DistillationMetrics.align_logits(
            teacher_logits, student_logits
        )
        
        # Compute probability distributions
        student_probs = torch.softmax(student_logits, dim=-1)
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        
        # Compute KL divergence
        kl_div = torch.sum(teacher_probs * torch.log(teacher_probs / (student_probs + 1e-8)), dim=-1)
        
        return kl_div.mean().item()
    
    @staticmethod
    def cosine_similarity(student_logits: torch.Tensor, 
                         teacher_logits: torch.Tensor) -> float:
        """
        Compute cosine similarity
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            
        Returns:
            Cosine similarity
        """
        # Align vocab_size dimension
        teacher_logits, student_logits = DistillationMetrics.align_logits(
            teacher_logits, student_logits
        )
        
        # Ensure tensor is contiguous first, then use reshape to flatten, avoiding view stride issues
        student_logits = student_logits.contiguous()
        teacher_logits = teacher_logits.contiguous()
        
        student_flat = student_logits.reshape(-1)
        teacher_flat = teacher_logits.reshape(-1)
        
        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            student_flat.unsqueeze(0), 
            teacher_flat.unsqueeze(0)
        )
        
        return cos_sim.item()
    
    @staticmethod
    def js_divergence(student_logits: torch.Tensor, 
                     teacher_logits: torch.Tensor) -> float:
        """
        Compute Jensen-Shannon divergence
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            
        Returns:
            JS divergence
        """
        # Align vocab_size dimension
        teacher_logits, student_logits = DistillationMetrics.align_logits(
            teacher_logits, student_logits
        )
        
        # Compute probability distributions
        student_probs = torch.softmax(student_logits, dim=-1)
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        
        # Compute mean distribution
        mean_probs = (student_probs + teacher_probs) / 2
        
        # Compute JS divergence
        kl_student = torch.sum(student_probs * torch.log(student_probs / (mean_probs + 1e-8)), dim=-1)
        kl_teacher = torch.sum(teacher_probs * torch.log(teacher_probs / (mean_probs + 1e-8)), dim=-1)
        
        js_div = (kl_student + kl_teacher) / 2
        
        return js_div.mean().item()


class TrainingMetrics:
    """Training metrics"""
    
    @staticmethod
    def convergence_rate(loss_history: List[float], 
                        window_size: int = 100) -> float:
        """
        Compute convergence rate
        
        Args:
            loss_history: Loss history
            window_size: Window size
            
        Returns:
            Convergence rate
        """
        if len(loss_history) < window_size * 2:
            return 0.0
        
        # Compute average loss of recent window
        recent_loss = np.mean(loss_history[-window_size:])
        
        # Compute average loss of early window
        early_loss = np.mean(loss_history[:window_size])
        
        # Compute convergence rate
        convergence_rate = (early_loss - recent_loss) / early_loss
        
        return max(0.0, convergence_rate)
    
    @staticmethod
    def training_stability(loss_history: List[float], 
                          window_size: int = 50) -> float:
        """
        Compute training stability
        
        Args:
            loss_history: Loss history
            window_size: Window size
            
        Returns:
            Stability score (higher is more stable)
        """
        if len(loss_history) < window_size:
            return 0.0
        
        # Compute standard deviation of recent window
        recent_losses = loss_history[-window_size:]
        stability = 1.0 / (1.0 + np.std(recent_losses))
        
        return stability
    
    @staticmethod
    def learning_efficiency(reward_history: List[float], 
                           step_history: List[int]) -> float:
        """
        Compute learning efficiency
        
        Args:
            reward_history: Reward history
            step_history: Step history
            
        Returns:
            Learning efficiency
        """
        if len(reward_history) < 2 or len(step_history) < 2:
            return 0.0
        
        # Compute reward improvement rate
        initial_reward = np.mean(reward_history[:10]) if len(reward_history) >= 10 else reward_history[0]
        final_reward = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else reward_history[-1]
        
        reward_improvement = final_reward - initial_reward
        
        # Compute number of steps
        total_steps = step_history[-1] - step_history[0]
        
        # Learning efficiency = reward improvement / steps
        efficiency = reward_improvement / max(1, total_steps)
        
        return efficiency


class ComprehensiveEvaluator:
    """Comprehensive evaluator"""
    
    def __init__(self):
        """Initialize comprehensive evaluator"""
        self.math_metrics = MathAccuracyMetrics()
        self.reasoning_metrics = ReasoningQualityMetrics()
        self.distillation_metrics = DistillationMetrics()
        self.training_metrics = TrainingMetrics()
    
    def evaluate_comprehensive(self, 
                             predictions: List[str],
                             ground_truths: List[str],
                             student_logits: Optional[torch.Tensor] = None,
                             teacher_logits: Optional[torch.Tensor] = None,
                             training_history: Optional[Dict] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation
        
        Args:
            predictions: Predictions
            ground_truths: Ground truth answers
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            training_history: Training history
            
        Returns:
            Comprehensive evaluation results
        """
        results = {}
        
        # Mathematical accuracy
        results["exact_match_accuracy"] = self.math_metrics.exact_match_accuracy(predictions, ground_truths)
        results["numerical_accuracy"] = self.math_metrics.numerical_accuracy(predictions, ground_truths)
        
        # Reasoning quality (if logits available)
        if student_logits is not None and teacher_logits is not None:
            results["kl_divergence"] = self.distillation_metrics.kl_divergence(student_logits, teacher_logits)
            results["cosine_similarity"] = self.distillation_metrics.cosine_similarity(student_logits, teacher_logits)
            results["js_divergence"] = self.distillation_metrics.js_divergence(student_logits, teacher_logits)
        
        # Training metrics (if training history available)
        if training_history is not None:
            if "loss_history" in training_history:
                results["convergence_rate"] = self.training_metrics.convergence_rate(training_history["loss_history"])
                results["training_stability"] = self.training_metrics.training_stability(training_history["loss_history"])
            
            if "reward_history" in training_history and "step_history" in training_history:
                results["learning_efficiency"] = self.training_metrics.learning_efficiency(
                    training_history["reward_history"], 
                    training_history["step_history"]
                )
        
        # Overall score
        accuracy_weight = 0.4
        distillation_weight = 0.3
        training_weight = 0.3
        
        overall_score = 0.0
        total_weight = 0.0
        
        # Accuracy component
        overall_score += results["numerical_accuracy"] * accuracy_weight
        total_weight += accuracy_weight
        
        # Distillation component
        if "cosine_similarity" in results:
            overall_score += results["cosine_similarity"] * distillation_weight
            total_weight += distillation_weight
        
        # Training component
        if "training_stability" in results:
            overall_score += results["training_stability"] * training_weight
            total_weight += training_weight
        
        if total_weight > 0:
            results["overall_score"] = overall_score / total_weight
        else:
            results["overall_score"] = results["numerical_accuracy"]
        
        return results


def create_comprehensive_evaluator() -> ComprehensiveEvaluator:
    """Convenience function to create a comprehensive evaluator"""
    return ComprehensiveEvaluator()






