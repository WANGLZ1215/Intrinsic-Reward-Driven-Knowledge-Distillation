"""
Reasoning Quality Evaluation Module
Function: Evaluate mathematical reasoning quality, including step coverage, logical consistency, etc.
"""

import re
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
from collections import Counter
import sympy as sp
from utils.math_utils import extract_answer_unified  # Import unified answer extraction function


class ReasoningEvaluator:
    """Reasoning Quality Evaluator"""
    
    def __init__(self):
        """Initialize evaluator"""
        # Reasoning step patterns
        self.step_patterns = [
            r'Step \d+:',      # "Step 1:", "Step 2:", etc.
            r'\d+\.',          # "1.", "2.", etc.
            r'First,',         # "First,", "Second,", etc.
            r'Then,',          # "Then,", "Next,", etc.
            r'Therefore,',     # "Therefore,", "Thus,", etc.
            r'So,',            # "So,", "Hence,", etc.
            r'Finally,',       # "Finally,", "In conclusion,", etc.
        ]
        
        # Mathematical operators
        self.math_operators = ['+', '-', '*', '/', '=', '^', '**']
        
        # Logical connectors
        self.logical_connectors = ['therefore', 'thus', 'so', 'hence', 'because', 'since']
        
        logging.info("Reasoning quality evaluator initialized")
    
    def extract_reasoning_steps(self, response: str) -> List[str]:
        """
        Extract reasoning steps
        
        Args:
            response: Model response text
            
        Returns:
            List of reasoning steps
        """
        steps = []
        
        # Split by lines
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check if matches step pattern
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in self.step_patterns):
                steps.append(line)
            elif line and self._is_math_expression(line):
                # If contains mathematical expression, also consider as reasoning step
                steps.append(line)
        
        return steps
    
    def _is_math_expression(self, text: str) -> bool:
        """
        Determine if text contains mathematical expression
        
        Args:
            text: Input text
            
        Returns:
            Whether text contains mathematical expression
        """
        # Check if contains mathematical operators
        has_operator = any(op in text for op in self.math_operators)
        
        # Check if contains numbers
        has_numbers = bool(re.search(r'\d+', text))
        
        # Check if contains mathematical keywords
        math_keywords = ['calculate', 'compute', 'solve', 'find', 'result', 'answer']
        has_math_keyword = any(keyword in text.lower() for keyword in math_keywords)
        
        return has_operator and (has_numbers or has_math_keyword)
    
    def evaluate_step_coverage(self, student_steps: List[str], 
                             teacher_steps: List[str]) -> Dict[str, float]:
        """
        Evaluate step coverage
        
        Args:
            student_steps: Student reasoning steps
            teacher_steps: Teacher reasoning steps
            
        Returns:
            Coverage evaluation results
        """
        if not teacher_steps:
            return {
                "step_coverage": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
        
        # Extract key mathematical expressions
        student_expressions = self._extract_math_expressions(student_steps)
        teacher_expressions = self._extract_math_expressions(teacher_steps)
        
        # Compute intersection
        common_expressions = set(student_expressions) & set(teacher_expressions)
        
        # Compute metrics
        precision = len(common_expressions) / max(1, len(student_expressions))
        recall = len(common_expressions) / len(teacher_expressions)
        f1_score = 2 * precision * recall / max(1, precision + recall)
        
        return {
            "step_coverage": recall,  # Use recall as coverage
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "student_steps": len(student_steps),
            "teacher_steps": len(teacher_steps),
            "common_steps": len(common_expressions)
        }
    
    def _extract_math_expressions(self, steps: List[str]) -> List[str]:
        """Extract mathematical expressions"""
        expressions = []
        
        for step in steps:
            # Find mathematical expression patterns
            patterns = [
                r'\d+\s*[+\-*/=]\s*\d+',  # Basic operations
                r'\d+\s*=\s*\d+',          # Equations
                r'[a-zA-Z]\s*=\s*\d+',     # Variable assignments
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, step)
                expressions.extend(matches)
        
        return expressions
    
    def evaluate_logical_consistency(self, response: str) -> Dict[str, float]:
        """
        Evaluate logical consistency
        
        Args:
            response: Model response
            
        Returns:
            Logical consistency evaluation results
        """
        # Extract number sequence
        numbers = self._extract_numbers(response)
        
        # Check reasonableness of numbers
        number_consistency = self._check_number_consistency(numbers)
        
        # Check usage of logical connectors
        logical_flow = self._check_logical_flow(response)
        
        # Check validity of mathematical expressions
        expression_validity = self._check_expression_validity(response)
        
        # Comprehensive scoring
        overall_consistency = (number_consistency + logical_flow + expression_validity) / 3
        
        return {
            "overall_consistency": overall_consistency,
            "number_consistency": number_consistency,
            "logical_flow": logical_flow,
            "expression_validity": expression_validity,
            "extracted_numbers": numbers
        }
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text"""
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(num) for num in numbers if num]
    
    def _check_number_consistency(self, numbers: List[float]) -> float:
        """
        Check number consistency (based on smoothness of number sequence)
        
        Removed heuristic negative/large number filtering because:
        - Problems themselves may involve negative numbers (losses, temperature, debt, etc.)
        - Problems may involve large numbers (population, distance, amount, etc.)
        """
        if not numbers:
            return 0.0
        
        if len(numbers) < 2:
            return 1.0
        
        # Compute smoothness of number sequence changes
        # Check if changes between adjacent numbers are reasonable (no sudden huge jumps)
        score = 1.0
        changes = []
        
        for i in range(len(numbers) - 1):
            if numbers[i] != 0:
                # Compute relative change
                relative_change = abs((numbers[i+1] - numbers[i]) / numbers[i])
                changes.append(relative_change)
        
        if changes:
            # If there are too many extreme changes (over 1000x), there may be a problem
            extreme_changes = sum(1 for change in changes if change > 1000)
            if extreme_changes > len(changes) * 0.5:
                score -= 0.3
        
        return max(0.0, score)
    
    def _check_logical_flow(self, text: str) -> float:
        """Check logical flow"""
        text_lower = text.lower()
        
        # Check usage of logical connectors
        connector_count = sum(1 for connector in self.logical_connectors 
                             if connector in text_lower)
        
        # Check if there is clear reasoning structure
        structure_score = 0.0
        
        if 'step' in text_lower or 'first' in text_lower:
            structure_score += 0.3
        if 'then' in text_lower or 'next' in text_lower:
            structure_score += 0.3
        if 'therefore' in text_lower or 'so' in text_lower or 'thus' in text_lower:
            structure_score += 0.4
        
        # Normalize score
        normalized_connector_score = min(1.0, connector_count / 3.0)
        
        return min(1.0, structure_score + normalized_connector_score * 0.3)
    
    def _check_expression_validity(self, text: str) -> float:
        """
        Check validity of mathematical expressions
        
        Improvement: Use more lenient standards, do not rely on eval (eval is ineffective for complex reasoning text)
        Only check if there is reasonable mathematical structure
        """
        # Extract mathematical expressions
        expressions = self._extract_math_expressions([text])
        
        if not expressions:
            # Lack of explicit mathematical expressions does not mean reasoning is invalid
            # Check if there are at least numbers and reasoning vocabulary
            has_numbers = bool(re.search(r'\d+', text))
            has_reasoning = any(word in text.lower() for word in 
                              ['calculate', 'compute', 'total', 'sum', 'multiply', 'divide', 'add', 'subtract'])
            if has_numbers and has_reasoning:
                return 0.7  # Has numbers and reasoning vocabulary, give higher score
            elif has_numbers:
                return 0.5  # Only numbers, give medium score
            else:
                return 0.3  # Lacks mathematical content
        
        # Check structural reasonableness of expressions
        valid_count = 0
        for expr in expressions:
            # Check if has basic mathematical structure (number + operator + number)
            if re.search(r'\d+\s*[+\-*/]\s*\d+', expr):
                valid_count += 1
            # Check equation structure (left = right, both have numbers)
            elif '=' in expr:
                parts = expr.split('=')
                if len(parts) == 2 and all(re.search(r'\d+', part) for part in parts):
                    valid_count += 1
        
        if len(expressions) == 0:
            return 0.5
        
        return max(0.3, valid_count / len(expressions))  # Give at least 0.3 points
    
    def extract_final_answer(self, text: str) -> Optional[float]:
        """
        Extract final answer from text
        
        Note: This method now calls utils.math_utils.extract_answer_unified for unified implementation
        Supports multiple formats: ####, \\boxed{}, "answer:", "The answer is"
        
        Args:
            text: Response text
            
        Returns:
            Extracted numerical answer, None if extraction fails
        """
        _, answer_num = extract_answer_unified(text)
        return answer_num
    
    def evaluate_answer_correctness(self, student_response: str, 
                                   ground_truth_answer: float,
                                   tolerance: float = 1e-4) -> Dict[str, Union[float, bool]]:
        """
        Evaluate correctness of final answer
        
        Args:
            student_response: Student response
            ground_truth_answer: Correct answer (numerical)
            tolerance: Tolerance range
            
        Returns:
            Answer correctness evaluation results
        """
        student_answer = self.extract_final_answer(student_response)
        
        if student_answer is None:
            # Unable to extract answer, treat as incorrect
            return {
                "is_correct": False,
                "correctness_score": 0.0,
                "student_answer": None,
                "ground_truth": ground_truth_answer,
                "error": "Unable to extract answer"
            }
        
        # Compute relative error
        if abs(ground_truth_answer) < 1e-10:
            # Ground truth close to 0, use absolute error
            is_correct = abs(student_answer - ground_truth_answer) < tolerance
            relative_error = abs(student_answer - ground_truth_answer)
        else:
            # Use relative error
            relative_error = abs(student_answer - ground_truth_answer) / abs(ground_truth_answer)
            is_correct = relative_error < tolerance
        
        # Compute continuous correctness score (give partial score even if not completely correct)
        if is_correct:
            correctness_score = 1.0
        else:
            # Give partial score based on error
            if relative_error < 0.01:  # Within 1% error
                correctness_score = 0.9
            elif relative_error < 0.05:  # Within 5% error
                correctness_score = 0.7
            elif relative_error < 0.1:  # Within 10% error
                correctness_score = 0.5
            elif relative_error < 0.5:  # Within 50% error
                correctness_score = 0.2
            else:
                correctness_score = 0.0
        
        return {
            "is_correct": is_correct,
            "correctness_score": correctness_score,
            "student_answer": student_answer,
            "ground_truth": ground_truth_answer,
            "relative_error": relative_error
        }
    
    def compute_kl_divergence(self, student_logits: torch.Tensor, 
                            teacher_logits: torch.Tensor) -> float:
        """
        Compute KL divergence
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            
        Returns:
            KL divergence
        """
        # 🔍 Safety check: ensure logits are not empty and dimensions are correct
        if student_logits is None or teacher_logits is None:
            return 0.0
        
        # Ensure tensor type
        if not isinstance(student_logits, torch.Tensor) or not isinstance(teacher_logits, torch.Tensor):
            return 0.0
        
        # Check if tensors are empty
        if student_logits.numel() == 0 or teacher_logits.numel() == 0:
            return 0.0
        
        # 🔍 Fix index out of bounds: ensure dimensions are correct
        try:
            # Check dimensions (should be [batch, seq_len, vocab_size])
            if len(student_logits.shape) < 2 or len(teacher_logits.shape) < 2:
                return 0.0
            
            # If only 2D, add batch dimension
            if len(student_logits.shape) == 2:
                student_logits = student_logits.unsqueeze(0)
            if len(teacher_logits.shape) == 2:
                teacher_logits = teacher_logits.unsqueeze(0)
            
            # Ensure dimensions match (take minimum sequence length to avoid index out of bounds)
            student_seq_len = student_logits.shape[1]
            teacher_seq_len = teacher_logits.shape[1]
            
            if student_seq_len == 0 or teacher_seq_len == 0:
                return 0.0
            
            min_len = min(student_seq_len, teacher_seq_len)
            
            # 🔍 Safe slicing: ensure indices are within valid range
            if min_len > 0:
                student_logits = student_logits[:, :min_len, :]
                teacher_logits = teacher_logits[:, :min_len, :]
            else:
                return 0.0
            
            # Compute probability distributions
            student_probs = F.softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            
            # Compute KL divergence
            kl_div = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                teacher_probs,
                reduction='batchmean'
            )
            
            return kl_div.item() if not torch.isnan(kl_div) else 0.0
            
        except (IndexError, RuntimeError, ValueError) as e:
            # Catch index out of bounds or other runtime errors
            logging.warning(f"Error computing KL divergence (possibly due to tensor dimension mismatch): {e}")
            return 0.0
    
    def evaluate_reasoning_quality(self, student_response: str, 
                                 teacher_response: str,
                                 ground_truth_answer: Optional[float] = None,
                                 student_logits: Optional[torch.Tensor] = None,
                                 teacher_logits: Optional[torch.Tensor] = None) -> Dict[str, Union[float, Dict]]:
        """
        Comprehensive evaluation of reasoning quality
        
        Improvement: Added final answer correctness as an independent dimension with higher weight
        
        Weight allocation:
        - Answer correctness: 50% (most important, especially in mathematical tasks like GSM8K)
        - Step coverage: 20% (completeness of reasoning process)
        - Logical consistency: 20% (reasonableness of reasoning process)
        - KL divergence: 10% (consistency with teacher model)
        
        Args:
            student_response: Student response
            teacher_response: Teacher response
            ground_truth_answer: Correct answer (optional, strongly recommended)
            student_logits: Student model logits (optional)
            teacher_logits: Teacher model logits (optional)
            
        Returns:
            Reasoning quality evaluation results
        """
        # Extract reasoning steps
        student_steps = self.extract_reasoning_steps(student_response)
        teacher_steps = self.extract_reasoning_steps(teacher_response)
        
        # Evaluate step coverage
        step_coverage_results = self.evaluate_step_coverage(student_steps, teacher_steps)
        
        # Evaluate logical consistency
        logical_consistency_results = self.evaluate_logical_consistency(student_response)
        
        # Evaluate final answer correctness
        answer_correctness_results = None
        if ground_truth_answer is not None:
            answer_correctness_results = self.evaluate_answer_correctness(
                student_response, ground_truth_answer
            )
        
        # Compute KL divergence (if logits available)
        kl_divergence = 0.0
        if student_logits is not None and teacher_logits is not None:
            kl_divergence = self.compute_kl_divergence(student_logits, teacher_logits)
        
        # Comprehensive scoring
        if answer_correctness_results is not None:
            # When correct answer is available, use new weight allocation
            overall_score = (
                answer_correctness_results["correctness_score"] * 0.5 +  # Answer correctness 50%
                step_coverage_results["step_coverage"] * 0.2 +           # Step coverage 20%
                logical_consistency_results["overall_consistency"] * 0.2 + # Logical consistency 20%
                (1.0 / (1.0 + kl_divergence)) * 0.1                       # KL divergence 10%
            )
        else:
            # When correct answer is not available, use old weights (but adjusted to more reasonable allocation)
            overall_score = (
                step_coverage_results["step_coverage"] * 0.35 +
                logical_consistency_results["overall_consistency"] * 0.35 +
                (1.0 / (1.0 + kl_divergence)) * 0.3
            )
        
        result = {
            "overall_score": overall_score,
            "step_coverage": step_coverage_results,
            "logical_consistency": logical_consistency_results,
            "kl_divergence": kl_divergence,
            "student_steps_count": len(student_steps),
            "teacher_steps_count": len(teacher_steps)
        }
        
        # If answer correctness results exist, add to return value
        if answer_correctness_results is not None:
            result["answer_correctness"] = answer_correctness_results
        
        return result


class BatchReasoningEvaluator:
    """Batch Reasoning Evaluator"""
    
    def __init__(self, evaluator: ReasoningEvaluator):
        """
        Initialize batch evaluator
        
        Args:
            evaluator: Reasoning evaluator
        """
        self.evaluator = evaluator
    
    def evaluate_batch(self, student_responses: List[str],
                      teacher_responses: List[str],
                      ground_truth_answers: Optional[List[float]] = None,
                      student_logits_list: Optional[List[torch.Tensor]] = None,
                      teacher_logits_list: Optional[List[torch.Tensor]] = None) -> Dict[str, List]:
        """
        Batch evaluation of reasoning quality
        
        Args:
            student_responses: List of student responses
            teacher_responses: List of teacher responses
            ground_truth_answers: List of correct answers (optional, recommended)
            student_logits_list: List of student logits (optional)
            teacher_logits_list: List of teacher logits (optional)
            
        Returns:
            Batch evaluation results
        """
        batch_results = []
        
        for i, (student_resp, teacher_resp) in enumerate(zip(student_responses, teacher_responses)):
            student_logits = student_logits_list[i] if student_logits_list else None
            teacher_logits = teacher_logits_list[i] if teacher_logits_list else None
            ground_truth = ground_truth_answers[i] if ground_truth_answers else None
            
            result = self.evaluator.evaluate_reasoning_quality(
                student_resp, teacher_resp, ground_truth, student_logits, teacher_logits
            )
            
            batch_results.append(result)
        
        # Compute batch statistics
        batch_stats = self._compute_batch_statistics(batch_results)
        
        return {
            "individual_results": batch_results,
            "batch_statistics": batch_stats
        }
    
    def _compute_batch_statistics(self, batch_results: List[Dict]) -> Dict[str, float]:
        """Compute batch statistics"""
        if not batch_results:
            return {}
        
        # Extract various metrics
        overall_scores = [result["overall_score"] for result in batch_results]
        step_coverage_scores = [result["step_coverage"]["step_coverage"] for result in batch_results]
        logical_consistency_scores = [result["logical_consistency"]["overall_consistency"] for result in batch_results]
        kl_divergences = [result["kl_divergence"] for result in batch_results]
        
        stats = {
            "mean_overall_score": np.mean(overall_scores),
            "std_overall_score": np.std(overall_scores),
            "mean_step_coverage": np.mean(step_coverage_scores),
            "mean_logical_consistency": np.mean(logical_consistency_scores),
            "mean_kl_divergence": np.mean(kl_divergences),
            "batch_size": len(batch_results)
        }
        
        # If answer correctness data exists, also compute its statistics
        if "answer_correctness" in batch_results[0]:
            correctness_scores = [
                result["answer_correctness"]["correctness_score"] 
                for result in batch_results 
                if "answer_correctness" in result
            ]
            is_correct_list = [
                result["answer_correctness"]["is_correct"] 
                for result in batch_results 
                if "answer_correctness" in result
            ]
            
            if correctness_scores:
                stats["mean_answer_correctness"] = np.mean(correctness_scores)
                stats["std_answer_correctness"] = np.std(correctness_scores)
                stats["accuracy"] = np.mean(is_correct_list)  # Proportion of completely correct
                stats["correct_count"] = sum(is_correct_list)
                stats["total_count"] = len(is_correct_list)
        
        return stats


def create_reasoning_evaluator() -> ReasoningEvaluator:
    """Convenience function to create a reasoning evaluator"""
    return ReasoningEvaluator()






