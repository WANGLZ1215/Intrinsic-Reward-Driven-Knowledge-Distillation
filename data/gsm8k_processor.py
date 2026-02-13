"""
GSM8K Dataset Processing Module
Function: Load, preprocess and format GSM8K mathematical reasoning dataset
"""

import json
import re
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer
import torch
from utils.math_utils import extract_answer_unified


PROMPT_TEMPLATE = (
    "You are an expert mathematical reasoning assistant.\n\n"
    "Let's think step by step.\n\n"
    "Solve the problem with a concise chain-of-thought using 1–3 short, well-structured steps in English.\n"
    "Each step should follow logically, justify the reasoning, and clearly connect intermediate results.\n\n"
    "Then, on a new line, provide the final numeric answer in GSM8K format:\n"
    "#### <number>\n\n"
    "Do not output anything after this answer line.\n\n"
    "Question:\n\n"
    "{question}\n"
)


def build_prompt(question: str) -> str:
    """Build unified GSM8K prompt template"""
    return PROMPT_TEMPLATE.format(question=question)


class GSM8KProcessor:
    """GSM8K Dataset Processor"""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_dataset(self, dataset_name: str = "gsm8k", config: str = "main") -> Dict[str, Dataset]:
        """Load GSM8K dataset"""
        try:
            dataset = load_dataset(dataset_name, config)
            print(f"Dataset loaded successfully: {dataset_name}")
            print(f"Training set size: {len(dataset['train'])}")
            print(f"Test set size: {len(dataset['test'])}")
            return dataset
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            raise

    def extract_answer(self, answer_text: str) -> Tuple[str, float]:
        """
        Extract final answer number from answer text
        
        Note: This method now calls utils.math_utils.extract_answer_unified for unified implementation.
        Supports multiple formats: #### (GSM8K standard), \\boxed{}, "answer:", "The answer is"

        Args:
            answer_text: Text containing reasoning process and final answer

        Returns:
            (Final answer text, answer number)
        """
        answer_text_result, answer_num = extract_answer_unified(answer_text)
        
        # If number is None, return 0.0 for backward compatibility
        if answer_num is None:
            answer_num = 0.0
        
        return answer_text_result, answer_num

    def format_prompt(self, question: str) -> str:
        """
        Format question as model input format

        Args:
            question: Original question

        Returns:
            Formatted prompt text
        """
        return build_prompt(question)

    def format_full_response(self, question: str, answer: str) -> str:
        """
        Format complete response (question + answer)

        Args:
            question: Question
            answer: Answer

        Returns:
            Formatted complete text
        """
        prompt = self.format_prompt(question)
        return prompt + answer + self.tokenizer.eos_token

    def preprocess_sft_data(self, examples: Dict) -> Dict[str, List[str]]:
        """
        Preprocess SFT training data

        Args:
            examples: Dictionary containing question and answer

        Returns:
            Preprocessed text data
        """
        texts = []
        for question, answer in zip(examples["question"], examples["answer"]):
            full_text = self.format_full_response(question, answer)
            texts.append(full_text)
        return {"text": texts}

    def preprocess_rl_data(self, examples: Dict) -> Dict[str, List]:
        """
        Preprocess RL training data

        Args:
            examples: Dictionary containing question and answer

        Returns:
            Preprocessed RL data
        """
        questions = []
        answers = []
        correct_answers = []

        for question, answer in zip(examples["question"], examples["answer"]):
            questions.append(self.format_prompt(question))
            answers.append(answer)
            # Extract numerical ground truth from answer (for correctness reward calculation)
            _, correct_num = self.extract_answer(answer)
            correct_answers.append(correct_num)

        return {
            "questions": questions,
            "answers": answers,
            "correct_answers": correct_answers
        }

    def tokenize_function(self, examples: Dict) -> Dict:
        """
        Tokenization function

        Args:
            examples: Dictionary containing text

        Returns:
            Tokenized data
        """
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def create_data_collator(self):
        """Create data collator"""
        from transformers import DataCollatorForLanguageModeling

        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )

    def validate_data(self, dataset: Dataset, num_samples: int = 5) -> None:
        """
        Validate dataset quality

        Args:
            dataset: Dataset
            num_samples: Number of validation samples
        """
        print(f"Validating dataset, sample count: {len(dataset)}")

        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            print(f"\nSample {i+1}:")
            print(f"Question: {sample.get('question', 'N/A')[:100]}...")
            print(f"Answer: {sample.get('answer', 'N/A')[:100]}...")

            if 'answer' in sample:
                answer_text, answer_num = self.extract_answer(sample['answer'])
                print(f"Extracted answer: {answer_text} (number: {answer_num})")


class MathDataUtils:
    """Mathematical Data Processing Tools"""

    @staticmethod
    def is_valid_math_expression(expression: str) -> bool:
        """Check if mathematical expression is valid"""
        try:
            # Simple mathematical expression validation
            import sympy as sp
            sp.sympify(expression)
            return True
        except:
            return False

    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """Extract all numbers from text"""
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(num) for num in numbers if num]

    @staticmethod
    def calculate_answer_accuracy(predicted: str, ground_truth: str, tolerance: float = 1e-6) -> bool:
        """
        Calculate answer accuracy

        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            tolerance: Tolerance

        Returns:
            Whether the answer is accurate
        """
        try:
            pred_numbers = MathDataUtils.extract_numbers(predicted)
            gt_numbers = MathDataUtils.extract_numbers(ground_truth)

            if not pred_numbers or not gt_numbers:
                return False

            pred_answer = pred_numbers[-1]
            gt_answer = gt_numbers[-1]
            return abs(pred_answer - gt_answer) < tolerance
        except:
            return False


def create_dataloader(dataset: Dataset, tokenizer: PreTrainedTokenizer,
                     batch_size: int = 8, shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    Create data loader

    Args:
        dataset: Dataset
        tokenizer: Tokenizer
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Returns:
        Data loader
    """
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        # Extract text
        texts = [item["text"] for item in batch]

        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

        # Create labels (same as input) - common setting for autoregressive language models
        labels = tokenized["input_ids"].clone()

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=2
    )
