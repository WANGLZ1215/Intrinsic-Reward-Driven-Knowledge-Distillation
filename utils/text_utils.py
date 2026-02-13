"""
Text processing utility functions
Functionality: Provides text processing, cleaning, formatting, and other utility functions

This module provides:
1. General text cleaning and formatting functions
2. Text statistics and analysis functions
3. Text splitting and extraction functions
4. Simplified mathematical text processing (for complex functions, please use math_utils.py)

Note:
- Mathematical answer extraction functionality has been unified to math_utils.py's extract_answer_unified()
- Mathematical-related functions in this file are mainly for simple text preprocessing
- For complex mathematical calculations and validation, please use math_utils.py
"""

import re
import string
from typing import List, Optional, Dict, Union
import logging


def clean_text(text: str, remove_extra_whitespace: bool = True,
               remove_special_chars: bool = False) -> str:
    """
    Cleans text
    
    Args:
        text: Input text
        remove_extra_whitespace: Whether to remove extra whitespace
        remove_special_chars: Whether to remove special characters
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    if remove_extra_whitespace:
        text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters
    if remove_special_chars:
        # Keep letters, numbers, basic punctuation, and spaces
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalizes whitespace characters
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Replace all whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text


def extract_sentences(text: str) -> List[str]:
    """
    Extracts sentences
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Simple sentence splitting (based on periods, question marks, exclamation marks)
    sentences = re.split(r'[.!?]+', text)
    
    # Clean and filter
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def extract_paragraphs(text: str) -> List[str]:
    """
    Extracts paragraphs
    
    Args:
        text: Input text
        
    Returns:
        List of paragraphs
    """
    if not text:
        return []
    
    # Split paragraphs by double newlines
    paragraphs = text.split('\n\n')
    
    # Clean and filter
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return paragraphs


def remove_punctuation(text: str, keep_basic: bool = True) -> str:
    """
    Removes punctuation
    
    Args:
        text: Input text
        keep_basic: Whether to keep basic punctuation
        
    Returns:
        Text with punctuation removed
    """
    if not text:
        return ""
    
    if keep_basic:
        # Keep basic punctuation
        punctuation = string.punctuation.replace('.', '').replace(',', '').replace('?', '').replace('!', '')
        text = text.translate(str.maketrans('', '', punctuation))
    else:
        # Remove all punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text


def extract_numbers(text: str, include_decimals: bool = True) -> List[str]:
    """
    Extracts numbers
    
    Args:
        text: Input text
        include_decimals: Whether to include decimals
        
    Returns:
        List of number strings
    """
    if not text:
        return []
    
    if include_decimals:
        pattern = r'-?\d+\.?\d*'
    else:
        pattern = r'-?\d+'
    
    numbers = re.findall(pattern, text)
    return numbers


def extract_words(text: str, min_length: int = 1) -> List[str]:
    """
    Extracts words
    
    Args:
        text: Input text
        min_length: Minimum length
        
    Returns:
        List of words
    """
    if not text:
        return []
    
    # Extract words (letters and numbers)
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter by minimum length
    words = [w for w in words if len(w) >= min_length]
    
    return words


def count_words(text: str) -> int:
    """
    Counts the number of words
    
    Args:
        text: Input text
        
    Returns:
        Number of words
    """
    words = extract_words(text)
    return len(words)


def count_characters(text: str, include_spaces: bool = True) -> int:
    """
    Counts the number of characters
    
    Args:
        text: Input text
        include_spaces: Whether to include spaces
        
    Returns:
        Number of characters
    """
    if not text:
        return 0
    
    if include_spaces:
        return len(text)
    else:
        return len(text.replace(' ', ''))


def find_keywords(text: str, keywords: List[str], 
                 case_sensitive: bool = False) -> Dict[str, List[int]]:
    """
    Finds keywords
    
    Args:
        text: Input text
        keywords: List of keywords
        case_sensitive: Whether to be case-sensitive
        
    Returns:
        Keywords and their positions
    """
    if not text or not keywords:
        return {}
    
    if not case_sensitive:
        text = text.lower()
        keywords = [kw.lower() for kw in keywords]
    
    result = {}
    
    for keyword in keywords:
        positions = []
        start = 0
        
        while True:
            pos = text.find(keyword, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        if positions:
            result[keyword] = positions
    
    return result


def format_text_for_model(text: str, max_length: Optional[int] = None,
                         add_prefix: str = "", add_suffix: str = "") -> str:
    """
    Formats text for model
    
    Args:
        text: Input text
        max_length: Maximum length
        add_prefix: Prefix to add
        add_suffix: Suffix to add
        
    Returns:
        Formatted text
    """
    # Clean text
    text = clean_text(text)
    
    # Add prefix and suffix
    if add_prefix:
        text = add_prefix + text
    if add_suffix:
        text = text + add_suffix
    
    # Truncate to maximum length
    if max_length and len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0]  # Truncate at word boundary
    
    return text


def split_text_into_chunks(text: str, chunk_size: int = 1000,
                          overlap: int = 100) -> List[str]:
    """
    Splits text into chunks
    
    Args:
        text: Input text
        chunk_size: Chunk size
        overlap: Overlap size
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If not the last chunk, try to split at word boundary
        if end < len(text):
            # Search forward for space
            while end > start and text[end] not in ' \n\t':
                end -= 1
            
            # If no space found, force split
            if end == start:
                end = start + chunk_size
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Calculate start position of next chunk
        start = end - overlap if end < len(text) else end
    
    return chunks


def extract_math_expressions(text: str) -> List[str]:
    """
    Extracts mathematical expressions (simplified version)
    
    Note: This function is for simple text preprocessing
    For more complex mathematical expression extraction, please use math_utils.extract_math_operations
    
    Args:
        text: Input text
        
    Returns:
        List of mathematical expressions
    """
    if not text:
        return []
    
    # Mathematical expression patterns (basic patterns)
    patterns = [
        r'\d+\s*[+\-*/]\s*\d+',           # Basic operations
        r'\d+\s*=\s*\d+',                 # Equations
        r'[a-zA-Z]\s*=\s*\d+',            # Variable assignments
    ]
    
    expressions = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        expressions.extend(matches)
    
    return list(set(expressions))  # Remove duplicates


def clean_math_text(text: str) -> str:
    """
    Cleans mathematical text
    
    Args:
        text: Input text
        
    Returns:
        Cleaned mathematical text
    """
    if not text:
        return ""
    
    # Normalize mathematical symbols
    text = text.replace('×', '*')
    text = text.replace('÷', '/')
    text = text.replace('²', '^2')
    text = text.replace('³', '^3')
    
    # Remove extra whitespace
    text = normalize_whitespace(text)
    
    # Ensure spaces around operators
    text = re.sub(r'(\d)\s*([+\-*/=])\s*(\d)', r'\1 \2 \3', text)
    
    return text


def format_math_answer(answer: str) -> str:
    """
    Formats mathematical answer (simplified version)
    
    Note: This function is for simple answer formatting
    For more complex mathematical answer formatting, please use math_utils.format_math_answer
    
    Args:
        answer: Answer text
        
    Returns:
        Formatted answer
    """
    if not answer:
        return ""
    
    # Lazy import to avoid circular dependencies
    from utils.math_utils import format_math_answer as format_math_answer_util
    
    # If answer is a pure number or simple number string, use math_utils formatting
    try:
        num = float(answer.strip())
        return format_math_answer_util(num)
    except ValueError:
        # If not a pure number, return original answer
        return answer.strip()


def validate_text_format(text: str, expected_format: str = "math") -> bool:
    """
    Validates text format
    
    Args:
        text: Input text
        expected_format: Expected format
        
    Returns:
        Whether the format matches
    """
    if not text:
        return False
    
    if expected_format == "math":
        # Check if contains mathematical content
        has_numbers = bool(re.search(r'\d+', text))
        has_operators = bool(re.search(r'[+\-*/=]', text))
        return has_numbers and has_operators
    
    elif expected_format == "reasoning":
        # Check if contains reasoning keywords
        reasoning_words = ['step', 'first', 'then', 'therefore', 'so', 'thus', 'hence']
        return any(word in text.lower() for word in reasoning_words)
    
    return True


def extract_final_answer_from_text(text: str) -> str:
    """
    Extracts final answer from text (text format)
    
    Note: This function calls math_utils.extract_answer_unified for unified implementation
    Avoids code duplication and ensures logical consistency
    
    Args:
        text: Input text
        
    Returns:
        Final answer text
    """
    # Lazy import to avoid circular dependencies
    from utils.math_utils import extract_answer_unified
    
    answer_text, _ = extract_answer_unified(text)
    return answer_text
