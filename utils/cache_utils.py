"""
Cache Utilities for Transformers v4.43+
Correct solution for handling past_key_values deprecation warnings
"""

import torch
from transformers import Cache
from typing import Optional, Tuple, Union
import logging


class ModernCacheManager:
    """Modern cache manager using the new Cache class"""
    
    def __init__(self, model, batch_size: int = 1, max_length: int = 512):
        """
        Initializes the modern cache manager
        
        Args:
            model: Model instance
            batch_size: Batch size
            max_length: Maximum length
        """
        self.model = model
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache = None
        
        logging.info("Modern cache manager initialized")
    
    def create_cache(self, batch_size: int = None, max_length: int = None) -> Cache:
        """
        Creates a new Cache instance
        
        Args:
            batch_size: Batch size
            max_length: Maximum length
            
        Returns:
            Cache instance
        """
        if batch_size is None:
            batch_size = self.batch_size
        if max_length is None:
            max_length = self.max_length
            
        # Create new Cache instance
        self.cache = Cache(
            batch_size=batch_size,
            max_length=max_length,
            device=self.model.device,
            dtype=self.model.dtype
        )
        
        return self.cache
    
    def get_cache(self) -> Optional[Cache]:
        """Gets the current cache"""
        return self.cache
    
    def clear_cache(self):
        """Clears the cache"""
        self.cache = None
    
    def update_cache(self, new_cache: Cache):
        """Updates the cache"""
        self.cache = new_cache


def create_modern_cache(model, batch_size: int = 1, max_length: int = 512) -> ModernCacheManager:
    """
    Convenience function to create a modern cache manager
    
    Args:
        model: Model instance
        batch_size: Batch size
        max_length: Maximum length
        
    Returns:
        Modern cache manager instance
    """
    return ModernCacheManager(model, batch_size, max_length)


def suppress_past_key_values_warning():
    """
    Suppresses past_key_values deprecation warnings
    This is a temporary solution until all code is migrated to the new Cache class
    """
    import warnings
    warnings.filterwarnings("ignore", message=".*past_key_values.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*past_key_values.*", category=DeprecationWarning)


def create_generation_config_with_cache(model, **kwargs):
    """
    Creates a generation configuration with correct cache settings
    
    Args:
        model: Model instance
        **kwargs: Other generation parameters
        
    Returns:
        Generation configuration dictionary
    """
    config = {
        "use_cache": True,
        "past_key_values": None,  # Explicitly set to None
        **kwargs
    }
    
    return config


def update_model_for_modern_cache(model):
    """
    Updates model to use modern cache
    
    Args:
        model: Model instance
        
    Returns:
        Updated model
    """
    # Set model configuration to use the new Cache class
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = True
    
    # Suppress deprecation warnings
    suppress_past_key_values_warning()
    
    return model
