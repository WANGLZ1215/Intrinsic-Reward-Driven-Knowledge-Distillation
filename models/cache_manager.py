"""
Cache Management Module
Function: Manage teacher model logits cache to improve training efficiency
"""

import torch
import pickle
import hashlib
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import time
import json


class CacheManager:
    """Cache manager"""
    
    def __init__(self, max_cache_size: int = 10000, 
                 eviction_policy: str = "LRU",
                 cache_file: Optional[str] = None):
        """
        Initialize cache manager
        
        Args:
            max_cache_size: Maximum cache size
            eviction_policy: Eviction policy (LRU, LFU, FIFO)
            FIFO (First-In, First-Out) - First in, first out
            LRU (Least Recently Used) - Least recently used
            LFU (Least Frequently Used) - Least frequently used
            cache_file: Cache file path
        """
        self.max_cache_size = max_cache_size
        self.eviction_policy = eviction_policy
        self.cache_file = cache_file or "./cache/teacher_cache.pkl"
        
        # Initialize cache
        self.cache = OrderedDict()
        self.access_count = {}  # Access count (for LFU)
        self.access_time = {}   # Access time (for LRU)
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        
        # Load existing cache
        self._load_cache()
        
        logging.info(f"Cache manager initialized: size={max_cache_size}, policy={eviction_policy}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _update_access_info(self, key: str):
        """Update access information"""
        current_time = time.time()
        self.access_time[key] = current_time
        
        if key in self.access_count:
            self.access_count[key] += 1
        else:
            self.access_count[key] = 1
    
    def _evict_item(self):
        """Evict item according to policy"""
        if not self.cache:
            return
        
        if self.eviction_policy == "LRU":
            # Remove least recently used item
            oldest_key = min(self.access_time.keys(), key=lambda k: self.access_time[k])
            self._remove_item(oldest_key)
            
        elif self.eviction_policy == "LFU":
            # Remove least frequently used item
            least_frequent_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            self._remove_item(least_frequent_key)
            
        elif self.eviction_policy == "FIFO":
            # Remove earliest added item
            oldest_key = next(iter(self.cache))
            self._remove_item(oldest_key)
    
    def _remove_item(self, key: str):
        """Remove cache item"""
        if key in self.cache:
            del self.cache[key]
            del self.access_count[key]
            del self.access_time[key]
            self.cache_evictions += 1
    
    def _load_cache(self):
        """Load cache"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.cache = OrderedDict(cache_data.get("cache", {}))
                self.access_count = cache_data.get("access_count", {})
                self.access_time = cache_data.get("access_time", {})
                self.cache_hits = cache_data.get("cache_hits", 0)
                self.cache_misses = cache_data.get("cache_misses", 0)
                self.cache_evictions = cache_data.get("cache_evictions", 0)
                
                logging.info(f"Cache loaded: {len(self.cache)} items")
                
            except Exception as e:
                logging.warning(f"Cache loading failed: {e}")
                self.cache = OrderedDict()
    
    def _save_cache(self):
        """Save cache"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            cache_data = {
                "cache": dict(self.cache),
                "access_count": self.access_count,
                "access_time": self.access_time,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_evictions": self.cache_evictions
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logging.info(f"Cache saved: {len(self.cache)} items")
            
        except Exception as e:
            logging.error(f"Cache saving failed: {e}")
    
    def get(self, text: str) -> Optional[torch.Tensor]:
        """
        Get cached logits
        
        Args:
            text: Input text
            
        Returns:
            Cached logits or None
        """
        key = self._get_cache_key(text)
        
        if key in self.cache:
            self.cache_hits += 1
            self._update_access_info(key)
            
            # Move to end (LRU)
            self.cache.move_to_end(key)
            
            return self.cache[key]
        else:
            self.cache_misses += 1
            return None
    
    def put(self, text: str, logits: torch.Tensor):
        """
        Store logits to cache
        
        Args:
            text: Input text
            logits: Logits to cache
        """
        key = self._get_cache_key(text)
        
        # If cache is full, evict some items first
        while len(self.cache) >= self.max_cache_size:
            self._evict_item()
        
        # Store to cache
        self.cache[key] = logits.clone().detach().cpu()  # Move to CPU to save GPU memory
        self._update_access_info(key)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_count.clear()
        self.access_time.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        logging.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "cache_evictions": self.cache_evictions,
            "eviction_policy": self.eviction_policy
        }
    
    def save_stats(self, filepath: str):
        """Save statistics"""
        stats = self.get_stats()
        stats["timestamp"] = time.time()
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def cleanup(self):
        """Cleanup resources"""
        self._save_cache()
        logging.info("Cache manager cleanup completed")
    
    def save_cache(self, filepath: str):
        """Public method: Save cache to specified path"""
        # Temporarily save original path
        original_cache_file = self.cache_file
        # Use new path
        self.cache_file = filepath
        # Save cache
        self._save_cache()
        # Restore original path
        self.cache_file = original_cache_file
    
    def load_cache(self, filepath: str):
        """Public method: Load cache from specified path"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.cache = OrderedDict(cache_data.get("cache", {}))
                self.access_count = cache_data.get("access_count", {})
                self.access_time = cache_data.get("access_time", {})
                self.cache_hits = cache_data.get("cache_hits", 0)
                self.cache_misses = cache_data.get("cache_misses", 0)
                self.cache_evictions = cache_data.get("cache_evictions", 0)
                
                logging.info(f"Cache loaded from {filepath}: {len(self.cache)} items")
                
            except Exception as e:
                logging.warning(f"Failed to load cache from {filepath}: {e}")
                self.cache = OrderedDict()
        else:
            logging.warning(f"Cache file does not exist: {filepath}")


class BatchCacheManager:
    """Batch cache manager"""
    
    def __init__(self, cache_manager: CacheManager, batch_size: int = 8):
        """
        Initialize batch cache manager
        
        Args:
            cache_manager: Base cache manager
            batch_size: Batch size
        """
        self.cache_manager = cache_manager
        self.batch_size = batch_size
    
    def get_batch(self, texts: List[str]) -> Tuple[List[Optional[torch.Tensor]], List[str]]:
        """
        Batch get cache
        
        Args:
            texts: List of texts
            
        Returns:
            (List of cached logits, List of uncached texts)
        """
        cached_logits = []
        uncached_texts = []
        
        for text in texts:
            logits = self.cache_manager.get(text)
            if logits is not None:
                cached_logits.append(logits.to('cuda' if torch.cuda.is_available() else 'cpu'))
            else:
                cached_logits.append(None)
                uncached_texts.append(text)
        
        return cached_logits, uncached_texts
    
    def put_batch(self, texts: List[str], logits_list: List[torch.Tensor]):
        """
        Batch store cache
        
        Args:
            texts: List of texts
            logits_list: List of logits
        """
        for text, logits in zip(texts, logits_list):
            self.cache_manager.put(text, logits)


class CacheMonitor:
    """Cache monitor"""
    
    def __init__(self, cache_manager: CacheManager, log_interval: int = 100):
        """
        Initialize monitor
        
        Args:
            cache_manager: Cache manager
            log_interval: Log interval
        """
        self.cache_manager = cache_manager
        self.log_interval = log_interval
        self.last_log_time = time.time()
        self.request_count = 0
    
    def log_request(self):
        """Log request"""
        self.request_count += 1
        current_time = time.time()
        
        if current_time - self.last_log_time >= self.log_interval:
            stats = self.cache_manager.get_stats()
            logging.info(f"Cache statistics: {stats}")
            self.last_log_time = current_time
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        stats = self.cache_manager.get_stats()
        current_time = time.time()
        
        return {
            **stats,
            "request_count": self.request_count,
            "uptime": current_time - self.last_log_time,
            "requests_per_second": self.request_count / max(1, current_time - self.last_log_time)
        }


def create_cache_manager(config: Dict) -> CacheManager:
    """
    Convenience function to create cache manager
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Cache manager instance
    """
    return CacheManager(
        max_cache_size=config["teacher_model"]["cache_size"],
        eviction_policy=config["teacher_model"]["cache_policy"],
        cache_file=config.get("cache_file", "./cache/teacher_cache.pkl")
    )


