"""
Multi-Level Caching System for Production-Scale RAG Systems

This module provides a sophisticated caching architecture with multiple
cache levels, intelligent eviction policies, and performance optimization
for large-scale document retrieval and processing.
"""

import asyncio
import logging
import time
import json
import pickle
import hashlib
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict, defaultdict
import weakref
import sys
from pathlib import Path

# External dependencies
import redis
import memcache
import sqlite3
import aiofiles
import numpy as np
from diskcache import Cache as DiskCache

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    avg_access_time: float = 0.0
    memory_usage: int = 0

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    size: int
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    ttl: Optional[float] = None
    priority: int = 1

class CacheLayer(ABC):
    """Abstract base class for cache layers"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        pass

class MemoryCache(CacheLayer):
    """High-performance in-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 1000, max_memory: int = 512 * 1024 * 1024):
        self.max_size = max_size
        self.max_memory = max_memory
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._current_memory = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        start_time = time.time()
        
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check TTL
                if entry.ttl and time.time() - entry.timestamp > entry.ttl:
                    del self._cache[key]
                    self._current_memory -= entry.size
                    self._stats.misses += 1
                    return None
                
                # Update access information
                entry.access_count += 1
                entry.last_access = time.time()
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                
                self._stats.hits += 1
                access_time = time.time() - start_time
                self._update_avg_access_time(access_time)
                
                return entry.value
            else:
                self._stats.misses += 1
                return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in memory cache"""
        try:
            # Calculate entry size
            entry_size = sys.getsizeof(value) + sys.getsizeof(key)
            
            with self._lock:
                # Remove existing entry if present
                if key in self._cache:
                    old_entry = self._cache[key]
                    self._current_memory -= old_entry.size
                    del self._cache[key]
                
                # Check memory constraints
                while (self._current_memory + entry_size > self.max_memory or 
                       len(self._cache) >= self.max_size):
                    if not self._cache:
                        break
                    self._evict_lru()
                
                # Create new entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    size=entry_size,
                    timestamp=time.time(),
                    ttl=ttl
                )
                
                self._cache[key] = entry
                self._current_memory += entry_size
                self._stats.total_size = len(self._cache)
                self._stats.memory_usage = self._current_memory
                
                return True
        except Exception as e:
            logger.error(f"Failed to set cache entry: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                del self._cache[key]
                self._current_memory -= entry.size
                self._stats.total_size = len(self._cache)
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
            self._stats.total_size = 0
            self._stats.memory_usage = 0
            return True
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            self._current_memory -= entry.size
            self._stats.evictions += 1
    
    def _update_avg_access_time(self, access_time: float):
        """Update average access time"""
        total_accesses = self._stats.hits + self._stats.misses
        if total_accesses > 1:
            self._stats.avg_access_time = (
                (self._stats.avg_access_time * (total_accesses - 1) + access_time) 
                / total_accesses
            )
        else:
            self._stats.avg_access_time = access_time
    
    def get_stats(self) -> CacheStats:
        """Get memory cache statistics"""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                total_size=len(self._cache),
                avg_access_time=self._stats.avg_access_time,
                memory_usage=self._current_memory
            )

class RedisCache(CacheLayer):
    """Redis-based distributed cache"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 key_prefix: str = "rag_cache:", serializer: str = "pickle"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.serializer = serializer
        self._redis = None
        self._stats = CacheStats()
        self._connect()
    
    def _connect(self):
        """Connect to Redis"""
        try:
            self._redis = redis.from_url(self.redis_url, decode_responses=False)
            self._redis.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._redis = None
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        if self.serializer == "pickle":
            return pickle.dumps(value)
        elif self.serializer == "json":
            return json.dumps(value).encode('utf-8')
        else:
            raise ValueError(f"Unsupported serializer: {self.serializer}")
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        if self.serializer == "pickle":
            return pickle.loads(data)
        elif self.serializer == "json":
            return json.loads(data.decode('utf-8'))
        else:
            raise ValueError(f"Unsupported serializer: {self.serializer}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self._redis:
            return None
        
        start_time = time.time()
        
        try:
            prefixed_key = f"{self.key_prefix}{key}"
            data = self._redis.get(prefixed_key)
            
            if data:
                value = self._deserialize(data)
                self._stats.hits += 1
                access_time = time.time() - start_time
                self._update_avg_access_time(access_time)
                return value
            else:
                self._stats.misses += 1
                return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in Redis cache"""
        if not self._redis:
            return False
        
        try:
            prefixed_key = f"{self.key_prefix}{key}"
            serialized_value = self._serialize(value)
            
            if ttl:
                self._redis.setex(prefixed_key, int(ttl), serialized_value)
            else:
                self._redis.set(prefixed_key, serialized_value)
            
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        if not self._redis:
            return False
        
        try:
            prefixed_key = f"{self.key_prefix}{key}"
            result = self._redis.delete(prefixed_key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries with prefix"""
        if not self._redis:
            return False
        
        try:
            pattern = f"{self.key_prefix}*"
            keys = self._redis.keys(pattern)
            if keys:
                self._redis.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    def _update_avg_access_time(self, access_time: float):
        """Update average access time"""
        total_accesses = self._stats.hits + self._stats.misses
        if total_accesses > 1:
            self._stats.avg_access_time = (
                (self._stats.avg_access_time * (total_accesses - 1) + access_time) 
                / total_accesses
            )
        else:
            self._stats.avg_access_time = access_time
    
    def get_stats(self) -> CacheStats:
        """Get Redis cache statistics"""
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            evictions=0,  # Redis handles eviction internally
            total_size=0,  # Would need Redis INFO to get exact size
            avg_access_time=self._stats.avg_access_time,
            memory_usage=0
        )

class DiskCacheLayer(CacheLayer):
    """Persistent disk-based cache"""
    
    def __init__(self, cache_dir: str = "./cache", max_size: int = 10 * 1024 * 1024 * 1024):
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self._cache = None
        self._stats = CacheStats()
        self._init_cache()
    
    def _init_cache(self):
        """Initialize disk cache"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache = DiskCache(str(self.cache_dir), size_limit=self.max_size)
            logger.info(f"Initialized disk cache at {self.cache_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize disk cache: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        if not self._cache:
            return None
        
        start_time = time.time()
        
        try:
            value = self._cache.get(key)
            if value is not None:
                self._stats.hits += 1
                access_time = time.time() - start_time
                self._update_avg_access_time(access_time)
                return value
            else:
                self._stats.misses += 1
                return None
        except Exception as e:
            logger.error(f"Disk cache get error: {e}")
            self._stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in disk cache"""
        if not self._cache:
            return False
        
        try:
            expire_time = time.time() + ttl if ttl else None
            self._cache.set(key, value, expire=expire_time)
            return True
        except Exception as e:
            logger.error(f"Disk cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from disk cache"""
        if not self._cache:
            return False
        
        try:
            return self._cache.delete(key)
        except Exception as e:
            logger.error(f"Disk cache delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all disk cache entries"""
        if not self._cache:
            return False
        
        try:
            self._cache.clear()
            return True
        except Exception as e:
            logger.error(f"Disk cache clear error: {e}")
            return False
    
    def _update_avg_access_time(self, access_time: float):
        """Update average access time"""
        total_accesses = self._stats.hits + self._stats.misses
        if total_accesses > 1:
            self._stats.avg_access_time = (
                (self._stats.avg_access_time * (total_accesses - 1) + access_time) 
                / total_accesses
            )
        else:
            self._stats.avg_access_time = access_time
    
    def get_stats(self) -> CacheStats:
        """Get disk cache statistics"""
        try:
            size = len(self._cache) if self._cache else 0
            volume = self._cache.volume() if self._cache else 0
        except:
            size = 0
            volume = 0
        
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            evictions=0,
            total_size=size,
            avg_access_time=self._stats.avg_access_time,
            memory_usage=volume
        )

class SmartCache:
    """Intelligent cache that chooses optimal storage layer"""
    
    def __init__(self, value: Any, access_pattern: str = "unknown"):
        self.value = value
        self.access_pattern = access_pattern
        self.size = sys.getsizeof(value)
        self.access_count = 0
        self.last_access = time.time()
        self.creation_time = time.time()
    
    def get_optimal_layer(self) -> str:
        """Determine optimal cache layer based on access patterns"""
        # Hot data: frequent access, keep in memory
        if self.access_count > 10 and (time.time() - self.last_access) < 300:
            return "memory"
        
        # Warm data: moderate access, keep in Redis
        elif self.access_count > 3 and (time.time() - self.last_access) < 3600:
            return "redis"
        
        # Cold data: infrequent access, move to disk
        else:
            return "disk"

class MultiLevelCachingSystem:
    """Comprehensive multi-level caching system"""
    
    def __init__(self, 
                 memory_cache_size: int = 1000,
                 memory_cache_mb: int = 512,
                 redis_url: Optional[str] = None,
                 disk_cache_dir: str = "./cache",
                 disk_cache_gb: int = 10,
                 enable_smart_routing: bool = True):
        
        # Initialize cache layers
        self.memory_cache = MemoryCache(
            max_size=memory_cache_size,
            max_memory=memory_cache_mb * 1024 * 1024
        )
        
        self.redis_cache = RedisCache(redis_url) if redis_url else None
        
        self.disk_cache = DiskCacheLayer(
            cache_dir=disk_cache_dir,
            max_size=disk_cache_gb * 1024 * 1024 * 1024
        )
        
        # Smart routing
        self.enable_smart_routing = enable_smart_routing
        self._smart_entries: Dict[str, SmartCache] = {}
        
        # Statistics
        self._global_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "layer_hits": defaultdict(int),
            "layer_misses": defaultdict(int)
        }
        
        # Background tasks
        self._optimization_task = None
        self._running = False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache"""
        self._global_stats["total_requests"] += 1
        
        # Try memory cache first
        value = await self.memory_cache.get(key)
        if value is not None:
            self._global_stats["cache_hits"] += 1
            self._global_stats["layer_hits"]["memory"] += 1
            await self._update_smart_entry(key, "memory")
            return value
        
        self._global_stats["layer_misses"]["memory"] += 1
        
        # Try Redis cache
        if self.redis_cache:
            value = await self.redis_cache.get(key)
            if value is not None:
                self._global_stats["cache_hits"] += 1
                self._global_stats["layer_hits"]["redis"] += 1
                
                # Promote to memory if accessed frequently
                if self.enable_smart_routing:
                    await self._maybe_promote_to_memory(key, value)
                
                await self._update_smart_entry(key, "redis")
                return value
            
            self._global_stats["layer_misses"]["redis"] += 1
        
        # Try disk cache
        value = await self.disk_cache.get(key)
        if value is not None:
            self._global_stats["cache_hits"] += 1
            self._global_stats["layer_hits"]["disk"] += 1
            
            # Promote to higher levels if accessed frequently
            if self.enable_smart_routing:
                await self._maybe_promote_to_higher_level(key, value)
            
            await self._update_smart_entry(key, "disk")
            return value
        
        self._global_stats["layer_misses"]["disk"] += 1
        self._global_stats["cache_misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None,
                 layer: Optional[str] = None) -> bool:
        """Set value in multi-level cache"""
        if self.enable_smart_routing and not layer:
            # Determine optimal layer
            smart_entry = SmartCache(value)
            optimal_layer = smart_entry.get_optimal_layer()
            self._smart_entries[key] = smart_entry
        else:
            optimal_layer = layer or "memory"
        
        # Set in specified or optimal layer
        if optimal_layer == "memory":
            success = await self.memory_cache.set(key, value, ttl)
        elif optimal_layer == "redis" and self.redis_cache:
            success = await self.redis_cache.set(key, value, ttl)
        elif optimal_layer == "disk":
            success = await self.disk_cache.set(key, value, ttl)
        else:
            # Fallback to memory
            success = await self.memory_cache.set(key, value, ttl)
        
        # Also cache in lower-cost layers for redundancy
        if success and optimal_layer == "memory" and self.redis_cache:
            await self.redis_cache.set(key, value, ttl)
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache layers"""
        success = True
        
        success &= await self.memory_cache.delete(key)
        
        if self.redis_cache:
            success &= await self.redis_cache.delete(key)
        
        success &= await self.disk_cache.delete(key)
        
        # Remove from smart entries
        self._smart_entries.pop(key, None)
        
        return success
    
    async def clear(self) -> bool:
        """Clear all cache layers"""
        success = True
        
        success &= await self.memory_cache.clear()
        
        if self.redis_cache:
            success &= await self.redis_cache.clear()
        
        success &= await self.disk_cache.clear()
        
        self._smart_entries.clear()
        
        return success
    
    async def _update_smart_entry(self, key: str, accessed_layer: str):
        """Update smart cache entry statistics"""
        if key in self._smart_entries:
            entry = self._smart_entries[key]
            entry.access_count += 1
            entry.last_access = time.time()
    
    async def _maybe_promote_to_memory(self, key: str, value: Any):
        """Maybe promote frequently accessed item to memory"""
        if key in self._smart_entries:
            entry = self._smart_entries[key]
            if entry.access_count > 5:  # Promotion threshold
                await self.memory_cache.set(key, value)
    
    async def _maybe_promote_to_higher_level(self, key: str, value: Any):
        """Maybe promote item to higher cache level"""
        if key in self._smart_entries:
            entry = self._smart_entries[key]
            if entry.access_count > 3:
                if self.redis_cache:
                    await self.redis_cache.set(key, value)
                    if entry.access_count > 7:
                        await self.memory_cache.set(key, value)
    
    def start_optimization(self):
        """Start background optimization task"""
        if self._optimization_task and not self._optimization_task.done():
            return
        
        self._running = True
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info("Started cache optimization")
    
    async def _optimization_loop(self):
        """Background cache optimization"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Analyze access patterns and optimize placement
                await self._optimize_cache_placement()
                
                # Clean up expired smart entries
                await self._cleanup_smart_entries()
                
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")
    
    async def _optimize_cache_placement(self):
        """Optimize cache placement based on access patterns"""
        current_time = time.time()
        
        for key, entry in list(self._smart_entries.items()):
            optimal_layer = entry.get_optimal_layer()
            
            # Demote cold data
            if optimal_layer == "disk" and entry.access_count < 2:
                # Move from memory/redis to disk
                value = await self.memory_cache.get(key)
                if value is None and self.redis_cache:
                    value = await self.redis_cache.get(key)
                
                if value is not None:
                    await self.disk_cache.set(key, value)
                    await self.memory_cache.delete(key)
                    if self.redis_cache:
                        await self.redis_cache.delete(key)
    
    async def _cleanup_smart_entries(self):
        """Clean up old smart cache entries"""
        current_time = time.time()
        cutoff_time = current_time - 86400  # 24 hours
        
        expired_keys = [
            key for key, entry in self._smart_entries.items()
            if entry.last_access < cutoff_time
        ]
        
        for key in expired_keys:
            del self._smart_entries[key]
    
    def stop_optimization(self):
        """Stop background optimization"""
        self._running = False
        if self._optimization_task:
            self._optimization_task.cancel()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        redis_stats = self.redis_cache.get_stats() if self.redis_cache else None
        disk_stats = self.disk_cache.get_stats()
        
        total_hits = self._global_stats["cache_hits"]
        total_requests = self._global_stats["total_requests"]
        hit_rate = total_hits / max(total_requests, 1)
        
        return {
            "global": {
                "total_requests": total_requests,
                "cache_hits": total_hits,
                "cache_misses": self._global_stats["cache_misses"],
                "hit_rate": hit_rate,
                "layer_hits": dict(self._global_stats["layer_hits"]),
                "layer_misses": dict(self._global_stats["layer_misses"])
            },
            "layers": {
                "memory": asdict(memory_stats),
                "redis": asdict(redis_stats) if redis_stats else None,
                "disk": asdict(disk_stats)
            },
            "smart_cache": {
                "entries": len(self._smart_entries),
                "optimization_running": self._running
            }
        }

# Factory function
def create_caching_system(
    memory_cache_mb: int = 512,
    redis_url: Optional[str] = None,
    disk_cache_gb: int = 10,
    enable_smart_routing: bool = True,
    auto_optimize: bool = True
) -> MultiLevelCachingSystem:
    """Factory function to create caching system with sensible defaults"""
    
    cache_system = MultiLevelCachingSystem(
        memory_cache_mb=memory_cache_mb,
        redis_url=redis_url,
        disk_cache_gb=disk_cache_gb,
        enable_smart_routing=enable_smart_routing
    )
    
    if auto_optimize:
        cache_system.start_optimization()
    
    return cache_system

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create caching system
        cache = create_caching_system(
            memory_cache_mb=256,
            redis_url="redis://localhost:6379",
            enable_smart_routing=True
        )
        
        # Test caching operations
        test_data = {
            "doc1": "This is a test document",
            "doc2": ["list", "of", "tokens"],
            "doc3": {"key": "value", "number": 42}
        }
        
        # Set values
        for key, value in test_data.items():
            await cache.set(key, value)
            print(f"Set {key}")
        
        # Get values
        for key in test_data.keys():
            value = await cache.get(key)
            print(f"Got {key}: {value is not None}")
        
        # Get stats
        stats = cache.get_comprehensive_stats()
        print("\nCache Stats:")
        print(json.dumps(stats, indent=2, default=str))
        
        # Cleanup
        cache.stop_optimization()
    
    asyncio.run(main())