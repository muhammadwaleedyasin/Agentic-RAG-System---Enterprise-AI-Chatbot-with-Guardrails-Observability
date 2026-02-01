"""Unit tests for MemoryCache from caching layer module."""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from src.optimization.caching_layer import MemoryCache, CacheStats, CacheEntry


class TestMemoryCache:
    """Test suite for MemoryCache class."""
    
    @pytest.fixture
    def memory_cache(self):
        """Create MemoryCache instance for testing."""
        return MemoryCache(max_size=5, max_memory=1024)
    
    @pytest.fixture
    def large_memory_cache(self):
        """Create MemoryCache instance with large limits for testing."""
        return MemoryCache(max_size=1000, max_memory=10 * 1024 * 1024)
    
    def test_init_with_default_params(self):
        """Test MemoryCache initialization with default parameters."""
        cache = MemoryCache()
        
        assert cache.max_size == 1000
        assert cache.max_memory == 512 * 1024 * 1024
        assert len(cache._cache) == 0
        assert cache._current_memory == 0
        assert isinstance(cache._stats, CacheStats)
    
    def test_init_with_custom_params(self):
        """Test MemoryCache initialization with custom parameters."""
        cache = MemoryCache(max_size=100, max_memory=1024)
        
        assert cache.max_size == 100
        assert cache.max_memory == 1024
    
    @pytest.mark.asyncio
    async def test_set_and_get_basic(self, memory_cache):
        """Test basic set and get operations."""
        key = "test_key"
        value = "test_value"
        
        success = await memory_cache.set(key, value)
        assert success is True
        
        retrieved_value = await memory_cache.get(key)
        assert retrieved_value == value
        
        # Check stats
        stats = memory_cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 0
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, memory_cache):
        """Test getting a nonexistent key."""
        retrieved_value = await memory_cache.get("nonexistent")
        assert retrieved_value is None
        
        # Check stats
        stats = memory_cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 1
    
    @pytest.mark.asyncio
    async def test_set_with_ttl_valid(self, memory_cache):
        """Test setting and getting with TTL while valid."""
        key = "ttl_key"
        value = "ttl_value"
        ttl = 1.0  # 1 second
        
        await memory_cache.set(key, value, ttl=ttl)
        
        # Should be available immediately
        retrieved_value = await memory_cache.get(key)
        assert retrieved_value == value
    
    @pytest.mark.asyncio
    async def test_set_with_ttl_expired(self, memory_cache):
        """Test getting expired TTL entry."""
        key = "ttl_key"
        value = "ttl_value"
        ttl = 0.1  # 100ms
        
        await memory_cache.set(key, value, ttl=ttl)
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Should be None (expired)
        retrieved_value = await memory_cache.get(key)
        assert retrieved_value is None
        
        # Should be removed from cache
        assert key not in memory_cache._cache
        
        # Check stats
        stats = memory_cache.get_stats()
        assert stats.misses == 1
    
    @pytest.mark.asyncio
    async def test_delete_existing_key(self, memory_cache):
        """Test deleting an existing key."""
        key = "delete_key"
        value = "delete_value"
        
        await memory_cache.set(key, value)
        
        # Verify it exists
        assert await memory_cache.get(key) == value
        
        # Delete it
        success = await memory_cache.delete(key)
        assert success is True
        
        # Verify it's gone
        assert await memory_cache.get(key) is None
        assert key not in memory_cache._cache
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, memory_cache):
        """Test deleting a nonexistent key."""
        success = await memory_cache.delete("nonexistent")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, memory_cache):
        """Test clearing all cache entries."""
        # Add multiple entries
        for i in range(3):
            await memory_cache.set(f"key_{i}", f"value_{i}")
        
        assert len(memory_cache._cache) == 3
        
        # Clear cache
        success = await memory_cache.clear()
        assert success is True
        
        # Verify cache is empty
        assert len(memory_cache._cache) == 0
        assert memory_cache._current_memory == 0
        
        stats = memory_cache.get_stats()
        assert stats.total_size == 0
        assert stats.memory_usage == 0
    
    @pytest.mark.asyncio
    async def test_lru_eviction_by_size(self, memory_cache):
        """Test LRU eviction when max_size is reached."""
        # Fill cache to max_size (5)
        for i in range(6):  # One more than max_size
            await memory_cache.set(f"key_{i}", f"value_{i}")
        
        # Should only have 5 entries (max_size)
        assert len(memory_cache._cache) == 5
        
        # First entry should be evicted (LRU)
        assert "key_0" not in memory_cache._cache
        assert "key_1" in memory_cache._cache
        assert "key_5" in memory_cache._cache
        
        # Check eviction stats
        stats = memory_cache.get_stats()
        assert stats.evictions == 1
    
    @pytest.mark.asyncio
    async def test_lru_eviction_by_memory(self):
        """Test LRU eviction when max_memory is reached."""
        # Create cache with very small memory limit
        cache = MemoryCache(max_size=100, max_memory=100)
        
        # Add entries that will exceed memory limit
        large_value = "x" * 50  # Large string
        
        await cache.set("key1", large_value)
        await cache.set("key2", large_value)
        
        # Adding third large entry should trigger eviction
        await cache.set("key3", large_value)
        
        # Should have evicted some entries
        stats = cache.get_stats()
        assert stats.evictions > 0
        assert len(cache._cache) < 3
    
    @pytest.mark.asyncio
    async def test_lru_ordering(self, memory_cache):
        """Test that LRU ordering is maintained correctly."""
        # Add entries
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")
        await memory_cache.set("key3", "value3")
        
        # Access key1 to make it most recently used
        await memory_cache.get("key1")
        
        # Add more entries to trigger eviction
        await memory_cache.set("key4", "value4")
        await memory_cache.set("key5", "value5")
        await memory_cache.set("key6", "value6")  # Should evict key2 (least recently used)
        
        # key1 should still be present (was accessed recently)
        assert await memory_cache.get("key1") == "value1"
        # key2 should be evicted
        assert await memory_cache.get("key2") is None
    
    @pytest.mark.asyncio
    async def test_update_existing_entry(self, memory_cache):
        """Test updating an existing cache entry."""
        key = "update_key"
        old_value = "old_value"
        new_value = "new_value"
        
        # Set initial value
        await memory_cache.set(key, old_value)
        old_memory = memory_cache._current_memory
        
        # Update value
        await memory_cache.set(key, new_value)
        
        # Should have new value
        assert await memory_cache.get(key) == new_value
        
        # Should only have one entry
        assert len(memory_cache._cache) == 1
        
        # Memory should be updated
        assert memory_cache._current_memory != old_memory
    
    @pytest.mark.asyncio
    async def test_access_count_tracking(self, memory_cache):
        """Test that access count is tracked correctly."""
        key = "access_key"
        value = "access_value"
        
        await memory_cache.set(key, value)
        
        # Access multiple times
        for _ in range(3):
            await memory_cache.get(key)
        
        entry = memory_cache._cache[key]
        assert entry.access_count == 3
        assert entry.last_access > entry.timestamp
    
    @pytest.mark.asyncio
    async def test_thread_safety(self, large_memory_cache):
        """Test thread safety of concurrent operations."""
        import threading
        
        results = []
        errors = []
        
        async def worker(worker_id):
            try:
                # Each worker sets and gets its own keys
                key = f"worker_{worker_id}_key"
                value = f"worker_{worker_id}_value"
                
                await large_memory_cache.set(key, value)
                retrieved = await large_memory_cache.get(key)
                results.append((worker_id, retrieved == value))
            except Exception as e:
                errors.append((worker_id, e))
        
        # Create multiple concurrent tasks
        tasks = []
        for i in range(10):
            task = asyncio.create_task(worker(i))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert all(success for _, success in results)
    
    @pytest.mark.asyncio
    async def test_memory_usage_calculation(self, memory_cache):
        """Test that memory usage is calculated correctly."""
        initial_memory = memory_cache._current_memory
        assert initial_memory == 0
        
        # Add entry and check memory increased
        await memory_cache.set("test_key", "test_value")
        assert memory_cache._current_memory > initial_memory
        
        # Delete entry and check memory decreased
        await memory_cache.delete("test_key")
        assert memory_cache._current_memory == 0
    
    @pytest.mark.asyncio
    async def test_stats_accuracy(self, memory_cache):
        """Test that cache statistics are accurate."""
        # Initial stats
        stats = memory_cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.total_size == 0
        
        # Add entries
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")
        
        # Access one, miss one
        await memory_cache.get("key1")  # hit
        await memory_cache.get("nonexistent")  # miss
        
        # Trigger eviction by filling cache
        for i in range(10):  # More than max_size (5)
            await memory_cache.set(f"evict_key_{i}", f"evict_value_{i}")
        
        stats = memory_cache.get_stats()
        assert stats.hits >= 1
        assert stats.misses >= 1
        assert stats.evictions > 0
        assert stats.total_size == len(memory_cache._cache)
        assert stats.memory_usage == memory_cache._current_memory
    
    @pytest.mark.asyncio
    async def test_average_access_time_calculation(self, memory_cache):
        """Test that average access time is calculated correctly."""
        key = "timing_key"
        value = "timing_value"
        
        await memory_cache.set(key, value)
        
        # Add small delay to make access time measurable
        await asyncio.sleep(0.001)
        await memory_cache.get(key)
        
        stats = memory_cache.get_stats()
        assert stats.avg_access_time > 0
    
    @pytest.mark.asyncio
    async def test_cache_entry_structure(self, memory_cache):
        """Test that cache entries have correct structure."""
        key = "structure_key"
        value = "structure_value"
        ttl = 10.0
        
        await memory_cache.set(key, value, ttl=ttl)
        
        entry = memory_cache._cache[key]
        assert isinstance(entry, CacheEntry)
        assert entry.key == key
        assert entry.value == value
        assert entry.size > 0
        assert entry.timestamp > 0
        assert entry.access_count >= 0
        assert entry.last_access >= entry.timestamp
        assert entry.ttl == ttl
        assert entry.priority == 1  # default priority
    
    @pytest.mark.asyncio
    async def test_set_operation_failure_handling(self):
        """Test handling of failures during set operations."""
        # Create cache with impossible constraints to trigger failure
        cache = MemoryCache(max_size=0, max_memory=0)
        
        # Should handle the constraint gracefully
        success = await cache.set("key", "value")
        
        # The specific behavior depends on implementation details
        # but should not raise exceptions
        assert isinstance(success, bool)
    
    @patch('src.optimization.caching_layer.sys.getsizeof', side_effect=Exception("Size calculation error"))
    @pytest.mark.asyncio
    async def test_set_with_size_calculation_error(self, mock_getsizeof, memory_cache):
        """Test handling of errors during size calculation."""
        success = await memory_cache.set("key", "value")
        
        # Should handle the error gracefully
        assert success is False
    
    @pytest.mark.asyncio
    async def test_empty_cache_operations(self, memory_cache):
        """Test operations on empty cache."""
        # Get from empty cache
        result = await memory_cache.get("any_key")
        assert result is None
        
        # Delete from empty cache
        success = await memory_cache.delete("any_key")
        assert success is False
        
        # Clear empty cache
        success = await memory_cache.clear()
        assert success is True
        
        # Stats from empty cache
        stats = memory_cache.get_stats()
        assert stats.total_size == 0
        assert stats.memory_usage == 0
    
    @pytest.mark.asyncio
    async def test_large_value_handling(self, memory_cache):
        """Test handling of large values."""
        # Create a large value
        large_value = "x" * 1000
        
        success = await memory_cache.set("large_key", large_value)
        
        if success:
            retrieved = await memory_cache.get("large_key")
            assert retrieved == large_value
        
        # Should handle large values according to memory constraints
        assert isinstance(success, bool)
    
    @pytest.mark.asyncio
    async def test_special_value_types(self, memory_cache):
        """Test caching of different value types."""
        test_values = [
            ("string", "test_string"),
            ("integer", 42),
            ("float", 3.14),
            ("list", [1, 2, 3]),
            ("dict", {"key": "value"}),
            ("none", None),
            ("boolean", True),
        ]
        
        for key, value in test_values:
            success = await memory_cache.set(key, value)
            assert success is True
            
            retrieved = await memory_cache.get(key)
            assert retrieved == value
    
    def test_cache_stats_dataclass(self):
        """Test CacheStats dataclass functionality."""
        stats = CacheStats(
            hits=10,
            misses=5,
            evictions=2,
            total_size=100,
            avg_access_time=0.05,
            memory_usage=1024
        )
        
        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.evictions == 2
        assert stats.total_size == 100
        assert stats.avg_access_time == 0.05
        assert stats.memory_usage == 1024
    
    def test_cache_entry_dataclass(self):
        """Test CacheEntry dataclass functionality."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            size=100,
            timestamp=time.time(),
            access_count=5,
            last_access=time.time(),
            ttl=3600,
            priority=2
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.size == 100
        assert entry.access_count == 5
        assert entry.ttl == 3600
        assert entry.priority == 2