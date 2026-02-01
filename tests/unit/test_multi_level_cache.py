"""Unit tests for MultiLevelCachingSystem from caching layer module."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from src.optimization.caching_layer import (
    MultiLevelCachingSystem, 
    MemoryCache, 
    RedisCache, 
    DiskCacheLayer,
    SmartCache
)


class TestMultiLevelCachingSystem:
    """Test suite for MultiLevelCachingSystem class."""
    
    @pytest.fixture
    def mock_redis_cache(self):
        """Create mock Redis cache."""
        mock_redis = Mock(spec=RedisCache)
        mock_redis.get = AsyncMock()
        mock_redis.set = AsyncMock()
        mock_redis.delete = AsyncMock()
        mock_redis.clear = AsyncMock()
        mock_redis.get_stats = Mock()
        return mock_redis
    
    @pytest.fixture
    def cache_system(self, mock_redis_cache):
        """Create MultiLevelCachingSystem instance for testing."""
        with patch('src.optimization.caching_layer.MemoryCache') as mock_memory, \
             patch('src.optimization.caching_layer.RedisCache') as mock_redis_class, \
             patch('src.optimization.caching_layer.DiskCacheLayer') as mock_disk:
            
            # Configure mock instances
            mock_memory_instance = Mock(spec=MemoryCache)
            mock_memory_instance.get = AsyncMock()
            mock_memory_instance.set = AsyncMock()
            mock_memory_instance.delete = AsyncMock()
            mock_memory_instance.clear = AsyncMock()
            mock_memory_instance.get_stats = Mock()
            mock_memory.return_value = mock_memory_instance
            
            mock_disk_instance = Mock(spec=DiskCacheLayer)
            mock_disk_instance.get = AsyncMock()
            mock_disk_instance.set = AsyncMock()
            mock_disk_instance.delete = AsyncMock()
            mock_disk_instance.clear = AsyncMock()
            mock_disk_instance.get_stats = Mock()
            mock_disk.return_value = mock_disk_instance
            
            mock_redis_class.return_value = mock_redis_cache
            
            system = MultiLevelCachingSystem(
                memory_cache_size=100,
                memory_cache_mb=64,
                redis_url="redis://localhost:6379",
                disk_cache_dir="./test_cache",
                disk_cache_gb=1,
                enable_smart_routing=True
            )
            
            # Store mock instances for access in tests
            system._mock_memory = mock_memory_instance
            system._mock_redis = mock_redis_cache
            system._mock_disk = mock_disk_instance
            
            return system
    
    def test_init_with_redis(self, mock_redis_cache):
        """Test initialization with Redis enabled."""
        with patch('src.optimization.caching_layer.MemoryCache'), \
             patch('src.optimization.caching_layer.RedisCache') as mock_redis_class, \
             patch('src.optimization.caching_layer.DiskCacheLayer'):
            
            mock_redis_class.return_value = mock_redis_cache
            
            system = MultiLevelCachingSystem(
                redis_url="redis://localhost:6379",
                enable_smart_routing=True
            )
            
            assert system.redis_cache is not None
            assert system.enable_smart_routing is True
            assert isinstance(system._smart_entries, dict)
            assert isinstance(system._global_stats, dict)
    
    def test_init_without_redis(self):
        """Test initialization without Redis."""
        with patch('src.optimization.caching_layer.MemoryCache'), \
             patch('src.optimization.caching_layer.DiskCacheLayer'):
            
            system = MultiLevelCachingSystem(redis_url=None)
            
            assert system.redis_cache is None
    
    @pytest.mark.asyncio
    async def test_get_memory_cache_hit(self, cache_system):
        """Test get operation with memory cache hit."""
        key = "test_key"
        value = "test_value"
        
        # Configure memory cache to return value
        cache_system._mock_memory.get.return_value = value
        
        result = await cache_system.get(key)
        
        assert result == value
        cache_system._mock_memory.get.assert_called_once_with(key)
        
        # Should not check other layers
        cache_system._mock_redis.get.assert_not_called()
        cache_system._mock_disk.get.assert_not_called()
        
        # Check stats
        assert cache_system._global_stats["cache_hits"] == 1
        assert cache_system._global_stats["layer_hits"]["memory"] == 1
    
    @pytest.mark.asyncio
    async def test_get_redis_cache_hit(self, cache_system):
        """Test get operation with Redis cache hit."""
        key = "test_key"
        value = "test_value"
        
        # Configure memory cache to miss, Redis to hit
        cache_system._mock_memory.get.return_value = None
        cache_system._mock_redis.get.return_value = value
        
        result = await cache_system.get(key)
        
        assert result == value
        cache_system._mock_memory.get.assert_called_once_with(key)
        cache_system._mock_redis.get.assert_called_once_with(key)
        cache_system._mock_disk.get.assert_not_called()
        
        # Check stats
        assert cache_system._global_stats["cache_hits"] == 1
        assert cache_system._global_stats["layer_hits"]["redis"] == 1
        assert cache_system._global_stats["layer_misses"]["memory"] == 1
    
    @pytest.mark.asyncio
    async def test_get_disk_cache_hit(self, cache_system):
        """Test get operation with disk cache hit."""
        key = "test_key"
        value = "test_value"
        
        # Configure memory and Redis to miss, disk to hit
        cache_system._mock_memory.get.return_value = None
        cache_system._mock_redis.get.return_value = None
        cache_system._mock_disk.get.return_value = value
        
        result = await cache_system.get(key)
        
        assert result == value
        cache_system._mock_memory.get.assert_called_once_with(key)
        cache_system._mock_redis.get.assert_called_once_with(key)
        cache_system._mock_disk.get.assert_called_once_with(key)
        
        # Check stats
        assert cache_system._global_stats["cache_hits"] == 1
        assert cache_system._global_stats["layer_hits"]["disk"] == 1
        assert cache_system._global_stats["layer_misses"]["memory"] == 1
        assert cache_system._global_stats["layer_misses"]["redis"] == 1
    
    @pytest.mark.asyncio
    async def test_get_cache_miss_all_layers(self, cache_system):
        """Test get operation with miss on all layers."""
        key = "nonexistent_key"
        
        # Configure all layers to miss
        cache_system._mock_memory.get.return_value = None
        cache_system._mock_redis.get.return_value = None
        cache_system._mock_disk.get.return_value = None
        
        result = await cache_system.get(key)
        
        assert result is None
        
        # All layers should be checked
        cache_system._mock_memory.get.assert_called_once_with(key)
        cache_system._mock_redis.get.assert_called_once_with(key)
        cache_system._mock_disk.get.assert_called_once_with(key)
        
        # Check stats
        assert cache_system._global_stats["cache_misses"] == 1
        assert cache_system._global_stats["layer_misses"]["memory"] == 1
        assert cache_system._global_stats["layer_misses"]["redis"] == 1
        assert cache_system._global_stats["layer_misses"]["disk"] == 1
    
    @pytest.mark.asyncio
    async def test_get_without_redis(self):
        """Test get operation when Redis is not available."""
        with patch('src.optimization.caching_layer.MemoryCache') as mock_memory, \
             patch('src.optimization.caching_layer.DiskCacheLayer') as mock_disk:
            
            mock_memory_instance = Mock(spec=MemoryCache)
            mock_memory_instance.get = AsyncMock(return_value=None)
            mock_memory_instance.set = AsyncMock()
            mock_memory_instance.delete = AsyncMock()
            mock_memory_instance.clear = AsyncMock()
            mock_memory_instance.get_stats = Mock()
            mock_memory.return_value = mock_memory_instance
            
            mock_disk_instance = Mock(spec=DiskCacheLayer)
            mock_disk_instance.get = AsyncMock(return_value="disk_value")
            mock_disk_instance.set = AsyncMock()
            mock_disk_instance.delete = AsyncMock()
            mock_disk_instance.clear = AsyncMock()
            mock_disk_instance.get_stats = Mock()
            mock_disk.return_value = mock_disk_instance
            
            system = MultiLevelCachingSystem(redis_url=None)
            
            result = await system.get("test_key")
            
            assert result == "disk_value"
            # Should skip Redis layer
            mock_memory_instance.get.assert_called_once()
            mock_disk_instance.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_set_to_memory_layer(self, cache_system):
        """Test set operation to memory layer."""
        key = "test_key"
        value = "test_value"
        
        cache_system._mock_memory.set.return_value = True
        
        success = await cache_system.set(key, value, layer="memory")
        
        assert success is True
        cache_system._mock_memory.set.assert_called_once_with(key, value, None)
    
    @pytest.mark.asyncio
    async def test_set_to_redis_layer(self, cache_system):
        """Test set operation to Redis layer."""
        key = "test_key"
        value = "test_value"
        ttl = 3600
        
        cache_system._mock_redis.set.return_value = True
        
        success = await cache_system.set(key, value, ttl=ttl, layer="redis")
        
        assert success is True
        cache_system._mock_redis.set.assert_called_once_with(key, value, ttl)
    
    @pytest.mark.asyncio
    async def test_set_to_disk_layer(self, cache_system):
        """Test set operation to disk layer."""
        key = "test_key"
        value = "test_value"
        
        cache_system._mock_disk.set.return_value = True
        
        success = await cache_system.set(key, value, layer="disk")
        
        assert success is True
        cache_system._mock_disk.set.assert_called_once_with(key, value, None)
    
    @pytest.mark.asyncio
    async def test_set_with_smart_routing(self, cache_system):
        """Test set operation with smart routing enabled."""
        key = "smart_key"
        value = "smart_value"
        
        cache_system._mock_memory.set.return_value = True
        
        with patch('src.optimization.caching_layer.SmartCache') as mock_smart:
            mock_smart_instance = Mock()
            mock_smart_instance.get_optimal_layer.return_value = "memory"
            mock_smart.return_value = mock_smart_instance
            
            success = await cache_system.set(key, value)
            
            assert success is True
            cache_system._mock_memory.set.assert_called_once()
            
            # Should store smart entry
            assert key in cache_system._smart_entries
    
    @pytest.mark.asyncio
    async def test_set_with_redundancy(self, cache_system):
        """Test set operation with redundancy to lower layers."""
        key = "redundant_key"
        value = "redundant_value"
        
        cache_system._mock_memory.set.return_value = True
        cache_system._mock_redis.set.return_value = True
        
        success = await cache_system.set(key, value, layer="memory")
        
        assert success is True
        
        # Should set in both memory and Redis for redundancy
        cache_system._mock_memory.set.assert_called_once()
        cache_system._mock_redis.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_set_fallback_layer(self, cache_system):
        """Test set operation fallback when specified layer fails."""
        key = "fallback_key"
        value = "fallback_value"
        
        cache_system._mock_redis.set.return_value = False
        cache_system._mock_memory.set.return_value = True
        
        success = await cache_system.set(key, value, layer="redis")
        
        # Should attempt Redis first, then fallback to memory
        cache_system._mock_redis.set.assert_called_once()
        cache_system._mock_memory.set.assert_called_once()
        assert success is True
    
    @pytest.mark.asyncio
    async def test_delete_from_all_layers(self, cache_system):
        """Test delete operation removes from all layers."""
        key = "delete_key"
        
        cache_system._mock_memory.delete.return_value = True
        cache_system._mock_redis.delete.return_value = True
        cache_system._mock_disk.delete.return_value = True
        
        success = await cache_system.delete(key)
        
        assert success is True
        cache_system._mock_memory.delete.assert_called_once_with(key)
        cache_system._mock_redis.delete.assert_called_once_with(key)
        cache_system._mock_disk.delete.assert_called_once_with(key)
        
        # Should remove smart entry
        assert key not in cache_system._smart_entries
    
    @pytest.mark.asyncio
    async def test_delete_partial_success(self, cache_system):
        """Test delete operation with partial success."""
        key = "partial_key"
        
        cache_system._mock_memory.delete.return_value = True
        cache_system._mock_redis.delete.return_value = False
        cache_system._mock_disk.delete.return_value = True
        
        success = await cache_system.delete(key)
        
        # Should return False if any layer fails
        assert success is False
    
    @pytest.mark.asyncio
    async def test_clear_all_layers(self, cache_system):
        """Test clear operation clears all layers."""
        cache_system._mock_memory.clear.return_value = True
        cache_system._mock_redis.clear.return_value = True
        cache_system._mock_disk.clear.return_value = True
        
        success = await cache_system.clear()
        
        assert success is True
        cache_system._mock_memory.clear.assert_called_once()
        cache_system._mock_redis.clear.assert_called_once()
        cache_system._mock_disk.clear.assert_called_once()
        
        # Should clear smart entries
        assert len(cache_system._smart_entries) == 0
    
    @pytest.mark.asyncio
    async def test_maybe_promote_to_memory(self, cache_system):
        """Test promotion to memory based on access count."""
        key = "promote_key"
        value = "promote_value"
        
        # Create smart entry with high access count
        smart_entry = SmartCache(value)
        smart_entry.access_count = 6  # Above threshold
        cache_system._smart_entries[key] = smart_entry
        
        cache_system._mock_memory.set.return_value = True
        
        await cache_system._maybe_promote_to_memory(key, value)
        
        cache_system._mock_memory.set.assert_called_once_with(key, value)
    
    @pytest.mark.asyncio
    async def test_maybe_promote_to_higher_level(self, cache_system):
        """Test promotion to higher cache levels."""
        key = "promote_key"
        value = "promote_value"
        
        # Create smart entry with moderate access count
        smart_entry = SmartCache(value)
        smart_entry.access_count = 5  # Above threshold for Redis
        cache_system._smart_entries[key] = smart_entry
        
        cache_system._mock_redis.set.return_value = True
        
        await cache_system._maybe_promote_to_higher_level(key, value)
        
        cache_system._mock_redis.set.assert_called_once_with(key, value)
    
    @pytest.mark.asyncio
    async def test_maybe_promote_to_memory_from_higher_level(self, cache_system):
        """Test promotion to memory from higher access count."""
        key = "promote_key"
        value = "promote_value"
        
        # Create smart entry with very high access count
        smart_entry = SmartCache(value)
        smart_entry.access_count = 8  # Above threshold for memory
        cache_system._smart_entries[key] = smart_entry
        
        cache_system._mock_redis.set.return_value = True
        cache_system._mock_memory.set.return_value = True
        
        await cache_system._maybe_promote_to_higher_level(key, value)
        
        # Should promote to both Redis and memory
        cache_system._mock_redis.set.assert_called_once()
        cache_system._mock_memory.set.assert_called_once()
    
    def test_start_optimization(self, cache_system):
        """Test starting background optimization."""
        cache_system.start_optimization()
        
        assert cache_system._running is True
        assert cache_system._optimization_task is not None
        
        # Cleanup
        cache_system.stop_optimization()
    
    def test_start_optimization_already_running(self, cache_system):
        """Test starting optimization when already running."""
        cache_system._optimization_task = Mock()
        cache_system._optimization_task.done.return_value = False
        
        cache_system.start_optimization()
        
        # Should not create new task
        assert cache_system._optimization_task.done.return_value is False
    
    def test_stop_optimization(self, cache_system):
        """Test stopping background optimization."""
        cache_system._running = True
        cache_system._optimization_task = Mock()
        
        cache_system.stop_optimization()
        
        assert cache_system._running is False
        cache_system._optimization_task.cancel.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_optimization_loop_cleanup(self, cache_system):
        """Test optimization loop performs cleanup."""
        current_time = time.time()
        old_time = current_time - 86500  # Older than 24 hours
        
        # Add old smart entries
        old_entry = SmartCache("old_value")
        old_entry.last_access = old_time
        cache_system._smart_entries["old_key"] = old_entry
        
        recent_entry = SmartCache("recent_value")
        recent_entry.last_access = current_time
        cache_system._smart_entries["recent_key"] = recent_entry
        
        await cache_system._cleanup_smart_entries()
        
        # Old entry should be removed
        assert "old_key" not in cache_system._smart_entries
        assert "recent_key" in cache_system._smart_entries
    
    @pytest.mark.asyncio
    async def test_optimize_cache_placement(self, cache_system):
        """Test cache placement optimization."""
        # Create smart entry that should be demoted
        cold_entry = SmartCache("cold_value")
        cold_entry.access_count = 1  # Low access count
        cache_system._smart_entries["cold_key"] = cold_entry
        
        # Mock getting value from memory
        cache_system._mock_memory.get.return_value = "cold_value"
        cache_system._mock_memory.delete.return_value = True
        cache_system._mock_disk.set.return_value = True
        
        await cache_system._optimize_cache_placement()
        
        # Should move cold data to disk
        cache_system._mock_disk.set.assert_called_once_with("cold_key", "cold_value")
        cache_system._mock_memory.delete.assert_called_once_with("cold_key")
    
    def test_get_comprehensive_stats(self, cache_system):
        """Test getting comprehensive cache statistics."""
        from src.optimization.caching_layer import CacheStats
        
        # Mock stats from all layers
        memory_stats = CacheStats(hits=10, misses=2, evictions=1, total_size=50, avg_access_time=0.01, memory_usage=1024)
        redis_stats = CacheStats(hits=5, misses=3, evictions=0, total_size=20, avg_access_time=0.02, memory_usage=512)
        disk_stats = CacheStats(hits=2, misses=1, evictions=0, total_size=100, avg_access_time=0.05, memory_usage=2048)
        
        cache_system._mock_memory.get_stats.return_value = memory_stats
        cache_system._mock_redis.get_stats.return_value = redis_stats
        cache_system._mock_disk.get_stats.return_value = disk_stats
        
        # Set some global stats
        cache_system._global_stats["total_requests"] = 20
        cache_system._global_stats["cache_hits"] = 15
        cache_system._global_stats["cache_misses"] = 5
        
        stats = cache_system.get_comprehensive_stats()
        
        assert "global" in stats
        assert "layers" in stats
        assert "smart_cache" in stats
        
        # Check global stats
        assert stats["global"]["total_requests"] == 20
        assert stats["global"]["cache_hits"] == 15
        assert stats["global"]["hit_rate"] == 0.75
        
        # Check layer stats
        assert stats["layers"]["memory"]["hits"] == 10
        assert stats["layers"]["redis"]["hits"] == 5
        assert stats["layers"]["disk"]["hits"] == 2
        
        # Check smart cache stats
        assert "entries" in stats["smart_cache"]
        assert "optimization_running" in stats["smart_cache"]


class TestSmartCache:
    """Test suite for SmartCache class."""
    
    def test_init_with_defaults(self):
        """Test SmartCache initialization with defaults."""
        value = "test_value"
        smart_cache = SmartCache(value)
        
        assert smart_cache.value == value
        assert smart_cache.access_pattern == "unknown"
        assert smart_cache.size > 0
        assert smart_cache.access_count == 0
        assert smart_cache.last_access > 0
        assert smart_cache.creation_time > 0
    
    def test_init_with_access_pattern(self):
        """Test SmartCache initialization with access pattern."""
        value = "test_value"
        pattern = "frequent"
        smart_cache = SmartCache(value, access_pattern=pattern)
        
        assert smart_cache.access_pattern == pattern
    
    def test_get_optimal_layer_hot_data(self):
        """Test optimal layer selection for hot data."""
        smart_cache = SmartCache("value")
        smart_cache.access_count = 15  # High access count
        smart_cache.last_access = time.time()  # Recent access
        
        optimal_layer = smart_cache.get_optimal_layer()
        
        assert optimal_layer == "memory"
    
    def test_get_optimal_layer_warm_data(self):
        """Test optimal layer selection for warm data."""
        smart_cache = SmartCache("value")
        smart_cache.access_count = 5  # Moderate access count
        smart_cache.last_access = time.time() - 1800  # 30 minutes ago
        
        optimal_layer = smart_cache.get_optimal_layer()
        
        assert optimal_layer == "redis"
    
    def test_get_optimal_layer_cold_data(self):
        """Test optimal layer selection for cold data."""
        smart_cache = SmartCache("value")
        smart_cache.access_count = 1  # Low access count
        smart_cache.last_access = time.time() - 7200  # 2 hours ago
        
        optimal_layer = smart_cache.get_optimal_layer()
        
        assert optimal_layer == "disk"
    
    def test_get_optimal_layer_new_data(self):
        """Test optimal layer selection for new data."""
        smart_cache = SmartCache("value")
        # Default values: access_count=0, recent creation
        
        optimal_layer = smart_cache.get_optimal_layer()
        
        # Should go to disk for new, unaccessed data
        assert optimal_layer == "disk"