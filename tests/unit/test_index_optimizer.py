"""Unit tests for IndexOptimizer from search optimization module."""

import pytest
import numpy as np
import threading
from unittest.mock import Mock, patch, MagicMock
from src.optimization.search_optimizer import IndexOptimizer, OptimizationConfig, IndexStats


class TestIndexOptimizer:
    """Test suite for IndexOptimizer class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return OptimizationConfig(
            faiss_index_type="IVF",
            nlist=100,
            nprobe=10
        )
    
    @pytest.fixture
    def mock_faiss(self):
        """Mock FAISS operations."""
        with patch('src.optimization.search_optimizer.faiss', create=True) as mock_faiss_module:
            # Mock index classes
            mock_flat_index = Mock()
            mock_flat_index.ntotal = 100
            mock_flat_index.d = 128
            mock_flat_index.is_trained = True
            mock_flat_index.add = Mock()
            
            mock_ivf_index = Mock()
            mock_ivf_index.ntotal = 1000
            mock_ivf_index.d = 128
            mock_ivf_index.is_trained = False
            mock_ivf_index.add = Mock()
            mock_ivf_index.train = Mock()
            
            mock_ivfpq_index = Mock()
            mock_ivfpq_index.ntotal = 100000
            mock_ivfpq_index.d = 128
            mock_ivfpq_index.is_trained = False
            mock_ivfpq_index.add = Mock()
            mock_ivfpq_index.train = Mock()
            
            mock_faiss_module.IndexFlatIP.return_value = mock_flat_index
            mock_faiss_module.IndexIVFFlat.return_value = mock_ivf_index
            mock_faiss_module.IndexIVFPQ.return_value = mock_ivfpq_index
            mock_faiss_module.ParameterSpace.return_value.set_index_parameters = Mock()
            
            yield {
                'module': mock_faiss_module,
                'flat': mock_flat_index,
                'ivf': mock_ivf_index,
                'ivfpq': mock_ivfpq_index
            }
    
    @pytest.fixture
    def index_optimizer(self, config, mock_faiss):
        """Create IndexOptimizer instance for testing."""
        return IndexOptimizer(config)
    
    def test_init_with_valid_config(self, config, mock_faiss):
        """Test IndexOptimizer initialization with valid configuration."""
        optimizer = IndexOptimizer(config)
        
        assert optimizer.config == config
        assert isinstance(optimizer.indexes, dict)
        assert isinstance(optimizer.index_metadata, dict)
        assert isinstance(optimizer._lock, threading.RLock)
    
    def test_create_optimized_index_small_dataset(self, index_optimizer, mock_faiss):
        """Test creating optimized index for small dataset (< 1000 vectors)."""
        vectors = np.random.rand(500, 128).astype(np.float32)
        
        index = index_optimizer.create_optimized_index(vectors, "small_index")
        
        # Should use flat index for small datasets
        mock_faiss['module'].IndexFlatIP.assert_called_once_with(128)
        assert index == mock_faiss['flat']
        
        # Verify metadata storage
        assert "small_index" in index_optimizer.indexes
        assert "small_index" in index_optimizer.index_metadata
        
        metadata = index_optimizer.index_metadata["small_index"]
        assert metadata["type"] == "Flat"
        assert metadata["dimension"] == 128
        assert metadata["total_vectors"] == 500
    
    def test_create_optimized_index_medium_dataset(self, index_optimizer, mock_faiss):
        """Test creating optimized index for medium dataset (1000-100000 vectors)."""
        vectors = np.random.rand(5000, 128).astype(np.float32)
        
        index = index_optimizer.create_optimized_index(vectors, "medium_index")
        
        # Should use IVF index for medium datasets
        mock_faiss['module'].IndexIVFFlat.assert_called_once()
        assert index == mock_faiss['ivf']
        
        # Should train index if not trained
        mock_faiss['ivf'].train.assert_called_once_with(vectors)
        mock_faiss['ivf'].add.assert_called_once_with(vectors)
        
        metadata = index_optimizer.index_metadata["medium_index"]
        assert metadata["type"] == "IVF"
        assert metadata["nlist"] == 100
    
    def test_create_optimized_index_large_dataset(self, index_optimizer, mock_faiss):
        """Test creating optimized index for large dataset (> 100000 vectors)."""
        vectors = np.random.rand(150000, 128).astype(np.float32)
        
        index = index_optimizer.create_optimized_index(vectors, "large_index")
        
        # Should use IVF-PQ index for large datasets
        mock_faiss['module'].IndexIVFPQ.assert_called_once()
        assert index == mock_faiss['ivfpq']
        
        metadata = index_optimizer.index_metadata["large_index"]
        assert metadata["type"] == "IVF-PQ"
    
    def test_create_optimized_index_already_trained(self, index_optimizer, mock_faiss):
        """Test creating index when it's already trained."""
        vectors = np.random.rand(5000, 128).astype(np.float32)
        mock_faiss['ivf'].is_trained = True
        
        index_optimizer.create_optimized_index(vectors, "trained_index")
        
        # Should not call train if already trained
        mock_faiss['ivf'].train.assert_not_called()
        mock_faiss['ivf'].add.assert_called_once_with(vectors)
    
    def test_optimize_search_params_nonexistent_index(self, index_optimizer):
        """Test optimizing search parameters for nonexistent index."""
        with pytest.raises(ValueError, match="Index 'nonexistent' not found"):
            index_optimizer.optimize_search_params("nonexistent")
    
    def test_optimize_search_params_flat_index(self, index_optimizer, mock_faiss):
        """Test optimizing search parameters for flat index."""
        # Create a flat index first
        vectors = np.random.rand(500, 128).astype(np.float32)
        index_optimizer.create_optimized_index(vectors, "flat_index")
        
        params = index_optimizer.optimize_search_params("flat_index")
        
        # Flat index should return basic nprobe parameter
        assert params["nprobe"] == 10  # config.nprobe
        
        # Should not call set_index_parameters for flat index
        mock_faiss['module'].ParameterSpace.return_value.set_index_parameters.assert_not_called()
    
    def test_optimize_search_params_ivf_high_recall(self, index_optimizer, mock_faiss):
        """Test optimizing search parameters for IVF index with high recall target."""
        # Create an IVF index first
        vectors = np.random.rand(5000, 128).astype(np.float32)
        index_optimizer.create_optimized_index(vectors, "ivf_index")
        
        params = index_optimizer.optimize_search_params("ivf_index", target_recall=0.95)
        
        # Should increase nprobe for higher recall
        assert params["nprobe"] == 20  # config.nprobe * 2
        
        # Should set index parameters
        mock_faiss['module'].ParameterSpace.return_value.set_index_parameters.assert_called_once()
    
    def test_optimize_search_params_ivf_medium_recall(self, index_optimizer, mock_faiss):
        """Test optimizing search parameters for IVF index with medium recall target."""
        # Create an IVF index first
        vectors = np.random.rand(5000, 128).astype(np.float32)
        index_optimizer.create_optimized_index(vectors, "ivf_index")
        
        params = index_optimizer.optimize_search_params("ivf_index", target_recall=0.90)
        
        # Should moderately increase nprobe
        assert params["nprobe"] == 15  # config.nprobe * 1.5
    
    def test_optimize_search_params_ivf_low_recall(self, index_optimizer, mock_faiss):
        """Test optimizing search parameters for IVF index with low recall target."""
        # Create an IVF index first
        vectors = np.random.rand(5000, 128).astype(np.float32)
        index_optimizer.create_optimized_index(vectors, "ivf_index")
        
        params = index_optimizer.optimize_search_params("ivf_index", target_recall=0.80)
        
        # Should use default nprobe
        assert params["nprobe"] == 10  # config.nprobe
    
    def test_optimize_search_params_nprobe_clamping(self, index_optimizer, mock_faiss):
        """Test that nprobe is clamped to nlist maximum."""
        # Create an IVF index first
        vectors = np.random.rand(5000, 128).astype(np.float32)
        index_optimizer.create_optimized_index(vectors, "ivf_index")
        
        # Set a very high target recall
        params = index_optimizer.optimize_search_params("ivf_index", target_recall=0.99)
        
        # nprobe should not exceed nlist (100)
        assert params["nprobe"] <= 100
    
    def test_get_index_stats_nonexistent_index(self, index_optimizer):
        """Test getting stats for nonexistent index."""
        with pytest.raises(ValueError, match="Index 'nonexistent' not found"):
            index_optimizer.get_index_stats("nonexistent")
    
    def test_get_index_stats_valid_index(self, index_optimizer, mock_faiss):
        """Test getting stats for valid index."""
        # Create an index first
        vectors = np.random.rand(1000, 128).astype(np.float32)
        mock_faiss['ivf'].ntotal = 1000
        index_optimizer.create_optimized_index(vectors, "test_index")
        
        stats = index_optimizer.get_index_stats("test_index")
        
        assert isinstance(stats, IndexStats)
        assert stats.index_name == "test_index"
        assert stats.total_documents == 1000
        assert stats.index_size_bytes == 1000 * 128 * 4  # ntotal * dimension * 4 bytes
        assert stats.avg_query_time == 0.0  # Default value
        assert stats.cache_hit_rate == 0.0  # Default value
        assert stats.fragmentation_ratio == 0.0  # Default value
    
    def test_thread_safety_concurrent_index_creation(self, index_optimizer, mock_faiss):
        """Test thread safety during concurrent index creation."""
        import threading
        
        def create_index(name_suffix):
            vectors = np.random.rand(100, 128).astype(np.float32)
            index_optimizer.create_optimized_index(vectors, f"concurrent_index_{name_suffix}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_index, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all indexes were created
        assert len(index_optimizer.indexes) == 5
        assert len(index_optimizer.index_metadata) == 5
        
        for i in range(5):
            assert f"concurrent_index_{i}" in index_optimizer.indexes
    
    def test_create_index_with_different_dimensions(self, index_optimizer, mock_faiss):
        """Test creating indexes with different vector dimensions."""
        dimensions = [64, 128, 256, 512]
        
        for dim in dimensions:
            vectors = np.random.rand(100, dim).astype(np.float32)
            index_name = f"index_dim_{dim}"
            
            index_optimizer.create_optimized_index(vectors, index_name)
            
            metadata = index_optimizer.index_metadata[index_name]
            assert metadata["dimension"] == dim
    
    def test_create_index_overwrites_existing(self, index_optimizer, mock_faiss):
        """Test that creating an index with existing name overwrites it."""
        vectors1 = np.random.rand(100, 128).astype(np.float32)
        vectors2 = np.random.rand(200, 128).astype(np.float32)
        
        # Create first index
        index_optimizer.create_optimized_index(vectors1, "test_index")
        assert index_optimizer.index_metadata["test_index"]["total_vectors"] == 100
        
        # Create second index with same name
        index_optimizer.create_optimized_index(vectors2, "test_index")
        assert index_optimizer.index_metadata["test_index"]["total_vectors"] == 200
    
    def test_index_metadata_structure(self, index_optimizer, mock_faiss):
        """Test the structure of index metadata."""
        vectors = np.random.rand(5000, 128).astype(np.float32)
        index_optimizer.create_optimized_index(vectors, "metadata_test")
        
        metadata = index_optimizer.index_metadata["metadata_test"]
        
        # Check all required metadata fields
        assert "type" in metadata
        assert "dimension" in metadata
        assert "total_vectors" in metadata
        assert "created_at" in metadata
        assert "nlist" in metadata
        
        # Check metadata types and values
        assert isinstance(metadata["type"], str)
        assert isinstance(metadata["dimension"], int)
        assert isinstance(metadata["total_vectors"], int)
        assert isinstance(metadata["created_at"], float)
        assert metadata["created_at"] > 0
    
    @patch('src.optimization.search_optimizer.logger')
    def test_logging_during_index_creation(self, mock_logger, index_optimizer, mock_faiss):
        """Test that appropriate log messages are generated during index creation."""
        vectors = np.random.rand(1000, 128).astype(np.float32)
        
        index_optimizer.create_optimized_index(vectors, "logged_index")
        
        # Check that info logs were called
        mock_logger.info.assert_called()
        
        # Verify log messages contain expected information
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Creating optimized index" in msg for msg in log_calls)
        assert any("Training" in msg for msg in log_calls)
        assert any("Created" in msg and "logged_index" in msg for msg in log_calls)
    
    def test_ivf_pq_parameter_calculation(self, index_optimizer, mock_faiss):
        """Test IVF-PQ parameter calculation for large datasets."""
        vectors = np.random.rand(150000, 256).astype(np.float32)
        
        index_optimizer.create_optimized_index(vectors, "ivfpq_test")
        
        # Check that IndexIVFPQ was called with correct parameters
        mock_faiss['module'].IndexIVFPQ.assert_called_once()
        call_args = mock_faiss['module'].IndexIVFPQ.call_args[0]
        
        # quantizer, dimension, nlist, m, nbits
        dimension = call_args[1]
        nlist = call_args[2]
        m = call_args[3]  # Number of sub-quantizers
        nbits = call_args[4]
        
        assert dimension == 256
        assert nlist == 100  # config.nlist
        assert m == min(64, 256 // 4)  # Should be 64
        assert nbits == 8
    
    def test_empty_vectors_handling(self, index_optimizer, mock_faiss):
        """Test handling of empty vector arrays."""
        vectors = np.array([]).reshape(0, 128).astype(np.float32)
        
        index = index_optimizer.create_optimized_index(vectors, "empty_index")
        
        # Should still create index but with 0 vectors
        assert "empty_index" in index_optimizer.indexes
        metadata = index_optimizer.index_metadata["empty_index"]
        assert metadata["total_vectors"] == 0
    
    def test_single_vector_handling(self, index_optimizer, mock_faiss):
        """Test handling of single vector."""
        vectors = np.random.rand(1, 128).astype(np.float32)
        
        index = index_optimizer.create_optimized_index(vectors, "single_vector")
        
        # Should create flat index for single vector
        mock_faiss['module'].IndexFlatIP.assert_called()
        metadata = index_optimizer.index_metadata["single_vector"]
        assert metadata["type"] == "Flat"
        assert metadata["total_vectors"] == 1