"""
Embedding service for generating text embeddings.
"""
import asyncio
import time
from typing import List, Optional, Dict, Any
import logging
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

from ..config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using sentence transformers."""
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on (cpu, cuda)
        """
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.model = None
        self.dimension = None
        
        # Ensure CUDA is available if requested
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
    
    async def initialize(self):
        """Initialize the embedding model asynchronously."""
        try:
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(self.model_name, device=self.device)
            )
            
            # Get embedding dimension
            test_embedding = await self.embed_single("test")
            self.dimension = len(test_embedding)
            
            logger.info(f"Embedding service initialized with model {self.model_name} on {self.device}")
            logger.info(f"Embedding dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {str(e)}")
            raise
    
    async def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        if not self.model:
            await self.initialize()
        
        try:
            # Run embedding in thread pool
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self.model.encode(text, convert_to_tensor=False)
            )
            
            # Convert to list if numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {str(e)}")
            raise
    
    async def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding lists
        """
        if not self.model:
            await self.initialize()
        
        if not texts:
            return []
        
        try:
            embeddings = []
            loop = asyncio.get_event_loop()
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Run batch embedding in thread pool
                batch_embeddings = await loop.run_in_executor(
                    None,
                    lambda: self.model.encode(batch, convert_to_tensor=False)
                )
                
                # Convert to list format
                if isinstance(batch_embeddings, np.ndarray):
                    batch_embeddings = batch_embeddings.tolist()
                
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {str(e)}")
            raise
    
    async def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between -1 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {str(e)}")
            return 0.0
    
    async def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar embeddings to a query.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                sim = await self.similarity(query_embedding, candidate)
                similarities.append((i, sim))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find similar embeddings: {str(e)}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.dimension,
            "initialized": self.model is not None
        }
    
    async def health_check(self) -> bool:
        """Check if the embedding service is healthy."""
        try:
            if not self.model:
                await self.initialize()
            
            # Test with a simple embedding
            test_embedding = await self.embed_single("health check")
            return len(test_embedding) > 0
            
        except Exception:
            return False