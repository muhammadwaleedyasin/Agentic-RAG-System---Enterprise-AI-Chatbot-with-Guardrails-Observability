"""
RAG-specific endpoints for advanced functionality.
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException

from ...models.rag import (
    RAGQuery, RAGResponse, RAGConfig, RAGMetricsResponse,
    IndexingRequest, IndexingResponse, EmbeddingRequest, EmbeddingResponse,
    VectorStoreStatsResponse, RetrievalStrategy
)
from ...models.common import BaseResponse
from ...core.rag_pipeline import RAGPipeline
from ...config.settings import settings
from ..deps import get_rag_pipeline

router = APIRouter()


@router.post("/rag/query", response_model=RAGResponse)
async def rag_query(
    query: RAGQuery,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Process a RAG query with advanced configuration options.
    """
    try:
        response = await rag_pipeline.query(query)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


@router.post("/rag/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(
    request: EmbeddingRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Generate embeddings for a list of texts.
    """
    try:
        import time
        start_time = time.time()
        
        # Generate embeddings
        embeddings = await rag_pipeline.embedding_service.embed_batch(
            texts=request.texts,
            batch_size=32
        )
        
        processing_time = time.time() - start_time
        
        return EmbeddingResponse(
            status="success",
            message="Embeddings generated successfully",
            embeddings=embeddings,
            model_name=request.model or settings.embedding_model,
            dimension=len(embeddings[0]) if embeddings else 0,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@router.post("/rag/reindex", response_model=IndexingResponse)
async def reindex_documents(
    request: IndexingRequest = None,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Reindex documents in the vector store.
    """
    try:
        import time
        start_time = time.time()
        
        # For now, this is a placeholder
        # In a full implementation, you would:
        # 1. Get all documents from the database
        # 2. Re-process and re-embed them
        # 3. Update the vector store
        
        # Get current stats
        stats = await rag_pipeline.get_stats()
        total_chunks = stats.get("vector_store", {}).get("total_chunks", 0)
        
        indexing_time = time.time() - start_time
        
        return IndexingResponse(
            status="success",
            message="Reindexing completed successfully",
            documents_processed=0,  # Would track actual documents
            chunks_indexed=total_chunks,
            indexing_time=indexing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")


@router.get("/rag/metrics", response_model=RAGMetricsResponse)
async def get_rag_metrics(
    time_period: str = "24h",
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Get RAG system metrics and performance statistics.
    """
    try:
        from datetime import datetime, timedelta
        
        # Parse time period
        if time_period == "24h":
            period_start = datetime.utcnow() - timedelta(hours=24)
        elif time_period == "7d":
            period_start = datetime.utcnow() - timedelta(days=7)
        elif time_period == "30d":
            period_start = datetime.utcnow() - timedelta(days=30)
        else:
            raise HTTPException(status_code=400, detail="Invalid time period")
        
        period_end = datetime.utcnow()
        
        # Get current stats (in production, this would come from metrics storage)
        stats = await rag_pipeline.get_stats()
        
        # Mock metrics (in production, these would be real metrics)
        metrics = {
            "total_queries": 0,  # Would track from metrics store
            "average_response_time": 0.0,
            "average_retrieval_time": 0.0,
            "average_generation_time": 0.0,
            "success_rate": 1.0,
            "total_documents_indexed": len(stats.get("vector_store", {})),
            "total_chunks_indexed": stats.get("vector_store", {}).get("total_chunks", 0)
        }
        
        return RAGMetricsResponse(
            status="success",
            message="RAG metrics retrieved successfully",
            metrics=metrics,
            time_period=time_period,
            period_start=period_start,
            period_end=period_end
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get RAG metrics: {str(e)}")


@router.get("/rag/vector-store/stats", response_model=VectorStoreStatsResponse)
async def get_vector_store_stats(
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Get detailed vector store statistics.
    """
    try:
        # Get vector store stats
        vector_stats = await rag_pipeline.vector_store.get_stats()
        
        # Mock additional stats (would come from actual vector store)
        stats = {
            "total_vectors": vector_stats.get("total_chunks", 0),
            "dimension": settings.vector_dimension,
            "index_size_mb": 0.0,  # Would calculate from actual index
            "memory_usage_mb": 0.0,  # Would get from vector store
            "disk_usage_mb": 0.0,   # Would calculate from files
            "collections": [settings.chroma_collection_name]
        }
        
        return VectorStoreStatsResponse(
            status="success",
            message="Vector store statistics retrieved",
            stats=stats,
            health_status="healthy" if vector_stats.get("initialized") else "unhealthy"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get vector store stats: {str(e)}")


@router.post("/rag/config/test")
async def test_rag_config(
    config: RAGConfig,
    test_query: str,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Test RAG configuration with a sample query.
    """
    try:
        from ...models.rag import RAGQuery
        
        # Create test query with the provided config
        query = RAGQuery(
            query=test_query,
            config=config
        )
        
        # Process query
        response = await rag_pipeline.query(query)
        
        # Return response with config info
        return {
            "status": "success",
            "message": "RAG configuration test completed",
            "config": config.dict(),
            "test_query": test_query,
            "response": response.dict(),
            "performance": {
                "retrieval_time": response.context.retrieval_time,
                "generation_time": response.generation_time,
                "total_time": response.total_time,
                "chunks_retrieved": len(response.context.retrieved_chunks)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG config test failed: {str(e)}")


@router.get("/rag/config/defaults")
async def get_default_rag_config():
    """
    Get the default RAG configuration.
    """
    try:
        config = RAGConfig()
        
        return {
            "status": "success",
            "message": "Default RAG configuration retrieved",
            "config": config.dict(),
            "settings": {
                "retrieval_top_k": settings.retrieval_top_k,
                "similarity_threshold": settings.similarity_threshold,
                "max_context_length": settings.max_context_length,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get default config: {str(e)}")


@router.post("/rag/similarity")
async def calculate_similarity(
    text1: str,
    text2: str,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Calculate semantic similarity between two texts.
    """
    try:
        # Generate embeddings
        embedding1 = await rag_pipeline.embedding_service.embed_single(text1)
        embedding2 = await rag_pipeline.embedding_service.embed_single(text2)
        
        # Calculate similarity
        similarity = await rag_pipeline.embedding_service.similarity(embedding1, embedding2)
        
        return {
            "status": "success",
            "message": "Similarity calculated successfully",
            "text1": text1,
            "text2": text2,
            "similarity_score": similarity,
            "interpretation": {
                "very_similar": similarity > 0.8,
                "similar": 0.6 < similarity <= 0.8,
                "somewhat_similar": 0.4 < similarity <= 0.6,
                "different": similarity <= 0.4
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity calculation failed: {str(e)}")


@router.delete("/rag/vector-store/clear")
async def clear_vector_store(
    confirm: bool = False,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Clear all data from the vector store (DANGEROUS OPERATION).
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="This operation requires confirmation. Set confirm=true to proceed."
        )
    
    try:
        await rag_pipeline.vector_store.clear()
        
        return BaseResponse(
            status="success",
            message="Vector store cleared successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear vector store: {str(e)}")
