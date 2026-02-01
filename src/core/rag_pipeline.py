"""
Core RAG pipeline orchestration.
"""
import asyncio
import time
from typing import List, Dict, Any, Optional
import logging

from ..config.settings import settings
from ..providers.provider_factory import create_llm_provider
from ..providers.base_provider import LLMMessage
from ..models.rag import RAGQuery, RAGResponse, RAGContext, RetrievedChunk
from ..models.documents import Document, DocumentChunk
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .document_processor import DocumentProcessor
from .retriever import Retriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Core RAG pipeline orchestration service."""
    
    def __init__(self):
        """Initialize the RAG pipeline."""
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.document_processor = DocumentProcessor()
        self.retriever = None
        self.llm_provider = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize all pipeline components."""
        try:
            logger.info("Initializing RAG pipeline...")
            
            # Initialize embedding service
            await self.embedding_service.initialize()
            
            # Initialize vector store
            await self.vector_store.initialize()
            
            # Initialize retriever
            self.retriever = Retriever(self.vector_store, self.embedding_service)
            
            # Initialize LLM provider
            self.llm_provider = create_llm_provider()
            
            self.initialized = True
            logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            raise
    
    async def ingest_document(self, file_path: str = None, text_content: str = None, 
                            filename: str = None, metadata: Dict[str, Any] = None) -> str:
        """
        Ingest a document into the RAG system.
        
        Args:
            file_path: Path to file to ingest
            text_content: Raw text content to ingest
            filename: Filename for text content
            metadata: Document metadata
            
        Returns:
            Document ID of the ingested document
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            # Process document
            if file_path:
                document = await self.document_processor.process_file(file_path, metadata)
            elif text_content:
                document = await self.document_processor.process_text(text_content, filename, metadata)
            else:
                raise ValueError("Either file_path or text_content must be provided")
            
            # Chunk document
            chunks = await self.document_processor.chunk_document(document)
            
            # Generate embeddings for chunks
            if chunks:
                # Extract content for embedding
                chunk_texts = [chunk.content for chunk in chunks]
                embeddings = await self.embedding_service.embed_batch(chunk_texts)
                
                # Add embeddings to chunks
                for chunk, embedding in zip(chunks, embeddings):
                    chunk.embedding = embedding
                
                # Store chunks in vector store
                await self.vector_store.add_chunks(chunks)
            
            processing_time = time.time() - start_time
            logger.info(f"Ingested document {document.document_id} with {len(chunks)} chunks in {processing_time:.2f}s")
            
            return document.document_id
            
        except Exception as e:
            logger.error(f"Failed to ingest document: {str(e)}")
            raise
    
    async def query(self, query: RAGQuery) -> RAGResponse:
        """
        Process a RAG query and generate a response.
        
        Args:
            query: RAG query object
            
        Returns:
            RAG response with answer and context
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            # Retrieve relevant chunks
            retrieval_start = time.time()
            retrieved_chunks = await self.retriever.retrieve(query)
            retrieval_time = time.time() - retrieval_start
            
            # Build context
            context = await self._build_context(query.query, retrieved_chunks, retrieval_time)
            
            # Generate response
            generation_start = time.time()
            answer = await self._generate_answer(query.query, context)
            generation_time = time.time() - generation_start
            
            # Build sources information
            sources = self._build_sources(retrieved_chunks)
            
            total_time = time.time() - start_time
            
            # Create response
            response = RAGResponse(
                status="success",
                message="Query processed successfully",
                query=query.query,
                answer=answer.content,
                context=context,
                sources=sources,
                generation_time=generation_time,
                total_time=total_time,
                usage=answer.usage
            )
            
            logger.info(f"Processed query in {total_time:.2f}s (retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process query: {str(e)}")
            raise
    
    async def _build_context(self, query: str, chunks: List[RetrievedChunk], retrieval_time: float) -> RAGContext:
        """Build RAG context from retrieved chunks."""
        # Combine chunk content
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[{i+1}] {chunk.content}")
        
        context_text = "\\n\\n".join(context_parts)
        
        # Truncate if too long
        max_length = settings.max_context_length
        if len(context_text) > max_length:
            context_text = context_text[:max_length] + "..."
        
        # Count tokens (rough approximation)
        total_tokens = len(context_text.split()) + len(query.split())
        
        return RAGContext(
            query=query,
            retrieved_chunks=chunks,
            total_chunks=len(chunks),
            context_text=context_text,
            retrieval_time=retrieval_time,
            total_tokens=total_tokens
        )
    
    async def _generate_answer(self, query: str, context: RAGContext) -> Any:
        """Generate answer using LLM based on context."""
        # Build prompt
        system_message = (
            "You are a helpful AI assistant. Use the provided context to answer the user's question. "
            "If the context doesn't contain enough information to answer the question, say so. "
            "Always cite the relevant parts of the context in your response using [1], [2], etc."
        )
        
        user_message = f"""Context:
{context.context_text}

Question: {query}

Please provide a comprehensive answer based on the context above."""
        
        # Create messages
        messages = [
            LLMMessage(role="system", content=system_message),
            LLMMessage(role="user", content=user_message)
        ]
        
        # Generate response
        response = await self.llm_provider.generate(messages)
        return response
    
    def _build_sources(self, chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """Build sources information from retrieved chunks."""
        sources = []
        for chunk in chunks:
            source = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "similarity_score": chunk.similarity_score,
                "rank": chunk.rank,
                "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                "metadata": chunk.metadata
            }
            sources.append(source)
        
        return sources
    
    async def delete_document(self, document_id: str):
        """
        Delete a document and all its chunks from the system.
        
        Args:
            document_id: ID of the document to delete
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            await self.vector_store.delete_document(document_id)
            logger.info(f"Deleted document {document_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            raise
    
    async def search_documents(self, query: str, top_k: int = 10, 
                             filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for documents without generating an answer.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Search filters
            
        Returns:
            List of search results
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_single(query)
            
            # Build filters
            where_filter = self.retriever._build_filters(filters)
            
            # Search
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                where=where_filter
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {str(e)}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        if not self.initialized:
            await self.initialize()
        
        try:
            vector_stats = await self.vector_store.get_stats()
            embedding_info = self.embedding_service.get_model_info()
            llm_info = self.llm_provider.get_model_info()
            
            return {
                "vector_store": vector_stats,
                "embedding_service": embedding_info,
                "llm_provider": llm_info,
                "initialized": self.initialized
            }
            
        except Exception as e:
            logger.error(f"Failed to get pipeline stats: {str(e)}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all pipeline components."""
        health = {
            "overall": False,
            "embedding_service": False,
            "vector_store": False,
            "retriever": False,
            "llm_provider": False
        }
        
        try:
            if not self.initialized:
                await self.initialize()
            
            # Check each component
            health["embedding_service"] = await self.embedding_service.health_check()
            health["vector_store"] = await self.vector_store.health_check()
            health["retriever"] = await self.retriever.health_check()
            health["llm_provider"] = await self.llm_provider.health_check()
            
            # Overall health
            health["overall"] = all([
                health["embedding_service"],
                health["vector_store"],
                health["retriever"],
                health["llm_provider"]
            ])
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
        
        return health
