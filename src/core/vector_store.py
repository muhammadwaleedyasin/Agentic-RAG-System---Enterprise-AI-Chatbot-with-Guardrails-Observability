"""
Vector store implementation using ChromaDB.
"""
import asyncio
import uuid
from typing import List, Dict, Any, Optional, Tuple
import logging
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from ..config.settings import settings
from ..models.documents import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store implementation using ChromaDB."""
    
    def __init__(self):
        """Initialize the vector store."""
        self.client = None
        self.collection = None
        self.collection_name = settings.chroma_collection_name
        self.initialized = False
    
    async def initialize(self):
        """Initialize the ChromaDB client and collection."""
        try:
            # Initialize ChromaDB client
            if settings.vector_db_path:
                # Persistent client
                self.client = chromadb.PersistentClient(
                    path=settings.vector_db_path,
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
            else:
                # In-memory client
                self.client = chromadb.Client(
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "RAG document chunks"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            self.initialized = True
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    async def add_chunks(self, chunks: List[DocumentChunk]):
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks to add
        """
        if not self.initialized:
            await self.initialize()
        
        if not chunks:
            return
        
        try:
            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = []
            embeddings = []
            
            for chunk in chunks:
                # Prepare metadata
                metadata = {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
                
                # Add chunk metadata if available
                if chunk.metadata:
                    metadata.update(chunk.metadata)
                
                metadatas.append(metadata)
                
                # Add embedding if available
                if chunk.embedding:
                    embeddings.append(chunk.embedding)
            
            # Add to collection
            if embeddings:
                # Use provided embeddings
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:
                # Let ChromaDB generate embeddings
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add chunks to vector store: {str(e)}")
            raise
    
    async def search(
        self,
        query: str = None,
        query_embedding: List[float] = None,
        top_k: int = 5,
        where: Dict[str, Any] = None,
        where_document: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in the vector store.
        
        Args:
            query: Text query (will be embedded)
            query_embedding: Pre-computed query embedding
            top_k: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions
            
        Returns:
            List of search results with metadata
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Build query parameters
            query_params = {
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
            }
            
            if where:
                query_params["where"] = where
            if where_document:
                query_params["where_document"] = where_document
            
            # Execute query
            if query_embedding:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    **query_params
                )
            elif query:
                results = self.collection.query(
                    query_texts=[query],
                    **query_params
                )
            else:
                raise ValueError("Either query or query_embedding must be provided")
            
            # Format results
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    result = {
                        "chunk_id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "similarity_score": 1 - results["distances"][0][i]  # Convert distance to similarity
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search vector store: {str(e)}")
            raise
    
    async def delete_chunks(self, chunk_ids: List[str]):
        """
        Delete chunks from the vector store.
        
        Args:
            chunk_ids: List of chunk IDs to delete
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            self.collection.delete(ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks from vector store")
            
        except Exception as e:
            logger.error(f"Failed to delete chunks: {str(e)}")
            raise
    
    async def delete_document(self, document_id: str):
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID to delete
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            self.collection.delete(where={"document_id": document_id})
            logger.info(f"Deleted all chunks for document: {document_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete document chunks: {str(e)}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if not self.initialized:
            await self.initialize()
        
        try:
            count = self.collection.count()
            
            return {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "initialized": self.initialized
            }
            
        except Exception as e:
            logger.error(f"Failed to get vector store stats: {str(e)}")
            return {"error": str(e)}
    
    async def clear(self):
        """Clear all data from the vector store."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document chunks"}
            )
            logger.info("Vector store cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear vector store: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Try to get collection count
            count = self.collection.count()
            return True
            
        except Exception:
            return False