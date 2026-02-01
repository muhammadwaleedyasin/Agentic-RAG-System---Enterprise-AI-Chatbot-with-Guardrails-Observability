"""Vector database interfaces and implementations."""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import pinecone
import weaviate
from pymilvus import Collection, connections, utility

from ..config.settings import get_settings
from ..models.base import Chunk, SearchQuery, SearchResult, VectorDBConfig
from ..utils.exceptions import VectorDBError
from ..utils.logging import log_vector_db_operation


class BaseVectorDB(ABC):
    """Abstract base class for vector database implementations."""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.settings = get_settings()
        self.client = None
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to vector database."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from vector database."""
        pass
    
    @abstractmethod
    async def create_collection(self, dimension: int) -> None:
        """Create collection/index for storing vectors."""
        pass
    
    @abstractmethod
    async def delete_collection(self) -> None:
        """Delete collection/index."""
        pass
    
    @abstractmethod
    async def upsert_chunks(self, chunks: List[Chunk]) -> None:
        """Insert or update chunks in the vector database."""
        pass
    
    @abstractmethod
    async def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunks by IDs."""
        pass
    
    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        pass


class WeaviateVectorDB(BaseVectorDB):
    """Weaviate vector database implementation."""
    
    def __init__(self, config: VectorDBConfig):
        super().__init__(config)
        self.class_name = config.config.get("class_name", "Document")
    
    async def connect(self) -> None:
        """Connect to Weaviate instance."""
        try:
            auth_config = None
            if self.config.config.get("api_key"):
                auth_config = weaviate.AuthApiKey(api_key=self.config.config["api_key"])
            
            self.client = weaviate.Client(
                url=self.config.connection_string,
                auth_client_secret=auth_config
            )
            
            # Test connection
            self.client.schema.get()
            
        except Exception as e:
            raise VectorDBError(f"Failed to connect to Weaviate: {str(e)}")
    
    async def disconnect(self) -> None:
        """Disconnect from Weaviate."""
        if self.client:
            # Weaviate client doesn't require explicit disconnection
            self.client = None
    
    async def create_collection(self, dimension: int) -> None:
        """Create Weaviate class (collection)."""
        try:
            # Check if class already exists
            existing_classes = self.client.schema.get()["classes"]
            class_names = [cls["class"] for cls in existing_classes]
            
            if self.class_name in class_names:
                return  # Class already exists
            
            # Define class schema
            class_schema = {
                "class": self.class_name,
                "description": "RAG document chunks",
                "vectorizer": "none",  # We provide our own vectors
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Chunk content"
                    },
                    {
                        "name": "document_id",
                        "dataType": ["string"],
                        "description": "Parent document ID"
                    },
                    {
                        "name": "chunk_index",
                        "dataType": ["int"],
                        "description": "Chunk position in document"
                    },
                    {
                        "name": "start_char",
                        "dataType": ["int"],
                        "description": "Start character position"
                    },
                    {
                        "name": "end_char",
                        "dataType": ["int"],
                        "description": "End character position"
                    },
                    {
                        "name": "app",
                        "dataType": ["string"],
                        "description": "Application name"
                    },
                    {
                        "name": "version",
                        "dataType": ["string"],
                        "description": "Document version"
                    },
                    {
                        "name": "audience",
                        "dataType": ["string"],
                        "description": "Target audience"
                    },
                    {
                        "name": "department",
                        "dataType": ["string"],
                        "description": "Department"
                    },
                    {
                        "name": "sensitivity",
                        "dataType": ["string"],
                        "description": "Data sensitivity level"
                    },
                    {
                        "name": "tags",
                        "dataType": ["string[]"],
                        "description": "Document tags"
                    }
                ]
            }
            
            self.client.schema.create_class(class_schema)
            
            log_vector_db_operation(
                operation="create_collection",
                provider="weaviate",
                collection=self.class_name,
                count=0
            )
            
        except Exception as e:
            raise VectorDBError(f"Failed to create Weaviate class: {str(e)}")
    
    async def delete_collection(self) -> None:
        """Delete Weaviate class."""
        try:
            self.client.schema.delete_class(self.class_name)
            
        except Exception as e:
            raise VectorDBError(f"Failed to delete Weaviate class: {str(e)}")
    
    async def upsert_chunks(self, chunks: List[Chunk]) -> None:
        """Insert chunks into Weaviate."""
        start_time = time.time()
        
        try:
            with self.client.batch as batch:
                batch.batch_size = 100
                
                for chunk in chunks:
                    properties = {
                        "content": chunk.content,
                        "document_id": str(chunk.document_id),
                        "chunk_index": chunk.chunk_index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "app": chunk.metadata.app,
                        "version": chunk.metadata.version,
                        "audience": chunk.metadata.audience,
                        "department": chunk.metadata.department or "",
                        "sensitivity": chunk.metadata.sensitivity,
                        "tags": chunk.metadata.tags
                    }
                    
                    batch.add_data_object(
                        data_object=properties,
                        class_name=self.class_name,
                        uuid=str(chunk.id),
                        vector=chunk.embedding
                    )
            
            duration = time.time() - start_time
            
            log_vector_db_operation(
                operation="upsert",
                provider="weaviate",
                collection=self.class_name,
                count=len(chunks),
                duration=duration
            )
            
        except Exception as e:
            log_vector_db_operation(
                operation="upsert",
                provider="weaviate",
                collection=self.class_name,
                count=len(chunks),
                error=str(e)
            )
            raise VectorDBError(f"Failed to upsert chunks to Weaviate: {str(e)}")
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar vectors in Weaviate."""
        start_time = time.time()
        
        try:
            # Build where filter
            where_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        conditions.append({
                            "path": [key],
                            "operator": "ContainsAny",
                            "valueTextArray": value
                        })
                    else:
                        conditions.append({
                            "path": [key],
                            "operator": "Equal",
                            "valueString": str(value)
                        })
                
                if len(conditions) == 1:
                    where_filter = conditions[0]
                else:
                    where_filter = {
                        "operator": "And",
                        "operands": conditions
                    }
            
            # Execute search
            result = (
                self.client.query
                .get(self.class_name, [
                    "content", "document_id", "chunk_index", 
                    "start_char", "end_char", "app", "version", 
                    "audience", "department", "sensitivity", "tags"
                ])
                .with_near_vector({"vector": query_embedding})
                .with_limit(top_k)
                .with_additional(["distance", "id"])
            )
            
            if where_filter:
                result = result.with_where(where_filter)
            
            search_results = result.do()
            
            # Parse results
            chunks_with_scores = []
            
            if "data" in search_results and "Get" in search_results["data"]:
                items = search_results["data"]["Get"].get(self.class_name, [])
                
                for item in items:
                    # Reconstruct chunk
                    from ..models.base import DocumentMetadata
                    
                    metadata = DocumentMetadata(
                        app=item.get("app", ""),
                        version=item.get("version", ""),
                        audience=item.get("audience", ""),
                        last_reviewed=None,  # Not stored in this example
                        tags=item.get("tags", []),
                        department=item.get("department"),
                        sensitivity=item.get("sensitivity", "internal")
                    )
                    
                    chunk = Chunk(
                        id=item["_additional"]["id"],
                        document_id=item["document_id"],
                        content=item["content"],
                        chunk_index=item["chunk_index"],
                        start_char=item["start_char"],
                        end_char=item["end_char"],
                        metadata=metadata
                    )
                    
                    # Calculate similarity score (1 - distance)
                    distance = item["_additional"]["distance"]
                    similarity = 1.0 - distance
                    
                    chunks_with_scores.append((chunk, similarity))
            
            duration = time.time() - start_time
            
            log_vector_db_operation(
                operation="search",
                provider="weaviate",
                collection=self.class_name,
                count=len(chunks_with_scores),
                duration=duration
            )
            
            return chunks_with_scores
            
        except Exception as e:
            log_vector_db_operation(
                operation="search",
                provider="weaviate",
                collection=self.class_name,
                count=0,
                error=str(e)
            )
            raise VectorDBError(f"Failed to search Weaviate: {str(e)}")
    
    async def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunks by IDs from Weaviate."""
        try:
            for chunk_id in chunk_ids:
                self.client.data_object.delete(
                    uuid=chunk_id,
                    class_name=self.class_name
                )
            
        except Exception as e:
            raise VectorDBError(f"Failed to delete chunks from Weaviate: {str(e)}")
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get Weaviate collection statistics."""
        try:
            # Get object count
            result = (
                self.client.query
                .aggregate(self.class_name)
                .with_meta_count()
                .do()
            )
            
            count = 0
            if "data" in result and "Aggregate" in result["data"]:
                aggregate_data = result["data"]["Aggregate"].get(self.class_name, [])
                if aggregate_data and "meta" in aggregate_data[0]:
                    count = aggregate_data[0]["meta"]["count"]
            
            return {
                "total_vectors": count,
                "collection_name": self.class_name,
                "provider": "weaviate"
            }
            
        except Exception as e:
            raise VectorDBError(f"Failed to get Weaviate stats: {str(e)}")


class PineconeVectorDB(BaseVectorDB):
    """Pinecone vector database implementation."""
    
    def __init__(self, config: VectorDBConfig):
        super().__init__(config)
        self.index_name = config.config.get("index_name", "rag-documents")
        self.index = None
    
    async def connect(self) -> None:
        """Connect to Pinecone."""
        try:
            pinecone.init(
                api_key=self.config.config["api_key"],
                environment=self.config.connection_string
            )
            
            # Connect to index
            if self.index_name in pinecone.list_indexes():
                self.index = pinecone.Index(self.index_name)
            else:
                raise VectorDBError(f"Pinecone index '{self.index_name}' does not exist")
            
        except Exception as e:
            raise VectorDBError(f"Failed to connect to Pinecone: {str(e)}")
    
    async def disconnect(self) -> None:
        """Disconnect from Pinecone."""
        # Pinecone doesn't require explicit disconnection
        self.index = None
    
    async def create_collection(self, dimension: int) -> None:
        """Create Pinecone index."""
        try:
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=self.config.distance_metric
                )
            
            self.index = pinecone.Index(self.index_name)
            
        except Exception as e:
            raise VectorDBError(f"Failed to create Pinecone index: {str(e)}")
    
    async def delete_collection(self) -> None:
        """Delete Pinecone index."""
        try:
            pinecone.delete_index(self.index_name)
            
        except Exception as e:
            raise VectorDBError(f"Failed to delete Pinecone index: {str(e)}")
    
    async def upsert_chunks(self, chunks: List[Chunk]) -> None:
        """Insert chunks into Pinecone."""
        start_time = time.time()
        
        try:
            vectors = []
            
            for chunk in chunks:
                metadata = {
                    "content": chunk.content,
                    "document_id": str(chunk.document_id),
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "app": chunk.metadata.app,
                    "version": chunk.metadata.version,
                    "audience": chunk.metadata.audience,
                    "department": chunk.metadata.department or "",
                    "sensitivity": chunk.metadata.sensitivity,
                    "tags": ",".join(chunk.metadata.tags)
                }
                
                vectors.append((
                    str(chunk.id),
                    chunk.embedding,
                    metadata
                ))
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            duration = time.time() - start_time
            
            log_vector_db_operation(
                operation="upsert",
                provider="pinecone",
                collection=self.index_name,
                count=len(chunks),
                duration=duration
            )
            
        except Exception as e:
            log_vector_db_operation(
                operation="upsert",
                provider="pinecone",
                collection=self.index_name,
                count=len(chunks),
                error=str(e)
            )
            raise VectorDBError(f"Failed to upsert chunks to Pinecone: {str(e)}")
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar vectors in Pinecone."""
        start_time = time.time()
        
        try:
            # Build metadata filter
            metadata_filter = {}
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        metadata_filter[key] = {"$in": value}
                    else:
                        metadata_filter[key] = {"$eq": str(value)}
            
            # Execute search
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=metadata_filter if metadata_filter else None
            )
            
            # Parse results
            chunks_with_scores = []
            
            for match in query_response.matches:
                metadata_dict = match.metadata
                
                # Reconstruct chunk
                from ..models.base import DocumentMetadata
                
                metadata = DocumentMetadata(
                    app=metadata_dict.get("app", ""),
                    version=metadata_dict.get("version", ""),
                    audience=metadata_dict.get("audience", ""),
                    last_reviewed=None,
                    tags=metadata_dict.get("tags", "").split(",") if metadata_dict.get("tags") else [],
                    department=metadata_dict.get("department"),
                    sensitivity=metadata_dict.get("sensitivity", "internal")
                )
                
                chunk = Chunk(
                    id=match.id,
                    document_id=metadata_dict["document_id"],
                    content=metadata_dict["content"],
                    chunk_index=int(metadata_dict["chunk_index"]),
                    start_char=int(metadata_dict["start_char"]),
                    end_char=int(metadata_dict["end_char"]),
                    metadata=metadata
                )
                
                chunks_with_scores.append((chunk, match.score))
            
            duration = time.time() - start_time
            
            log_vector_db_operation(
                operation="search",
                provider="pinecone",
                collection=self.index_name,
                count=len(chunks_with_scores),
                duration=duration
            )
            
            return chunks_with_scores
            
        except Exception as e:
            log_vector_db_operation(
                operation="search",
                provider="pinecone",
                collection=self.index_name,
                count=0,
                error=str(e)
            )
            raise VectorDBError(f"Failed to search Pinecone: {str(e)}")
    
    async def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunks by IDs from Pinecone."""
        try:
            self.index.delete(ids=chunk_ids)
            
        except Exception as e:
            raise VectorDBError(f"Failed to delete chunks from Pinecone: {str(e)}")
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        try:
            stats = self.index.describe_index_stats()
            
            return {
                "total_vectors": stats.total_vector_count,
                "collection_name": self.index_name,
                "provider": "pinecone",
                "dimension": stats.dimension
            }
            
        except Exception as e:
            raise VectorDBError(f"Failed to get Pinecone stats: {str(e)}")


# Factory for creating vector database instances
class VectorDBFactory:
    """Factory for creating vector database instances."""
    
    _providers = {
        "weaviate": WeaviateVectorDB,
        "pinecone": PineconeVectorDB,
        # Add other providers as needed
    }
    
    @classmethod
    def create_vector_db(cls, config: VectorDBConfig) -> BaseVectorDB:
        """Create vector database instance."""
        provider_class = cls._providers.get(config.provider.value)
        
        if not provider_class:
            raise VectorDBError(f"Unknown vector database provider: {config.provider}")
        
        return provider_class(config)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available providers."""
        return list(cls._providers.keys())