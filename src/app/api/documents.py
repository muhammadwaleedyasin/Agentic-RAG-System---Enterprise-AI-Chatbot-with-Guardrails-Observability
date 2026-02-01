"""
Document management endpoints.
"""
import os
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse

from ...config.settings import settings
from ...core.rag_pipeline import RAGPipeline
from ...models.common import BaseResponse, PaginationParams
from ...models.documents import (
    DocumentUploadRequest, DocumentUploadResponse, DocumentProcessRequest,
    DocumentProcessResponse, DocumentSearchRequest, DocumentSearchResponse,
    DocumentListResponse, DocumentStatsResponse, BulkDocumentOperation,
    BulkDocumentResponse, DocumentMetadata, DocumentType
)
from ..deps import get_rag_pipeline

router = APIRouter()

# In-memory document storage (replace with database in production)
documents_db = {}


async def process_document_background(file_path: str, document_id: str, rag_pipeline: RAGPipeline):
    """Background task to process uploaded document."""
    try:
        await rag_pipeline.ingest_document(file_path=file_path)
        # Update document status in database
        if document_id in documents_db:
            documents_db[document_id]["status"] = "completed"
    except Exception as e:
        # Update document status with error
        if document_id in documents_db:
            documents_db[document_id]["status"] = "failed"
            documents_db[document_id]["error_message"] = str(e)


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: str = Form(None),
    auto_process: bool = Form(True),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Upload a document file for processing.
    """
    try:
        # Validate file type
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.allowed_file_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed types: {settings.allowed_file_types}"
            )
        
        # Read file content to check size
        content = await file.read()
        if len(content) > settings.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.max_file_size} bytes"
            )
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Create upload directory if it doesn't exist
        upload_dir = Path(settings.upload_path)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = upload_dir / f"{document_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Parse metadata if provided
        doc_metadata = None
        if metadata:
            import json
            try:
                metadata_dict = json.loads(metadata)
                doc_metadata = DocumentMetadata(**metadata_dict)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid metadata format: {str(e)}")
        
        # Store document info
        documents_db[document_id] = {
            "document_id": document_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "status": "processing" if auto_process else "pending",
            "metadata": doc_metadata.dict() if doc_metadata else {},
            "file_size": len(content),
            "content_type": file.content_type
        }
        
        # Process document in background if auto_process is True
        if auto_process:
            background_tasks.add_task(
                process_document_background,
                str(file_path),
                document_id,
                rag_pipeline
            )
        
        return DocumentUploadResponse(
            status="success",
            message="Document uploaded successfully",
            document_id=document_id,
            filename=file.filename,
            upload_status="processing" if auto_process else "pending"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")


@router.post("/documents/upload-text", response_model=DocumentUploadResponse)
async def upload_text_document(
    text_content: str = Form(...),
    filename: str = Form(...),
    metadata: str = Form(None),
    auto_process: bool = Form(True),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Upload text content as a document.
    """
    try:
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Parse metadata if provided
        doc_metadata = None
        if metadata:
            import json
            try:
                metadata_dict = json.loads(metadata)
                doc_metadata = DocumentMetadata(**metadata_dict)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid metadata format: {str(e)}")
        
        # Store document info
        documents_db[document_id] = {
            "document_id": document_id,
            "filename": filename,
            "content": text_content,
            "status": "processing" if auto_process else "pending",
            "metadata": doc_metadata.dict() if doc_metadata else {},
            "file_size": len(text_content.encode('utf-8')),
            "content_type": "text/plain"
        }
        
        # Process document in background if auto_process is True
        if auto_process:
            async def process_text_background():
                try:
                    await rag_pipeline.ingest_document(
                        text_content=text_content,
                        filename=filename,
                        metadata=doc_metadata.dict() if doc_metadata else None
                    )
                    documents_db[document_id]["status"] = "completed"
                except Exception as e:
                    documents_db[document_id]["status"] = "failed"
                    documents_db[document_id]["error_message"] = str(e)
            
            background_tasks.add_task(process_text_background)
        
        return DocumentUploadResponse(
            status="success",
            message="Text document uploaded successfully",
            document_id=document_id,
            filename=filename,
            upload_status="processing" if auto_process else "pending"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text document upload failed: {str(e)}")


@router.post("/documents/{document_id}/process", response_model=DocumentProcessResponse)
async def process_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    request: DocumentProcessRequest = None,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Process a previously uploaded document.
    """
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_info = documents_db[document_id]
    
    try:
        # Update status
        documents_db[document_id]["status"] = "processing"
        
        # Process document in background
        if "file_path" in doc_info:
            background_tasks.add_task(
                process_document_background,
                doc_info["file_path"],
                document_id,
                rag_pipeline
            )
        elif "content" in doc_info:
            async def process_text():
                try:
                    await rag_pipeline.ingest_document(
                        text_content=doc_info["content"],
                        filename=doc_info["filename"],
                        metadata=doc_info.get("metadata")
                    )
                    documents_db[document_id]["status"] = "completed"
                except Exception as e:
                    documents_db[document_id]["status"] = "failed"
                    documents_db[document_id]["error_message"] = str(e)
            
            background_tasks.add_task(process_text)
        else:
            raise HTTPException(status_code=400, detail="Document has no content to process")
        
        return DocumentProcessResponse(
            status="success",
            message="Document processing started",
            document_id=document_id,
            process_status="processing"
        )
        
    except Exception as e:
        documents_db[document_id]["status"] = "failed"
        documents_db[document_id]["error_message"] = str(e)
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@router.post("/documents/search", response_model=DocumentSearchResponse)
async def search_documents(
    request: DocumentSearchRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Search for documents using semantic similarity.
    """
    try:
        # Build filters
        filters = {}
        if request.document_ids:
            filters["document_ids"] = request.document_ids
        if request.tags:
            filters["tags"] = request.tags
        if request.document_types:
            filters["document_types"] = [dt.value for dt in request.document_types]
        
        # Search documents
        search_results = await rag_pipeline.search_documents(
            query=request.query,
            top_k=request.top_k,
            filters=filters if filters else None
        )
        
        # Format results
        formatted_results = []
        for result in search_results:
            if result["similarity_score"] >= request.similarity_threshold:
                search_result = {
                    "document_id": result["metadata"]["document_id"],
                    "chunk_id": result["chunk_id"],
                    "content": result["content"] if request.include_content else result["content"][:200] + "...",
                    "similarity_score": result["similarity_score"],
                    "metadata": result["metadata"],
                    "document_metadata": {}  # Would need to join with document metadata
                }
                formatted_results.append(search_result)
        
        return DocumentSearchResponse(
            status="success",
            message="Document search completed",
            query=request.query,
            results=formatted_results,
            total_results=len(formatted_results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document search failed: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = 1,
    size: int = 10,
    status: str = None,
    document_type: str = None
):
    """
    List all documents with pagination and filtering.
    """
    try:
        # Apply filters
        filtered_docs = []
        for doc_info in documents_db.values():
            if status and doc_info.get("status") != status:
                continue
            if document_type and doc_info.get("document_type") != document_type:
                continue
            filtered_docs.append(doc_info)
        
        # Apply pagination
        start = (page - 1) * size
        end = start + size
        paginated_docs = filtered_docs[start:end]
        
        return DocumentListResponse(
            items=paginated_docs,
            total=len(filtered_docs),
            page=page,
            size=size,
            pages=(len(filtered_docs) + size - 1) // size
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """
    Get a specific document by ID.
    """
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return documents_db[document_id]


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Delete a document and all its chunks.
    """
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Delete from vector store
        await rag_pipeline.delete_document(document_id)
        
        # Delete file if it exists
        doc_info = documents_db[document_id]
        if "file_path" in doc_info:
            file_path = Path(doc_info["file_path"])
            if file_path.exists():
                file_path.unlink()
        
        # Remove from database
        del documents_db[document_id]
        
        return BaseResponse(
            status="success",
            message="Document deleted successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document deletion failed: {str(e)}")


@router.get("/documents/stats", response_model=DocumentStatsResponse)
async def get_document_stats(rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """
    Get document statistics.
    """
    try:
        # Get stats from vector store
        vector_stats = await rag_pipeline.get_stats()
        
        # Count documents by status and type
        status_counts = {}
        type_counts = {}
        total_size = 0
        
        for doc_info in documents_db.values():
            status = doc_info.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            
            doc_type = doc_info.get("content_type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            total_size += doc_info.get("file_size", 0)
        
        return DocumentStatsResponse(
            status="success",
            message="Document statistics retrieved",
            total_documents=len(documents_db),
            total_chunks=vector_stats.get("vector_store", {}).get("total_chunks", 0),
            documents_by_type=type_counts,
            documents_by_status=status_counts,
            total_size_bytes=total_size,
            average_chunk_size=0  # Would need to calculate from vector store
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document stats: {str(e)}")


@router.post("/documents/bulk", response_model=BulkDocumentResponse)
async def bulk_document_operation(
    operation: BulkDocumentOperation,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Perform bulk operations on multiple documents.
    """
    try:
        successful = 0
        failed = 0
        failed_documents = []
        errors = []
        
        for document_id in operation.document_ids:
            try:
                if operation.operation == "delete":
                    await rag_pipeline.delete_document(document_id)
                    if document_id in documents_db:
                        del documents_db[document_id]
                    successful += 1
                elif operation.operation == "reprocess":
                    if document_id in documents_db:
                        doc_info = documents_db[document_id]
                        if "file_path" in doc_info:
                            await rag_pipeline.ingest_document(file_path=doc_info["file_path"])
                        elif "content" in doc_info:
                            await rag_pipeline.ingest_document(
                                text_content=doc_info["content"],
                                filename=doc_info["filename"]
                            )
                        successful += 1
                    else:
                        failed += 1
                        failed_documents.append(document_id)
                        errors.append(f"Document {document_id} not found")
                else:
                    failed += 1
                    failed_documents.append(document_id)
                    errors.append(f"Unsupported operation: {operation.operation}")
                    
            except Exception as e:
                failed += 1
                failed_documents.append(document_id)
                errors.append(f"Document {document_id}: {str(e)}")
        
        return BulkDocumentResponse(
            status="success" if failed == 0 else "partial",
            message=f"Bulk operation completed: {successful} successful, {failed} failed",
            operation=operation.operation,
            total_documents=len(operation.document_ids),
            successful=successful,
            failed=failed,
            failed_documents=failed_documents if failed_documents else None,
            errors=errors if errors else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk operation failed: {str(e)}")
