"""
Advanced ingestion endpoints with batch processing capabilities.
"""
import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field
import zipfile
import tempfile
import shutil

from ....security.access_control import User, Permission
from ....core.rag_pipeline import RAGPipeline
from ....config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class BatchIngestRequest(BaseModel):
    """Request model for batch ingestion."""
    auto_process: bool = True
    metadata_template: Optional[Dict[str, Any]] = None
    chunk_settings: Optional[Dict[str, Any]] = None


class BatchIngestResponse(BaseModel):
    """Response model for batch ingestion."""
    status: str
    message: str
    batch_id: str
    total_files: int
    processing_status: str
    failed_files: Optional[List[str]] = None


class IngestJobStatus(BaseModel):
    """Status model for ingestion jobs."""
    job_id: str
    status: str  # pending, processing, completed, failed
    total_files: int
    processed_files: int
    failed_files: int
    error_messages: Optional[List[str]] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# In-memory job tracking (replace with database in production)
ingestion_jobs: Dict[str, Dict[str, Any]] = {}


# Dependencies
from ...deps import get_current_user, get_rag_pipeline


# Permission check for write operations
async def require_write_permission(current_user: User = Depends(get_current_user)):
    """Require write permission for document operations."""
    if not current_user.can_perform_action(Permission.WRITE_DOCUMENTS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Document write permission required"
        )
    return current_user


@router.post("/batch/upload", response_model=BatchIngestResponse)
async def batch_upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    metadata: str = Form("{}"),
    auto_process: bool = Form(True),
    current_user: User = Depends(require_write_permission),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Upload multiple files for batch processing.
    """
    try:
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        
        # Parse metadata
        import json
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid metadata JSON format"
            )
        
        # Validate files
        valid_files = []
        invalid_files = []
        
        for file in files:
            if not file.filename:
                continue
                
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in settings.allowed_file_types:
                invalid_files.append(f"{file.filename}: Unsupported file type")
                continue
                
            if file.size and file.size > settings.max_file_size:
                invalid_files.append(f"{file.filename}: File too large")
                continue
                
            valid_files.append(file)
        
        if not valid_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid files to process"
            )
        
        # Create job entry
        job_info = {
            "batch_id": batch_id,
            "status": "processing" if auto_process else "pending",
            "total_files": len(valid_files),
            "processed_files": 0,
            "failed_files": 0,
            "error_messages": invalid_files,
            "started_at": None,
            "completed_at": None,
            "user_id": current_user.user_id,
            "file_paths": []
        }
        
        ingestion_jobs[batch_id] = job_info
        
        # Save files to disk
        upload_dir = Path(settings.upload_path) / "batch" / batch_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for file in valid_files:
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            saved_files.append({
                "filename": file.filename,
                "file_path": str(file_path),
                "size": len(content)
            })
        
        job_info["file_paths"] = saved_files
        
        # Process in background if auto_process
        if auto_process:
            background_tasks.add_task(
                process_batch_job,
                batch_id,
                metadata_dict,
                rag_pipeline
            )
        
        return BatchIngestResponse(
            status="success",
            message=f"Batch upload completed. {len(valid_files)} files uploaded.",
            batch_id=batch_id,
            total_files=len(valid_files),
            processing_status="processing" if auto_process else "pending",
            failed_files=invalid_files if invalid_files else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch upload failed: {str(e)}"
        )


@router.post("/batch/archive", response_model=BatchIngestResponse)
async def upload_archive(
    background_tasks: BackgroundTasks,
    archive: UploadFile = File(...),
    metadata: str = Form("{}"),
    auto_process: bool = Form(True),
    current_user: User = Depends(require_write_permission),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Upload and extract archive (ZIP) for batch processing.
    """
    try:
        # Validate archive file
        if not archive.filename or not archive.filename.endswith('.zip'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only ZIP archives are supported"
            )
        
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        
        # Parse metadata
        import json
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid metadata JSON format"
            )
        
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded archive
            archive_path = Path(temp_dir) / archive.filename
            with open(archive_path, "wb") as buffer:
                content = await archive.read()
                buffer.write(content)
            
            # Extract archive
            extract_dir = Path(temp_dir) / "extracted"
            extract_dir.mkdir()
            
            try:
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            except zipfile.BadZipFile:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or corrupted ZIP file"
                )
            
            # Find all files in extracted directory
            extracted_files = []
            invalid_files = []
            
            for file_path in extract_dir.rglob('*'):
                if file_path.is_file():
                    file_extension = file_path.suffix.lower()
                    if file_extension in settings.allowed_file_types:
                        if file_path.stat().st_size <= settings.max_file_size:
                            extracted_files.append(file_path)
                        else:
                            invalid_files.append(f"{file_path.name}: File too large")
                    else:
                        invalid_files.append(f"{file_path.name}: Unsupported file type")
            
            if not extracted_files:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid files found in archive"
                )
            
            # Move files to permanent location
            upload_dir = Path(settings.upload_path) / "batch" / batch_id
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            saved_files = []
            for file_path in extracted_files:
                dest_path = upload_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                
                saved_files.append({
                    "filename": file_path.name,
                    "file_path": str(dest_path),
                    "size": dest_path.stat().st_size
                })
        
        # Create job entry
        job_info = {
            "batch_id": batch_id,
            "status": "processing" if auto_process else "pending",
            "total_files": len(saved_files),
            "processed_files": 0,
            "failed_files": 0,
            "error_messages": invalid_files,
            "started_at": None,
            "completed_at": None,
            "user_id": current_user.user_id,
            "file_paths": saved_files
        }
        
        ingestion_jobs[batch_id] = job_info
        
        # Process in background if auto_process
        if auto_process:
            background_tasks.add_task(
                process_batch_job,
                batch_id,
                metadata_dict,
                rag_pipeline
            )
        
        return BatchIngestResponse(
            status="success",
            message=f"Archive extracted successfully. {len(saved_files)} files ready for processing.",
            batch_id=batch_id,
            total_files=len(saved_files),
            processing_status="processing" if auto_process else "pending",
            failed_files=invalid_files if invalid_files else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Archive upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Archive upload failed: {str(e)}"
        )


async def process_batch_job(batch_id: str, metadata: Dict[str, Any], rag_pipeline: RAGPipeline):
    """Background task to process batch ingestion job."""
    if batch_id not in ingestion_jobs:
        return
    
    job_info = ingestion_jobs[batch_id]
    job_info["status"] = "processing"
    job_info["started_at"] = str(asyncio.get_event_loop().time())
    
    try:
        for file_info in job_info["file_paths"]:
            try:
                # Add file-specific metadata
                file_metadata = metadata.copy()
                file_metadata.update({
                    "batch_id": batch_id,
                    "original_filename": file_info["filename"],
                    "file_size": file_info["size"]
                })
                
                # Ingest document
                await rag_pipeline.ingest_document(
                    file_path=file_info["file_path"],
                    metadata=file_metadata
                )
                
                job_info["processed_files"] += 1
                logger.info(f"Processed file {file_info['filename']} in batch {batch_id}")
                
            except Exception as e:
                job_info["failed_files"] += 1
                if "error_messages" not in job_info:
                    job_info["error_messages"] = []
                job_info["error_messages"].append(f"{file_info['filename']}: {str(e)}")
                logger.error(f"Failed to process file {file_info['filename']} in batch {batch_id}: {str(e)}")
        
        job_info["status"] = "completed"
        job_info["completed_at"] = str(asyncio.get_event_loop().time())
        
        logger.info(f"Batch job {batch_id} completed: {job_info['processed_files']} processed, {job_info['failed_files']} failed")
        
    except Exception as e:
        job_info["status"] = "failed"
        job_info["completed_at"] = str(asyncio.get_event_loop().time())
        if "error_messages" not in job_info:
            job_info["error_messages"] = []
        job_info["error_messages"].append(f"Batch processing failed: {str(e)}")
        logger.error(f"Batch job {batch_id} failed: {str(e)}")


@router.get("/batch/{batch_id}/status", response_model=IngestJobStatus)
async def get_batch_status(
    batch_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get status of a batch ingestion job."""
    if batch_id not in ingestion_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch job not found"
        )
    
    job_info = ingestion_jobs[batch_id]
    
    # Check if user has access to this job
    if (job_info.get("user_id") != current_user.user_id and 
        not current_user.can_perform_action(Permission.ADMIN_ACCESS)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return IngestJobStatus(
        job_id=batch_id,
        status=job_info["status"],
        total_files=job_info["total_files"],
        processed_files=job_info["processed_files"],
        failed_files=job_info["failed_files"],
        error_messages=job_info.get("error_messages"),
        started_at=job_info.get("started_at"),
        completed_at=job_info.get("completed_at")
    )


@router.post("/batch/{batch_id}/process")
async def process_batch(
    batch_id: str,
    background_tasks: BackgroundTasks,
    metadata: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(require_write_permission),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Manually trigger processing of a pending batch."""
    if batch_id not in ingestion_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch job not found"
        )
    
    job_info = ingestion_jobs[batch_id]
    
    # Check if user has access to this job
    if (job_info.get("user_id") != current_user.user_id and 
        not current_user.can_perform_action(Permission.ADMIN_ACCESS)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    if job_info["status"] != "pending":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch is not in pending status (current: {job_info['status']})"
        )
    
    # Start processing
    background_tasks.add_task(
        process_batch_job,
        batch_id,
        metadata or {},
        rag_pipeline
    )
    
    return {
        "status": "success",
        "message": "Batch processing started",
        "batch_id": batch_id
    }


@router.get("/batch/jobs")
async def list_batch_jobs(
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 50
):
    """List batch ingestion jobs for current user or all jobs for admins."""
    try:
        jobs = []
        
        for job_id, job_info in ingestion_jobs.items():
            # Filter jobs based on user permissions
            if (job_info.get("user_id") == current_user.user_id or 
                current_user.can_perform_action(Permission.ADMIN_ACCESS)):
                
                jobs.append({
                    "job_id": job_id,
                    "status": job_info["status"],
                    "total_files": job_info["total_files"],
                    "processed_files": job_info["processed_files"],
                    "failed_files": job_info["failed_files"],
                    "started_at": job_info.get("started_at"),
                    "completed_at": job_info.get("completed_at"),
                    "user_id": job_info.get("user_id")
                })
        
        # Apply pagination
        paginated_jobs = jobs[skip:skip + limit]
        
        return {
            "status": "success",
            "message": "Batch jobs retrieved successfully",
            "jobs": paginated_jobs,
            "total": len(jobs)
        }
        
    except Exception as e:
        logger.error(f"Error listing batch jobs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list batch jobs: {str(e)}"
        )


@router.delete("/batch/{batch_id}")
async def delete_batch_job(
    batch_id: str,
    current_user: User = Depends(require_write_permission)
):
    """Delete a batch ingestion job and its files."""
    if batch_id not in ingestion_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch job not found"
        )
    
    job_info = ingestion_jobs[batch_id]
    
    # Check if user has access to this job
    if (job_info.get("user_id") != current_user.user_id and 
        not current_user.can_perform_action(Permission.ADMIN_ACCESS)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    try:
        # Remove files from disk
        upload_dir = Path(settings.upload_path) / "batch" / batch_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        
        # Remove job from memory
        del ingestion_jobs[batch_id]
        
        logger.info(f"Batch job {batch_id} deleted by {current_user.username}")
        
        return {
            "status": "success",
            "message": "Batch job deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting batch job {batch_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete batch job: {str(e)}"
        )
