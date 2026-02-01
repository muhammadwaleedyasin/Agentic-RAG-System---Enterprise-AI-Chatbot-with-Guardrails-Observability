"""Utility helper functions for the RAG system."""

import hashlib
import mimetypes
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from ..models.base import DocumentType


def get_file_hash(file_path: str) -> str:
    """Generate SHA-256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def get_file_type(file_path: str) -> Optional[DocumentType]:
    """Determine document type from file extension."""
    file_extension = Path(file_path).suffix.lower()
    
    type_mapping = {
        ".pdf": DocumentType.PDF,
        ".docx": DocumentType.DOCX,
        ".doc": DocumentType.DOCX,
        ".txt": DocumentType.TXT,
        ".md": DocumentType.MD,
        ".markdown": DocumentType.MD,
        ".html": DocumentType.HTML,
        ".htm": DocumentType.HTML,
        ".json": DocumentType.JSON,
    }
    
    return type_mapping.get(file_extension)


def validate_file_type(file_path: str, supported_types: List[str]) -> bool:
    """Validate if file type is supported."""
    file_extension = Path(file_path).suffix.lower()
    return file_extension in supported_types


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip('_.')
    # Ensure filename is not empty
    if not sanitized:
        sanitized = f"file_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    return sanitized


def get_mime_type(file_path: str) -> str:
    """Get MIME type of a file."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if not."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)


def is_binary_file(file_path: str) -> bool:
    """Check if file is binary."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(1024)
        return False
    except (UnicodeDecodeError, UnicodeError):
        return True


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
    # Normalize quotes
    text = re.sub(r'[""''‛‚„‟‵‶]', '"', text)
    text = re.sub(r'[''‛‚‵]', "'", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def extract_text_from_html(html_content: str) -> str:
    """Extract clean text from HTML content."""
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text and clean it
    text = soup.get_text()
    return clean_text(text)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def parse_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """Extract metadata from structured filename."""
    # Example: "user_guide_v2.1_internal_20241201.pdf"
    # Pattern: {title}_{version}_{audience}_{date}.{ext}
    
    metadata = {}
    
    # Remove extension
    name_without_ext = Path(filename).stem
    
    # Try to parse structured filename
    parts = name_without_ext.split('_')
    
    if len(parts) >= 4:
        # Try to identify version pattern
        version_pattern = r'v?\d+\.\d+(\.\d+)?'
        date_pattern = r'\d{8}'
        
        for i, part in enumerate(parts):
            if re.match(version_pattern, part):
                metadata['version'] = part
                # Assume title is everything before version
                metadata['title'] = '_'.join(parts[:i])
                # Look for audience and date after version
                remaining = parts[i+1:]
                if remaining:
                    if re.match(date_pattern, remaining[-1]):
                        metadata['date'] = remaining[-1]
                        if len(remaining) > 1:
                            metadata['audience'] = '_'.join(remaining[:-1])
                    else:
                        metadata['audience'] = '_'.join(remaining)
                break
    
    # Fallback to simple parsing
    if 'title' not in metadata:
        metadata['title'] = name_without_ext.replace('_', ' ').title()
    
    return metadata


def validate_uuid(uuid_string: str) -> bool:
    """Validate UUID string format."""
    try:
        UUID(uuid_string)
        return True
    except ValueError:
        return False


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def calculate_similarity_score(text1: str, text2: str) -> float:
    """Calculate simple similarity score between two texts."""
    # This is a basic implementation - you might want to use more sophisticated methods
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries, with later ones taking precedence."""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def safe_get_nested(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Safely get nested dictionary value."""
    try:
        for key in keys:
            data = data[key]
        return data
    except (KeyError, TypeError):
        return default


def format_citations(sources: List[Dict[str, Any]]) -> List[str]:
    """Format source citations for display."""
    citations = []
    
    for i, source in enumerate(sources, 1):
        metadata = source.get('metadata', {})
        title = metadata.get('title', 'Unknown Document')
        app = metadata.get('app', 'Unknown App')
        version = metadata.get('version', 'Unknown Version')
        
        citation = f"[{i}] {title} ({app} v{version})"
        citations.append(citation)
    
    return citations


def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Estimate token count for text (rough approximation)."""
    # Simple approximation: ~4 characters per token for English text
    # This is a rough estimate and should be replaced with proper tokenization
    return len(text) // 4


def retry_with_exponential_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple = (Exception,)
):
    """Decorator for retrying functions with exponential backoff."""
    import time
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        raise e
                    
                    time.sleep(min(delay, max_delay))
                    delay *= 2
            
        return wrapper
    return decorator