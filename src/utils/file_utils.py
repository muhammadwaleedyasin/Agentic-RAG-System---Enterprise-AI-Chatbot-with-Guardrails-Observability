"""
File utility functions.
"""
import os
from pathlib import Path
from typing import List


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure
        
    Returns:
        Path object for the directory
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_extension(filename: str) -> str:
    """
    Get the file extension from a filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension including the dot (e.g., '.txt')
    """
    return Path(filename).suffix.lower()


def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
    """
    Validate if a file type is allowed.
    
    Args:
        filename: Name of the file
        allowed_types: List of allowed file extensions
        
    Returns:
        True if file type is allowed, False otherwise
    """
    extension = get_file_extension(filename)
    return extension in allowed_types


def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path)


def is_file_too_large(file_path: str, max_size: int) -> bool:
    """
    Check if a file is too large.
    
    Args:
        file_path: Path to the file
        max_size: Maximum allowed size in bytes
        
    Returns:
        True if file is too large, False otherwise
    """
    return get_file_size(file_path) > max_size