"""File utility functions for cleanup and management."""
import os
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from utils.config import VIDEO_UPLOAD_FOLDER, VIDEO_PROCESSED_FOLDER

logger = logging.getLogger(__name__)


async def cleanup_old_files(
    max_age_hours: int = 24,
    upload_folder: Path = VIDEO_UPLOAD_FOLDER,
    processed_folder: Path = VIDEO_PROCESSED_FOLDER
) -> int:
    """
    Remove files older than max_age_hours from upload and processed folders.
    
    Returns:
        Number of files deleted
    """
    deleted_count = 0
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    for folder in [upload_folder, processed_folder]:
        if not folder.exists():
            continue
            
        for file_path in folder.iterdir():
            if file_path.is_file():
                try:
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
    
    return deleted_count


def cleanup_job_files(job_id: str) -> None:
    """Clean up all files associated with a specific job/upload."""
    patterns = [
        VIDEO_UPLOAD_FOLDER / f"{job_id}*",
        VIDEO_PROCESSED_FOLDER / f"{job_id}*",
    ]
    
    for folder in [VIDEO_UPLOAD_FOLDER, VIDEO_PROCESSED_FOLDER]:
        for file_path in folder.glob(f"*{job_id}*"):
            try:
                file_path.unlink()
                logger.info(f"Cleaned up: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up {file_path}: {e}")


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes."""
    return file_path.stat().st_size / (1024 * 1024)


def safe_delete(file_path: Path) -> bool:
    """Safely delete a file, returning True if successful."""
    try:
        if file_path.exists():
            file_path.unlink()
            return True
    except Exception as e:
        logger.error(f"Error deleting {file_path}: {e}")
    return False


async def get_folder_size_mb(folder: Path) -> float:
    """Get total size of folder contents in MB."""
    total_size = 0
    if folder.exists():
        for file_path in folder.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    return total_size / (1024 * 1024)


async def ensure_storage_space(
    required_mb: float,
    max_storage_mb: float = 5000,
    folder: Path = VIDEO_PROCESSED_FOLDER
) -> bool:
    """
    Ensure there's enough storage space by cleaning old files if needed.
    
    Returns:
        True if space is available, False otherwise
    """
    current_size = await get_folder_size_mb(folder)
    
    if current_size + required_mb <= max_storage_mb:
        return True
    
    # Try to clean up old files
    await cleanup_old_files(max_age_hours=1)
    
    current_size = await get_folder_size_mb(folder)
    return current_size + required_mb <= max_storage_mb
