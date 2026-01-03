"""
Multi-Object Tracking API
FastAPI application for video object detection and tracking using YOLOv8 + DeepSORT.
"""
import asyncio
import logging
import shutil
import subprocess
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from utils.config import (
    VIDEO_UPLOAD_FOLDER, 
    VIDEO_PROCESSED_FOLDER, 
    DEVICE,
    MAX_FILE_SIZE_MB,
    is_valid_video_extension
)
from utils.processing import process_video
from utils.file_utils import (
    cleanup_old_files, 
    get_file_size_mb, 
    safe_delete,
    ensure_storage_space
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown tasks."""
    # Startup
    logger.info(f"Starting Multi-Object Tracking API on device: {DEVICE}")
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Shutdown
    cleanup_task.cancel()
    logger.info("Shutting down Multi-Object Tracking API")


# Initialize FastAPI with lifespan handler
app = FastAPI(
    title="Multi-Object Tracking API",
    description="Upload videos for AI-powered object detection and tracking",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


async def periodic_cleanup():
    """Background task to periodically clean up old files."""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            deleted = await cleanup_old_files(max_age_hours=24)
            if deleted > 0:
                logger.info(f"Periodic cleanup: removed {deleted} old files")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")


async def run_ffmpeg_async(command: str) -> tuple[bool, str]:
    """Run FFmpeg command asynchronously."""
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        success = process.returncode == 0
        return success, stderr.decode() if stderr else ""
    except Exception as e:
        return False, str(e)


async def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extracts audio from the original video asynchronously."""
    command = f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}" -y -loglevel error'
    success, error = await run_ffmpeg_async(command)
    if not success:
        logger.warning(f"Audio extraction warning: {error}")
    return success


async def merge_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    """Merges audio back into processed video asynchronously."""
    command = (
        f'ffmpeg -i "{video_path}" -i "{audio_path}" '
        f'-c:v libx264 -crf 23 -preset fast -c:a aac -strict experimental '
        f'"{output_path}" -y -loglevel error'
    )
    success, error = await run_ffmpeg_async(command)
    if not success:
        logger.error(f"Audio merge error: {error}")
    return success


async def check_video_validity(video_path: Path) -> bool:
    """Check if video file contains a valid video stream."""
    command = f'ffprobe -i "{video_path}" -show_streams -select_streams v -loglevel error'
    success, _ = await run_ffmpeg_async(command)
    return video_path.exists() and video_path.stat().st_size > 0


async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    """Save uploaded file asynchronously with chunked writing."""
    async with aiofiles.open(destination, 'wb') as out_file:
        # Read and write in chunks for memory efficiency
        chunk_size = 1024 * 1024  # 1MB chunks
        while content := await upload_file.read(chunk_size):
            await out_file.write(content)


def cleanup_processing_files(
    video_path: Path, 
    audio_path: Path, 
    processed_path: Path
) -> None:
    """Background task to clean up intermediate processing files."""
    for path in [video_path, audio_path, processed_path]:
        safe_delete(path)
    logger.info(f"Cleaned up processing files for job")


@app.post("/track/")
async def track_objects(
    request: Request, 
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a video and process it for multi-object tracking.
    
    - Validates file format and size
    - Extracts and preserves audio
    - Processes video with YOLOv8 + DeepSORT
    - Returns processed video with tracking overlays
    """
    # Validate file extension
    if not file.filename or not is_valid_video_extension(file.filename):
        raise HTTPException(
            status_code=400, 
            detail="Invalid video format. Supported: MP4, AVI, MOV, MKV, WebM"
        )
    
    # Generate unique job ID and filenames
    job_id = str(uuid.uuid4())[:8]
    safe_filename = f"{job_id}_{Path(file.filename).stem}"
    
    video_path = VIDEO_UPLOAD_FOLDER / f"{safe_filename}{Path(file.filename).suffix}"
    audio_path = VIDEO_UPLOAD_FOLDER / f"{safe_filename}.mp3"
    processed_video_path = VIDEO_PROCESSED_FOLDER / f"{safe_filename}_processed.mp4"
    final_output_path = VIDEO_PROCESSED_FOLDER / f"{safe_filename}_final.mp4"
    
    try:
        # Save uploaded file
        await save_upload_file(file, video_path)
        
        # Validate file size
        file_size_mb = get_file_size_mb(video_path)
        if file_size_mb > MAX_FILE_SIZE_MB:
            safe_delete(video_path)
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
            )
        
        logger.info(f"Processing job {job_id}: {file.filename} ({file_size_mb:.1f}MB)")
        
        # Check storage space
        if not await ensure_storage_space(file_size_mb * 2):
            raise HTTPException(
                status_code=507,
                detail="Insufficient storage space. Please try again later."
            )
        
        # Extract audio (non-blocking)
        has_audio = await extract_audio(video_path, audio_path)
        
        # Process video (CPU-bound, runs in thread pool)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, 
            process_video, 
            video_path, 
            processed_video_path
        )
        
        # Validate processed video
        if not await check_video_validity(processed_video_path):
            raise HTTPException(
                status_code=500, 
                detail="Video processing failed. Please try a different video."
            )
        
        # Merge audio if extracted successfully
        if has_audio and audio_path.exists():
            merge_success = await merge_audio(
                processed_video_path, 
                audio_path, 
                final_output_path
            )
            if not merge_success:
                # Fall back to video without audio
                shutil.copy(processed_video_path, final_output_path)
        else:
            # No audio to merge
            shutil.copy(processed_video_path, final_output_path)
        
        # Schedule cleanup of intermediate files
        background_tasks.add_task(
            cleanup_processing_files,
            video_path,
            audio_path,
            processed_video_path
        )
        
        logger.info(f"Job {job_id} completed successfully")
        
        final_output_url = f"/static/videos/processed/{final_output_path.name}"
        return templates.TemplateResponse(
            "result.html", 
            {"request": request, "processed_video": final_output_url}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        # Cleanup on error
        for path in [video_path, audio_path, processed_video_path, final_output_path]:
            safe_delete(path)
        raise HTTPException(
            status_code=500, 
            detail=f"Processing error: {str(e)}"
        )


@app.get("/download/{filename}")
async def download_video(filename: str):
    """Download processed video by filename."""
    # Sanitize filename to prevent path traversal
    safe_filename = Path(filename).name
    file_path = VIDEO_PROCESSED_FOLDER / safe_filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        str(file_path), 
        media_type="video/mp4", 
        filename=safe_filename
    )


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return JSONResponse({
        "status": "healthy",
        "device": DEVICE,
        "version": "2.0.0"
    })


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with upload form."""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "max_file_size_mb": MAX_FILE_SIZE_MB}
    )


# API endpoint for programmatic access
@app.post("/api/track/")
async def api_track_objects(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    API endpoint for programmatic video processing.
    Returns JSON response with download URL.
    """
    if not file.filename or not is_valid_video_extension(file.filename):
        raise HTTPException(
            status_code=400, 
            detail="Invalid video format"
        )
    
    job_id = str(uuid.uuid4())[:8]
    safe_filename = f"{job_id}_{Path(file.filename).stem}"
    
    video_path = VIDEO_UPLOAD_FOLDER / f"{safe_filename}{Path(file.filename).suffix}"
    audio_path = VIDEO_UPLOAD_FOLDER / f"{safe_filename}.mp3"
    processed_video_path = VIDEO_PROCESSED_FOLDER / f"{safe_filename}_processed.mp4"
    final_output_path = VIDEO_PROCESSED_FOLDER / f"{safe_filename}_final.mp4"
    
    try:
        await save_upload_file(file, video_path)
        
        file_size_mb = get_file_size_mb(video_path)
        if file_size_mb > MAX_FILE_SIZE_MB:
            safe_delete(video_path)
            raise HTTPException(status_code=413, detail="File too large")
        
        has_audio = await extract_audio(video_path, audio_path)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, process_video, video_path, processed_video_path)
        
        if not await check_video_validity(processed_video_path):
            raise HTTPException(status_code=500, detail="Processing failed")
        
        if has_audio and audio_path.exists():
            await merge_audio(processed_video_path, audio_path, final_output_path)
        else:
            shutil.copy(processed_video_path, final_output_path)
        
        background_tasks.add_task(
            cleanup_processing_files, video_path, audio_path, processed_video_path
        )
        
        return JSONResponse({
            "success": True,
            "job_id": job_id,
            "download_url": f"/download/{final_output_path.name}",
            "video_url": f"/static/videos/processed/{final_output_path.name}"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        for path in [video_path, audio_path, processed_video_path, final_output_path]:
            safe_delete(path)
        raise HTTPException(status_code=500, detail=str(e))
