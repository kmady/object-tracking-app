from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import subprocess
import uuid
from starlette.requests import Request

from utils.config import VIDEO_UPLOAD_FOLDER, VIDEO_PROCESSED_FOLDER
from utils.processing import process_video


# Initialize FastAPI
app = FastAPI(title="Multi-Object Tracking API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Mount static files to serve videos and CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

def extract_audio(video_path: Path, audio_path: Path):
    """Extracts audio from the original video."""
    command = f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}" -y'
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def merge_audio(video_path: Path, audio_path: Path, output_path: Path):
    """Merges the extracted audio back into the processed video."""
    command = f'ffmpeg -i "{video_path}" -i "{audio_path}" -c:v libx264 -crf 23 -preset fast -c:a aac -strict experimental "{output_path}" -y'
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
def check_video_validity(video_path: Path):
    """Check if a video file contains a valid video stream."""
    command = f'ffprobe -i "{video_path}" -show_streams -select_streams v -loglevel error'
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return bool(result.stdout)  # True if video stream exists


@app.post("/track/")
async def track_objects(request: Request, file: UploadFile = File(...)):
    """Upload a video and process it for multi-object tracking."""
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Invalid video format. Use MP4, AVI, MOV, or MKV.")
    
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    video_path = VIDEO_UPLOAD_FOLDER / unique_filename
    with video_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Paths for processing
    audio_path = VIDEO_UPLOAD_FOLDER / f"{video_path.stem}.mp3"
    processed_video_path = VIDEO_PROCESSED_FOLDER / f"{video_path.stem}_processed.mp4"
    final_output_path = VIDEO_PROCESSED_FOLDER / f"{video_path.stem}_final.mp4"

    # Extract audio
    extract_audio(video_path, audio_path)

    # Process video (object detection, etc.)
    process_video(video_path, processed_video_path)
    
    if not check_video_validity(processed_video_path):
        raise HTTPException(status_code=500, detail="Error: Processed video has no video stream.")

    # Merge audio back into processed video
    merge_audio(processed_video_path, audio_path, final_output_path)

    final_output_url = f"/static/videos/processed/{final_output_path.name}"
    return templates.TemplateResponse(
        "result.html", {"request": request, "processed_video": final_output_url}
    )
    

@app.get("/download/{filename}")
async def download_video(filename: str):
    """Download processed video."""
    file_path = VIDEO_PROCESSED_FOLDER / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(str(file_path), media_type="video/mp4", filename=filename)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with an upload form."""
    return templates.TemplateResponse("index.html", {"request": request})
