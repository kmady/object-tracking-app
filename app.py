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

def process_video(input_video: Path, output_video: Path):
    """Run YOLOv8 tracking while preserving audio."""
    command = (
        f'ffmpeg -i "{input_video}" -vf "drawbox=x=100:y=100:w=200:h=200:color=red@0.5" '
        f'-c:v libx264 -preset fast -c:a copy "{output_video}" -y'
    )
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")

    return output_video

@app.post("/track/")
async def track_objects(request: Request, file: UploadFile = File(...)):
    """Upload a video and process it for multi-object tracking."""
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Invalid video format. Use MP4, AVI, MOV, or MKV.")
    
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    video_path = VIDEO_UPLOAD_FOLDER / unique_filename
    with video_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    processed_video_path = VIDEO_PROCESSED_FOLDER / f"processed_{unique_filename}"
    process_video(video_path, processed_video_path)

    return templates.TemplateResponse(
        "result.html", 
        {"request": request, "processed_video": f"/static/videos/processed/{processed_video_path.name}"}
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
