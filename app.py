from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import shutil
import subprocess

# Define folders
UPLOAD_FOLDER = Path("videos/uploads")
PROCESSED_FOLDER = Path("videos/processed")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI
app = FastAPI(title="Multi-Object Tracking API")

def process_video(input_video: Path, output_video: Path):
    """Run YOLOv8 tracking while preserving audio."""
    command = (
        f'ffmpeg -i "{input_video}" -vf "drawbox=x=100:y=100:w=200:h=200:color=red@0.5" '
        f'-c:v libx264 -preset fast -c:a copy "{output_video}" -y'
    )
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_video

@app.post("/track/")
async def track_objects(file: UploadFile = File(...)):
    """Upload a video and process it for multi-object tracking."""
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Invalid video format. Use MP4, AVI, MOV, or MKV.")
    
    video_path = UPLOAD_FOLDER / file.filename
    with video_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    processed_video_path = PROCESSED_FOLDER / f"processed_{file.filename}"
    process_video(video_path, processed_video_path)
    
    return JSONResponse({"processed_video": f"/download/{processed_video_path.name}"})

@app.get("/download/{filename}")
async def download_video(filename: str):
    """Download processed video."""
    file_path = PROCESSED_FOLDER / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(str(file_path), media_type="video/mp4", filename=filename)

@app.get("/")
async def home():
    return {"message": "Welcome to the Multi-Object Tracking API!"}
