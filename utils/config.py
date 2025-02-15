from pathlib import Path
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

# Define static folders
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_UPLOAD_FOLDER = BASE_DIR / "static/images/uploads"
IMAGE_PROCESSED_FOLDER = BASE_DIR / "static/images/processed"
VIDEO_UPLOAD_FOLDER = BASE_DIR / "static/videos/uploads"
VIDEO_PROCESSED_FOLDER = BASE_DIR / "static/videos/processed"

# Create directories if they don't exist
for folder in [IMAGE_UPLOAD_FOLDER, IMAGE_PROCESSED_FOLDER, VIDEO_UPLOAD_FOLDER, VIDEO_PROCESSED_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")
