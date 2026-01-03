from pathlib import Path
from ultralytics import YOLO
import torch
import os

# Define base directory first
BASE_DIR = Path(__file__).resolve().parent.parent

# Environment configuration
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
MODEL_NAME = os.getenv("YOLO_MODEL", "models/yolo11n.pt")  # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))

# Tracker configuration - using ultralytics built-in trackers
# Options: "botsort" (recommended - best accuracy) or "bytetrack" (fastest)
TRACKER_TYPE = os.getenv("TRACKER_TYPE", "botsort")

# Custom tracker config path (use project config files)
TRACKER_CONFIG_DIR = BASE_DIR / "config"
TRACKER_CONFIG_PATH = TRACKER_CONFIG_DIR / f"{TRACKER_TYPE}.yaml"

# Verify tracker config exists, fallback to built-in if not
if not TRACKER_CONFIG_PATH.exists():
    TRACKER_CONFIG_PATH = f"{TRACKER_TYPE}.yaml"  # Use ultralytics built-in

# Define static folders
VIDEO_UPLOAD_FOLDER = BASE_DIR / "static/videos/uploads"
VIDEO_PROCESSED_FOLDER = BASE_DIR / "static/videos/processed"

# Create directories if they don't exist
for folder in [VIDEO_UPLOAD_FOLDER, VIDEO_PROCESSED_FOLDER, TRACKER_CONFIG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Detect available device (GPU/CPU)
def get_device() -> str:
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"

DEVICE = get_device()

# Load YOLO model with optimizations
def load_model() -> YOLO:
    """Load YOLO model with device optimization."""
    model = YOLO(MODEL_NAME)
    model.to(DEVICE)
    return model

model = load_model()

# Supported video formats
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

def is_valid_video_extension(filename: str) -> bool:
    """Check if file has a valid video extension."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS
