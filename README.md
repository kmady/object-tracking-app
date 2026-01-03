# Multi-Object Tracking API v2.0

## 📌 Overview
This project is an **optimized FastAPI-based web application** for uploading videos and performing multi-object tracking using **YOLOv8** for detection and **DeepSORT** for tracking. The processed videos retain their original audio, and users can download the final output.

## 🚀 Features
- **AI-Powered Detection**: YOLOv8 neural network for accurate object detection
- **Real-time Tracking**: DeepSORT algorithm maintains unique IDs across frames
- **GPU Acceleration**: Automatic CUDA/MPS detection for faster processing
- **Async Processing**: Non-blocking file uploads and FFmpeg operations
- **Background Cleanup**: Automatic cleanup of old files (24-hour retention)
- **Modern UI**: Drag-and-drop uploads with progress indicators
- **Audio Preservation**: Extracts and re-merges original audio
- **REST API**: Both web interface and JSON API endpoints
- Supports: **MP4, AVI, MOV, MKV, WebM**

## ⚡ Performance Optimizations (v2.0)
| Optimization | Description |
|-------------|-------------|
| GPU Support | Auto-detects CUDA/Apple MPS for hardware acceleration |
| Async FFmpeg | Non-blocking audio extraction and merging |
| Chunked Uploads | 1MB chunks for memory-efficient large file handling |
| Thread Pool | CPU-bound video processing runs in executor |
| Frame Skip | Optional frame skipping for faster processing |
| Background Tasks | Intermediate file cleanup runs asynchronously |
| Connection Pooling | Persistent DeepSORT tracker per video session |

## 🏗️ Project Structure
```
project/
├── app.py                    # FastAPI main application (async optimized)
├── utils/
│   ├── __init__.py          # Package exports
│   ├── config.py            # Configuration, GPU detection, model loading
│   ├── processing.py        # YOLOv8 + DeepSORT video processing
│   └── file_utils.py        # Async file cleanup utilities
├── templates/
│   ├── index.html           # Upload page with drag-and-drop
│   └── result.html          # Results page with video player
├── static/
│   ├── css/style.css        # Stylesheet
│   └── videos/              # Upload and processed video storage
├── Dockerfile               # Container configuration
├── DOCKER_README.md         # Docker documentation
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- FFmpeg (for audio processing)
- CUDA toolkit (optional, for GPU acceleration)

### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/kmady/object-tracking-app.git
cd object-tracking-app
```

### 2️⃣ **Create a Virtual Environment**
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ **Install Dependencies**
```sh
pip install -r requirements.txt
```

### 4️⃣ **Run the Application**
```sh
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 🎯 Usage

### **Web Interface**
1. Open `http://127.0.0.1:8000/` in a browser
2. Drag & drop a video or click to browse
3. Wait for processing (progress indicator shown)
4. View, download, or share the processed video

### **REST API**
```bash
# Upload and process video (returns JSON)
curl -X POST "http://localhost:8000/api/track/" \
  -F "file=@video.mp4"

# Response:
{
  "success": true,
  "job_id": "abc12345",
  "download_url": "/download/abc12345_video_final.mp4",
  "video_url": "/static/videos/processed/abc12345_video_final.mp4"
}
```

## ⚙️ API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Home page (upload form) |
| `POST` | `/track/` | Upload & process video (HTML response) |
| `POST` | `/api/track/` | Upload & process video (JSON response) |
| `GET` | `/download/{filename}` | Download processed video |
| `GET` | `/health` | Health check endpoint |

## 🔧 Configuration

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model (n/s/m/l/x) |
| `CONFIDENCE_THRESHOLD` | `0.5` | Detection confidence threshold |
| `MAX_FILE_SIZE_MB` | `500` | Maximum upload file size |
| `DEBUG` | `false` | Enable debug mode |

### Example
```sh
export YOLO_MODEL=yolov8m.pt
export CONFIDENCE_THRESHOLD=0.6
export MAX_FILE_SIZE_MB=1000
uvicorn app:app --reload
```

### YOLO Model Options
| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolov8n.pt` | 6MB | ⚡ Fastest | Good |
| `yolov8s.pt` | 22MB | Fast | Better |
| `yolov8m.pt` | 50MB | Medium | High |
| `yolov8l.pt` | 83MB | Slow | Higher |
| `yolov8x.pt` | 131MB | Slowest | Best |

## 🐳 Docker Deployment
```sh
docker build -t object-tracking-app .
docker run -p 8000:8000 -v ./static/videos:/app/static/videos object-tracking-app
```

See [DOCKER_README.md](DOCKER_README.md) for detailed Docker instructions.

## 🛠 Troubleshooting

### Video processing fails
- Check if FFmpeg is installed: `ffmpeg -version`
- Ensure video file is not corrupted
- Check logs for detailed error messages

### Slow processing
- Use GPU: ensure CUDA is installed and detected
- Use smaller YOLO model (yolov8n.pt)
- Reduce video resolution before upload

### Out of memory
- Reduce `MAX_FILE_SIZE_MB` environment variable
- Use a smaller YOLO model
- Process shorter video clips

### Health check
```sh
curl http://localhost:8000/health
# Returns: {"status": "healthy", "device": "cuda", "version": "2.0.0"}
```

## 🤝 Contributions
Pull requests are welcome! Areas for improvement:
- [ ] WebSocket progress updates during processing
- [ ] Batch video processing
- [ ] Custom object class filtering
- [ ] Video streaming output
- [ ] Multi-GPU support

## 📜 License
This project is licensed under the **MIT License**.

---
### 👨‍💻 Author: Kartoue Mady Demdah
For questions, reach out at: **kartoue@gmail.com**
