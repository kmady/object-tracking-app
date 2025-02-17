# Multi-Object Tracking API

## 📌 Overview
This project is a FastAPI-based web application for uploading videos and performing multi-object tracking. The processed videos retain their original audio, and users can download the final output.

## 🚀 Features
- Upload videos in formats: **MP4, AVI, MOV, MKV**
- Extract audio from the original video
- Process video with **object tracking**
- Merge the original audio back into the processed video
- Download the final processed video
- Friendly web interface using **HTML + Jinja2 Templates**

## 🏗️ Project Structure
```
project/
│── static/                   # Static files (CSS, processed videos)
│── templates/                # HTML templates (index.html, result.html)
│── utils/
│   │── config.py             # Configuration for file paths
│   │── processing.py         # Object tracking logic
│── app.py                    # FastAPI main application
│── Dockerfile                # Project dockerfile
│── requirements.txtx         # Project list of required packages
│── README.md                 # Project documentation
```

## 🛠️ Installation
### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/your-repo/multi-object-tracking-api.git
cd multi-object-tracking-api
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
uvicorn app:app --reload
```

## 🎯 Usage
### **Uploading and Processing a Video**
1. Open `http://127.0.0.1:8000/` in a browser.
2. Click **"Choose File"** to upload a video.
3. Press **"Upload and Process"** to start tracking objects.
4. View the processed video and download the final version.

## ⚙️ API Endpoints
| Method | Endpoint       | Description |
|--------|---------------|-------------|
| `GET`  | `/`           | Home page (upload form) |
| `POST` | `/track/`     | Upload and process a video |
| `GET`  | `/download/{filename}` | Download processed video |

## 🔧 Configuration
Modify `utils/config.py` to change paths for **video uploads and processing output.**

```python
VIDEO_UPLOAD_FOLDER = Path("static/videos/uploaded")
VIDEO_PROCESSED_FOLDER = Path("static/videos/processed")
```

## 🛠 Troubleshooting
- **Processed video has no visuals?**
  - Check if `process_video()` is producing a valid output.
  - Ensure `merge_audio()` is correctly merging the video and audio.
- **Server not starting?**
  - Make sure **FastAPI** and **Uvicorn** are installed.

## 🤝 Contributions
Pull requests are welcome! Feel free to improve the UI, optimize video processing, or enhance object tracking algorithms.

## 📜 License
This project is licensed under the **MIT License**.

---
### 👨‍💻 Author: Kartoue Mady Demdah
For questions, reach out at: **kartoue@gmail.com**


