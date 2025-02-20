import cv2
from utils.config import VIDEO_PROCESSED_FOLDER, model
from pathlib import Path  # Import Path explicitly

from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=5, nn_budget=100)

def process_video(video_path: Path, output_path: Path):
    """Process a video frame-by-frame using YOLOv8 and DeepSORT."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Error opening video file.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # YOLO detection
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append(([x1, y1, x2, y2], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)  # ✅ This should be inside the loop

    cap.release()  # ✅ Now correctly placed outside the loop
    out.release()  # ✅ Now correctly placed outside the loop
    return output_path  
