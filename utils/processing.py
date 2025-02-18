import cv2
from utils.config import VIDEO_PROCESSED_FOLDER, model
from pathlib import Path  # Import Path explicitly

def process_video(video_path: Path, output_path: Path):  # Take output_path as an argument
    """Process a video frame-by-frame using YOLOv8 and save the output."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Error opening video file.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or *'XVID' if you prefer AVI
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use the provided output_path here!
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))  # Corrected!
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame)
    
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = float(box.conf[0])  # Confidence score
                cls = int(box.cls[0])  # Class ID
                detections.append(([x1, y1, x2, y2], conf, cls))  # DeepSORT format

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()

            # Draw bounding box and ID
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)

        cap.release()
        out.release()

    return output_path  # Return the correct output path