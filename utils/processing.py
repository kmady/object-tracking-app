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

        # Draw bounding boxes
        processed_frame = results[0].plot()

        # Write frame to output video
        out.write(processed_frame)

    cap.release()
    out.release()

    return output_path  # Return the correct output path