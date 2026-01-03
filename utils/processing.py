import cv2
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

from utils.config import (
    VIDEO_PROCESSED_FOLDER, 
    model, 
    DEVICE, 
    CONFIDENCE_THRESHOLD,
    TRACKER_TYPE,
    TRACKER_CONFIG_PATH
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Video metadata container."""
    fps: int
    width: int
    height: int
    total_frames: int
    duration_seconds: float


def get_video_metadata(cap: cv2.VideoCapture) -> VideoMetadata:
    """Extract video metadata from capture object."""
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    return VideoMetadata(
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
        duration_seconds=duration
    )


def draw_tracking_box(
    frame, 
    bbox: Tuple[int, int, int, int], 
    track_id: int,
    class_name: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> None:
    """Draw bounding box and label on frame."""
    x1, y1, x2, y2 = bbox
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Prepare label
    label = f"ID:{track_id}"
    if class_name:
        label = f"{class_name} {label}"
    
    # Calculate label background
    (label_width, label_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
    )
    
    # Draw label background
    cv2.rectangle(
        frame, 
        (x1, y1 - label_height - 10), 
        (x1 + label_width + 5, y1), 
        color, 
        -1
    )
    
    # Draw label text
    cv2.putText(
        frame, 
        label, 
        (x1 + 2, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        (0, 0, 0),  # Black text on colored background
        2
    )


def process_frame_with_tracking(
    frame,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    persist: bool = True
) -> List[Tuple[Tuple[int, int, int, int], int, str]]:
    """
    Process a single frame using ultralytics built-in tracking (BoT-SORT).
    
    Args:
        frame: Input frame
        conf_threshold: Confidence threshold for detections
        persist: Whether to persist tracks between frames
    
    Returns:
        List of (bbox, track_id, class_name) tuples
    """
    # Run YOLO tracking with BoT-SORT (built into ultralytics)
    # Using model.track() instead of model() enables built-in tracking
    results = model.track(
        frame, 
        verbose=False,
        conf=conf_threshold,
        device=DEVICE,
        persist=persist,
        tracker=str(TRACKER_CONFIG_PATH)  # Custom config or built-in tracker
    )
    
    tracked_objects = []
    
    for result in results:
        # Check if tracking IDs are available
        if result.boxes.id is None:
            continue
            
        boxes = result.boxes
        for i in range(len(boxes)):
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            
            # Get track ID
            track_id = int(boxes.id[i])
            
            # Get class name
            cls = int(boxes.cls[i])
            class_name = result.names.get(cls, "object")
            
            tracked_objects.append(((x1, y1, x2, y2), track_id, class_name))
    
    return tracked_objects


def reset_tracker() -> None:
    """Reset the tracker state for a new video."""
    # Reset the model's tracker by calling track with a dummy reset
    # The tracker state is automatically managed by ultralytics
    model.predictor = None  # This resets the internal state


def process_video(
    video_path: Path, 
    output_path: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    frame_skip: int = 0
) -> Path:
    """
    Process a video frame-by-frame using YOLOv11 and BoT-SORT tracker.
    
    Args:
        video_path: Path to input video
        output_path: Path for output video
        progress_callback: Optional callback(current_frame, total_frames)
        frame_skip: Process every Nth frame (0 = process all frames)
    
    Returns:
        Path to processed video
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {video_path}")
    
    try:
        # Get video metadata
        metadata = get_video_metadata(cap)
        logger.info(
            f"Processing video: {metadata.width}x{metadata.height}, "
            f"{metadata.fps}fps, {metadata.total_frames} frames, "
            f"{metadata.duration_seconds:.1f}s"
        )
        logger.info(f"Using tracker: {TRACKER_TYPE}")
        
        # Initialize video writer with H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path), 
            fourcc, 
            metadata.fps, 
            (metadata.width, metadata.height)
        )
        
        if not out.isOpened():
            raise RuntimeError(f"Error creating output video: {output_path}")
        
        # Reset tracker for new video
        reset_tracker()
        
        frame_count = 0
        processed_count = 0
        last_tracked_objects = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Frame skipping logic
            should_process = (frame_skip == 0) or (frame_count % (frame_skip + 1) == 1)
            
            if should_process:
                # Process frame with BoT-SORT tracking
                tracked_objects = process_frame_with_tracking(frame)
                last_tracked_objects = tracked_objects
                processed_count += 1
            else:
                # Use last known tracking results for skipped frames
                tracked_objects = last_tracked_objects
            
            # Draw tracking results on frame
            for bbox, track_id, class_name in tracked_objects:
                draw_tracking_box(frame, bbox, track_id, class_name)
            
            # Write processed frame
            out.write(frame)
            
            # Progress callback
            if progress_callback and frame_count % 10 == 0:
                progress_callback(frame_count, metadata.total_frames)
        
        logger.info(
            f"Processing complete: {frame_count} frames, "
            f"{processed_count} processed (skip={frame_skip})"
        )
        
    finally:
        cap.release()
        out.release()
    
    return output_path


def process_video_async_generator(
    video_path: Path, 
    output_path: Path,
    frame_skip: int = 0
):
    """
    Generator version for async processing with progress updates.
    
    Yields:
        (current_frame, total_frames, status) tuples
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {video_path}")
    
    try:
        metadata = get_video_metadata(cap)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path), 
            fourcc, 
            metadata.fps, 
            (metadata.width, metadata.height)
        )
        
        # Reset tracker for new video
        reset_tracker()
        
        frame_count = 0
        last_tracked_objects = []
        
        yield (0, metadata.total_frames, "started")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            should_process = (frame_skip == 0) or (frame_count % (frame_skip + 1) == 1)
            
            if should_process:
                tracked_objects = process_frame_with_tracking(frame)
                last_tracked_objects = tracked_objects
            else:
                tracked_objects = last_tracked_objects
            
            for bbox, track_id, class_name in tracked_objects:
                draw_tracking_box(frame, bbox, track_id, class_name)
            
            out.write(frame)
            
            # Yield progress every 30 frames
            if frame_count % 30 == 0:
                yield (frame_count, metadata.total_frames, "processing")
        
        yield (frame_count, metadata.total_frames, "completed")
        
    finally:
        cap.release()
        out.release()
