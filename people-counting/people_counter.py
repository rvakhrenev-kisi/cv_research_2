import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import os
import datetime
from tqdm import tqdm
import subprocess
import sys
import platform

class LineCounter:
    def __init__(self, start_point, end_point, direction_point=None, counting_region=30):
        """
        Initialize a line counter for tracking objects crossing a line.
        
        Args:
            start_point: Starting point of the line (x1, y1)
            end_point: Ending point of the line (x2, y2)
            direction_point: Point indicating the "IN" direction (x, y)
            counting_region: Width of the region around the line to detect crossings
        """
        self.start_point = np.array(start_point)
        self.end_point = np.array(end_point)
        self.direction_point = np.array(direction_point) if direction_point else None
        self.counting_region = counting_region
        
        # Calculate line properties
        self.line_vector = self.end_point - self.start_point
        self.line_length = np.linalg.norm(self.line_vector)
        self.unit_line_vector = self.line_vector / self.line_length
        self.normal_vector = np.array([-self.unit_line_vector[1], self.unit_line_vector[0]])
        
        # Calculate direction reference if direction_point is provided
        if self.direction_point is not None:
            # Calculate which side of the line the direction point is on
            self.direction_side = self.get_distance_from_line(self.direction_point)
        else:
            self.direction_side = None
        
        # Tracking data
        self.object_tracks = defaultdict(list)
        self.crossed_objects = set()
        self.up_count = 0
        self.down_count = 0

    def get_distance_from_line(self, point):
        """Calculate the signed distance from a point to the line."""
        point = np.array(point)
        return np.dot(point - self.start_point, self.normal_vector)

    def is_point_in_counting_region(self, point):
        """Check if a point is within the counting region around the line."""
        # Project the point onto the line
        point = np.array(point)
        point_vector = point - self.start_point
        projection_length = np.dot(point_vector, self.unit_line_vector)
        
        # Check if projection is within line segment
        if projection_length < 0 or projection_length > self.line_length:
            return False
        
        # Check distance from line
        distance = abs(self.get_distance_from_line(point))
        return distance <= self.counting_region

    def update(self, object_id, center_point):
        """
        Update tracking for an object and check if it crossed the line.
        
        Args:
            object_id: Unique identifier for the object
            center_point: Current center point of the object (x, y)
            
        Returns:
            "up", "down", or False if no crossing occurred
        """
        # Allow multiple crossings per person - removed the crossed_objects check
        
        # Add current position to track
        self.object_tracks[object_id].append(center_point)
        
        # Need at least 2 points to detect crossing
        if len(self.object_tracks[object_id]) < 2:
            return False
        
        # Get current and previous positions
        current_pos = np.array(self.object_tracks[object_id][-1])
        prev_pos = np.array(self.object_tracks[object_id][-2])
        
        # Calculate distances from line
        current_distance = self.get_distance_from_line(current_pos)
        prev_distance = self.get_distance_from_line(prev_pos)
        
        # Check if the object crossed the line (sign change in distance)
        # At least one point should be within the counting region to avoid false positives
        # But also check if the person is moving towards the line from far away
        if (current_distance * prev_distance <= 0 and 
            (abs(current_distance) <= self.counting_region or 
                abs(prev_distance) <= self.counting_region or
                # Also detect crossings when person is moving towards the line from far away
                (abs(current_distance) <= self.counting_region * 2 and 
                abs(prev_distance) <= self.counting_region * 2 and
                abs(current_distance - prev_distance) > self.counting_region))):
            
            # Allow multiple crossings - don't add to crossed_objects
            
            # Determine direction of crossing
            if self.direction_side is not None:
                # Use user-defined direction arrow to determine "in" vs "out"
                # If moving towards the direction side, it's "in" (up)
                # If moving away from the direction side, it's "out" (down)
                if (current_distance > prev_distance and self.direction_side > 0) or \
                    (current_distance < prev_distance and self.direction_side < 0):
                    self.up_count += 1
                    return "up"
                else:
                    self.down_count += 1
                    return "down"
            else:
                # Fallback to original logic if no direction specified
                if current_distance > prev_distance:
                    self.up_count += 1
                    return "up"
                else:
                    self.down_count += 1
                    return "down"
        
        return False

def _ocsort_color_for_id(track_id: int):
    h = (int(track_id) * 2654435761) & 0xFFFFFFFF
    r = 50 + (h & 0xFF) % 206
    g = 50 + ((h >> 8) & 0xFF) % 206
    b = 50 + ((h >> 16) & 0xFF) % 206
    return int(b), int(g), int(r)

def setup_local_ffmpeg():
    """
    Configure the environment to use local ffmpeg and ffprobe binaries.
    
    This function:
    1. Gets the absolute paths to the local ffmpeg and ffprobe binaries
    2. Sets environment variables to tell OpenCV and other libraries to use these binaries
    3. Verifies that the binaries are executable
    
    Returns:
        tuple: (ffmpeg_path, ffprobe_path) - Absolute paths to the binaries
    """
    # Get the absolute paths to the binaries
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_path = os.path.join(script_dir, "bin", "ffmpeg")
    ffprobe_path = os.path.join(script_dir, "bin", "ffprobe")
    
    # Make sure the paths are absolute
    ffmpeg_path = os.path.abspath(ffmpeg_path)
    ffprobe_path = os.path.abspath(ffprobe_path)
    
    # Verify that the binaries exist
    if not os.path.isfile(ffmpeg_path):
        raise FileNotFoundError(f"ffmpeg binary not found at {ffmpeg_path}")
    if not os.path.isfile(ffprobe_path):
        raise FileNotFoundError(f"ffprobe binary not found at {ffprobe_path}")
    
    # Ensure the binaries are executable
    if not os.access(ffmpeg_path, os.X_OK):
        os.chmod(ffmpeg_path, 0o755)  # rwxr-xr-x
    if not os.access(ffprobe_path, os.X_OK):
        os.chmod(ffprobe_path, 0o755)  # rwxr-xr-x
    
    # Set environment variables for OpenCV and other libraries
    os.environ["OPENCV_FFMPEG_BINARY"] = ffmpeg_path
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;file,rtp,udp,tcp,https,tls"
    
    # Set environment variables for ultralytics/YOLO
    os.environ["FFMPEG_BINARY"] = ffmpeg_path
    os.environ["FFPROBE_BINARY"] = ffprobe_path
    
    print(f"Using local ffmpeg: {ffmpeg_path}")
    print(f"Using local ffprobe: {ffprobe_path}")
    
    return ffmpeg_path, ffprobe_path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Count people crossing a line in a video using YOLO12")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, default="models/yolo12n.pt", help="Path to YOLO model")
    parser.add_argument("--model-type", type=str, choices=["yolo12"], default="yolo12", 
                        help="Type of YOLO model to use (yolo12)")
    parser.add_argument("--line-start", type=int, nargs=2, default=[0, 0], help="Starting point of counting line (x y)")
    parser.add_argument("--line-end", type=int, nargs=2, default=[0, 0], help="Ending point of counting line (x y)")
    parser.add_argument("--direction-point", type=int, nargs=2, default=None, help="Direction point indicating 'IN' side (x y)")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.3, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size for inference")
    parser.add_argument("--agnostic-nms", action="store_true", help="Use class-agnostic NMS")
    parser.add_argument("--track-high-thresh", type=float, default=0.6, help="High confidence threshold for tracking")
    parser.add_argument("--track-low-thresh", type=float, default=0.1, help="Low confidence threshold for tracking")
    parser.add_argument("--new-track-thresh", type=float, default=0.7, help="New track initialization threshold")
    parser.add_argument("--match-thresh", type=float, default=0.8, help="Matching threshold for tracking")
    parser.add_argument("--output", type=str, default="", help="Path to output video file")
    parser.add_argument("--output-height", type=int, default=480, 
                        help="Output video height in pixels (default: 480, 0 for original resolution)")
    parser.add_argument("--show", action="store_true", help="Display the video while processing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--classes", type=int, nargs="+", default=[0], help="Classes to detect (default: 0 for person)")
    parser.add_argument("--dataset", type=str, default="unknown", help="Dataset name to control line-cross logic (e.g., courtyard/cisco/vortex)")
    parser.add_argument("--tracker-yaml", type=str, default="", help="Optional Ultralytics tracker YAML (e.g., trackers/botsort.yaml)")
    parser.add_argument("--tracker-type", type=str, choices=["bytetrack", "botsort", "ocsort"], default="bytetrack", 
                        help="Tracker type to use (bytetrack, botsort, ocsort)")
    return parser.parse_args()

def process_video(video_path, line_start, line_end, model_path, confidence=0.3, classes=[0], 
                 output_path="object_counting_output.mp4", show=False, verbose=False, output_height=480,
                 iou=0.3, imgsz=640, agnostic_nms=False, track_high_thresh=0.6, track_low_thresh=0.1, 
                 new_track_thresh=0.7, match_thresh=0.8, dataset="unknown", direction_point=None, tracker_type="bytetrack"):
    """
    Process a video to count people crossing a line with progress tracking.
    
    Args:
        video_path: Path to the input video
        line_start: Starting point of the counting line (x, y)
        line_end: Ending point of the counting line (x, y)
        model_path: Path to the YOLO model
        confidence: Detection confidence threshold
        classes: List of classes to detect
        output_path: Path to save the output video
        show: Whether to display the video while processing
        verbose: Whether to enable verbose output for the YOLO model
        output_height: Height of the output video in pixels (default: 480)
                      Width is calculated to maintain the original aspect ratio.
                      If set to 0, the original video resolution will be used.
        
    Note:
        Displays a progress bar showing frames processed, percentage complete, and estimated time remaining.
        
    Returns:
        tuple: (output_path, frame_count, up_count, down_count)
    """
    # Check if video exists
    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' not found")
        return None, 0, 0, 0
    
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return None, 0, 0, 0
    
    # Get video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Handle special cases for output_height
    if output_height < 0:
        print(f"Warning: Negative output height ({output_height}) is invalid. Using default (480).")
        output_height = 480
    
    # Handle case where output_height is 0 (use original resolution)
    if output_height == 0 or output_height == orig_height:
        output_height = orig_height
        output_width = orig_width
        scale_x = 1.0
        scale_y = 1.0
    else:
        # Calculate output dimensions while maintaining aspect ratio
        output_width = int(orig_width * (output_height / orig_height))
        
        # Calculate scale factors for coordinate conversion
        scale_x = output_width / orig_width
        scale_y = output_height / orig_height
    
    # Set default line if not provided
    if line_start == [0, 0] and line_end == [0, 0]:
        # Default to a horizontal line in the middle of the frame
        line_start = [0, orig_height // 2]
        line_end = [orig_width, orig_height // 2]
    
    # Scale line coordinates for output resolution
    output_line_start = [int(line_start[0] * scale_x), int(line_start[1] * scale_y)]
    output_line_end = [int(line_end[0] * scale_x), int(line_end[1] * scale_y)]
    
    # Initialize video writer with output dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    # Print resolution information
    print(f"Input resolution: {orig_width}x{orig_height}")
    print(f"Output resolution: {output_width}x{output_height}")
    
    frame_count = 0
    up_count = 0
    down_count = 0
    
    # Use the LineCounter approach
    line_counter = LineCounter(line_start, line_end, direction_point)
    
    # Check for GPU availability
    print(f"🔍 Starting GPU detection...")
    try:
        import torch
        print(f"   PyTorch imported successfully")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Using device: {device}")
        
        if device == 'cuda':
            print(f"   CUDA is available, checking devices...")
            try:
                device_count = torch.cuda.device_count()
                print(f"   Device count: {device_count}")
                if device_count > 0:
                    print(f"   Getting device name...")
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"   GPU: {gpu_name}")
                    print(f"   Getting device properties...")
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    print(f"   Memory: {gpu_memory:.1f} GB")
                else:
                    print(f"   GPU: CUDA available but no devices found")
                    device = 'cpu'  # Fallback to CPU
            except Exception as e:
                print(f"   GPU: Error accessing GPU info ({e})")
                print(f"   Error type: {type(e).__name__}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                device = 'cpu'  # Fallback to CPU
        else:
            print(f"   GPU: Not available")
    except Exception as e:
        print(f"❌ Error in GPU detection: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        device = 'cpu'
        print(f"   Falling back to CPU")
    
    print(f"✅ GPU detection completed, device: {device}")
    
    # Initialize YOLO model with GPU support
    print(f"model_path = {model_path}")
    try:
        if device == 'cuda':
            # Initialize model directly on GPU
            model = YOLO(model_path, task='detect', verbose=verbose)
            model.to(device)
            print(f"✅ Model initialized and moved to GPU")
        else:
            # Initialize model on CPU
            model = YOLO(model_path, task='detect', verbose=verbose)
            print(f"✅ Model initialized on CPU")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        print(f"   Falling back to CPU mode")
        device = 'cpu'
        model = YOLO(model_path, task='detect', verbose=verbose)
        print(f"✅ Model initialized on CPU (fallback)")
    
    print(f"✅ Model initialization completed")

    # Short-lived highlight for crossings (frames remaining per track id)
    highlight_remaining = defaultdict(int)
    # Track IDs that have crossed at least once (always show these)
    crossed_ids = set()
    # Track first-detected frame index per ID to allow backfill display
    first_seen_frame = {}

    # Optional: two-pass to know crossing IDs upfront for pre-cross display
    crossing_ids_precomputed = set()
    try:
        cap_probe = cv2.VideoCapture(video_path)
        if not cap_probe.isOpened():
            raise RuntimeError("Could not open video for probe")
        frame_idx_probe = 0
        # Minimal line counter for probe
        probe_counter = LineCounter(line_start, line_end, direction_point)
        while cap_probe.isOpened():
            ret_p, frm = cap_probe.read()
            if not ret_p:
                break
            frame_idx_probe += 1
            # Inference
            try:
                res_p = model(frm, conf=float(confidence), classes=classes, verbose=False, imgsz=imgsz, iou=iou, agnostic_nms=agnostic_nms, device=device)
            except Exception:
                break
            det_p = sv.Detections.from_ultralytics(res_p[0])
            # Track
            if hasattr(tracker, "update_with_detections"):
                det_p = tracker.update_with_detections(det_p, img_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), img_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), input_size=imgsz)
            else:
                det_p = tracker.update_with_detections(det_p)
            # Examine
            det_ids_p = det_p.tracker_id if getattr(det_p, "tracker_id", None) is not None else []
            for i_p, tid_p in enumerate(det_ids_p):
                if tid_p is None:
                    continue
                xyxy_p = det_p.xyxy[i_p]
                x1p, y1p, x2p, y2p = xyxy_p
                if dataset.lower() == "courtyard":
                    cxp = (x1p + x2p) / 2
                    cyp = y2p
                else:
                    cxp = (x1p + x2p) / 2
                    cyp = (y1p + y2p) / 2
                crossing = probe_counter.update(int(tid_p), (cxp, cyp))
                if crossing in ("up", "down"):
                    crossing_ids_precomputed.add(int(tid_p))
        cap_probe.release()
        # Recreate tracker for main pass to reset state
        if tracker_type == "ocsort":
            from trackers import OCSortWrapper
            tracker = OCSortWrapper(det_thresh=confidence, max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False)
        else:
            try:
                if tracker_type == "botsort":
                    tracker = sv.BoTSORT(frame_rate=fps)
                else:
                    tracker = sv.ByteTrack(frame_rate=fps)
            except Exception:
                tracker = sv.ByteTrack()
    except Exception:
        crossing_ids_precomputed = set()

    # Initialize tracker based on type
    if tracker_type == "ocsort":
        # Use our custom OC-SORT wrapper
        from trackers import OCSortWrapper
        tracker = OCSortWrapper(
            det_thresh=confidence,
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            delta_t=3,
            asso_func="iou",
            inertia=0.2,
            use_byte=False
        )
        print(f"   Using OC-SORT tracker")
    else:
        # Use supervision trackers (ByteTrack or BoT-SORT)
        if tracker_type == "botsort":
            try:
                tracker = sv.BoTSORT(
                    track_high_thresh=track_high_thresh,
                    track_low_thresh=track_low_thresh,
                    new_track_thresh=new_track_thresh,
                    match_thresh=match_thresh,
                    frame_rate=fps
                )
                print(f"   Using BoT-SORT with custom parameters")
            except (TypeError, AttributeError):
                try:
                    tracker = sv.BoTSORT(frame_rate=fps)
                    print("   Using BoT-SORT with frame_rate only")
                except (TypeError, AttributeError):
                    tracker = sv.BoTSORT()
                    print("   Using basic BoT-SORT initialization")
        else:  # bytetrack
            try:
                # Try with custom parameters first
                tracker = sv.ByteTrack(
                    track_high_thresh=track_high_thresh,
                    track_low_thresh=track_low_thresh,
                    new_track_thresh=new_track_thresh,
                    match_thresh=match_thresh,
                    frame_rate=fps
                )
                print(f"   Using ByteTrack with custom parameters")
            except TypeError:
                try:
                    # Try with frame_rate parameter only
                    tracker = sv.ByteTrack(frame_rate=fps)
                    print("   Using ByteTrack with frame_rate only")
                except TypeError:
                    # Fallback to basic initialization
                    tracker = sv.ByteTrack()
                    print("   Using basic ByteTrack initialization")
    
    # Get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process the video with progress bar
    progress_bar = tqdm(total=total_frames, desc="Processing video", 
                        unit="frames", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    
    consecutive_errors = 0
    max_consecutive_errors = 10  # Maximum number of consecutive errors before giving up
    
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            progress_bar.update(1)
            consecutive_errors = 0  # Reset error counter on successful frame read
        
            # Run YOLO inference on the frame with configurable parameters
            # Ensure confidence is a Python native float, not float32
            try:
                results = model(frame, 
                              conf=float(confidence), 
                              classes=classes, 
                              verbose=False,
                              imgsz=imgsz,  # Configurable input size
                              iou=iou,      # Configurable IoU threshold
                              agnostic_nms=agnostic_nms,  # Configurable NMS type
                              device=device)  # Use GPU if available
            except Exception as e:
                print(f"❌ Error in YOLO inference: {e}")
                print(f"   Error type: {type(e).__name__}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                break
            
            # Get detections
            detections = sv.Detections.from_ultralytics(results[0])
        
            # Update tracker
            if hasattr(tracker, "update_with_detections"):
                detections = tracker.update_with_detections(
                    detections,
                    img_h=orig_height,
                    img_w=orig_width,
                    input_size=imgsz,
                )
            else:
                detections = tracker.update_with_detections(detections)
            
            # Create a clean frame for display and a clean frame for output
            display_frame = frame.copy()
            
            # Process each detection and update tracking
            detection_info = []  # Store (xyxy, tracker_id) for drawing later
            
            # Normalize detections arrays to avoid None iterations
            det_xyxy = detections.xyxy if getattr(detections, "xyxy", None) is not None else []
            num_dets = len(det_xyxy)
            det_conf = detections.confidence if getattr(detections, "confidence", None) is not None else np.ones(num_dets, dtype=float)
            det_cls = detections.class_id if getattr(detections, "class_id", None) is not None else np.zeros(num_dets, dtype=int)
            det_ids = detections.tracker_id if getattr(detections, "tracker_id", None) is not None else [None] * num_dets

            for i in range(num_dets):
                xyxy = det_xyxy[i]
                _confidence = det_conf[i] if i < len(det_conf) else 1.0
                class_id = det_cls[i] if i < len(det_cls) else 0
                tracker_id = det_ids[i] if i < len(det_ids) else None
                if tracker_id is None:
                    continue
                # Record first-seen frame for each tracker_id
                tid = int(tracker_id)
                if tid not in first_seen_frame:
                    first_seen_frame[tid] = frame_count
                    
                # Calculate point of interest for crossing check based on dataset
                x1, y1, x2, y2 = xyxy
                if dataset.lower() == "courtyard":
                    # Use bottom-center of the box
                    center_x = (x1 + x2) / 2
                    center_y = y2
                else:
                    # Use middle of the box (default for cisco/vortex)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                
                # Update line counter
                crossing = line_counter.update(tracker_id, (center_x, center_y))
                
                # Debug output for first few frames
                if frame_count < 10 and verbose:
                    print(f"Frame {frame_count}, ID {tracker_id}: center=({center_x:.1f}, {center_y:.1f}), crossing={crossing}")
                
                # If crossing detected, trigger short red highlight for this ID
                if crossing in ("up", "down"):
                    highlight_remaining[int(tracker_id)] = 10
                    crossed_ids.add(int(tracker_id))
                    if verbose:
                        arrow = "UP" if crossing == "up" else "DOWN"
                        print(f"  -> {arrow} crossing detected for ID {tracker_id}")

                # Store detection info for drawing
                detection_info.append((xyxy, tracker_id))
            
            # Draw on display frame (original resolution)
            for xyxy, tracker_id in detection_info:
                x1, y1, x2, y2 = xyxy
                tid = int(tracker_id)
                # Uniform box color; override with red when highlighted
                default_box_color = (0, 255, 255)  # yellow
                box_color = (0, 0, 255) if highlight_remaining.get(tid, 0) > 0 else default_box_color
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                # No ID text on the box
            
            # Draw the counting line on display frame (courtyard-style: green)
            cv2.line(display_frame, tuple(line_start), tuple(line_end), (0, 255, 0), 2)
            
            # Remove magenta counting region box around the line
            
            # Draw counts on display frame (top-right corner)
            frame_height, frame_width = display_frame.shape[:2]
            count_text = f"Crossing In: {line_counter.up_count} | Crossing Out: {line_counter.down_count}"
            text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = frame_width - text_size[0] - 10
            text_y = 30
            
            # Draw background rectangle for better visibility
            cv2.rectangle(display_frame, (text_x - 5, text_y - 25), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(display_frame, count_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Create output frame with appropriate resolution
            if output_height != orig_height:
                # Start with a clean resized frame
                output_frame = cv2.resize(frame.copy(), (output_width, output_height))
                
                # Draw the counting line on output frame
                # Draw the counting line on output frame (courtyard-style: green)
                cv2.line(output_frame, tuple(output_line_start), tuple(output_line_end), (0, 255, 0), 2)
                
                # Remove magenta counting region box around the line
                
                # Draw detections on output frame with scaled coordinates
                for xyxy, tracker_id in detection_info:
                    x1, y1, x2, y2 = xyxy
                    scaled_x1 = int(x1 * scale_x)
                    scaled_y1 = int(y1 * scale_y)
                    scaled_x2 = int(x2 * scale_x)
                    scaled_y2 = int(y2 * scale_y)
                    tid = int(tracker_id)
                    
                    # Uniform box color; override with red when highlighted
                    default_box_color = (0, 255, 255)  # yellow
                    box_color = (0, 0, 255) if highlight_remaining.get(tid, 0) > 0 else default_box_color
                    cv2.rectangle(output_frame, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), box_color, 2)
                    # No ID text on the box

                # Decrement highlight timers for IDs present this frame
            # Decrement highlight timers for IDs present this frame
            for _, tracker_id in detection_info:
                tid = int(tracker_id)
                if highlight_remaining.get(tid, 0) > 0:
                    highlight_remaining[tid] -= 1
                
                # Draw counts with scaled font size and position (top-right corner)
                count_text = f"Crossing In: {line_counter.up_count} | Crossing Out: {line_counter.down_count}"
                font_scale = 0.8 * scale_y
                thickness = max(1, int(2 * scale_y))
                text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = output_width - text_size[0] - 10
                text_y = int(30 * scale_y)
                
                # Draw background rectangle for better visibility
                cv2.rectangle(output_frame, (text_x - 5, text_y - int(25 * scale_y)), 
                             (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(output_frame, count_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (0, 255, 0), thickness)
            else:
                # If no resizing is needed, use the display frame
                output_frame = display_frame.copy()
            
            # Display the frame (original size for display)
            if show:
                cv2.imshow('People Counter', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write the output frame to video
            output_writer.write(output_frame)
            
        except Exception as e:
            consecutive_errors += 1
            print(f"\nError processing frame: {e}")
            
            # Skip to the next frame if there's an error
            # If we've had too many consecutive errors, break the loop
            if consecutive_errors >= max_consecutive_errors:
                print(f"Too many consecutive errors ({max_consecutive_errors}). Stopping processing.")
                break
            
            # Try to seek to the next frame
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos + 1)
    
    # Close progress bar
    progress_bar.close()
    
    # Get counts
    up_count = line_counter.up_count
    down_count = line_counter.down_count
    
    # Release resources
    cap.release()
    output_writer.release()
    cv2.destroyAllWindows()
    
    # Print summary of processing
    print(f"\nVideo processing summary:")
    print(f"  Total frames processed: {frame_count}")
    print(f"  People count - Up: {up_count}, Down: {down_count}, Total: {up_count + down_count}")
    
    if consecutive_errors > 0:
        print(f"  Warning: {consecutive_errors} errors encountered during processing.")
        print(f"  The video may be corrupted or have encoding issues.")
        print(f"  Results represent partial processing up to the point of failure.")
    
    # Return results
    return output_path, frame_count, up_count, down_count

def main():
    # Set up local ffmpeg and ffprobe binaries
    try:
        ffmpeg_path, ffprobe_path = setup_local_ffmpeg()
        print("Successfully configured local ffmpeg and ffprobe binaries")
    except Exception as e:
        print(f"Warning: Failed to set up local ffmpeg/ffprobe: {e}")
        print("Falling back to system-installed ffmpeg/ffprobe if available")
    
    args = parse_arguments()
    
    # Generate output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    original_filename = os.path.splitext(os.path.basename(args.video))[0]
    output_filename = f"{original_filename}_counting_{timestamp}.mp4"
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Set output path
    output_path_arg = args.output if args.output else os.path.join("output", output_filename)
    
    # If Ultralytics tracker YAML provided, run tracking-only flow (IDs) and exit
    if getattr(args, "tracker_yaml", ""):
        # Run Ultralytics tracker but preserve our output format and draw the line and filter to person only
        print(f"🔄 Using Ultralytics tracker: {args.tracker_yaml}")
        model = YOLO(args.model)

        # Output path handling
        os.makedirs(os.path.dirname(output_path_arg) or ".", exist_ok=True)

        # Video IO setup to mirror standard branch
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: Could not open video '{args.video}'")
            return
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        out_h = args.output_height if args.output_height not in (None, -1) else 480
        if out_h == 0:
            out_h = orig_height
        out_w = int(orig_width * (out_h / orig_height)) if out_h != orig_height else orig_width
        scale_x = out_w / orig_width
        scale_y = out_h / orig_height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path_arg, fourcc, fps, (out_w, out_h))

        # Coordinates for line
        lx1, ly1 = args.line_start
        lx2, ly2 = args.line_end

        # Initialize line counter for Ultralytics tracker
        line_counter = LineCounter(args.line_start, args.line_end, args.direction_point)

        # Stream tracking to avoid RAM accumulation and to let us draw overlays
        stream = model.track(
            source=args.video,
            tracker=args.tracker_yaml,
            conf=args.confidence,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=args.classes,  # person-only (class 0)
            persist=True,
            stream=True,
            verbose=args.verbose,
        )

        frame_count = 0
        for res in stream:
            frame = res.orig_img
            frame_count += 1
            
            # Get tracking results
            if res.boxes is not None and len(res.boxes) > 0:
                # Process each tracked object
                for box in res.boxes:
                    if box.id is not None:  # Only process tracked objects
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Calculate point of interest for crossing check based on dataset
                        if args.dataset.lower() == "courtyard":
                            # Use bottom-center of the box
                            center_x = (x1 + x2) / 2
                            center_y = y2
                        else:
                            # Use middle of the box (default for cisco/vortex)
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                        
                        # Update line counter
                        crossing = line_counter.update(int(box.id), (center_x, center_y))
                        
                        # Additional debug info for line crossing analysis
                        if args.verbose and frame_count <= 20:  # Show detailed info for first 20 frames
                            distance = line_counter.get_distance_from_line((center_x, center_y))
                            print(f"    -> Distance from line: {distance:.1f}, Line y=532, Person y={center_y:.1f}")
                        
                        # Debug output for all frames when verbose
                        if args.verbose:
                            print(f"Frame {frame_count}, ID {int(box.id)}: center=({center_x:.1f}, {center_y:.1f}), crossing={crossing}")
                        
                        if crossing == "up" and args.verbose:
                            print(f"  -> UP crossing detected for ID {int(box.id)}")
                        elif crossing == "down" and args.verbose:
                            print(f"  -> DOWN crossing detected for ID {int(box.id)}")
            
            # Render tracker annotations (IDs) onto frame
            annotated = res.plot()  # includes boxes and IDs
            
            # Resize if needed
            if out_h != orig_height:
                annotated = cv2.resize(annotated, (out_w, out_h))
                # scale line
                sx1, sy1 = int(lx1 * scale_x), int(ly1 * scale_y)
                sx2, sy2 = int(lx2 * scale_x), int(ly2 * scale_y)
            else:
                sx1, sy1, sx2, sy2 = lx1, ly1, lx2, ly2
            
            # Draw counting line for visual consistency
            cv2.line(annotated, (sx1, sy1), (sx2, sy2), (0, 255, 255), 3)
            
            # Draw counts on frame (top-right corner)
            frame_height, frame_width = annotated.shape[:2]
            count_text = f"Crossing In: {line_counter.up_count} | Crossing Out: {line_counter.down_count}"
            text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = frame_width - text_size[0] - 10
            text_y = 30
            
            # Draw background rectangle for better visibility
            cv2.rectangle(annotated, (text_x - 5, text_y - 25), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(annotated, count_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            writer.write(annotated)

        writer.release()
        cap.release()
        
        # Print summary
        print(f"✅ Tracking output saved to: {output_path_arg}")
        print(f"Video processing summary:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  People count - Up: {line_counter.up_count}, Down: {line_counter.down_count}, Total: {line_counter.up_count + line_counter.down_count}")
        return

    # Process the video
    output_path, frame_count, up_count, down_count = process_video(
        video_path=args.video,
        line_start=args.line_start,
        line_end=args.line_end,
        model_path=args.model,
        confidence=args.confidence,
        classes=args.classes,
        output_path=output_path_arg,
        show=args.show,
        verbose=args.verbose,
        output_height=args.output_height,
        iou=args.iou,
        imgsz=args.imgsz,
        agnostic_nms=args.agnostic_nms,
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        match_thresh=args.match_thresh,
        dataset=args.dataset,
        direction_point=args.direction_point,
        tracker_type=args.tracker_type
    )
    
    # Print results
    if output_path:
        print(f"Processing complete. {frame_count} frames processed.")
        print(f"People count - Up: {up_count}, Down: {down_count}")

if __name__ == "__main__":
    main()
