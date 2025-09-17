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
    def __init__(self, start_point, end_point, counting_region=30):
        """
        Initialize a line counter for tracking objects crossing a line.
        
        Args:
            start_point: Starting point of the line (x1, y1)
            end_point: Ending point of the line (x2, y2)
            counting_region: Width of the region around the line to detect crossings
        """
        self.start_point = np.array(start_point)
        self.end_point = np.array(end_point)
        self.counting_region = counting_region
        
        # Calculate line properties
        self.line_vector = self.end_point - self.start_point
        self.line_length = np.linalg.norm(self.line_vector)
        self.unit_line_vector = self.line_vector / self.line_length
        self.normal_vector = np.array([-self.unit_line_vector[1], self.unit_line_vector[0]])
        
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
        if object_id in self.crossed_objects:
            return False
        
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
            
            self.crossed_objects.add(object_id)
            
            # Determine direction of crossing
            if current_distance > prev_distance:
                self.up_count += 1
                return "up"
            else:
                self.down_count += 1
                return "down"
        
        return False

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
    return parser.parse_args()

def process_video(video_path, line_start, line_end, model_path, confidence=0.3, classes=[0], 
                 output_path="object_counting_output.mp4", show=False, verbose=False, output_height=480,
                 iou=0.3, imgsz=640, agnostic_nms=False, track_high_thresh=0.6, track_low_thresh=0.1, 
                 new_track_thresh=0.7, match_thresh=0.8, dataset="unknown"):
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
    line_counter = LineCounter(line_start, line_end)
    
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
    
    # Initialize tracker with configurable parameters
    # Note: ByteTrack parameters may vary by supervision version
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
            detections = tracker.update_with_detections(detections)
            
            # Create a clean frame for display and a clean frame for output
            display_frame = frame.copy()
            
            # Process each detection and update tracking
            detection_info = []  # Store detection info for drawing later
            
            for i, (xyxy, _confidence, class_id, tracker_id) in enumerate(zip(
                detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id
            )):
                if tracker_id is None:
                    continue
                    
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
                
                # Determine color based on crossing status
                color = (0, 255, 0)  # Green for default
                if crossing == "up":
                    color = (0, 0, 255)  # Red for up crossing
                    if verbose:
                        print(f"  -> UP crossing detected for ID {tracker_id}")
                elif crossing == "down":
                    color = (255, 0, 0)  # Blue for down crossing
                    if verbose:
                        print(f"  -> DOWN crossing detected for ID {tracker_id}")
                
                # Store detection info for drawing
                detection_info.append((xyxy, tracker_id, color))
            
            # Draw on display frame (original resolution)
            for xyxy, tracker_id, color in detection_info:
                x1, y1, x2, y2 = xyxy
                # Draw bounding box
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                # Draw ID
                cv2.putText(display_frame, f"ID: {tracker_id}", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw the counting line on display frame
            cv2.line(display_frame, tuple(line_start), tuple(line_end), (255, 0, 255), 2)
            
            # Draw counting region on display frame
            region_points = []
            for t in np.linspace(0, 1, 100):
                point = np.array(line_start) + t * (np.array(line_end) - np.array(line_start))
                region_points.append(point + line_counter.normal_vector * line_counter.counting_region)
            
            for t in np.linspace(1, 0, 100):
                point = np.array(line_start) + t * (np.array(line_end) - np.array(line_start))
                region_points.append(point - line_counter.normal_vector * line_counter.counting_region)
            
            region_points = np.array(region_points, dtype=np.int32)
            cv2.polylines(display_frame, [region_points], True, (255, 0, 255), 1)
            
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
                cv2.line(output_frame, tuple(output_line_start), tuple(output_line_end), (255, 0, 255), 2)
                
                # Draw counting region on output frame
                output_region_points = []
                # Scale the counting region
                scaled_counting_region = int(line_counter.counting_region * scale_y)
                
                # Calculate normal vector for output resolution
                output_line_vector = np.array(output_line_end) - np.array(output_line_start)
                output_line_length = np.linalg.norm(output_line_vector)
                output_unit_line_vector = output_line_vector / output_line_length
                output_normal_vector = np.array([-output_unit_line_vector[1], output_unit_line_vector[0]])
                
                for t in np.linspace(0, 1, 100):
                    point = np.array(output_line_start) + t * (np.array(output_line_end) - np.array(output_line_start))
                    output_region_points.append(point + output_normal_vector * scaled_counting_region)
                
                for t in np.linspace(1, 0, 100):
                    point = np.array(output_line_start) + t * (np.array(output_line_end) - np.array(output_line_start))
                    output_region_points.append(point - output_normal_vector * scaled_counting_region)
                
                output_region_points = np.array(output_region_points, dtype=np.int32)
                cv2.polylines(output_frame, [output_region_points], True, (255, 0, 255), 1)
                
                # Draw detections on output frame with scaled coordinates
                for xyxy, tracker_id, color in detection_info:
                    x1, y1, x2, y2 = xyxy
                    scaled_x1 = int(x1 * scale_x)
                    scaled_y1 = int(y1 * scale_y)
                    scaled_x2 = int(x2 * scale_x)
                    scaled_y2 = int(y2 * scale_y)
                    
                    # Draw bounding box with scaled coordinates
                    cv2.rectangle(output_frame, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), color, 2)
                    
                    # Draw ID with scaled coordinates
                    cv2.putText(output_frame, f"ID: {tracker_id}", (scaled_x1, scaled_y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale_y, color, max(1, int(2 * scale_y)))
                
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
        line_counter = LineCounter(args.line_start, args.line_end)

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
        dataset=args.dataset
    )
    
    # Print results
    if output_path:
        print(f"Processing complete. {frame_count} frames processed.")
        print(f"People count - Up: {up_count}, Down: {down_count}")

if __name__ == "__main__":
    main()
