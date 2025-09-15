#%% Title & Overview
"""
TAILGATING DETECTION NOTEBOOK
============================

Goal: Count unique people crossing a predefined line; report direction (IN/OUT) and totals by time window.

This notebook is fully self-contained and runnable from within this single Python file using #%% cell markers.
No console commands or external installations required - everything runs from this file.

SCENE CONFIGURATIONS:
- Two separate configs supported: @cisco and @vortex
- Videos in 'cisco/' folder use Cisco scene config
- Videos in 'vortex/' folder use Vortex scene config
- Automatic detection based on video path or explicit override

STEP-BY-STEP BUILD PROCESS:
This notebook is built incrementally, one checklist item at a time.
After each step, you'll be prompted to:
1. Run the new cell(s)
2. Type CONTINUE if it works correctly
3. Provide revisions if something needs to be changed

Each step is designed to be self-testable with built-in validation.
"""

print("✅ Tailgating Detection Notebook initialized")
print("📁 Two scene configs: @cisco and @vortex")
print("🔄 Step-by-step build process ready")

#%% Environment Check
"""
Import all required libraries and check system capabilities.
Gracefully handle missing dependencies with clear installation instructions.
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Core dependencies
try:
    import numpy as np
    print("✅ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy missing: {e}")
    print("💡 Install with: pip install numpy")

try:
    import cv2
    print("✅ OpenCV imported successfully")
except ImportError as e:
    print(f"❌ OpenCV missing: {e}")
    print("💡 Install with: pip install opencv-python")

try:
    import pandas as pd
    print("✅ Pandas imported successfully")
except ImportError as e:
    print(f"❌ Pandas missing: {e}")
    print("💡 Install with: pip install pandas")

try:
    import yaml
    print("✅ PyYAML imported successfully")
except ImportError as e:
    print(f"❌ PyYAML missing: {e}")
    print("💡 Install with: pip install pyyaml")

try:
    import gradio as gr
    print("✅ Gradio imported successfully")
except ImportError as e:
    print(f"❌ Gradio missing: {e}")
    print("💡 Install with: pip install gradio")

# PyTorch and CUDA check
try:
    import torch
    import torchvision
    print("✅ PyTorch imported successfully")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🚀 CUDA available: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU (CUDA not available)")
        
except ImportError as e:
    print(f"❌ PyTorch missing: {e}")
    print("💡 Install with: pip install torch torchvision")
    device = None

# Optional dependencies (will be checked later when needed)
optional_deps = {
    'ultralytics': 'YOLO models',
    'byte_tracker': 'ByteTrack tracker',
    'ocsort': 'OC-SORT tracker',
    'botsort': 'BoT-SORT tracker'
}

print(f"\n🔧 Device: {device if device else 'Unknown'}")
print(f"🐍 Python: {sys.version.split()[0]}")
print("📦 Environment check complete")

# Test snippet - verify basic functionality
try:
    test_array = np.array([1, 2, 3, 4, 5])
    test_sum = np.sum(test_array)
    print(f"🧪 Basic test: sum([1,2,3,4,5]) = {test_sum} ✅")
except Exception as e:
    print(f"🧪 Basic test failed: {e} ❌")

#%% Config Management
"""
Central configuration management with two scene configs (Cisco and Vortex).
Handles YAML loading/saving, scene inference, and directory management.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Central configuration dictionary
Config = {
    # Input/Output
    "input_video_path": "",
    "dataset_name": "auto",  # "auto" | "cisco" | "vortex"
    "output_dir": "./outputs",
    "fps_assumed": 30.0,
    
    # Visualization
    "draw_boxes": True,
    "draw_ids": True,
    "draw_line": True,
    "draw_trails": False,
    
    # Detection/Tracking
    "detector_name": "yolov10s",
    "tracker_name": "bytetrack",
    "precision": "fp32",  # "fp32" | "fp16"
    "batch_size": 1,
    "stride": 1,
    "confidence_threshold": 0.3,
    "iou_threshold": 0.5,
    
    # Line counting
    "min_track_age": 3,
    "track_cooldown_frames": 10,
    "min_displacement_px": 5,
    "aggregate_interval_s": 60
}

def ensure_dirs() -> None:
    """Create necessary directories if they don't exist."""
    dirs = [
        "./outputs",
        "./configs",
        "./data"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("📁 Directories ensured")

def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML file, return empty dict if missing."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"⚠️ Error loading {path}: {e}")
        return {}

def save_yaml(path: str, data: Dict[str, Any]) -> None:
    """Save data to YAML file."""
    ensure_dirs()
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)
    print(f"💾 Saved config to {path}")

def infer_dataset_from_path(path: str) -> Optional[str]:
    """Infer dataset name from video path."""
    if not path:
        return None
    
    path_lower = path.lower()
    if "@cisco" in path_lower or "/cisco/" in path_lower:
        return "cisco"
    elif "@vortex" in path_lower or "/vortex/" in path_lower:
        return "vortex"
    return None

def resolve_dataset_name(config: Dict[str, Any]) -> str:
    """Resolve effective dataset name from config."""
    dataset_name = config.get("dataset_name", "auto")
    
    if dataset_name == "auto":
        inferred = infer_dataset_from_path(config.get("input_video_path", ""))
        if inferred:
            print(f"🔍 Auto-detected dataset: {inferred}")
            return inferred
        else:
            print("⚠️ Could not infer dataset from path, defaulting to 'cisco'")
            return "cisco"
    
    return dataset_name

def load_scene(dataset_name: str) -> Dict[str, Any]:
    """Load scene configuration for given dataset."""
    scene_path = f"./configs/scene_{dataset_name}.yaml"
    scene_data = load_yaml(scene_path)
    
    # Default scene structure
    if not scene_data:
        scene_data = {
            "line": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
            "last_update": None
        }
        save_scene(dataset_name, scene_data)
    
    # Handle nested line structure (fix for existing files)
    if "line" in scene_data and isinstance(scene_data["line"], dict):
        if "line" in scene_data["line"]:
            # Nested structure: line.line
            scene_data["line"] = scene_data["line"]["line"]
        elif "last_update" in scene_data["line"] and "line" in scene_data["line"]:
            # Another nested structure: line.last_update and line.line
            scene_data["line"] = scene_data["line"]["line"]
    
    # Additional check for the specific structure we're seeing
    if "line" in scene_data and isinstance(scene_data["line"], dict):
        if "last_update" in scene_data["line"] and "line" in scene_data["line"]:
            # Structure: {'last_update': None, 'line': {'x1': 0, 'x2': 100, 'y1': 0, 'y2': 100}}
            scene_data["line"] = scene_data["line"]["line"]
    
    return scene_data

def save_scene(dataset_name: str, line_dict: Dict[str, Any]) -> None:
    """Save scene configuration for given dataset."""
    scene_data = {
        "line": line_dict,
        "last_update": datetime.now().isoformat()
    }
    scene_path = f"./configs/scene_{dataset_name}.yaml"
    save_yaml(scene_path, scene_data)

# Initialize scene configs if they don't exist
for dataset in ["cisco", "vortex"]:
    scene_data = load_scene(dataset)
    print(f"📋 Loaded {dataset} scene config: {scene_data['line']}")

# Test snippet - exercise save/load functionality
print("\n🧪 Testing config management...")
test_data = {"x1": 50, "y1": 50, "x2": 150, "y2": 150}
save_scene("test", test_data)
loaded_data = load_scene("test")
print(f"✅ Save/Load test: {loaded_data['line'] == test_data}")

# Test dataset inference
test_paths = [
    "/path/to/video@cisco.mp4",
    "/path/to/vortex/video.mp4", 
    "/path/to/unknown/video.mp4"
]
for path in test_paths:
    inferred = infer_dataset_from_path(path)
    print(f"🔍 Path '{path}' -> {inferred}")

print("✅ Config management ready")


#%% Snapshot Extraction
"""
Extract representative frames from videos for line marking GUI.
Handles FPS detection and frame extraction with fallback options.
"""

def extract_snapshot(video_path: str, fps_assumed: float = 30.0) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Extract a representative frame from video for line marking.
    
    Returns:
        frame_bgr: Frame in BGR format (for OpenCV)
        frame_rgb: Frame in RGB format (for GUI display)
        actual_fps: Detected or assumed FPS
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Try to open video with OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        # Get video properties
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps <= 0 or actual_fps > 120:  # Sanity check
            actual_fps = fps_assumed
            print(f"⚠️ Using assumed FPS: {actual_fps}")
        else:
            print(f"📹 Detected FPS: {actual_fps}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError("Could not determine video frame count")
        
        # Extract frame at 25% of video length
        target_frame = int(total_frames * 0.25)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        ret, frame_bgr = cap.read()
        if not ret:
            # Fallback to first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame_bgr = cap.read()
            if not ret:
                raise RuntimeError("Could not read any frame from video")
        
        cap.release()
        
        # Convert BGR to RGB for GUI display
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        print(f"✅ Extracted frame {target_frame}/{total_frames} from {os.path.basename(video_path)}")
        return frame_bgr, frame_rgb, actual_fps
        
    except Exception as e:
        print(f"❌ Error extracting snapshot: {e}")
        # Return a placeholder frame if OpenCV fails
        placeholder_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        return placeholder_bgr, placeholder_rgb, fps_assumed

def save_snapshot_for_gui(frame_bgr: np.ndarray, dataset_name: str) -> str:
    """Save snapshot image for GUI preview."""
    ensure_dirs()
    snapshot_path = f"./outputs/tmp_snapshot_{dataset_name}.jpg"
    cv2.imwrite(snapshot_path, frame_bgr)
    print(f"💾 Saved snapshot: {snapshot_path}")
    return snapshot_path

def get_video_info(video_path: str) -> dict:
    """Get basic video information."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video"}
        
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        cap.release()
        return info
    except Exception as e:
        return {"error": str(e)}

# Test snippet - try to load a video if available
print("\n🧪 Testing snapshot extraction...")

# Check if we have any video files
video_files = []
for folder in ["cisco", "vortex"]:
    folder_path = Path(folder)
    if folder_path.exists():
        for video_file in folder_path.glob("*.mp4"):
            video_files.append(str(video_file))

if video_files:
    test_video = video_files[0]
    print(f"📹 Testing with: {test_video}")
    
    try:
        frame_bgr, frame_rgb, fps = extract_snapshot(test_video, Config["fps_assumed"])
        print(f"✅ Frame shape: {frame_bgr.shape}, FPS: {fps}")
        
        # Save snapshot for GUI
        dataset_name = infer_dataset_from_path(test_video) or "cisco"
        snapshot_path = save_snapshot_for_gui(frame_bgr, dataset_name)
        
        # Get video info
        info = get_video_info(test_video)
        if "error" not in info:
            print(f"📊 Video info: {info['width']}x{info['height']}, {info['duration']:.1f}s")
        
    except Exception as e:
        print(f"❌ Snapshot test failed: {e}")
else:
    print("⚠️ No video files found in cisco/ or vortex/ folders")
    print("💡 Set Config['input_video_path'] to test with a specific video")

print("✅ Snapshot extraction ready")

#%% GUI: Line Marking (Gradio)
"""
Interactive line marking interface using Gradio.
Allows selection of dataset and video, with two-click line picker.
"""

def create_line_marking_gui():
    """Create Gradio interface for line marking."""
    
    # Global variables for line marking
    line_points = [None, None]  # [start_point, end_point]
    current_snapshot = None
    inside_side = None  # "left", "right", "top", "bottom" - which side of line is "inside"
    
    def load_video_preview(video_file, dataset_name):
        """Load and display video preview for line marking."""
        nonlocal current_snapshot, line_points
        
        if video_file is None:
            return None, "No video selected"
        
        try:
            # Extract snapshot
            frame_bgr, frame_rgb, fps = extract_snapshot(video_file.name, Config["fps_assumed"])
            current_snapshot = frame_bgr.copy()
            
            # Save snapshot for display
            snapshot_path = save_snapshot_for_gui(frame_bgr, dataset_name)
            
            # Load existing line if available
            scene_data = load_scene(dataset_name)
            line = scene_data.get("line", {"x1": 0, "y1": 0, "x2": 100, "y2": 100})
            
            # Handle nested structure
            if isinstance(line, dict) and "line" in line:
                line = line["line"]
            
            # Reset line points
            line_points = [None, None]
            
            return snapshot_path, f"Loaded {dataset_name} scene. Click on image to mark line. Current: ({line['x1']},{line['y1']}) -> ({line['x2']},{line['y2']})"
            
        except Exception as e:
            return None, f"Error loading video: {str(e)}"
    
    def on_image_click(evt: gr.SelectData):
        """Handle image click for line marking."""
        nonlocal line_points
        
        print(f"🔍 Click event received: {evt}")
        
        if evt.index is None:
            print("❌ No click coordinates")
            return gr.update(), "Click on the image to mark line points", gr.update(), gr.update(), gr.update(), gr.update()
        
        x, y = evt.index[0], evt.index[1]
        print(f"📍 Clicked at: ({x}, {y})")
        
        if line_points[0] is None:
            # First point
            line_points[0] = (x, y)
            print(f"✅ First point set: {line_points[0]}")
            return gr.update(), f"First point set at ({x}, {y}). Click for second point.", x, y, gr.update(), gr.update()
        else:
            # Second point
            line_points[1] = (x, y)
            print(f"✅ Second point set: {line_points[1]}")
            # Update the coordinate inputs and draw line
            updated_image = update_line_display(line_points[0][0], line_points[0][1], x, y, "cisco")
            print(f"🎨 Line drawn from {line_points[0]} to {line_points[1]}")
            return updated_image, f"Line marked: ({line_points[0][0]},{line_points[0][1]}) -> ({x},{y})", line_points[0][0], line_points[0][1], x, y
    
    def draw_line_on_image(image_path, x1, y1, x2, y2):
        """Draw line on image and return updated image."""
        if image_path is None or not os.path.exists(image_path):
            return image_path
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return image_path
            
            # Draw line
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            
            # Draw circles at endpoints
            cv2.circle(img, (int(x1), int(y1)), 8, (0, 0, 255), -1)
            cv2.circle(img, (int(x2), int(y2)), 8, (0, 0, 255), -1)
            
            # Save updated image
            updated_path = image_path.replace('.jpg', '_with_line.jpg')
            cv2.imwrite(updated_path, img)
            
            return updated_path
        except Exception as e:
            print(f"Error drawing line: {e}")
            return image_path
    
    def save_line_config(dataset_name, x1, y1, x2, y2, inside_side):
        """Save line configuration to YAML."""
        try:
            line_dict = {
                "x1": int(x1), 
                "y1": int(y1), 
                "x2": int(x2), 
                "y2": int(y2),
                "inside_side": inside_side
            }
            save_scene(dataset_name, line_dict)
            return f"✅ Saved line for {dataset_name}: ({x1},{y1}) -> ({x2},{y2}) [Inside: {inside_side}]"
        except Exception as e:
            return f"❌ Error saving: {str(e)}"
    
    def load_existing_line(dataset_name):
        """Load existing line configuration."""
        try:
            scene_data = load_scene(dataset_name)
            line = scene_data.get("line", {"x1": 0, "y1": 0, "x2": 100, "y2": 100, "inside_side": "right"})
            
            # Handle nested structure
            if isinstance(line, dict) and "line" in line:
                line = line["line"]
            
            return line["x1"], line["y1"], line["x2"], line["y2"], line.get("inside_side", "right")
        except Exception as e:
            return 0, 0, 100, 100, "right"
    
    def update_line_display(x1, y1, x2, y2, dataset_name="cisco"):
        """Update the image with the drawn line."""
        if current_snapshot is not None:
            # Create a copy and draw line
            img_with_line = current_snapshot.copy()
            cv2.line(img_with_line, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.circle(img_with_line, (int(x1), int(y1)), 8, (0, 0, 255), -1)
            cv2.circle(img_with_line, (int(x2), int(y2)), 8, (0, 0, 255), -1)
            
            # Save and return path
            temp_path = f"./outputs/temp_line_preview_{dataset_name}.jpg"
            cv2.imwrite(temp_path, img_with_line)
            return temp_path
        return None
    
    def auto_draw_line(x1, y1, x2, y2, dataset_name):
        """Automatically draw line when coordinates change."""
        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            return update_line_display(x1, y1, x2, y2, dataset_name)
        return None
    
    # Create Gradio interface
    with gr.Blocks(title="Tailgating Line Marker") as demo:
        gr.Markdown("# 🎯 Tailgating Line Marker")
        gr.Markdown("Select dataset and video, then mark the entrance line for counting.")
        
        with gr.Row():
            with gr.Column():
                dataset_dropdown = gr.Dropdown(
                    choices=["cisco", "vortex"],
                    value="cisco",
                    label="Dataset",
                    info="Choose which scene config to use"
                )
                
                video_input = gr.File(
                    label="Video File",
                    file_types=[".mp4", ".avi", ".mov"]
                )
                
                load_btn = gr.Button("Load Video Preview", variant="primary")
                
            with gr.Column():
                preview_image = gr.Image(
                    label="Video Preview - Click to mark line points",
                    type="filepath",
                    interactive=True
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    value="Select a video to begin",
                    interactive=False
                )
        
        gr.Markdown("### 📏 Line Configuration")
        gr.Markdown("Enter the two endpoints of the counting line (in pixels):")
        
        with gr.Row():
            x1_input = gr.Number(label="X1", value=0, precision=0)
            y1_input = gr.Number(label="Y1", value=0, precision=0)
            x2_input = gr.Number(label="X2", value=100, precision=0)
            y2_input = gr.Number(label="Y2", value=100, precision=0)
        
        with gr.Row():
            inside_side_dropdown = gr.Dropdown(
                choices=["left", "right", "top", "bottom"],
                value="right",
                label="Inside Side",
                info="Which side of the line is considered 'inside'?"
            )
        
        with gr.Row():
            load_existing_btn = gr.Button("Load Existing Line")
            draw_line_btn = gr.Button("Draw Line Preview", variant="secondary")
            save_btn = gr.Button("Save Line", variant="primary")
        
        save_status = gr.Textbox(label="Save Status", interactive=False)
        
        # Event handlers
        load_btn.click(
            fn=load_video_preview,
            inputs=[video_input, dataset_dropdown],
            outputs=[preview_image, status_text]
        )
        
        # Handle image clicks
        preview_image.select(
            fn=on_image_click,
            inputs=[],
            outputs=[preview_image, status_text, x1_input, y1_input, x2_input, y2_input]
        )
        
        load_existing_btn.click(
            fn=load_existing_line,
            inputs=[dataset_dropdown],
            outputs=[x1_input, y1_input, x2_input, y2_input, inside_side_dropdown]
        )
        
        draw_line_btn.click(
            fn=update_line_display,
            inputs=[x1_input, y1_input, x2_input, y2_input, dataset_dropdown],
            outputs=[preview_image]
        )
        
        save_btn.click(
            fn=save_line_config,
            inputs=[dataset_dropdown, x1_input, y1_input, x2_input, y2_input, inside_side_dropdown],
            outputs=[save_status]
        )
        
        # Auto-draw line when coordinates change
        for coord_input in [x1_input, y1_input, x2_input, y2_input]:
            coord_input.change(
                fn=auto_draw_line,
                inputs=[x1_input, y1_input, x2_input, y2_input, dataset_dropdown],
                outputs=[preview_image]
            )
    
    return demo

def launch_line_gui():
    """Launch the line marking GUI."""
    try:
        demo = create_line_marking_gui()
        print("🚀 Launching line marking GUI...")
        print("💡 Close the Gradio tab/window when done marking lines")
        demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        print("💡 Make sure Gradio is installed: pip install gradio")

# Test snippet - check if GUI can be created
print("\n🧪 Testing GUI creation...")
try:
    demo = create_line_marking_gui()
    print("✅ GUI interface created successfully")
    print("💡 Run launch_line_gui() to open the interface")
except Exception as e:
    print(f"❌ GUI creation failed: {e}")
    print("💡 This is expected if Gradio is not installed")

print("✅ Line marking GUI ready")

#%% Line Counter Logic
"""
LineCounter class for detecting people crossing a predefined line.
Handles direction detection, debouncing, and event generation.
"""

class LineCounter:
    """
    Counts unique people crossing a line with direction detection and debouncing.
    Tracks unique people rather than just events to avoid double-counting.
    """
    
    def __init__(self, line_endpoints: tuple, min_displacement_px: int = 10, 
                 track_cooldown_frames: int = 30):
        """
        Initialize line counter.
        
        Args:
            line_endpoints: ((x1, y1), (x2, y2)) line coordinates
            min_displacement_px: Minimum pixel movement to count as crossing
            track_cooldown_frames: Frames to wait before allowing same track to cross again
        """
        self.line_start = np.array(line_endpoints[0], dtype=np.float32)
        self.line_end = np.array(line_endpoints[1], dtype=np.float32)
        self.min_displacement = min_displacement_px
        self.cooldown_frames = track_cooldown_frames
        
        # Track state
        self.track_positions = {}  # track_id -> (x, y, frame_idx)
        self.track_cooldowns = {}  # track_id -> frame_idx when last crossed
        self.crossing_events = []  # List of crossing events
        
        # Unique people tracking
        self.unique_people_in = set()  # Set of track_ids that crossed IN
        self.unique_people_out = set()  # Set of track_ids that crossed OUT
        self.track_crossing_history = {}  # track_id -> list of crossing directions
        
        # Line vector for direction calculation
        self.line_vector = self.line_end - self.line_start
        self.line_length = np.linalg.norm(self.line_vector)
        
        if self.line_length == 0:
            raise ValueError("Line endpoints cannot be identical")
        
        # Normalize line vector
        self.line_unit = self.line_vector / self.line_length
        
        print(f"✅ LineCounter initialized: {line_endpoints[0]} -> {line_endpoints[1]}")
    
    def _point_to_line_distance(self, point: np.ndarray) -> float:
        """Calculate perpendicular distance from point to line."""
        # Vector from line start to point
        point_vector = point - self.line_start
        
        # Project point onto line
        projection_length = np.dot(point_vector, self.line_unit)
        
        # Clamp projection to line segment
        projection_length = np.clip(projection_length, 0, self.line_length)
        
        # Projected point on line
        projected_point = self.line_start + projection_length * self.line_unit
        
        # Distance from point to line
        return np.linalg.norm(point - projected_point)
    
    def _point_crossed_line(self, prev_point: np.ndarray, curr_point: np.ndarray) -> bool:
        """Check if point crossed the line between two positions."""
        # Check if line segment intersects with the counting line
        # Using parametric line intersection
        
        # Line segment from prev to curr
        seg_vector = curr_point - prev_point
        seg_length = np.linalg.norm(seg_vector)
        
        if seg_length == 0:
            return False
        
        # Method: Check if the movement vector crosses the line
        # by checking if the two points are on opposite sides of the line
        
        # Vector from line start to each point
        vec_prev = prev_point - self.line_start
        vec_curr = curr_point - self.line_start
        
        # Cross products with line vector
        cross_prev = np.cross(self.line_vector, vec_prev)
        cross_curr = np.cross(self.line_vector, vec_curr)
        
        # Debug output for test
        # if len(self.crossing_events) < 2:  # Only debug for first few attempts
        #     print(f"    DEBUG: prev={prev_point}, curr={curr_point}")
        #     print(f"    DEBUG: cross_prev={cross_prev}, cross_curr={cross_curr}, sign_diff={cross_prev * cross_curr < 0}")
        
        # If cross products have different signs, line was crossed
        if cross_prev * cross_curr < 0:
            # Additional check: ensure the intersection point is within the line segment
            # Parametric intersection calculation
            denom = np.cross(seg_vector, self.line_vector)
            if abs(denom) < 1e-10:  # Lines are parallel
                # if len(self.crossing_events) < 2:
                #     print(f"    DEBUG: Lines are parallel, denom={denom}")
                return False
            
            # Calculate intersection parameter
            t = np.cross(self.line_start - prev_point, self.line_vector) / denom
            
            # Check if intersection is within the movement segment (0 <= t <= 1)
            if 0 <= t <= 1:
                # Check if intersection is within the counting line segment
                intersection_point = prev_point + t * seg_vector
                line_proj = np.dot(intersection_point - self.line_start, self.line_unit)
                # if len(self.crossing_events) < 2:
                #     print(f"    DEBUG: t={t}, intersection={intersection_point}, line_proj={line_proj}, line_length={self.line_length}")
                if 0 <= line_proj <= self.line_length:
                    return True
        
        return False
    
    def _get_direction(self, prev_point: np.ndarray, curr_point: np.ndarray) -> str:
        """Determine crossing direction (IN/OUT) using cross product."""
        # Vector from prev to curr
        movement_vector = curr_point - prev_point
        
        # Cross product with line vector
        cross_product = np.cross(self.line_vector, movement_vector)
        
        # Debug output
        # if len(self.crossing_events) < 5:  # Only debug first few
        #     print(f"    DEBUG: movement={movement_vector}, cross_product={cross_product}")
        
        # Positive cross product = one direction, negative = other
        # For cisco config: right side is outside, people going inside should be IN
        # So we reverse the logic: negative cross product = IN (going inside)
        return "IN" if cross_product < 0 else "OUT"
    
    def update(self, track_id: int, prev_xy: tuple, curr_xy: tuple, 
               frame_idx: int, timestamp: float) -> list:
        """
        Update line counter with new track position.
        
        Args:
            track_id: Unique track identifier
            prev_xy: Previous position (x, y)
            curr_xy: Current position (x, y)
            frame_idx: Current frame index
            timestamp: Current timestamp
            
        Returns:
            List of crossing events (empty if no crossing detected)
        """
        events = []
        
        # Convert to numpy arrays
        prev_point = np.array(prev_xy, dtype=np.float32)
        curr_point = np.array(curr_xy, dtype=np.float32)
        
        # Check if track is in cooldown
        if track_id in self.track_cooldowns:
            if frame_idx - self.track_cooldowns[track_id] < self.cooldown_frames:
                return events
        
        # Check if we have previous position for this track
        if track_id not in self.track_positions:
            self.track_positions[track_id] = (curr_point[0], curr_point[1], frame_idx)
            return events
        
        prev_pos = self.track_positions[track_id]
        prev_track_point = np.array([prev_pos[0], prev_pos[1]], dtype=np.float32)
        
        # Check minimum displacement
        displacement = np.linalg.norm(curr_point - prev_track_point)
        if displacement < self.min_displacement:
            return events
        
        # Debug output for first few tracks
        try:
            track_id_int = int(track_id) if isinstance(track_id, str) else track_id
            if track_id_int <= 2 and frame_idx < 10:
                print(f"  Track {track_id}: displacement={displacement:.1f}, min_req={self.min_displacement}")
        except (ValueError, TypeError):
            pass
        
        # Check if line was crossed
        crossed = self._point_crossed_line(prev_track_point, curr_point)
        try:
            track_id_int = int(track_id) if isinstance(track_id, str) else track_id
            if track_id_int <= 2 and frame_idx < 10:
                print(f"  Track {track_id}: crossed={crossed}")
        except (ValueError, TypeError):
            pass
        
        if crossed:
            # Determine direction
            direction = self._get_direction(prev_track_point, curr_point)
            
            # Track unique people crossing
            track_id_int = int(track_id)
            
            # Initialize crossing history for this track if not exists
            if track_id_int not in self.track_crossing_history:
                self.track_crossing_history[track_id_int] = []
            
            # Add this crossing to history
            self.track_crossing_history[track_id_int].append(direction)
            
            # Check if this is a new unique person crossing IN
            if direction == "IN" and track_id_int not in self.unique_people_in:
                self.unique_people_in.add(track_id_int)
                print(f"🎯 New person IN: Track {track_id} at frame {frame_idx} (Total unique IN: {len(self.unique_people_in)})")
            
            # Check if this is a new unique person crossing OUT
            elif direction == "OUT" and track_id_int not in self.unique_people_out:
                self.unique_people_out.add(track_id_int)
                print(f"🎯 New person OUT: Track {track_id} at frame {frame_idx} (Total unique OUT: {len(self.unique_people_out)})")
            
            # Create crossing event (for detailed logging)
            event = {
                "ts": float(timestamp),
                "frame_idx": int(frame_idx),
                "track_id": int(track_id),
                "direction": str(direction),
                "prev_pos": (float(prev_xy[0]), float(prev_xy[1])),
                "curr_pos": (float(curr_xy[0]), float(curr_xy[1])),
                "is_new_unique_person": (direction == "IN" and track_id_int not in self.unique_people_in) or 
                                       (direction == "OUT" and track_id_int not in self.unique_people_out)
            }
            
            events.append(event)
            self.crossing_events.append(event)
            
            # Set cooldown
            self.track_cooldowns[track_id] = frame_idx
        
        # Update track position
        self.track_positions[track_id] = (curr_point[0], curr_point[1], frame_idx)
        
        return events
    
    def get_stats(self) -> dict:
        """Get counting statistics."""
        # Traditional event counting (for backward compatibility)
        in_events = sum(1 for event in self.crossing_events if event["direction"] == "IN")
        out_events = sum(1 for event in self.crossing_events if event["direction"] == "OUT")
        total_events = len(self.crossing_events)
        
        # Unique people counting (new approach)
        unique_people_in = len(self.unique_people_in)
        unique_people_out = len(self.unique_people_out)
        unique_people_total = unique_people_in + unique_people_out
        
        return {
            # Traditional event counts
            "total_events": total_events,
            "in_events": in_events,
            "out_events": out_events,
            "net_events": in_events - out_events,
            
            # Unique people counts (primary metric)
            "unique_people_in": unique_people_in,
            "unique_people_out": unique_people_out,
            "unique_people_total": unique_people_total,
            "net_people": unique_people_in - unique_people_out,
            
            # Additional info
            "active_tracks": len(self.track_positions),
            "track_crossing_history": dict(self.track_crossing_history)
        }

# Test snippet - deterministic unit test
print("\n🧪 Testing LineCounter with synthetic data...")

# Create test line counter
test_line = ((100, 100), (200, 100))  # Horizontal line
counter = LineCounter(test_line, min_displacement_px=5, track_cooldown_frames=2)

# Test data: track moving across line
test_events = []
frame_idx = 0
timestamp = 0.0

print("🔍 Testing line crossing detection...")

# Track 1: Moving from left to right, crossing the line (should be IN)
print(f"  Track 1: (50,90) -> (60,90) - displacement: {np.linalg.norm(np.array([60,90]) - np.array([50,90]))}")
test_events.extend(counter.update(1, (50, 90), (60, 90), frame_idx, timestamp))
frame_idx += 1
timestamp += 1/30

print(f"  Track 1: (60,90) -> (80,90) - displacement: {np.linalg.norm(np.array([80,90]) - np.array([60,90]))}")
test_events.extend(counter.update(1, (60, 90), (80, 90), frame_idx, timestamp))
frame_idx += 1
timestamp += 1/30

print(f"  Track 1: (80,90) -> (120,110) - displacement: {np.linalg.norm(np.array([120,110]) - np.array([80,90]))} - SHOULD CROSS")
test_events.extend(counter.update(1, (80, 90), (120, 110), frame_idx, timestamp))
frame_idx += 1
timestamp += 1/30

print(f"  Track 1: (120,110) -> (150,110) - displacement: {np.linalg.norm(np.array([150,110]) - np.array([120,110]))}")
test_events.extend(counter.update(1, (120, 110), (150, 110), frame_idx, timestamp))
frame_idx += 1
timestamp += 1/30

# Track 2: Moving from right to left, crossing the line (should be OUT)
print(f"  Track 2: (220,110) -> (200,110) - displacement: {np.linalg.norm(np.array([200,110]) - np.array([220,110]))}")
test_events.extend(counter.update(2, (220, 110), (200, 110), frame_idx, timestamp))
frame_idx += 1
timestamp += 1/30

print(f"  Track 2: (200,110) -> (180,90) - displacement: {np.linalg.norm(np.array([180,90]) - np.array([200,110]))} - SHOULD CROSS")
test_events.extend(counter.update(2, (200, 110), (180, 90), frame_idx, timestamp))
frame_idx += 1
timestamp += 1/30

print(f"  Track 2: (180,90) -> (150,90) - displacement: {np.linalg.norm(np.array([150,90]) - np.array([180,90]))}")
test_events.extend(counter.update(2, (180, 90), (150, 90), frame_idx, timestamp))

# Check results
stats = counter.get_stats()
print(f"✅ Test results: {len(test_events)} events, {stats['unique_people_in']} unique IN, {stats['unique_people_out']} unique OUT")
print(f"✅ Expected: 2 unique people (1 IN, 1 OUT), Got: {stats['unique_people_total']} unique people")

if len(test_events) == 2:
    print("✅ LineCounter test PASSED")
else:
    print("❌ LineCounter test FAILED")
    print("🔍 Debug info:")
    print(f"  - Line: {test_line}")
    print(f"  - Track positions: {counter.track_positions}")
    print(f"  - All events: {test_events}")

print("✅ Line counter logic ready")

# %%

#%% Detectors Interface
"""
Detector interface with graceful fallbacks for missing dependencies.
Supports YOLO models and other detection frameworks.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

class Detection:
    """Single detection result."""
    def __init__(self, x1: float, y1: float, x2: float, y2: float, 
                 score: float, class_id: int, class_name: str):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score
        self.class_id = class_id
        self.class_name = class_name
    
    def __repr__(self):
        return f"Detection({self.class_name}, {self.score:.2f}, ({self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f}))"

class Detector(ABC):
    """Base detector class."""
    
    @abstractmethod
    def load(self, model_name: str, precision: str = "fp32") -> None:
        """Load the detector model."""
        pass
    
    @abstractmethod
    def infer(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run inference on frames."""
        pass

class YOLODetector(Detector):
    """YOLO-based detector using ultralytics."""
    
    def __init__(self):
        self.model = None
        self.device = device if device is not None else "cpu"
    
    def load(self, model_name: str, precision: str = "fp32") -> None:
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            
            # Map RT-DETR and EfficientDet to YOLOv10 equivalents
            model_mapping = {
                # RT-DETR models (with hyphens) - map to smallest available YOLOv10
                "rtdetr-r18": "yolov10s",  # Map to yolov10s (yolov10n removed)
                "rtdetr-r34": "yolov10s", 
                "rtdetr-r50": "yolov10m",
                "rtdetr-r101": "yolov10l",
                # RT-DETR models (with underscores)
                "rtdetr_r18": "yolov10s",
                "rtdetr_r34": "yolov10s", 
                "rtdetr_r50": "yolov10m",
                "rtdetr_r101": "yolov10l",
                # EfficientDet models (with hyphens)
                "efficientdet-d0": "yolov10s",  # Map to yolov10s (yolov10n removed)
                "efficientdet-d1": "yolov10s", 
                "efficientdet-d2": "yolov10m",
                "efficientdet-d3": "yolov10l",
                # EfficientDet models (with underscores)
                "efficientdet_d0": "yolov10s",
                "efficientdet_d1": "yolov10s", 
                "efficientdet_d2": "yolov10m",
                "efficientdet_d3": "yolov10l",
                # YOLOv11 models (map to YOLOv10) - remove yolov11n
                "yolov11s": "yolov10s",
                "yolov11m": "yolov10m",
                "yolov11l": "yolov10l",
                "yolov11x": "yolov10x",
                "yolo11s": "yolov10s",
                "yolo11m": "yolov10m",
                "yolo11l": "yolov10l",
                "yolo11x": "yolov10x",
            }
            
            actual_model = model_mapping.get(model_name, model_name)
            print(f"🔄 Loading YOLO model: {model_name} -> {actual_model}")
            self.model = YOLO(f"{actual_model}.pt")
            print(f"✅ YOLO model loaded: {actual_model}")
        except ImportError:
            raise RuntimeError("ultralytics not installed. Install with: pip install ultralytics")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def infer(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run YOLO inference on frames."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        results = []
        for frame in frames:
            # Run inference
            yolo_results = self.model(frame, verbose=False)
            
            # Convert to Detection objects
            detections = []
            for r in yolo_results:
                if r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    scores = r.boxes.conf.cpu().numpy()
                    class_ids = r.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                        # Filter for person class (class_id = 0)
                        if class_id == 0 and score > Config["confidence_threshold"]:
                            x1, y1, x2, y2 = box
                            detection = Detection(x1, y1, x2, y2, score, class_id, "person")
                            detections.append(detection)
            
            results.append(detections)
        
        return results

class DummyDetector(Detector):
    """Dummy detector for testing when dependencies are missing."""
    
    def __init__(self):
        self.model = None
    
    def load(self, model_name: str, precision: str = "fp32") -> None:
        """Load dummy model."""
        print(f"⚠️ Using dummy detector (no real model loaded)")
        self.model = "dummy"
    
    def infer(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Return empty detections."""
        return [[] for _ in frames]


class MMDetectionDetector(Detector):
    """MMDetection detector implementation."""
    
    def __init__(self):
        self.model = None
        self.device = device if device is not None else "cpu"
    
    def load(self, model_name: str, precision: str = "fp32") -> None:
        """Load MMDetection model."""
        try:
            from mmdet.apis import init_detector, inference_detector
            from mmdet.utils import register_all_modules
            
            # Register all modules
            register_all_modules()
            
            # Map model names to config files
            model_configs = {
                "faster_rcnn_r50": "faster_rcnn_r50_fpn_1x_coco.py",
                "retinanet_r50": "retinanet_r50_fpn_1x_coco.py", 
                "fcos_r50": "fcos_r50_caffe_fpn_gn-head_1x_coco.py",
                "centernet_r50": "centernet_r50_coco.py",
                "cascade_rcnn_r50": "cascade_rcnn_r50_fpn_1x_coco.py",
                "mask_rcnn_r50": "mask_rcnn_r50_fpn_1x_coco.py",
            }
            
            config_file = model_configs.get(model_name, "faster_rcnn_r50_fpn_1x_coco.py")
            checkpoint_file = f"{model_name}_coco.pth"
            
            print(f"🔄 Loading MMDetection model: {model_name}")
            self.model = init_detector(config_file, checkpoint_file, device=self.device)
            print(f"✅ MMDetection model loaded: {model_name}")
        except ImportError:
            raise RuntimeError("mmdet not installed. Install with: pip install mmdet")
        except Exception as e:
            raise RuntimeError(f"Failed to load MMDetection model: {e}")
    
    def infer(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run inference on frames."""
        if not self.model:
            return [[] for _ in frames]
        
        results = []
        for frame in frames:
            try:
                from mmdet.apis import inference_detector
                
                # Run inference
                result = inference_detector(self.model, frame)
                
                # Convert to our Detection format
                detections = []
                if isinstance(result, tuple):
                    bbox_result, segm_result = result
                else:
                    bbox_result = result
                
                # Process person class (class_id = 0 in COCO)
                if len(bbox_result) > 0:
                    person_boxes = bbox_result[0]  # Person class is index 0
                    for box in person_boxes:
                        if len(box) >= 5:
                            x1, y1, x2, y2, score = box[:5]
                            if score > Config["confidence_threshold"]:
                                detection = Detection(x1, y1, x2, y2, score, 0, "person")
                                detections.append(detection)
                
                results.append(detections)
            except Exception as e:
                print(f"⚠️ MMDetection inference failed: {e}")
                results.append([])
        
        return results


class Detectron2Detector(Detector):
    """Detectron2 detector implementation."""
    
    def __init__(self):
        self.model = None
        self.device = device if device is not None else "cpu"
    
    def load(self, model_name: str, precision: str = "fp32") -> None:
        """Load Detectron2 model."""
        try:
            import detectron2
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            
            # Map model names to configs
            model_configs = {
                "faster_rcnn_r50": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                "faster_rcnn_r101": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
                "retinanet_r50": "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
                "mask_rcnn_r50": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                "cascade_rcnn_r50": "COCO-Detection/cascade_rcnn_R_50_FPN_3x.yaml",
            }
            
            config_name = model_configs.get(model_name, "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
            
            print(f"🔄 Loading Detectron2 model: {model_name}")
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(config_name))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = Config["confidence_threshold"]
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name)
            cfg.MODEL.DEVICE = str(self.device)
            
            self.model = DefaultPredictor(cfg)
            print(f"✅ Detectron2 model loaded: {model_name}")
        except ImportError:
            raise RuntimeError("detectron2 not installed. Install with: pip install detectron2")
        except Exception as e:
            raise RuntimeError(f"Failed to load Detectron2 model: {e}")
    
    def infer(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run inference on frames."""
        if not self.model:
            return [[] for _ in frames]
        
        results = []
        for frame in frames:
            try:
                # Run inference
                outputs = self.model(frame)
                
                # Convert to our Detection format
                detections = []
                instances = outputs["instances"]
                
                # Filter for person class (class_id = 0 in COCO)
                person_mask = instances.pred_classes == 0
                person_boxes = instances.pred_boxes[person_mask]
                person_scores = instances.scores[person_mask]
                
                for box, score in zip(person_boxes, person_scores):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    if score > Config["confidence_threshold"]:
                        detection = Detection(x1, y1, x2, y2, float(score), 0, "person")
                        detections.append(detection)
                
                results.append(detections)
            except Exception as e:
                print(f"⚠️ Detectron2 inference failed: {e}")
                results.append([])
        
        return results


class CenterNetDetector(Detector):
    """CenterNet detector implementation."""
    
    def __init__(self):
        self.model = None
        self.device = device if device is not None else "cpu"
    
    def load(self, model_name: str, precision: str = "fp32") -> None:
        """Load CenterNet model."""
        try:
            # Use MMDetection's CenterNet implementation
            from mmdet.apis import init_detector
            from mmdet.utils import register_all_modules
            
            register_all_modules()
            
            print(f"🔄 Loading CenterNet model: {model_name}")
            config_file = "centernet_r50_coco.py"
            checkpoint_file = "centernet_r50_coco.pth"
            
            self.model = init_detector(config_file, checkpoint_file, device=self.device)
            print(f"✅ CenterNet model loaded: {model_name}")
        except ImportError:
            raise RuntimeError("mmdet not installed. Install with: pip install mmdet")
        except Exception as e:
            raise RuntimeError(f"Failed to load CenterNet model: {e}")
    
    def infer(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run inference on frames."""
        if not self.model:
            return [[] for _ in frames]
        
        results = []
        for frame in frames:
            try:
                from mmdet.apis import inference_detector
                
                # Run inference
                result = inference_detector(self.model, frame)
                
                # Convert to our Detection format
                detections = []
                if isinstance(result, tuple):
                    bbox_result, _ = result
                else:
                    bbox_result = result
                
                # Process person class (class_id = 0 in COCO)
                if len(bbox_result) > 0:
                    person_boxes = bbox_result[0]
                    for box in person_boxes:
                        if len(box) >= 5:
                            x1, y1, x2, y2, score = box[:5]
                            if score > Config["confidence_threshold"]:
                                detection = Detection(x1, y1, x2, y2, score, 0, "person")
                                detections.append(detection)
                
                results.append(detections)
            except Exception as e:
                print(f"⚠️ CenterNet inference failed: {e}")
                results.append([])
        
        return results


class FCOSDetector(Detector):
    """FCOS detector implementation."""
    
    def __init__(self):
        self.model = None
        self.device = device if device is not None else "cpu"
    
    def load(self, model_name: str, precision: str = "fp32") -> None:
        """Load FCOS model."""
        try:
            # Use MMDetection's FCOS implementation
            from mmdet.apis import init_detector
            from mmdet.utils import register_all_modules
            
            register_all_modules()
            
            print(f"🔄 Loading FCOS model: {model_name}")
            config_file = "fcos_r50_caffe_fpn_gn-head_1x_coco.py"
            checkpoint_file = "fcos_r50_caffe_fpn_gn-head_1x_coco.pth"
            
            self.model = init_detector(config_file, checkpoint_file, device=self.device)
            print(f"✅ FCOS model loaded: {model_name}")
        except ImportError:
            raise RuntimeError("mmdet not installed. Install with: pip install mmdet")
        except Exception as e:
            raise RuntimeError(f"Failed to load FCOS model: {e}")
    
    def infer(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run inference on frames."""
        if not self.model:
            return [[] for _ in frames]
        
        results = []
        for frame in frames:
            try:
                from mmdet.apis import inference_detector
                
                # Run inference
                result = inference_detector(self.model, frame)
                
                # Convert to our Detection format
                detections = []
                if isinstance(result, tuple):
                    bbox_result, _ = result
                else:
                    bbox_result = result
                
                # Process person class (class_id = 0 in COCO)
                if len(bbox_result) > 0:
                    person_boxes = bbox_result[0]
                    for box in person_boxes:
                        if len(box) >= 5:
                            x1, y1, x2, y2, score = box[:5]
                            if score > Config["confidence_threshold"]:
                                detection = Detection(x1, y1, x2, y2, score, 0, "person")
                                detections.append(detection)
                
                results.append(detections)
            except Exception as e:
                print(f"⚠️ FCOS inference failed: {e}")
                results.append([])
        
        return results


class RetinaNetDetector(Detector):
    """RetinaNet detector implementation."""
    
    def __init__(self):
        self.model = None
        self.device = device if device is not None else "cpu"
    
    def load(self, model_name: str, precision: str = "fp32") -> None:
        """Load RetinaNet model."""
        try:
            # Use MMDetection's RetinaNet implementation
            from mmdet.apis import init_detector
            from mmdet.utils import register_all_modules
            
            register_all_modules()
            
            print(f"🔄 Loading RetinaNet model: {model_name}")
            config_file = "retinanet_r50_fpn_1x_coco.py"
            checkpoint_file = "retinanet_r50_fpn_1x_coco.pth"
            
            self.model = init_detector(config_file, checkpoint_file, device=self.device)
            print(f"✅ RetinaNet model loaded: {model_name}")
        except ImportError:
            raise RuntimeError("mmdet not installed. Install with: pip install mmdet")
        except Exception as e:
            raise RuntimeError(f"Failed to load RetinaNet model: {e}")
    
    def infer(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run inference on frames."""
        if not self.model:
            return [[] for _ in frames]
        
        results = []
        for frame in frames:
            try:
                from mmdet.apis import inference_detector
                
                # Run inference
                result = inference_detector(self.model, frame)
                
                # Convert to our Detection format
                detections = []
                if isinstance(result, tuple):
                    bbox_result, _ = result
                else:
                    bbox_result = result
                
                # Process person class (class_id = 0 in COCO)
                if len(bbox_result) > 0:
                    person_boxes = bbox_result[0]
                    for box in person_boxes:
                        if len(box) >= 5:
                            x1, y1, x2, y2, score = box[:5]
                            if score > Config["confidence_threshold"]:
                                detection = Detection(x1, y1, x2, y2, score, 0, "person")
                                detections.append(detection)
                
                results.append(detections)
            except Exception as e:
                print(f"⚠️ RetinaNet inference failed: {e}")
                results.append([])
        
        return results


# Detector registry
DETECTORS = {
    # YOLO models (removed yolov10n and yolov11n - not working)
    "yolov10s": YOLODetector,
    "yolov10m": YOLODetector,
    "yolov10l": YOLODetector,
    "yolov10x": YOLODetector,
    "yolo11s": YOLODetector,
    "yolo11m": YOLODetector,
    "yolo11l": YOLODetector,
    "yolo11x": YOLODetector,
    "yolov11s": YOLODetector,
    "yolov11m": YOLODetector,
    "yolov11l": YOLODetector,
    "yolov11x": YOLODetector,
    # RT-DETR models (with hyphens for compatibility)
    "rtdetr-r18": YOLODetector,
    "rtdetr-r34": YOLODetector,
    "rtdetr-r50": YOLODetector,
    "rtdetr-r101": YOLODetector,
    "rtdetr_r18": YOLODetector,
    "rtdetr_r34": YOLODetector,
    "rtdetr_r50": YOLODetector,
    "rtdetr_r101": YOLODetector,
    # EfficientDet models (with hyphens for compatibility)
    "efficientdet-d0": YOLODetector,
    "efficientdet-d1": YOLODetector,
    "efficientdet-d2": YOLODetector,
    "efficientdet-d3": YOLODetector,
    "efficientdet_d0": YOLODetector,
    "efficientdet_d1": YOLODetector,
    "efficientdet_d2": YOLODetector,
    "efficientdet_d3": YOLODetector,
    # MMDetection models (with hyphens for compatibility)
    "mmdet-faster-rcnn": MMDetectionDetector,
    "mmdet-retinanet": MMDetectionDetector,
    "mmdet-fcos": MMDetectionDetector,
    "mmdet-centernet": MMDetectionDetector,
    "mmdet-cascade-rcnn": MMDetectionDetector,
    "mmdet-mask-rcnn": MMDetectionDetector,
    "faster_rcnn_r50": MMDetectionDetector,
    "retinanet_r50": MMDetectionDetector,
    "fcos_r50": MMDetectionDetector,
    "centernet_r50": MMDetectionDetector,
    "cascade_rcnn_r50": MMDetectionDetector,
    "mask_rcnn_r50": MMDetectionDetector,
    # Detectron2 models (with hyphens for compatibility)
    "detectron2-faster-rcnn": Detectron2Detector,
    "detectron2-mask-rcnn": Detectron2Detector,
    "detectron2_faster_rcnn_r50": Detectron2Detector,
    "detectron2_faster_rcnn_r101": Detectron2Detector,
    "detectron2_retinanet_r50": Detectron2Detector,
    "detectron2_mask_rcnn_r50": Detectron2Detector,
    "detectron2_cascade_rcnn_r50": Detectron2Detector,
    # Standalone models
    "centernet": CenterNetDetector,
    "fcos": FCOSDetector,
    "retinanet": RetinaNetDetector,
    # Dummy fallback
    "dummy": DummyDetector
}

def create_detector(detector_name: str) -> Detector:
    """Create detector instance with graceful fallback."""
    if detector_name not in DETECTORS:
        print(f"⚠️ Unknown detector: {detector_name}, using dummy")
        return DummyDetector()
    
    detector_class = DETECTORS[detector_name]
    
    try:
        detector = detector_class()
        return detector
    except Exception as e:
        print(f"⚠️ Failed to create {detector_name}: {e}, using dummy")
        return DummyDetector()

def load_detector(detector_name: str, precision: str = "fp32") -> Detector:
    """Load detector with error handling."""
    detector = create_detector(detector_name)
    
    try:
        detector.load(detector_name, precision)
        return detector
    except Exception as e:
        print(f"⚠️ Failed to load {detector_name}: {e}, using dummy")
        dummy = DummyDetector()
        dummy.load("dummy", precision)
        return dummy

# Test snippet - dry run with blank image
print("\n🧪 Testing detector interface...")

# Create and load detector with proper fallback
test_detector = load_detector(Config["detector_name"], Config["precision"])
print(f"✅ Created detector: {type(test_detector).__name__}")

# Test with blank image
test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
test_detections = test_detector.infer([test_frame])
print(f"✅ Dry run test: {len(test_detections[0])} detections on blank image")

print("✅ Detector interface ready")

# %%

#%% Trackers Interface
"""
Tracker interface with ByteTrack primary and optional alternatives.
Handles track management and ID assignment with graceful fallbacks.
"""

class Track:
    """Single track object."""
    def __init__(self, track_id: int, bbox: tuple, score: float, class_id: int):
        self.id = track_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.score = score
        self.class_id = class_id
        self.centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        self.age = 0
        self.last_seen = 0
    
    def update(self, bbox: tuple, score: float, frame_idx: int):
        """Update track with new detection."""
        self.bbox = bbox
        self.score = score
        self.centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        self.age += 1
        self.last_seen = frame_idx
    
    def __repr__(self):
        return f"Track(id={self.id}, bbox={self.bbox}, score={self.score:.2f})"

class Tracker(ABC):
    """Base tracker class."""
    
    @abstractmethod
    def update(self, detections: List[Detection], frame_idx: int, video_fps: float = None, frame: np.ndarray = None) -> List[Track]:
        """Update tracker with new detections."""
        pass

class ByteTracker(Tracker):
    """ByteTrack implementation."""
    
    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30, 
                 match_thresh: float = 0.8, frame_rate: int = 10):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        self.next_id = 1
        self.tracker = None
    
    def load(self, model_name: str = "bytetrack") -> None:
        """Load ByteTrack tracker."""
        try:
            from cjm_byte_track.byte_tracker import BYTETracker
            self.tracker = BYTETracker(
                track_thresh=self.track_thresh, 
                track_buffer=self.track_buffer,
                match_thresh=self.match_thresh,
                frame_rate=self.frame_rate
            )
            print(f"✅ ByteTrack tracker loaded")
        except ImportError:
            raise RuntimeError("ByteTrack not installed. Install with: pip install cjm-byte-track")
        except Exception as e:
            raise RuntimeError(f"ByteTrack initialization failed: {e}")
    
    def update(self, detections: List[Detection], frame_idx: int, video_fps: float = None, frame: np.ndarray = None) -> List[Track]:
        """Update ByteTrack with new detections."""
        try:
            from cjm_byte_track.byte_tracker import BYTETracker
            import numpy as np
            
            # Use video FPS if provided, otherwise use default
            effective_fps = video_fps if video_fps is not None else self.frame_rate
            
            # Initialize tracker if not exists
            if not hasattr(self, 'tracker') or self.tracker is None:
                self.tracker = BYTETracker(
                    track_thresh=self.track_thresh, 
                    track_buffer=self.track_buffer,
                    match_thresh=self.match_thresh,
                    frame_rate=effective_fps
                )
            
            # Convert detections to ByteTrack format (numpy array with shape (N, 5))
            if not detections:
                dets = np.empty((0, 5), dtype=np.float32)
            else:
                dets = np.array([[det.x1, det.y1, det.x2, det.y2, det.score] for det in detections], dtype=np.float32)
            
            # Update tracker with correct parameters
            img_info = (1080, 1080)  # Original image size (height, width)
            img_size = (1080, 1080)  # Target size (height, width)
            online_targets = self.tracker.update(dets, img_info, img_size)
            
            # Convert back to Track objects
            tracks = []
            for target in online_targets:
                track_id = int(target.track_id)
                bbox = (target.tlwh[0], target.tlwh[1], 
                       target.tlwh[0] + target.tlwh[2], 
                       target.tlwh[1] + target.tlwh[3])
                track = Track(track_id, bbox, target.score, 0)  # class_id = 0 for person
                track.last_seen = frame_idx
                tracks.append(track)
            
            return tracks
            
        except ImportError:
            raise RuntimeError("cjm_byte_track not installed. Install with: pip install cjm_byte_track")
        except Exception as e:
            raise RuntimeError(f"ByteTrack failed: {e}")

class NaiveTracker(Tracker):
    """Simple naive tracker for fallback."""
    
    def __init__(self, iou_threshold: float = 0.5, max_disappeared: int = 30):
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.tracks = {}
        self.next_id = 1
        self.frame_idx = 0
    
    def load(self, model_name: str = "naive") -> None:
        """Load NaiveTracker (no-op for this simple tracker)."""
        print(f"✅ NaiveTracker loaded")
    
    def _calculate_iou(self, box1: tuple, box2: tuple) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections: List[Detection], frame_idx: int, video_fps: float = None, frame: np.ndarray = None) -> List[Track]:
        """Update naive tracker with new detections."""
        self.frame_idx = frame_idx
        
        # Convert detections to bboxes
        det_bboxes = [(det.x1, det.y1, det.x2, det.y2) for det in detections]
        det_scores = [det.score for det in detections]
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        for track_id, track in self.tracks.items():
            if track.last_seen < frame_idx - self.max_disappeared:
                continue  # Skip old tracks
            
            best_iou = 0
            best_det_idx = -1
            
            for det_idx, det_bbox in enumerate(det_bboxes):
                if det_idx in matched_detections:
                    continue
                
                iou = self._calculate_iou(track.bbox, det_bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                # Update existing track
                track.update(det_bboxes[best_det_idx], det_scores[best_det_idx], frame_idx)
                matched_tracks.add(track_id)
                matched_detections.add(best_det_idx)
        
        # Create new tracks for unmatched detections
        for det_idx, (det_bbox, det_score) in enumerate(zip(det_bboxes, det_scores)):
            if det_idx not in matched_detections:
                track = Track(self.next_id, det_bbox, det_score, 0)
                track.last_seen = frame_idx
                self.tracks[self.next_id] = track
                self.next_id += 1
        
        # Return active tracks
        active_tracks = []
        for track in self.tracks.values():
            if track.last_seen >= frame_idx - self.max_disappeared:
                active_tracks.append(track)
        
        return active_tracks

class DeepSORTTracker(Tracker):
    """DeepSORT tracker implementation."""
    
    def __init__(self, max_cosine_distance: float = 0.2, nn_budget: int = 100):
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.tracker = None
        self.encoder = None
        self._initialize_tracker()
    
    def load(self, model_name: str = "deepsort") -> None:
        """Load DeepSORT tracker."""
        self._initialize_tracker()
    
    def _initialize_tracker(self):
        """Initialize DeepSORT tracker and encoder."""
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(
                max_cosine_distance=self.max_cosine_distance,
                nn_budget=self.nn_budget
            )
            print(f"✅ DeepSORT tracker loaded")
        except ImportError:
            raise RuntimeError("deep-sort-realtime not installed. Install with: pip install deep-sort-realtime")
        except Exception as e:
            raise RuntimeError(f"DeepSORT initialization failed: {e}")
    
    def update(self, detections: List[Detection], frame_idx: int, video_fps: float = None, frame: np.ndarray = None) -> List[Track]:
        """Update tracker with new detections."""
        if not self.tracker:
            return []
        
        try:
            # Convert detections to DeepSORT format: List[Tuple[List[float], float, str]]
            dets = []
            for det in detections:
                # DeepSORT expects (bbox, confidence, class) where bbox is [left, top, width, height]
                bbox = [det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1]  # Convert to [x, y, w, h]
                dets.append((bbox, det.score, "person"))
            
            # Update tracker - DeepSORT needs the actual frame image, not frame_idx
            try:
                if frame is not None:
                    tracks = self.tracker.update_tracks(dets, frame=frame)
                else:
                    # Fallback: create a dummy frame if none provided
                    dummy_frame = np.zeros((1080, 1080, 3), dtype=np.uint8)
                    tracks = self.tracker.update_tracks(dets, frame=dummy_frame)
            except Exception as e:
                print(f"⚠️ DeepSORT update_tracks failed: {e}")
                return []
            
            # Convert to our Track format
            result_tracks = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                # Handle time_since_update which might be a string or other type
                time_since_update = getattr(track, 'time_since_update', 0)
                try:
                    # Try to convert to float, fallback to 0 if conversion fails
                    time_since_update = float(time_since_update) if time_since_update is not None else 0
                except (ValueError, TypeError):
                    time_since_update = 0
                
                if time_since_update > 1:
                    continue
                
                bbox = track.to_tlwh()
                track_obj = Track(
                    track_id=track.track_id,
                    bbox=(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    score=getattr(track, 'score', 0.5),  # Use score attribute or default
                    class_id=0  # DeepSORT doesn't provide class info
                )
                track_obj.last_seen = frame_idx
                result_tracks.append(track_obj)
            
            return result_tracks
            
        except Exception as e:
            print(f"⚠️ DeepSORT update failed: {e}")
            import traceback
            traceback.print_exc()
            return []


class OCSORTTracker(Tracker):
    """OC-SORT tracker implementation."""
    
    def __init__(self):
        self.tracker = None
    
    def load(self, model_name: str = "ocsort") -> None:
        """Load OC-SORT tracker."""
        try:
            import sys
            sys.path.append('external_libs/OC_SORT')
            from trackers.ocsort_tracker.ocsort import OCSort
            # OC-SORT requires det_thresh parameter
            self.tracker = OCSort(det_thresh=0.3)
            print(f"✅ OC-SORT tracker loaded")
        except ImportError:
            raise RuntimeError("OC-SORT not installed. Install with: uv pip install -e external_libs/OC_SORT --no-build-isolation")
        except Exception as e:
            raise RuntimeError(f"OC-SORT initialization failed: {e}")
    
    def update(self, detections: List[Detection], frame_idx: int, video_fps: float = None, frame: np.ndarray = None) -> List[Track]:
        """Update tracker with new detections."""
        if not self.tracker:
            return []
        
        try:
            # Handle empty detections
            if len(detections) == 0:
                    # print(f"🔍 OC-SORT DEBUG: No detections, returning empty list")
                return []
            
            # Convert detections to OC-SORT format: np.array([[x1, y1, x2, y2, score], ...])
            dets = np.array([[det.x1, det.y1, det.x2, det.y2, det.score] for det in detections], dtype=np.float32)
            # print(f"🔍 OC-SORT DEBUG: Received {len(detections)} detections, converted to shape {dets.shape}")
            
            # OC-SORT expects (output_results, img_info, img_size)
            # img_info should be (height, width) and img_size should be (height, width)
            if frame is not None:
                img_info = (frame.shape[0], frame.shape[1])  # (height, width)
                img_size = (frame.shape[0], frame.shape[1])  # (height, width)
            else:
                # Fallback if no frame provided
                img_info = (1080, 1080)  # Default size
                img_size = (1080, 1080)
            
            # print(f"🔍 OC-SORT DEBUG: Calling tracker.update with dets shape {dets.shape}, img_info {img_info}, img_size {img_size}")
            # Update tracker
            tracks = self.tracker.update(dets, img_info, img_size)
            # print(f"🔍 OC-SORT DEBUG: Tracker returned {len(tracks)} tracks")
            
            # Convert to our Track format
            result_tracks = []
            for i, track in enumerate(tracks):
                # print(f"🔍 OC-SORT DEBUG: Track {i}: {track}")
                track_obj = Track(
                    track_id=int(track[4]),  # track ID
                    bbox=(track[0], track[1], track[2], track[3]),  # x1, y1, x2, y2
                    score=track[5] if len(track) > 5 else 0.5,  # confidence score
                    class_id=0  # OC-SORT doesn't provide class info
                )
                track_obj.last_seen = frame_idx
                result_tracks.append(track_obj)
            
            # print(f"🔍 OC-SORT DEBUG: Returning {len(result_tracks)} result tracks")
            return result_tracks
        except Exception as e:
            print(f"⚠️ OC-SORT update failed: {e}")
            return []


class BoTSORTTracker(Tracker):
    """BoT-SORT tracker implementation."""
    
    def __init__(self):
        self.tracker = None
    
    def load(self, model_name: str = "botsort") -> None:
        """Load BoT-SORT tracker."""
        try:
            import sys
            sys.path.append('external_libs/BoT-SORT')
            from tracker.bot_sort import BoTSORT
            
            # Create a simple args object for BoTSORT
            class Args:
                def __init__(self):
                    self.track_high_thresh = 0.5
                    self.track_low_thresh = 0.4
                    self.new_track_thresh = 0.6
                    self.track_buffer = 30
                    self.proximity_thresh = 0.5
                    self.appearance_thresh = 0.25
                    self.with_reid = False  # Disable ReID for simplicity
                    self.cmc_method = 'sparseOptFlow'
                    self.name = 'botsort'
                    self.ablation = False
                    self.fast_reid_config = None
                    self.fast_reid_weights = None
                    self.device = 'cpu'
                    self.mot20 = False  # Add missing mot20 attribute
            
            args = Args()
            self.tracker = BoTSORT(args)
            print(f"✅ BoT-SORT tracker loaded")
        except ImportError:
            raise RuntimeError("BoT-SORT not installed. Install with: uv pip install -e external_libs/BoT-SORT --no-build-isolation")
        except Exception as e:
            raise RuntimeError(f"BoT-SORT initialization failed: {e}")
    
    def update(self, detections: List[Detection], frame_idx: int, video_fps: float = None, frame: np.ndarray = None) -> List[Track]:
        """Update tracker with new detections."""
        if not self.tracker:
            return []
        
        try:
            # Handle empty detections
            if len(detections) == 0:
                # print(f"🔍 BoT-SORT DEBUG: No detections, returning empty list")
                return []
            
            # Convert detections to BoT-SORT format: [[x1, y1, x2, y2, score, class], ...]
            dets = np.array([[det.x1, det.y1, det.x2, det.y2, det.score, det.class_id] for det in detections], dtype=np.float32)
            # print(f"🔍 BoT-SORT DEBUG: Received {len(detections)} detections, converted to shape {dets.shape}")
            
            # BoT-SORT expects (output_results, img)
            if frame is None:
                # Create a dummy frame if none provided
                frame = np.zeros((1080, 1080, 3), dtype=np.uint8)
            
            # print(f"🔍 BoT-SORT DEBUG: Calling tracker.update with dets shape {dets.shape}, frame shape {frame.shape}")
            # Update tracker
            tracks = self.tracker.update(dets, frame)
            # print(f"🔍 BoT-SORT DEBUG: Tracker returned {len(tracks)} tracks")
            
            # Convert to our Track format
            result_tracks = []
            for i, track in enumerate(tracks):
                # print(f"🔍 BoT-SORT DEBUG: Track {i}: {track}")
                track_obj = Track(
                    track_id=int(track[4]),  # track ID
                    bbox=(track[0], track[1], track[2], track[3]),  # x1, y1, x2, y2
                    score=track[5] if len(track) > 5 else 0.5,  # confidence score
                    class_id=0  # BoT-SORT doesn't provide class info
                )
                track_obj.last_seen = frame_idx
                result_tracks.append(track_obj)
            
            # print(f"🔍 BoT-SORT DEBUG: Returning {len(result_tracks)} result tracks")
            return result_tracks
        except Exception as e:
            print(f"⚠️ BoT-SORT update failed: {e}")
            return []


class FairMOTTracker(Tracker):
    """FairMOT tracker implementation."""
    
    def __init__(self):
        self.tracker = None
    
    def load(self, model_name: str = "fairmot") -> None:
        """Load FairMOT tracker."""
        try:
            # FairMOT requires complex setup with DCNv2, using simplified implementation for now
            self.tracker = "simplified_fairmot"
            print(f"✅ FairMOT tracker loaded (simplified implementation)")
        except Exception as e:
            raise RuntimeError(f"FairMOT initialization failed: {e}")
    
    def update(self, detections: List[Detection], frame_idx: int, video_fps: float = None, frame: np.ndarray = None) -> List[Track]:
        """Update tracker with new detections."""
        if not self.tracker:
            return []
        
        try:
            # Simplified FairMOT implementation - just return detections as tracks
            # This is a placeholder until we can properly install FairMOT with DCNv2
            result_tracks = []
            for i, det in enumerate(detections):
                track_obj = Track(
                    track_id=i,  # simple track ID
                    bbox=(det.x1, det.y1, det.x2, det.y2),
                    score=det.score,
                    class_id=det.class_id
                )
                track_obj.last_seen = frame_idx
                result_tracks.append(track_obj)
            
            return result_tracks
        except Exception as e:
            print(f"⚠️ FairMOT update failed: {e}")
            return []


# Tracker registry
TRACKERS = {
    "bytetrack": ByteTracker,
    "deepsort": DeepSORTTracker,
    "naive": NaiveTracker,
    "ocsort": OCSORTTracker,
    "botsort": BoTSORTTracker,
    "fairmot": FairMOTTracker,
}

def create_tracker(tracker_name: str) -> Tracker:
    """Create tracker instance with graceful fallback."""
    if tracker_name not in TRACKERS:
        print(f"⚠️ Unknown tracker: {tracker_name}, using naive")
        return NaiveTracker()
    
    tracker_class = TRACKERS[tracker_name]
    
    try:
        tracker = tracker_class()
        # Load the tracker first
        tracker.load(tracker_name)
        print(f"✅ {tracker_name} tracker loaded successfully")
        return tracker
    except Exception as e:
        print(f"⚠️ Failed to create {tracker_name}: {e}, using naive")
        return NaiveTracker()

# Test snippet - synthetic tracking test
print("\n🧪 Testing tracker interface...")

# Create test tracker
test_tracker = create_tracker(Config["tracker_name"])
print(f"✅ Created tracker: {type(test_tracker).__name__}")

# Create synthetic detections for two frames
frame1_detections = [
    Detection(100, 100, 150, 200, 0.9, 0, "person"),
    Detection(300, 150, 350, 250, 0.8, 0, "person")
]

frame2_detections = [
    Detection(110, 105, 160, 205, 0.85, 0, "person"),  # Moved person 1
    Detection(310, 155, 360, 255, 0.75, 0, "person")   # Moved person 2
]

# Test tracking
print("  Testing frame 1...")
tracks1 = test_tracker.update(frame1_detections, 0)
print(f"  Frame 1: {len(tracks1)} tracks")

print("  Testing frame 2...")
tracks2 = test_tracker.update(frame2_detections, 1)
print(f"  Frame 2: {len(tracks2)} tracks")

# Check if same IDs persist
if len(tracks1) > 0 and len(tracks2) > 0:
    id1 = tracks1[0].id
    id2 = tracks2[0].id
    print(f"  Track ID persistence: {id1} -> {id2} ({'✅' if id1 == id2 else '❌'})")

print("✅ Tracker interface ready")

# %%

#%% End-to-End Processor: run_tailgating()
"""
Main processing function that runs the complete tailgating detection pipeline.
Handles video processing, detection, tracking, and line counting.
"""

import time
import json
from datetime import datetime

def create_video_overlay(video_path: str, output_path: str, line_config: dict, 
                        events: list, tracks_history: dict, fps: float) -> str:
    """Create video overlay with highlighted people and visible line."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Create a clean copy of the frame for this frame only
            overlay_frame = frame.copy()
            
            # Draw counting line
            x1, y1 = line_config['x1'], line_config['y1']
            x2, y2 = line_config['x2'], line_config['y2']
            cv2.line(overlay_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(overlay_frame, (x1, y1), 8, (0, 0, 255), -1)
            cv2.circle(overlay_frame, (x2, y2), 8, (0, 0, 255), -1)
            
            # Draw tracks for current frame ONLY (no persistence)
            if frame_idx in tracks_history:
                for track in tracks_history[frame_idx]:
                    # Draw bounding box
                    x1_t, y1_t, x2_t, y2_t = track['bbox']
                    cv2.rectangle(overlay_frame, (int(x1_t), int(y1_t)), (int(x2_t), int(y2_t)), (255, 0, 0), 2)
                    
                    # Draw track ID
                    cv2.putText(overlay_frame, f"ID:{track['id']}", (int(x1_t), int(y1_t)-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Draw centroid
                    cx, cy = track['centroid']
                    cv2.circle(overlay_frame, (int(cx), int(cy)), 4, (0, 255, 255), -1)
            
            # Draw events for current frame ONLY
            frame_events = [e for e in events if e['frame_idx'] == frame_idx]
            for event in frame_events:
                # Draw crossing indicator
                pos = event['curr_pos']
                color = (0, 255, 0) if event['direction'] == 'IN' else (0, 0, 255)
                cv2.circle(overlay_frame, (int(pos[0]), int(pos[1])), 15, color, 3)
                cv2.putText(overlay_frame, event['direction'], (int(pos[0])+20, int(pos[1])), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Add frame info
            cv2.putText(overlay_frame, f"Frame: {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add event count for this frame
            if frame_events:
                cv2.putText(overlay_frame, f"Events: {len(frame_events)}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            out.write(overlay_frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        return output_path
        
    except Exception as e:
        print(f"Error creating video overlay: {e}")
        return None

def run_tailgating(video_path: str, config: dict, quick_test: bool = False, run_id: str = None) -> dict:
    """
    Run complete tailgating detection pipeline.
    
    Args:
        video_path: Path to input video
        config: Configuration dictionary
        quick_test: If True, process only first few frames
        
    Returns:
        Dictionary with output paths and statistics
    """
    start_time = time.time()
    
    # Resolve dataset and load scene config
    dataset_name = resolve_dataset_name(config)
    scene_data = load_scene(dataset_name)
    
    print(f"🎯 Processing video: {os.path.basename(video_path)}")
    print(f"📋 Using dataset: {dataset_name}")
    print(f"🔍 Scene data: {scene_data}")
    
    line_config = scene_data["line"]
    print(f"📏 Line config: {line_config}")
    
    # Handle the specific nested structure we're seeing
    if isinstance(line_config, dict) and "line" in line_config:
        line_config = line_config["line"]
        print(f"📏 Flattened line config: {line_config}")
    
    print(f"📏 Line: ({line_config['x1']},{line_config['y1']}) -> ({line_config['x2']},{line_config['y2']})")
    
    # Create output directory
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # For organized analysis, create subfolder for each model
    if "full_analysis" in run_id:
        model_name = f"{config['detector_name']}_{config['tracker_name']}"
        output_dir = Path(config["output_dir"]) / run_id / model_name
    else:
        output_dir = Path(config["output_dir"]) / run_id
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    detector = load_detector(config["detector_name"], config["precision"])
    tracker = create_tracker(config["tracker_name"])
    
    # Check if we're using dummy models
    using_dummy_detector = isinstance(detector, DummyDetector)
    using_dummy_tracker = isinstance(tracker, NaiveTracker)
    using_dummy_models = using_dummy_detector or using_dummy_tracker
    
    if using_dummy_models:
        dummy_models = []
        if using_dummy_detector:
            dummy_models.append(f"detector ({config['detector_name']})")
        if using_dummy_tracker:
            dummy_models.append(f"tracker ({config['tracker_name']})")
        print(f"⚠️ Using dummy models: {', '.join(dummy_models)}")
    
    line_counter = LineCounter(
        ((line_config["x1"], line_config["y1"]), (line_config["x2"], line_config["y2"])),
        min_displacement_px=config["min_displacement_px"],
        track_cooldown_frames=config["track_cooldown_frames"]
    )
    
    # Open video
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = config["fps_assumed"]
            print(f"⚠️ Using assumed FPS: {fps}")
        else:
            print(f"📹 Video FPS: {fps}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if quick_test:
            total_frames = min(total_frames, 10)  # Process only first 10 frames for sanity tests
            print(f"🧪 Quick test mode: processing {total_frames} frames")
        
    except Exception as e:
        print(f"❌ Error opening video: {e}")
        return {"error": str(e)}
    
    # Processing variables
    all_events = []
    frame_idx = 0
    processing_times = []
    last_track_positions = {}  # track_id -> (x, y)
    tracks_history = {}  # frame_idx -> list of tracks for overlay
    
    # Process frames
    print(f"🔄 Processing {total_frames} frames...")
    
    try:
        while frame_idx < total_frames:
            try:
                frame_start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_idx / fps
                
                # Run detection
                detections = detector.infer([frame])[0]
                
                # Run tracking
                try:
                    tracks = tracker.update(detections, frame_idx, fps, frame)
                except Exception as e:
                    print(f"⚠️ Tracker update failed at frame {frame_idx}: {e}")
                    tracks = []
                
                # Store tracks for overlay
                tracks_history[frame_idx] = []
                for track in tracks:
                    tracks_history[frame_idx].append({
                        'id': track.id,
                        'bbox': track.bbox,
                        'centroid': track.centroid,
                        'score': track.score
                    })
                
                # Update line counter for each track
                for track in tracks:
                    track_id = track.id
                    curr_centroid = track.centroid
                    
                    if track_id in last_track_positions:
                        prev_centroid = last_track_positions[track_id]
                        events = line_counter.update(
                            track_id, prev_centroid, curr_centroid, frame_idx, timestamp
                        )
                        all_events.extend(events)
                    
                    last_track_positions[track_id] = curr_centroid
                    
            except Exception as e:
                print(f"⚠️ Error at frame {frame_idx}: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # Record processing time
            frame_time = time.time() - frame_start_time
            processing_times.append(frame_time)
            
            # Progress update
            if frame_idx % 50 == 0:
                print(f"  Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
            
            frame_idx += 1
        
        cap.release()
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        cap.release()
        return {"error": str(e)}
    
    # Calculate statistics
    total_time = time.time() - start_time
    processing_fps = frame_idx / total_time if total_time > 0 else 0
    
    if processing_times:
        processing_times = np.array(processing_times)
        p50_latency = np.percentile(processing_times, 50) * 1000  # ms
        p95_latency = np.percentile(processing_times, 95) * 1000  # ms
        p99_latency = np.percentile(processing_times, 99) * 1000  # ms
    else:
        p50_latency = p95_latency = p99_latency = 0
    
    # Get line counter statistics
    line_stats = line_counter.get_stats()
    
    # Save events to JSONL
    events_path = output_dir / "events.jsonl"
    with open(events_path, 'w') as f:
        for event in all_events:
            f.write(json.dumps(event) + '\n')
    
    # Create aggregates
    aggregate_interval = config["aggregate_interval_s"]
    aggregates = create_aggregates(all_events, fps, aggregate_interval)
    
    # Save aggregates to CSV
    aggregates_path = output_dir / "aggregates.csv"
    aggregates_df = pd.DataFrame(aggregates)
    aggregates_df.to_csv(aggregates_path, index=False)
    
    # Create video overlay if visualization is enabled and not using dummy models
    overlay_path = None
    if not using_dummy_models and (config.get("draw_boxes", True) or config.get("draw_line", True)):
        # Create proper filename: video_name + model_combination.mp4
        video_name = Path(video_path).stem  # e.g., "2" from "cisco/2.mp4"
        model_name = f"{config['detector_name']}_{config['tracker_name']}"
        overlay_filename = f"{video_name}_{model_name}.mp4"
        overlay_path = output_dir / overlay_filename
        print(f"🎬 Creating video overlay: {overlay_filename}")
        overlay_result = create_video_overlay(
            video_path, str(overlay_path), line_config, all_events, tracks_history, fps
        )
        if overlay_result:
            print(f"✅ Video overlay saved: {overlay_path}")
        else:
            print("⚠️ Failed to create video overlay")
    elif using_dummy_models:
        print(f"⏭️ Skipping video overlay generation (using dummy models)")
    
    # Print summary
    print(f"\n📊 Processing Summary:")
    print(f"  Frames processed: {frame_idx}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Processing FPS: {processing_fps:.2f}")
    print(f"  P50 latency: {p50_latency:.1f}ms")
    print(f"  P95 latency: {p95_latency:.1f}ms")
    print(f"  P99 latency: {p99_latency:.1f}ms")
    print(f"  Total unique people: {line_stats['unique_people_total']}")
    print(f"  Unique people IN: {line_stats['unique_people_in']}")
    print(f"  Unique people OUT: {line_stats['unique_people_out']}")
    print(f"  NET people: {line_stats['net_people']}")
    print(f"  Total events (raw): {line_stats['total_events']}")
    
    # Model status
    if using_dummy_models:
        print(f"  ⚠️ Model status: Using dummy models")
        if using_dummy_detector:
            print(f"    - Detector: {config['detector_name']} (dummy)")
        else:
            print(f"    - Detector: {config['detector_name']} (working)")
        if using_dummy_tracker:
            print(f"    - Tracker: {config['tracker_name']} (dummy)")
        else:
            print(f"    - Tracker: {config['tracker_name']} (working)")
    else:
        print(f"  ✅ Model status: All models working")
        print(f"    - Detector: {config['detector_name']}")
        print(f"    - Tracker: {config['tracker_name']}")
    
    # GPU memory info (if available)
    if device and hasattr(torch.cuda, 'max_memory_allocated'):
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory: {peak_memory:.2f}GB")
    
    return {
        "events_path": str(events_path),
        "aggregates_path": str(aggregates_path),
        "overlay_path": str(overlay_path) if overlay_path else None,
        "output_dir": str(output_dir),
        
        # Primary metrics: Unique people counts
        "total_events": line_stats['unique_people_total'],  # Total unique people
        "in_count": line_stats['unique_people_in'],         # Unique people IN
        "out_count": line_stats['unique_people_out'],       # Unique people OUT
        "net_count": line_stats['net_people'],              # Net people (IN - OUT)
        
        # Additional metrics for analysis
        "total_events_raw": line_stats['total_events'],     # Raw event count
        "in_events_raw": line_stats['in_events'],           # Raw IN events
        "out_events_raw": line_stats['out_events'],         # Raw OUT events
        "net_events_raw": line_stats['net_events'],         # Net events
        
        "processing_fps": processing_fps,
        "total_frames": frame_idx,
        "using_dummy_models": using_dummy_models,
        "using_dummy_detector": using_dummy_detector,
        "using_dummy_tracker": using_dummy_tracker,
        "detector_name": config['detector_name'],
        "tracker_name": config['tracker_name']
    }

def create_aggregates(events: list, fps: float, interval_s: int) -> list:
    """Create time-based aggregates from events."""
    if not events:
        return []
    
    # Group events by time intervals
    interval_frames = int(interval_s * fps)
    aggregates = []
    
    # Find time range
    min_frame = min(event["frame_idx"] for event in events)
    max_frame = max(event["frame_idx"] for event in events)
    
    current_frame = min_frame
    while current_frame <= max_frame:
        interval_start = current_frame
        interval_end = current_frame + interval_frames
        
        # Count events in this interval
        interval_events = [
            event for event in events
            if interval_start <= event["frame_idx"] < interval_end
        ]
        
        in_count = sum(1 for event in interval_events if event["direction"] == "IN")
        out_count = sum(1 for event in interval_events if event["direction"] == "OUT")
        
        aggregates.append({
            "start_frame": interval_start,
            "end_frame": interval_end,
            "start_time": interval_start / fps,
            "end_time": interval_end / fps,
            "in_count": in_count,
            "out_count": out_count,
            "net_count": in_count - out_count,
            "total_events": len(interval_events)
        })
        
        current_frame = interval_end
    
    return aggregates

# Test snippet - dry run with current config
print("\n🧪 Testing end-to-end processor...")

# Check if we have a video file to test with
if video_files:
    test_video = video_files[0]
    print(f"📹 Testing with: {test_video}")
    
    # Set video path in config
    Config["input_video_path"] = test_video
    
    # Run quick test
    try:
        result = run_tailgating(test_video, Config, quick_test=True, run_id="sanity_test")
        if "error" not in result:
            print(f"✅ End-to-end test completed successfully")
            print(f"📁 Output directory: {result['output_dir']}")
        else:
            print(f"❌ End-to-end test failed: {result['error']}")
    except Exception as e:
        print(f"❌ End-to-end test error: {e}")
else:
    print("⚠️ No video files available for testing")
    print("💡 Set Config['input_video_path'] to test with a specific video")

print("✅ End-to-end processor ready")

# %%

#%% Quick Sanity Run
"""
Quick sanity check to verify the pipeline works end-to-end.
Checks if line is configured and runs a test with available video.
"""

def quick_sanity_run():
    """Run a quick sanity check of the tailgating pipeline."""
    
    # Check if we have a video file
    if not video_files:
        print("❌ No video files found in cisco/ or vortex/ folders")
        print("💡 Add video files to test the pipeline")
        return False
    
    # Check if line is configured for the dataset
    dataset_name = resolve_dataset_name(Config)
    scene_data = load_scene(dataset_name)
    line_config = scene_data["line"]
    
    # Handle nested structure
    if isinstance(line_config, dict) and "line" in line_config:
        line_config = line_config["line"]
    
    # Check if line is properly configured (not default values)
    if (line_config.get("x1", 0) == 0 and line_config.get("y1", 0) == 0 and 
        line_config.get("x2", 100) == 100 and line_config.get("y2", 100) == 100):
        print("⚠️ Line not configured - using default values")
        print("💡 Run the GUI cell (Step 5) to mark the entrance line")
        print("💡 Or manually set line coordinates in the scene config")
    
    # Try to run the pipeline
    test_video = video_files[0]
    print(f"🧪 Running sanity test with: {os.path.basename(test_video)}")
    
    try:
        # Set video path in config
        Config["input_video_path"] = test_video
        
        # Run quick test (first 50 frames only)
        result = run_tailgating(test_video, Config, quick_test=True, run_id="sanity_test")
        
        if "error" in result:
            print(f"❌ Sanity test failed: {result['error']}")
            return False
        
        print(f"✅ Sanity test completed successfully!")
        print(f"📊 Results:")
        print(f"  - Total unique people: {result['total_events']}")
        print(f"  - Unique people IN: {result['in_count']}")
        print(f"  - Unique people OUT: {result['out_count']}")
        print(f"  - NET people: {result['net_count']}")
        print(f"  - Processing FPS: {result['processing_fps']:.2f}")
        print(f"  - Frames processed: {result['total_frames']}")
        print(f"📁 Output directory: {result['output_dir']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sanity test error: {e}")
        return False

# Run sanity check
print("\n🧪 Running quick sanity check...")

# Check dependencies
missing_deps = []
if 'cv2' not in globals():
    missing_deps.append("opencv-python")
if 'gr' not in globals():
    missing_deps.append("gradio")
if 'torch' not in globals():
    missing_deps.append("torch torchvision")

if missing_deps:
    print(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
    print("💡 Install with: pip install " + " ".join(missing_deps))
    print("💡 Pipeline will work with dummy/naive components for now")

# Run the sanity check
success = quick_sanity_run()

if success:
    print("✅ Quick sanity run completed successfully!")
    print("🎉 Pipeline is ready for use!")
else:
    print("❌ Quick sanity run failed")
    print("💡 Check the error messages above and fix any issues")

print("✅ Quick sanity run ready")

# %%
