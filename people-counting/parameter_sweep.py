#!/usr/bin/env python3
"""
Parameter sweep script for people counting detection.
Tests a single parameter across a range and creates a grid comparison video.
"""

import argparse
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
from config_loader import ConfigLoader
import json
import datetime

class ParameterSweep:
    def __init__(self, config_file="config.yaml"):
        self.config_loader = ConfigLoader(config_file)
        self.config = self.config_loader.config
        
        # Get configuration sections
        self.model_config = self.config_loader.get_model_config()
        self.detection_config = self.config_loader.get_detection_config()
        self.tracking_config = self.config_loader.get_tracking_config()
        self.video_config = self.config_loader.get_video_config()
        self.output_config = self.config_loader.get_output_config()
        self.cctv_config = self.config_loader.get_cctv_config()
        
        # Detect supported CLI flags of people_counter.py
        self.supports = self._detect_supported_flags()

        # Set up directories
        self.output_dir = Path(self.output_config.get("base_dir", "outputs"))
        self.sweep_base_dir = self.output_dir / "parameter_sweep"
        self.sweep_base_dir.mkdir(exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sweep_dir = self.sweep_base_dir / f"run_{timestamp}"
        self.sweep_dir.mkdir(exist_ok=True)
        
        # Check for GPU availability
        try:
            import torch
            self.gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
            if self.gpu_available:
                try:
                    self.gpu_name = torch.cuda.get_device_name(0)
                    self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                except Exception as e:
                    self.gpu_name = f"GPU available but info unavailable ({e})"
                    self.gpu_memory = 0
            else:
                self.gpu_name = "None"
                self.gpu_memory = 0
        except ImportError:
            self.gpu_available = False
            self.gpu_name = "PyTorch not available"
            self.gpu_memory = 0

    def _detect_supported_flags(self) -> Dict[str, bool]:
        """Parse people_counter.py --help to detect supported flags for compatibility across machines."""
        help_text = ""
        try:
            proc = subprocess.run([sys.executable, "people_counter.py", "--help"], capture_output=True, text=True)
            help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        except Exception:
            pass
        def has(flag: str) -> bool:
            return flag in help_text
        return {
            "iou": has("--iou"),
            "imgsz": has("--imgsz"),
            "agnostic": has("--agnostic-nms"),
            "track_high": has("--track-high-thresh"),
            "track_low": has("--track-low-thresh"),
            "new_track": has("--new-track-thresh"),
            "match": has("--match-thresh"),
        }
    
    def generate_parameter_values(self, param_name: str, start: float, end: float, increment: float = 0.05) -> List[Any]:
        """Generate parameter values from start to end with given increment."""
        values = []
        
        # Handle different parameter types
        if param_name in ["detection.agnostic_nms", "video.verbose", "performance.gpu_enabled", 
                         "output.create_timestamped_dir", "output.save_individual_results", 
                         "output.save_summary", "output.save_plots"]:
            # Boolean parameters - test both True and False
            return [True, False]
        
        elif param_name == "model.name":
            # String parameter - predefined model options
            model_options = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
            return model_options
        
        elif param_name in ["detection.imgsz", "tracking.track_buffer", "performance.batch_size", 
                           "performance.num_workers", "video.output_height"]:
            # Integer parameters
            current = int(start)
            end_int = int(end)
            inc_int = max(1, int(increment))
            while current <= end_int:
                values.append(current)
                current += inc_int
            return values
        
        else:
            # Float parameters (default)
            current = start
            while current <= end + 1e-6:  # Add small epsilon for floating point precision
                values.append(round(current, 3))
                current += increment
            return values
    
    def get_parameter_path(self, config_dict: Dict[str, Any], param_name: str) -> List[str]:
        """Get the path to a parameter in the config dictionary."""
        # Handle nested parameters like detection.confidence, tracking.track_high_thresh
        if '.' in param_name:
            parts = param_name.split('.')
            current = config_dict
            for part in parts[:-1]:
                if part in current:
                    current = current[part]
                else:
                    return []
            return parts
        else:
            return [param_name]
    
    def set_parameter_value(self, param_name: str, value: float) -> None:
        """Set a parameter value in the config."""
        path = self.get_parameter_path(self.config, param_name)
        if not path:
            raise ValueError(f"Parameter '{param_name}' not found in config")
        
        # Navigate to the correct section
        current = self.config
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the value
        current[path[-1]] = value
    
    def get_parameter_value(self, param_name: str) -> Any:
        """Get current parameter value from config."""
        path = self.get_parameter_path(self.config, param_name)
        if not path:
            raise ValueError(f"Parameter '{param_name}' not found in config")
        
        current = self.config
        for part in path:
            if part in current:
                current = current[part]
            else:
                return None
        return current
    
    def load_line_config(self, dataset: str) -> Dict[str, Any]:
        """Load line configuration for a dataset from config file."""
        line_config = self.config_loader.get_line_config(dataset)
        
        if not line_config:
            raise ValueError(f"No line configuration found for {dataset}")
        
        # Convert to the format expected by people_counter.py
        return {
            "x1": line_config["start"][0],
            "y1": line_config["start"][1],
            "x2": line_config["end"][0],
            "y2": line_config["end"][1],
            "direction": line_config.get("direction", "unknown")
        }
    
    def process_video_with_parameter(self, video_path: Path, dataset: str, 
                                   param_name: str, param_value: float, 
                                   line_config: Dict[str, Any], model_path: str) -> Tuple[bool, str, int, int, int]:
        """Process a single video with a specific parameter value."""
        try:
            # Create temporary config file with modified parameter
            temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            import yaml
            yaml.dump(self.config, temp_config)
            temp_config.close()
            
            # Create output filename
            video_name = video_path.stem
            output_name = f"{dataset}_{video_name}_{param_name}_{param_value:.3f}.mp4"
            output_path = self.sweep_dir / output_name
            
            # Use the virtual environment Python explicitly
            venv_python = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
            if os.path.exists(venv_python):
                python_cmd = venv_python
            else:
                python_cmd = sys.executable
            
            # Get parameters from configuration
            confidence = self.detection_config.get("confidence", 0.1)
            model_type = self.model_config.get("type", "yolo12")
            output_height = self.video_config.get("output_height", 0)
            verbose = self.video_config.get("verbose", True)
            
            # Override the specific parameter for people_counter.py
            if param_name == "detection.confidence":
                confidence = param_value
            elif param_name == "detection.iou":
                # This will be passed as --iou parameter
                pass
            elif param_name == "detection.imgsz":
                # This will be passed as --imgsz parameter
                pass
            elif param_name == "detection.agnostic_nms":
                # This will be passed as --agnostic-nms flag
                pass
            elif param_name == "tracking.track_high_thresh":
                # This will be passed as --track-high-thresh parameter
                pass
            elif param_name == "tracking.track_low_thresh":
                # This will be passed as --track-low-thresh parameter
                pass
            elif param_name == "tracking.new_track_thresh":
                # This will be passed as --new-track-thresh parameter
                pass
            elif param_name == "tracking.match_thresh":
                # This will be passed as --match-thresh parameter
                pass
            elif param_name == "tracking.track_buffer":
                # Note: people_counter.py doesn't currently support track_buffer parameter
                pass
            elif param_name == "tracking.frame_rate":
                # Note: people_counter.py doesn't currently support frame_rate parameter
                pass
            elif param_name == "video.output_height":
                output_height = int(param_value) if param_value != 0 else 0
            elif param_name == "video.verbose":
                verbose = bool(param_value)
            elif param_name == "model.name":
                # Update model path
                model_path = f"models/{param_value}"
            # Note: Other parameters (performance, output) don't affect people_counter.py directly
            
            cmd = [
                python_cmd, "people_counter.py",
                "--video", str(video_path),
                "--model", model_path,
                "--model-type", model_type,
                "--line-start", str(line_config["x1"]), str(line_config["y1"]),
                "--line-end", str(line_config["x2"]), str(line_config["y2"]),
                "--confidence", str(confidence),
                "--output", str(output_path),
                "--output-height", str(output_height)
            ]
            
            # Add detection parameters
            if param_name == "detection.iou":
                cmd.extend(["--iou", str(param_value)])
            elif param_name == "detection.imgsz":
                cmd.extend(["--imgsz", str(int(param_value))])
            elif param_name == "detection.agnostic_nms":
                if param_value:
                    cmd.append("--agnostic-nms")
            else:
                # Add default detection parameters
                cmd.extend(["--iou", str(self.detection_config.get("iou", 0.3))])
                cmd.extend(["--imgsz", str(self.detection_config.get("imgsz", 640))])
                if self.detection_config.get("agnostic_nms", False):
                    cmd.append("--agnostic-nms")
            
            # Add tracking parameters
            if param_name == "tracking.track_high_thresh":
                cmd.extend(["--track-high-thresh", str(param_value)])
            elif param_name == "tracking.track_low_thresh":
                cmd.extend(["--track-low-thresh", str(param_value)])
            elif param_name == "tracking.new_track_thresh":
                cmd.extend(["--new-track-thresh", str(param_value)])
            elif param_name == "tracking.match_thresh":
                cmd.extend(["--match-thresh", str(param_value)])
            else:
                # Add default tracking parameters
                cmd.extend(["--track-high-thresh", str(self.tracking_config.get("track_high_thresh", 0.6))])
                cmd.extend(["--track-low-thresh", str(self.tracking_config.get("track_low_thresh", 0.1))])
                cmd.extend(["--new-track-thresh", str(self.tracking_config.get("new_track_thresh", 0.7))])
                cmd.extend(["--match-thresh", str(self.tracking_config.get("match_thresh", 0.8))])
            
            # Add verbose flag if enabled
            if verbose:
                cmd.append("--verbose")
            
            print(f"   Testing {param_name}={param_value:.3f}")
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            # Extract counts from output
            up_count, down_count, total_count = self.extract_count_from_output(result.stdout)
            
            if result.returncode == 0:
                print(f"   ✅ Success: {output_name} (Total: {total_count})")
                return True, str(output_path), up_count, down_count, total_count
            else:
                print(f"   ❌ Failed: {output_name} (exit code: {result.returncode})")
                return False, "", 0, 0, 0
                
        except Exception as e:
            print(f"   ❌ Error processing {video_path}: {e}")
            return False, "", 0, 0, 0
        finally:
            # Clean up temporary config file
            if 'temp_config' in locals():
                try:
                    os.unlink(temp_config.name)
                except:
                    pass
    
    def extract_count_from_output(self, output_text: str) -> Tuple[int, int, int]:
        """Extract people counts from people_counter.py output."""
        up_count = 0
        down_count = 0
        total_count = 0
        
        lines = output_text.split('\n')
        for line in lines:
            if "People count - Up:" in line:
                try:
                    up_count = int(line.split("Up:")[1].strip())
                except:
                    pass
            elif "People count - Down:" in line:
                try:
                    down_count = int(line.split("Down:")[1].strip())
                except:
                    pass
            elif "Total:" in line and "people" in line.lower():
                try:
                    total_count = int(line.split("Total:")[1].strip())
                except:
                    pass
        
        return up_count, down_count, total_count
    
    def add_text_overlay(self, frame: np.ndarray, text: str, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """Add text overlay to a frame."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)  # White
        thickness = 2
        
        # Add black background for better visibility
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        cv2.rectangle(frame, position, (position[0] + text_size[0] + 10, position[1] + text_size[1] + 10), (0, 0, 0), -1)
        
        # Add text
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
        return frame
    
    def create_grid_video(self, video_paths: List[str], param_values: List[float], 
                         param_name: str, output_path: str) -> None:
        """Create a grid video from multiple video files."""
        if not video_paths:
            print("No videos to create grid from")
            return
        
        # Open all videos
        caps = []
        for video_path in video_paths:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                caps.append(cap)
            else:
                print(f"Warning: Could not open {video_path}")
        
        if not caps:
            print("No valid videos found")
            return
        
        # Get video properties from first video
        width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(caps[0].get(cv2.CAP_PROP_FPS) or 30)
        
        # Calculate grid dimensions
        num_videos = len(caps)
        cols = min(3, num_videos)  # Max 3 columns
        rows = (num_videos + cols - 1) // cols
        
        # Calculate individual video size in grid
        grid_width = width // cols
        grid_height = height // rows
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Creating grid video: {rows}x{cols} grid")
        
        frame_count = 0
        while True:
            frames = []
            valid_frames = 0
            
            # Read frames from all videos
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if ret:
                    # Resize frame to fit grid
                    frame_resized = cv2.resize(frame, (grid_width, grid_height))
                    
                    # Add parameter value overlay
                    val = param_values[i]
                    if isinstance(val, float):
                        val_str = f"{val:.3f}"
                    else:
                        val_str = str(val)
                    param_text = f"{param_name}={val_str}"
                    frame_resized = self.add_text_overlay(frame_resized, param_text, (10, 30))
                    
                    frames.append(frame_resized)
                    valid_frames += 1
                else:
                    # Create black frame if video ended
                    black_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
                    frames.append(black_frame)
            
            if valid_frames == 0:
                break
            
            # Create grid frame
            grid_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            for i, frame in enumerate(frames):
                row = i // cols
                col = i % cols
                
                y_start = row * grid_height
                y_end = min((row + 1) * grid_height, height)
                x_start = col * grid_width
                x_end = min((col + 1) * grid_width, width)
                
                # Resize frame to fit the allocated space
                frame_final = cv2.resize(frame, (x_end - x_start, y_end - y_start))
                grid_frame[y_start:y_end, x_start:x_end] = frame_final
            
            out.write(grid_frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
        
        # Clean up
        for cap in caps:
            cap.release()
        out.release()
        
        print(f"✅ Grid video saved to: {output_path}")
    
    def run_sweep(self, dataset: str, file_name: str, param_name: str, 
                  start_value: float, end_value: float, increment: float = 0.05) -> None:
        """Run parameter sweep for a specific video."""
        print(f"🔬 Parameter Sweep: {param_name}")
        print(f"   Dataset: {dataset}")
        print(f"   File: {file_name}")
        print(f"   Range: {start_value} to {end_value} (increment: {increment})")
        print(f"   Run directory: {self.sweep_dir}")
        print()
        
        # Generate parameter values
        param_values = self.generate_parameter_values(param_name, start_value, end_value, increment)
        print(f"   Testing {len(param_values)} values: {param_values}")
        print()
        
        # Check if model exists
        model_name = self.model_config.get("name", "yolo11n.pt")
        model_path = f"models/{model_name}"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            print("   Please download YOLOv11 models first using: python download_yolov11.py")
            return
        
        # Load line configuration
        try:
            line_config = self.load_line_config(dataset)
        except ValueError as e:
            print(f"❌ {e}")
            return
        
        # Find video file
        video_path = Path(f"../{dataset}/{file_name}")
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            return
        
        print(f"✅ Found video: {video_path}")
        print(f"   Line: ({line_config['x1']},{line_config['y1']}) -> ({line_config['x2']},{line_config['y2']})")
        print()
        
        # Process video with each parameter value
        successful_videos = []
        successful_values = []
        results = []
        
        for param_value in param_values:
            # Set parameter value in config
            self.set_parameter_value(param_name, param_value)
            
            # Process video
            success, output_path, up_count, down_count, total_count = self.process_video_with_parameter(
                video_path, dataset, param_name, param_value, line_config, model_path
            )
            
            if success:
                successful_videos.append(output_path)
                successful_values.append(param_value)
            
            results.append({
                "parameter": param_name,
                "value": param_value,
                "success": success,
                "up_count": up_count,
                "down_count": down_count,
                "total_count": total_count,
                "output_file": output_path
            })
        
        print(f"\n📊 Results Summary:")
        print(f"   Successful: {len(successful_videos)}/{len(param_values)}")
        for result in results:
            status = "✅" if result["success"] else "❌"
            print(f"   {status} {param_name}={result['value']:.3f}: {result['total_count']} people")
        
        # Create grid video if we have successful results
        if successful_videos:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            grid_output = self.sweep_dir / f"{dataset}_{file_name}_{param_name}_sweep_{timestamp}.mp4"
            
            print(f"\n🎬 Creating grid comparison video...")
            self.create_grid_video(successful_videos, successful_values, param_name, str(grid_output))
        
        # Save results to JSON
        results_file = self.sweep_dir / f"{dataset}_{file_name}_{param_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Copy config file to run directory for reproducibility
        self.copy_config_to_output()
        
        print(f"💾 Results saved to: {results_file}")
        print(f"📁 Run directory: {self.sweep_dir}")
    
    def copy_config_to_output(self):
        """Copy the current config.yaml to run directory for reproducibility."""
        import shutil
        
        config_source = Path("config.yaml")
        config_dest = self.sweep_dir / "config_used.yaml"
        
        if config_source.exists():
            shutil.copy2(config_source, config_dest)
            print(f"   📋 Config copied to: {config_dest}")
        else:
            print(f"   ⚠️  Config file not found: {config_source}")

def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for people counting detection")
    parser.add_argument("dataset", choices=["cisco", "vortex"], help="Dataset name")
    parser.add_argument("file_name", help="Video file name (e.g., 1.mp4)")
    parser.add_argument("parameter", help="Parameter name (e.g., detection.confidence)")
    parser.add_argument("start_value", type=float, help="Start value for parameter range")
    parser.add_argument("end_value", type=float, help="End value for parameter range")
    parser.add_argument("--increment", type=float, default=0.05, help="Increment between values (default: 0.05)")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Validate parameter name
    valid_params = [
        # Model parameters
        "model.name",
        
        # Detection parameters
        "detection.confidence",
        "detection.iou", 
        "detection.imgsz",
        "detection.agnostic_nms",
        
        # Tracking parameters
        "tracking.track_high_thresh",
        "tracking.track_low_thresh",
        "tracking.new_track_thresh",
        "tracking.track_buffer",
        "tracking.match_thresh",
        "tracking.frame_rate",
        
        # Video parameters
        "video.output_height",
        "video.verbose",
        
        # Performance parameters
        "performance.gpu_enabled",
        "performance.batch_size",
        "performance.num_workers",
        
        # Output parameters
        "output.create_timestamped_dir",
        "output.save_individual_results",
        "output.save_summary",
        "output.save_plots"
    ]
    
    if args.parameter not in valid_params:
        print(f"❌ Invalid parameter: {args.parameter}")
        print(f"   Valid parameters: {', '.join(valid_params)}")
        return
    
    # Validate parameter-specific constraints
    if args.parameter in ["detection.agnostic_nms", "video.verbose", "performance.gpu_enabled", 
                         "output.create_timestamped_dir", "output.save_individual_results", 
                         "output.save_summary", "output.save_plots"]:
        # Boolean parameters - ignore start/end values
        print(f"   Note: {args.parameter} is a boolean parameter, will test both True and False")
    elif args.parameter == "model.name":
        # String parameter - ignore start/end values
        print(f"   Note: {args.parameter} is a string parameter, will test all model options")
    else:
        # Numeric parameters
        if args.start_value >= args.end_value:
            print("❌ Start value must be less than end value")
            return
        
        if args.increment <= 0:
            print("❌ Increment must be positive")
            return
    
    # Run sweep
    sweep = ParameterSweep(args.config)
    sweep.run_sweep(
        args.dataset, 
        args.file_name, 
        args.parameter, 
        args.start_value, 
        args.end_value, 
        args.increment
    )

if __name__ == "__main__":
    main()
