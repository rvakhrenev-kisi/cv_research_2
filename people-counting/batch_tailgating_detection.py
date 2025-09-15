#!/usr/bin/env python3
"""
Batch tailgating detection script for Cisco and Vortex videos.
Uses configuration file for parameter tuning.
Runs YOLOv11 on all videos with saved line configurations.
"""

import os
import cv2
import json
import subprocess
import datetime
from pathlib import Path
import argparse
from tqdm import tqdm
from config_loader import ConfigLoader

class BatchTailgatingDetector:
    def __init__(self, config_file="config.yaml"):
        # Load configuration
        self.config_loader = ConfigLoader(config_file)
        self.config = self.config_loader.config
        
        # Get configuration sections
        self.model_config = self.config_loader.get_model_config()
        self.detection_config = self.config_loader.get_detection_config()
        self.tracking_config = self.config_loader.get_tracking_config()
        self.video_config = self.config_loader.get_video_config()
        self.output_config = self.config_loader.get_output_config()
        self.cctv_config = self.config_loader.get_cctv_config()
        
        # Set up directories
        self.output_dir = Path(self.output_config.get("base_dir", "outputs"))
        self.datasets = ["cisco", "vortex"]
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamped run directory if enabled
        if self.output_config.get("create_timestamped_dir", True):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.output_dir / f"run_{timestamp}"
        else:
            self.run_dir = self.output_dir / "latest"
        self.run_dir.mkdir(exist_ok=True)
        
        print(f"📁 Run directory: {self.run_dir}")
        
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
        
        # Build tuned parameters from config
        self.tuned_parameters = {
            "model_type": self.model_config.get("name", "yolo11n.pt"),
            "model_type_param": self.model_config.get("type", "yolo12"),
            "confidence_threshold": self.detection_config.get("confidence", 0.1),
            "input_size": self.detection_config.get("imgsz", 640),
            "iou_threshold": self.detection_config.get("iou", 0.3),
            "agnostic_nms": self.detection_config.get("agnostic_nms", False),
            "track_high_thresh": self.tracking_config.get("track_high_thresh", 0.6),
            "track_low_thresh": self.tracking_config.get("track_low_thresh", 0.1),
            "new_track_thresh": self.tracking_config.get("new_track_thresh", 0.7),
            "track_buffer": self.tracking_config.get("track_buffer", 30),
            "match_thresh": self.tracking_config.get("match_thresh", 0.8),
            "frame_rate": self.tracking_config.get("frame_rate", 30),
            "optimization": "CCTV ceiling cameras" if self.cctv_config.get("optimized", True) else "Standard",
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": round(self.gpu_memory, 1)
        }
        
        # Print configuration summary
        self.config_loader.print_config_summary()
    
    def load_line_config(self, dataset):
        """Load line configuration for a dataset."""
        config_file = self.config_dir / f"{dataset}_line_config.json"
        
        if not config_file.exists():
            print(f"❌ No configuration found for {dataset} dataset")
            print(f"   Please run the line marker interface first to create {config_file}")
            return None
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            line = config.get("line", {})
            return {
                "x1": line.get("x1", 0),
                "y1": line.get("y1", 0),
                "x2": line.get("x2", 100),
                "y2": line.get("y2", 100)
            }
        except Exception as e:
            print(f"❌ Error loading {dataset} configuration: {e}")
            return None
    
    def get_video_files(self, dataset):
        """Get all video files for a dataset from the original folders."""
        # Use the original dataset folders
        dataset_dir = Path(f"../{dataset}")
        video_files = []
        
        if not dataset_dir.exists():
            print(f"❌ Dataset directory not found: {dataset_dir}")
            return video_files
        
        # Look for all video files in the dataset directory
        for ext in ['.mp4', '.avi', '.mov']:
            video_files.extend(dataset_dir.glob(f"*{ext}"))
        
        return sorted(video_files)
    
    def extract_count_from_output(self, output_text):
        """Extract people count from the people counter output."""
        import re
        # Look for patterns like "People count - Up: 5, Down: 3, Total: 8"
        up_pattern = r"Up:\s*(\d+)"
        down_pattern = r"Down:\s*(\d+)"
        total_pattern = r"Total:\s*(\d+)"
        
        up_count = 0
        down_count = 0
        total_count = 0
        
        # Try to extract counts from the output
        up_match = re.search(up_pattern, output_text)
        if up_match:
            up_count = int(up_match.group(1))
        
        down_match = re.search(down_pattern, output_text)
        if down_match:
            down_count = int(down_match.group(1))
        
        total_match = re.search(total_pattern, output_text)
        if total_match:
            total_count = int(total_match.group(1))
        else:
            total_count = up_count + down_count
        
        return up_count, down_count, total_count

    def process_video(self, video_path, dataset, line_config, model_path):
        """Process a single video with CCTV-optimized parameters."""
        try:
            # Create output filename
            video_name = video_path.stem
            output_name = f"{dataset}_{video_name}.mp4"
            output_path = self.run_dir / output_name
            
            # Build command with CCTV-optimized parameters
            # Use the virtual environment Python explicitly
            import sys
            import os
            
            # Try to use the virtual environment Python first
            venv_python = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
            if os.path.exists(venv_python):
                python_cmd = venv_python
            else:
                # Fallback to sys.executable
                python_cmd = sys.executable
                
            # Get parameters from configuration
            confidence = self.detection_config.get("confidence", 0.1)
            model_type = self.model_config.get("type", "yolo12")
            output_height = self.video_config.get("output_height", 0)
            verbose = self.video_config.get("verbose", True)
            
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
            
            # Add verbose flag if enabled
            if verbose:
                cmd.append("--verbose")
            
            print(f"🎬 Processing: {video_name}")
            print(f"   Command: {' '.join(cmd)}")
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            # Show all output from subprocess for debugging
            if result.stdout:
                print(f"   📤 Subprocess stdout:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        print(f"      {line}")
            
            if result.stderr:
                print(f"   📤 Subprocess stderr:")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        print(f"      {line}")
            
            # Show GPU detection output from subprocess
            if "Using device:" in result.stdout:
                device_lines = [line for line in result.stdout.split('\n') if 'Using device:' in line]
                if device_lines:
                    print(f"   {device_lines[0]}")
            if "GPU:" in result.stdout:
                gpu_lines = [line for line in result.stdout.split('\n') if 'GPU:' in line]
                if gpu_lines:
                    print(f"   {gpu_lines[0]}")
            if "Memory:" in result.stdout:
                memory_lines = [line for line in result.stdout.split('\n') if 'Memory:' in line]
                if memory_lines:
                    print(f"   {memory_lines[0]}")
            if "Model initialized" in result.stdout:
                model_lines = [line for line in result.stdout.split('\n') if 'Model initialized' in line]
                if model_lines:
                    print(f"   {model_lines[0]}")
            
            if result.returncode == 0:
                # Extract counts from the output
                up_count, down_count, total_count = self.extract_count_from_output(result.stdout)
                
                print(f"   ✅ Success: {output_name}")
                print(f"   📊 Counts - Up: {up_count}, Down: {down_count}, Total: {total_count}")
                
                return True, {
                    "output": output_name,
                    "up_count": up_count,
                    "down_count": down_count,
                    "total_count": total_count
                }
            else:
                print(f"   ❌ Error: {result.stderr}")
                return False, {"error": result.stderr}
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            return False, {"error": str(e)}
    
    def run_batch_detection(self, model_size="x", confidence=0.1):
        """Run batch detection optimized for CCTV ceiling cameras."""
        print(f"🚀 Starting CCTV-optimized batch tailgating detection")
        print(f"   Model: YOLOv10{model_size}")
        print(f"   Confidence: {confidence} (optimized for CCTV ceiling cameras)")
        print(f"   Output Quality: Original resolution")
        print(f"   Optimized for: Ceiling-mounted CCTV cameras")
        print(f"   GPU: {self.gpu_name} ({self.gpu_memory:.1f} GB)" if self.gpu_available else "   GPU: Not available")
        print(f"   Output: {self.run_dir}")
        print()
        
        # Update tuned parameters with current run settings
        self.tuned_parameters["confidence_threshold"] = confidence
        
        # Check if model exists (try YOLOv10 first, fallback to YOLOv8)
        model_path = f"models/yolov10{model_size}.pt"
        if not os.path.exists(model_path):
            model_path = f"models/yolov8{model_size}.pt"
            if not os.path.exists(model_path):
                print(f"❌ Model not found: models/yolov10{model_size}.pt or models/yolov8{model_size}.pt")
                print("   Please download YOLOv10 or YOLOv8 models first")
                return
            else:
                print(f"⚠️  Using YOLOv8{model_size} (YOLOv10{model_size} not found)")
                self.tuned_parameters["model_type"] = f"YOLOv8{model_size}"
        else:
            print(f"✅ Using YOLOv10{model_size} model")
            self.tuned_parameters["model_type"] = f"YOLOv10{model_size}"
        
        results = {
            "cisco": {"success": 0, "failed": 0, "files": [], "total_people": 0},
            "vortex": {"success": 0, "failed": 0, "files": [], "total_people": 0}
        }
        
        for dataset in self.datasets:
            print(f"📁 Processing {dataset.upper()} dataset...")
            
            # Load line configuration
            line_config = self.load_line_config(dataset)
            if line_config is None:
                print(f"   ⏭️  Skipping {dataset} dataset (no configuration)")
                continue
            
            print(f"   📏 Line: ({line_config['x1']},{line_config['y1']}) -> ({line_config['x2']},{line_config['y2']})")
            
            # Get video files
            video_files = self.get_video_files(dataset)
            if not video_files:
                print(f"   ⚠️  No videos found for {dataset} dataset")
                continue
            
            print(f"   🎬 Found {len(video_files)} videos")
            
            # Process each video
            dataset_total = 0
            for video_path in tqdm(video_files, desc=f"Processing {dataset}"):
                success, result = self.process_video(
                    video_path, dataset, line_config, model_path, confidence
                )
                
                file_result = {
                    "input": str(video_path),
                    "success": success
                }
                
                if success:
                    file_result.update(result)
                    dataset_total += result.get("total_count", 0)
                    results[dataset]["success"] += 1
                else:
                    file_result["error"] = result.get("error", "Unknown error")
                    results[dataset]["failed"] += 1
                
                results[dataset]["files"].append(file_result)
            
            results[dataset]["total_people"] = dataset_total
            print(f"   ✅ {dataset}: {results[dataset]['success']} success, {results[dataset]['failed']} failed")
            print(f"   👥 Total people detected: {dataset_total}")
            print()
        
        # Save results summary and parameters
        self.save_results_summary(results)
        self.save_tuned_parameters()
        
        # Print final summary
        print("📊 IMPROVED BATCH PROCESSING COMPLETE")
        print("=" * 60)
        total_people = 0
        for dataset in self.datasets:
            success = results[dataset]["success"]
            failed = results[dataset]["failed"]
            people = results[dataset]["total_people"]
            total_people += people
            print(f"{dataset.upper()}: {success}/{success+failed} videos, {people} people detected")
        print(f"TOTAL: {total_people} people detected across all videos")
        print(f"📁 Results saved in: {self.run_dir}")
    
    def save_results_summary(self, results):
        """Save processing results to JSON file."""
        summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "run_directory": str(self.run_dir),
            "tuned_parameters": self.tuned_parameters,
            "results": results
        }
        
        summary_file = self.run_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Summary saved: {summary_file}")
    
    def save_tuned_parameters(self):
        """Save tuned parameters to a readable text file."""
        params_file = self.run_dir / "tuned_parameters.txt"
        with open(params_file, 'w') as f:
            f.write("CCTV-Optimized Detection Parameters\n")
            f.write("=" * 40 + "\n\n")
            
            for key, value in self.tuned_parameters.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n" + "=" * 40 + "\n")
            f.write("Notes:\n")
            f.write("- Lower IoU threshold (0.3) prevents merging close people\n")
            f.write("- Higher tracking confidence (0.5) improves line crossing detection\n")
            f.write("- Optimized for ceiling-mounted CCTV cameras\n")
            f.write("- YOLOv10x provides best accuracy for person detection\n")
            f.write("- GPU acceleration enabled for faster processing\n")
        
        print(f"📋 Parameters saved: {params_file}")

def main():
    parser = argparse.ArgumentParser(description="Batch tailgating detection for Cisco and Vortex videos")
    parser.add_argument("--model-size", choices=["n", "s", "m", "l", "x"], default="x",
                       help="YOLOv10 model size (default: x for best accuracy)")
    parser.add_argument("--confidence", type=float, default=0.1,
                       help="Detection confidence threshold (default: 0.1, optimized for CCTV ceiling cameras)")
    parser.add_argument("--config-dir", default="configs",
                       help="Directory containing line configurations (default: configs)")
    parser.add_argument("--input-dir", default="input",
                       help="Directory containing input videos (default: input)")
    parser.add_argument("--output-dir", default="outputs",
                       help="Directory for output videos (default: outputs)")
    
    args = parser.parse_args()
    
    # Create detector
    detector = BatchTailgatingDetector(
        config_dir=args.config_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Run batch detection
    detector.run_batch_detection(
        model_size=args.model_size,
        confidence=args.confidence
    )

if __name__ == "__main__":
    main()
