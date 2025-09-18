#!/usr/bin/env python3
"""
Updated batch tailgating detection script with configuration file support.
Uses YOLOv11 and configurable parameters.
"""

import os
import cv2
import json
import subprocess
import datetime
import glob
from pathlib import Path
import argparse
from tqdm import tqdm
from config_loader import ConfigLoader

class BatchTailgatingDetectorV2:
    def __init__(self, config_file="configs/global.yaml"):
        # Load configuration
        self.config_loader = ConfigLoader(config_file)
        self.config = self.config_loader.config
        
        # Get configuration sections
        self.model_config = self.config_loader.get_model_config()
        self.detection_config = self.config_loader.get_detection_config()
        self.tracking_config = self.config_loader.get_tracking_config()
        self.video_config = self.config_loader.get_video_config()
        self.tracker_config = self.config_loader.get_tracker_config()
        self.output_config = self.config_loader.get_output_config()
        self.cctv_config = self.config_loader.get_cctv_config()
        
        # Set up directories
        self.output_dir = Path(self.output_config.get("base_dir", "outputs"))
        # Discover datasets from input/ to stay in sync with configs
        input_root = Path("input")
        if input_root.exists():
            self.datasets = [p.name for p in input_root.iterdir() if p.is_dir()]
        else:
            self.datasets = ["cisco", "vortex", "courtyard"]
        
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
        """Load line configuration for a dataset from config file."""
        line_config = self.config_loader.get_line_config(dataset)
        
        if not line_config:
            print(f"❌ No line configuration found for {dataset}")
            return None
        
        # Handle different JSON formats
        if "line" in line_config:
            # Format: {"line": {"x1": 675, "y1": 532, "x2": 1252, "y2": 532}}
            line_data = line_config["line"]
            return {
                "x1": line_data["x1"],
                "y1": line_data["y1"],
                "x2": line_data["x2"],
                "y2": line_data["y2"],
                "direction": line_data.get("direction", "unknown"),
                "direction_point": line_config.get("direction_point")
            }
        elif "start" in line_config and "end" in line_config:
            # Format: {"start": [x1, y1], "end": [x2, y2]}
            return {
                "x1": line_config["start"][0],
                "y1": line_config["start"][1],
                "x2": line_config["end"][0],
                "y2": line_config["end"][1],
                "direction": line_config.get("direction", "unknown"),
                "direction_point": line_config.get("direction_point")
            }
        else:
            # Direct format: {"x1": 675, "y1": 532, "x2": 1252, "y2": 532}
            return {
                "x1": line_config["x1"],
                "y1": line_config["y1"],
                "x2": line_config["x2"],
                "y2": line_config["y2"],
                "direction": line_config.get("direction", "unknown"),
                "direction_point": line_config.get("direction_point")
            }
    
    def get_video_files(self, dataset):
        """Get video files for a dataset from input/<dataset>/"""
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        video_files = []
        base = Path("input") / dataset
        for ext in video_extensions:
            video_files.extend([str(p) for p in base.glob(ext)])
        return sorted([Path(f) for f in video_files])
    
    def extract_count_from_output(self, output_text):
        """Extract people counts from people_counter.py output."""
        up_count = 0
        down_count = 0
        total_count = 0
        
        lines = output_text.split('\n')
        for line in lines:
            if "People count - Up:" in line and "Down:" in line and "Total:" in line:
                try:
                    # Parse format: "People count - Up: X, Down: Y, Total: Z"
                    parts = line.split("Up:")[1].strip()
                    up_count = int(parts.split(",")[0].strip())
                    
                    down_part = parts.split("Down:")[1].strip()
                    down_count = int(down_part.split(",")[0].strip())
                    
                    total_part = down_part.split("Total:")[1].strip()
                    total_count = int(total_part.strip())
                except Exception as e:
                    print(f"   ⚠️  Error parsing counts from line: {line.strip()}")
                    print(f"   ⚠️  Error: {e}")
                    pass
        
        return up_count, down_count, total_count
    
    def process_video(self, video_path, dataset, line_config, model_path):
        """Process a single video with configuration parameters."""
        try:
            # Create output filename
            video_name = video_path.stem
            # Per-dataset output folder
            dataset_out_dir = self.run_dir / dataset
            dataset_out_dir.mkdir(parents=True, exist_ok=True)
            output_name = f"{dataset}_{video_name}.mp4"
            output_path = dataset_out_dir / output_name
            
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
            
            # Get parameters from configuration (allow per-dataset overrides)
            # Detection params: dataset-specific only
            ds_detection = self.config_loader.get_dataset_detection_config(dataset)
            confidence = ds_detection.get("confidence", self.detection_config.get("confidence", 0.1))
            model_type = self.model_config.get("type", "yolo12")
            # Video params: prefer dataset-specific video.yaml; fallback to global video section
            ds_video = self.config_loader.get_dataset_video_config(dataset)
            output_height = ds_video.get("output_height", self.video_config.get("output_height", 0))
            verbose = ds_video.get("verbose", self.video_config.get("verbose", True))
            
            use_ultra_tracker = self.tracker_config.get("use_ultralytics", False)

            cmd = [
                python_cmd, "people_counter.py",
                "--video", str(video_path),
                "--model", model_path,
                "--model-type", model_type,
                "--dataset", dataset,
                "--line-start", str(line_config["x1"]), str(line_config["y1"]),
                "--line-end", str(line_config["x2"]), str(line_config["y2"]),
                "--confidence", str(confidence),
                "--output", str(output_path),
                "--output-height", str(output_height)
            ]
            
            # Add direction point if available
            if "direction_point" in line_config and line_config["direction_point"]:
                cmd.extend(["--direction-point", str(line_config["direction_point"][0]), str(line_config["direction_point"][1])])

            # Add detection params from config
            iou = ds_detection.get("iou", self.detection_config.get("iou", 0.3))
            imgsz = ds_detection.get("imgsz", self.detection_config.get("imgsz", 640))
            agnostic_nms = ds_detection.get("agnostic_nms", self.detection_config.get("agnostic_nms", False))
            cmd.extend(["--iou", str(iou)])
            cmd.extend(["--imgsz", str(imgsz)])
            if agnostic_nms:
                cmd.append("--agnostic-nms")

            # Add tracking params or Ultralytics tracker yaml
            if use_ultra_tracker:
                # Prefer per-dataset tracker yaml if present
                tracker_yaml = self.config_loader.get_dataset_tracker_yaml(dataset)
                cmd.extend(["--tracker-yaml", tracker_yaml])
            else:
                # Get tracker type from dataset config or default to bytetrack
                tracker_type = ds_detection.get("tracker_type", "bytetrack")
                cmd.extend(["--tracker-type", tracker_type])
                
                # Add tracking params for non-ultralytics trackers
                if tracker_type != "ocsort":
                    cmd.extend(["--track-high-thresh", str(self.tracking_config.get("track_high_thresh", 0.6))])
                    cmd.extend(["--track-low-thresh", str(self.tracking_config.get("track_low_thresh", 0.1))])
                    cmd.extend(["--new-track-thresh", str(self.tracking_config.get("new_track_thresh", 0.7))])
                    cmd.extend(["--match-thresh", str(self.tracking_config.get("match_thresh", 0.8))])
            
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
            
            # Extract counts from output
            up_count, down_count, total_count = self.extract_count_from_output(result.stdout)
            
            print(f"   📊 People count - Up: {up_count}, Down: {down_count}, Total: {total_count}")
            
            if result.returncode == 0:
                print(f"   ✅ Success: {output_name}")
                return True, up_count, down_count, total_count
            else:
                print(f"   ❌ Failed: {output_name} (exit code: {result.returncode})")
                return False, 0, 0, 0
                
        except Exception as e:
            print(f"   ❌ Error processing {video_path}: {e}")
            return False, 0, 0, 0
    
    def run_batch_detection(self):
        """Run batch detection using configuration parameters."""
        model_name = self.model_config.get("name", "yolo11n.pt")
        confidence = self.detection_config.get("confidence", 0.1)
        
        print(f"🚀 Starting batch tailgating detection")
        print(f"   Model: {model_name}")
        print(f"   Confidence: {confidence}")
        output_height = self.video_config.get('output_height', 0)
        output_quality = 'Original' if output_height == 0 else f'{output_height}px'
        print(f"   Output Quality: {output_quality}")
        print(f"   Optimization: {self.tuned_parameters['optimization']}")
        print(f"   GPU: {self.gpu_name} ({self.gpu_memory:.1f} GB)" if self.gpu_available else "   GPU: Not available")
        print(f"   Output: {self.run_dir}")
        print()
        
        # Check if model exists
        model_path = f"models/{model_name}"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            print("   Please download YOLOv11 models first using: python download_yolov11.py")
            return
        else:
            print(f"✅ Using {model_name} model")
        
        all_results = {}
        total_processed = 0
        total_successful = 0
        
        for dataset in self.datasets:
            print(f"\n📁 Processing {dataset.upper()} dataset...")
            
            # Load line configuration (fallback to a minimal default if missing)
            line_config = self.load_line_config(dataset)
            if not line_config:
                print(f"   ⚠️  No line configuration found for {dataset}. Proceeding without a line (visual only).")
                line_config = {"x1": 0, "y1": 0, "x2": 10, "y2": 0, "direction": "unknown"}
            
            print(f"   📏 Line: ({line_config['x1']},{line_config['y1']}) -> ({line_config['x2']},{line_config['y2']})")
            if line_config.get("direction_point"):
                dp = line_config["direction_point"]
                print(f"   🧭 Direction point (IN side): ({dp[0]},{dp[1]})")
            else:
                print(f"   🧭 Direction: {line_config.get('direction', 'unknown')}")
            
            # Get video files
            video_files = self.get_video_files(dataset)
            if not video_files:
                print(f"   ⚠️  No videos found in ../{dataset}/")
                continue
            
            print(f"   🎬 Found {len(video_files)} videos")
            
            # Process videos
            dataset_results = {}
            for video_path in tqdm(video_files, desc=f"Processing {dataset}"):
                success, up_count, down_count, total_count = self.process_video(
                    video_path, dataset, line_config, model_path
                )
                
                video_name = video_path.stem
                dataset_results[video_name] = {
                    "success": success,
                    "up_count": up_count,
                    "down_count": down_count,
                    "total_count": total_count,
                    "file": str(video_path)
                }
                
                total_processed += 1
                if success:
                    total_successful += 1
            
            all_results[dataset] = dataset_results

            # Copy per-dataset configs used
            self.copy_dataset_configs_to_output(dataset)
        
        # Save results summary
        self.save_results_summary(all_results, total_processed, total_successful)
        
        # Copy config file to output directory for reproducibility
        self.copy_config_to_output()
        # Copy tracker YAML if using Ultralytics tracker
        self.copy_tracker_yaml_to_output()
        
        # Calculate total crossing counts for console output
        total_up_count = 0
        total_down_count = 0
        total_people_count = 0
        
        for dataset, results in all_results.items():
            for video, data in results.items():
                if data["success"]:
                    total_up_count += data["up_count"]
                    total_down_count += data["down_count"]
                    total_people_count += data["total_count"]
        
        print(f"\n✅ Batch processing completed!")
        print(f"   📊 Processed: {total_processed} videos")
        print(f"   ✅ Successful: {total_successful} videos")
        print(f"   🚶 Crossing In: {total_up_count}")
        print(f"   🚶 Crossing Out: {total_down_count}")
        print(f"   🔄 Total Crossings: {total_up_count + total_down_count}")
        print(f"   📁 Results saved to: {self.run_dir}")
    
    def save_results_summary(self, all_results, total_processed, total_successful):
        """Save processing results summary."""
        # Calculate total crossing counts
        total_up_count = 0
        total_down_count = 0
        total_people_count = 0
        
        for dataset, results in all_results.items():
            for video, data in results.items():
                if data["success"]:
                    total_up_count += data["up_count"]
                    total_down_count += data["down_count"]
                    total_people_count += data["total_count"]
        
        summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_processed": total_processed,
            "total_successful": total_successful,
            "success_rate": total_successful / total_processed if total_processed > 0 else 0,
            "total_crossings": {
                "up_count": total_up_count,
                "down_count": total_down_count,
                "total_count": total_people_count
            },
            "tuned_parameters": self.tuned_parameters,
            "results": all_results
        }
        
        # Save JSON summary
        summary_file = self.run_dir / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save human-readable summary
        txt_file = self.run_dir / "processing_summary.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("BATCH TAILGATING DETECTION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {summary['timestamp']}\n")
            f.write(f"Total Processed: {total_processed}\n")
            f.write(f"Total Successful: {total_successful}\n")
            f.write(f"Success Rate: {summary['success_rate']:.2%}\n\n")
            
            f.write("CROSSING COUNTS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Crossing In: {total_up_count}\n")
            f.write(f"Crossing Out: {total_down_count}\n")
            f.write(f"Total Crossings: {total_up_count + total_down_count}\n\n")
            
            f.write("CONFIGURATION USED:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Model: {self.tuned_parameters['model_type']}\n")
            f.write(f"Confidence: {self.tuned_parameters['confidence_threshold']}\n")
            f.write(f"Input Size: {self.tuned_parameters['input_size']}\n")
            f.write(f"IoU Threshold: {self.tuned_parameters['iou_threshold']}\n")
            f.write(f"GPU: {self.tuned_parameters['gpu_name']}\n")
            f.write(f"Optimization: {self.tuned_parameters['optimization']}\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 30 + "\n")
            for dataset, results in all_results.items():
                f.write(f"\n{dataset.upper()} DATASET:\n")
                for video, data in results.items():
                    status = "✅" if data["success"] else "❌"
                    f.write(
                        f"  {status} {video}: Crossing In={data['up_count']}, "
                        f"Crossing Out={data['down_count']}, Total={data['total_count']}\n"
                    )
        
        print(f"💾 Results saved to {summary_file} and {txt_file}")
    
    def copy_config_to_output(self):
        """Copy the current config.yaml to output directory for reproducibility."""
        import shutil
        
        config_source = Path(self.config_loader.config_file)
        config_dest = self.run_dir / "global_config_used.yaml"
        
        if config_source.exists():
            shutil.copy2(config_source, config_dest)
            print(f"   📋 Config copied to: {config_dest}")
        else:
            print(f"   ⚠️  Config file not found: {config_source}")

    def copy_dataset_configs_to_output(self, dataset: str):
        """Copy dataset-level configs (detection.yaml, tracker.yaml, line.json) into outputs/<run>/<dataset>/configs_used"""
        import shutil
        ds_cfg_dir = Path("configs") / "datasets" / dataset
        dest_dir = self.run_dir / dataset / "configs_used"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # detection.yaml
        det_src = ds_cfg_dir / "detection.yaml"
        if det_src.exists():
            shutil.copy2(det_src, dest_dir / "detection.yaml")
        # tracker.yaml
        trk_src = ds_cfg_dir / "tracker.yaml"
        # If per-dataset not present but global specified and used, still copy global
        if trk_src.exists():
            shutil.copy2(trk_src, dest_dir / "tracker.yaml")
        else:
            global_yaml = self.tracker_config.get("yaml")
            if global_yaml and Path(global_yaml).exists():
                shutil.copy2(global_yaml, dest_dir / Path(global_yaml).name)
        # line.json
        line_src = ds_cfg_dir / "line.json"
        if line_src.exists():
            shutil.copy2(line_src, dest_dir / "line.json")

    def copy_tracker_yaml_to_output(self):
        """Copy the per-dataset tracker YAMLs into each dataset output folder's configs_used."""
        # Already handled in copy_dataset_configs_to_output per dataset; nothing to do here.
        return

def main():
    parser = argparse.ArgumentParser(description="Batch tailgating detection with configuration")
    parser.add_argument("--config", default="configs/global.yaml", help="Configuration file path")
    parser.add_argument("--datasets", nargs='*', help="Optional list of datasets to process (e.g., cisco vortex)")
    args = parser.parse_args()
    
    detector = BatchTailgatingDetectorV2(args.config)
    if args.datasets:
        # Restrict datasets to user-specified list
        detector.datasets = [ds for ds in detector.datasets if ds in set(args.datasets)]
    detector.run_batch_detection()

if __name__ == "__main__":
    main()
