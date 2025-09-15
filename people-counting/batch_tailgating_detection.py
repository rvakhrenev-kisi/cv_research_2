#!/usr/bin/env python3
"""
Batch tailgating detection script for Cisco and Vortex videos.
Runs YOLOv10 on all videos with saved line configurations.
"""

import os
import cv2
import json
import subprocess
import datetime
from pathlib import Path
import argparse
from tqdm import tqdm

class BatchTailgatingDetector:
    def __init__(self, config_dir="configs", input_dir="input", output_dir="outputs"):
        self.config_dir = Path(config_dir)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.datasets = ["cisco", "vortex"]
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        print(f"📁 Run directory: {self.run_dir}")
    
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

    def process_video(self, video_path, dataset, line_config, model_path, confidence=0.1):
        """Process a single video with CCTV-optimized parameters."""
        try:
            # Create output filename
            video_name = video_path.stem
            output_name = f"{dataset}_{video_name}.mp4"
            output_path = self.run_dir / output_name
            
            # Build command with CCTV-optimized parameters
            cmd = [
                "python", "people_counter.py",
                "--video", str(video_path),
                "--model", model_path,
                "--model-type", "yolo12",
                "--line-start", str(line_config["x1"]), str(line_config["y1"]),
                "--line-end", str(line_config["x2"]), str(line_config["y2"]),
                "--confidence", str(confidence),  # Much lower confidence for CCTV
                "--output", str(output_path),
                "--output-height", "0",  # 0 = original resolution for better quality
                "--verbose"  # Enable verbose output to get counts
            ]
            
            print(f"🎬 Processing: {video_name}")
            print(f"   Command: {' '.join(cmd)}")
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
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
    
    def run_batch_detection(self, model_size="n", confidence=0.1):
        """Run batch detection optimized for CCTV ceiling cameras."""
        print(f"🚀 Starting CCTV-optimized batch tailgating detection")
        print(f"   Model: YOLOv8{model_size}")
        print(f"   Confidence: {confidence} (very low for CCTV ceiling cameras)")
        print(f"   Output Quality: Original resolution")
        print(f"   Optimized for: Ceiling-mounted CCTV cameras")
        print(f"   Output: {self.run_dir}")
        print()
        
        # Check if model exists
        model_path = f"models/yolov8{model_size}.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            print("   Please download YOLOv8 models first")
            return
        
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
        
        # Save results summary
        self.save_results_summary(results)
        
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
            "results": results
        }
        
        summary_file = self.run_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Summary saved: {summary_file}")

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
