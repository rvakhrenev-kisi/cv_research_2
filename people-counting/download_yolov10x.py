#!/usr/bin/env python3
"""
Download YOLOv10x model for improved detection accuracy
"""
import os
import subprocess
import sys
from pathlib import Path

def download_yolov10x():
    """Download YOLOv10x model."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "yolov10x.pt"
    
    if model_path.exists():
        print(f"✅ YOLOv10x model already exists: {model_path}")
        return True
    
    print("📥 Downloading YOLOv10x model...")
    print("   This may take a few minutes (61MB download)")
    
    try:
        # Download from ultralytics releases
        cmd = [
            "wget", 
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10x.pt",
            "-O", str(model_path)
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✅ Successfully downloaded YOLOv10x model")
            print(f"   Size: {size_mb:.1f} MB")
            print(f"   Path: {model_path}")
            return True
        else:
            print("❌ Download failed - file not found after download")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print(f"   Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ wget not found. Please install wget or download manually:")
        print(f"   URL: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10x.pt")
        print(f"   Save to: {model_path}")
        return False

def main():
    """Main function."""
    print("🚀 YOLOv10x Model Downloader")
    print("=" * 40)
    
    success = download_yolov10x()
    
    if success:
        print("\n🎉 Ready to use YOLOv10x for improved detection!")
        print("   Run: python batch_tailgating_detection.py --model-size x")
    else:
        print("\n❌ Failed to download YOLOv10x model")
        sys.exit(1)

if __name__ == "__main__":
    main()
