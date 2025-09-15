#!/usr/bin/env python3
"""
Download YOLOv10 models for tailgating detection.
"""

import os
import subprocess
from pathlib import Path

def download_yolov10_models():
    """Download YOLOv10 models."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    models = {
        "yolov10n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt",
        "yolov10s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10s.pt",
        "yolov10m.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10m.pt"
    }
    
    for model_name, url in models.items():
        model_path = models_dir / model_name
        if model_path.exists():
            print(f"✅ {model_name} already exists")
            continue
        
        print(f"📥 Downloading {model_name}...")
        try:
            subprocess.run(["wget", "-O", str(model_path), url], check=True)
            print(f"✅ Downloaded {model_name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download {model_name}: {e}")
        except FileNotFoundError:
            print(f"❌ wget not found. Please install wget or download manually:")
            print(f"   {url} -> {model_path}")

if __name__ == "__main__":
    download_yolov10_models()
