#!/usr/bin/env python3
"""
Download YOLOv11 models for enhanced detection and tracking.
Based on Medium article recommendations.
"""

import os
import subprocess
import platform
from pathlib import Path

def download_yolov11_models():
    """Download YOLOv11 models"""
    print("🚀 YOLOv11 Model Downloader")
    print("=" * 50)
    print("Based on Medium article: YOLOv11 + ByteTrack recommendations")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # YOLOv11 model variants (from smallest to largest)
    models = {
        "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "yolo11s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt", 
        "yolo11m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        "yolo11l.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
        "yolo11x.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt"
    }
    
    print(f"📁 Models directory: {models_dir.absolute()}")
    print(f"🎯 Downloading {len(models)} YOLOv11 models...")
    
    success_count = 0
    
    for model_name, url in models.items():
        model_path = models_dir / model_name
        
        if model_path.exists():
            file_size = model_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ {model_name} already exists ({file_size:.1f} MB)")
            success_count += 1
            continue
        
        print(f"\n📥 Downloading {model_name}...")
        print(f"   URL: {url}")
        
        try:
            system = platform.system().lower()
            
            if system == "windows":
                # Use curl on Windows
                cmd = ["curl", "-L", "-o", str(model_path), url]
            else:
                # Use wget on Linux/Mac
                cmd = ["wget", url, "-O", str(model_path)]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if model_path.exists():
                file_size = model_path.stat().st_size / (1024 * 1024)  # MB
                print(f"   ✅ Downloaded successfully ({file_size:.1f} MB)")
                success_count += 1
            else:
                print(f"   ❌ Download failed - file not created")
                
        except FileNotFoundError:
            print(f"   ❌ curl/wget not found. Please download manually:")
            print(f"      {url}")
            print(f"      Save to: {model_path}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Download failed: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Successfully downloaded: {success_count}/{len(models)} models")
    print(f"   📁 Models saved to: {models_dir.absolute()}")
    
    if success_count > 0:
        print(f"\n🎯 Recommended models for your use case:")
        print(f"   🚀 yolo11n.pt - Fastest, good for real-time")
        print(f"   ⚖️  yolo11s.pt - Balanced speed/accuracy")
        print(f"   🎯 yolo11m.pt - Good accuracy for CCTV")
        print(f"   🔥 yolo11l.pt - High accuracy")
        print(f"   💎 yolo11x.pt - Maximum accuracy (slowest)")
        print(f"\n💡 For CCTV people counting, try yolo11m.pt or yolo11l.pt first")
    
    return success_count > 0

def main():
    success = download_yolov11_models()
    
    if success:
        print(f"\n✅ YOLOv11 models ready!")
        print(f"🚀 You can now use YOLOv11 in your tuning scripts")
    else:
        print(f"\n❌ Download failed. Please download models manually.")

if __name__ == "__main__":
    main()
