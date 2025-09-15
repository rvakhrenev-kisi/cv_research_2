#!/usr/bin/env python3
"""
Run improved batch detection with multiple confidence levels using YOLOv10x
and optimized parameters for CCTV ceiling cameras.
"""
import subprocess
import sys
import time
from datetime import datetime

def run_detection(confidence: float) -> None:
    """Run batch detection with given confidence level using improved parameters."""
    print(f"\n{'='*60}")
    print(f"🚀 Running IMPROVED detection with confidence: {confidence}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Using YOLOv10x with optimized tracking parameters")
    print(f"📊 Lower IoU (0.3) to prevent people merging")
    print(f"🎯 Higher tracking confidence (0.5) for better line crossing")
    print(f"{'='*60}")
    
    cmd = [
        "python", "batch_tailgating_detection.py",
        "--model-size", "x",  # Use YOLOv10x for best accuracy
        "--confidence", str(confidence)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ Confidence {confidence} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Confidence {confidence} failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print(f"⏹️  Confidence {confidence} interrupted by user")
        sys.exit(1)

def main():
    """Run all confidence levels with improved parameters."""
    confidence_levels = [0.05, 0.1, 0.3]
    
    print("🎯 Starting IMPROVED confidence level comparison")
    print(f"📊 Testing levels: {confidence_levels}")
    print(f"🔧 Model: YOLOv10x (most accurate)")
    print(f"🎯 Optimizations:")
    print(f"   - Lower IoU threshold (0.3) to prevent people merging")
    print(f"   - Higher tracking confidence (0.5) for better line crossing")
    print(f"   - Improved ByteTrack parameters")
    print(f"   - CCTV ceiling camera optimizations")
    print(f"🕐 Total start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for i, confidence in enumerate(confidence_levels, 1):
        print(f"\n📈 Test {i}/{len(confidence_levels)}: Confidence {confidence}")
        run_detection(confidence)
        
        if i < len(confidence_levels):
            print(f"⏳ Waiting 5 seconds before next test...")
            time.sleep(5)
    
    print(f"\n🎉 All improved confidence tests completed!")
    print(f"🕐 Total end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📋 Check the output directories for:")
    print(f"   - tuned_parameters.txt (saved parameters)")
    print(f"   - processing_summary.json (detailed results)")
    print(f"   - Individual video outputs with improved detection")

if __name__ == "__main__":
    main()
