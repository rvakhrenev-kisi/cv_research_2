#!/usr/bin/env python3
"""
Run batch detection with multiple confidence levels
"""
import subprocess
import sys
import time
from datetime import datetime

def run_detection(confidence: float) -> None:
    """Run batch detection with given confidence level"""
    print(f"\n{'='*60}")
    print(f"🚀 Running detection with confidence: {confidence}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    cmd = [
        "python", "batch_tailgating_detection.py",
        "--model-size", "x",
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
    """Run all confidence levels"""
    confidence_levels = [0.05, 0.1, 0.3]
    
    print("🎯 Starting confidence level comparison")
    print(f"📊 Testing levels: {confidence_levels}")
    print(f"🕐 Total start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for i, confidence in enumerate(confidence_levels, 1):
        print(f"\n📈 Test {i}/{len(confidence_levels)}: Confidence {confidence}")
        run_detection(confidence)
        
        if i < len(confidence_levels):
            print(f"⏳ Waiting 5 seconds before next test...")
            time.sleep(5)
    
    print(f"\n🎉 All confidence tests completed!")
    print(f"🕐 Total end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
