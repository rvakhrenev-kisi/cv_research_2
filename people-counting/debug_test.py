#!/usr/bin/env python3
"""
Debug script to test what's happening with people_counter.py
"""

import subprocess
import os

def test_single_run():
    """Test a single run of people_counter.py with detailed output"""
    print("🔍 Debug Test - Single Run")
    print("=" * 50)
    
    cmd = [
        "python", "people_counter.py",
        "--video", "../cisco/1.mp4",
        "--model", "models/yolov10x.pt",
        "--model-type", "yolo12",
        "--line-start", "521", "898",
        "--line-end", "737", "622",
        "--confidence", "0.1",
        "--output", "debug_test.mp4",
        "--output-height", "0",
        "--verbose"
    ]
    
    print(f"🚀 Running command:")
    print(f"   {' '.join(cmd)}")
    print()
    
    try:
        print("⏱️  Starting video processing...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), timeout=600)
        
        print(f"\n📤 Return code: {result.returncode}")
        print(f"📤 STDOUT length: {len(result.stdout)} characters")
        print(f"📤 STDERR length: {len(result.stderr)} characters")
        
        print(f"\n📤 STDOUT:")
        print("-" * 50)
        print(result.stdout)
        print("-" * 50)
        
        if result.stderr:
            print(f"\n📤 STDERR:")
            print("-" * 50)
            print(result.stderr)
            print("-" * 50)
        
        # Check if output file was created
        if os.path.exists("debug_test.mp4"):
            file_size = os.path.getsize("debug_test.mp4")
            print(f"\n✅ Output file created: debug_test.mp4 ({file_size} bytes)")
        else:
            print(f"\n❌ Output file NOT created")
        
        # Parse for counts
        print(f"\n🔍 Looking for count patterns in output...")
        lines = result.stdout.split('\n')
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['total', 'count', 'people', 'up', 'down']):
                print(f"   Line {i}: {line}")
        
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out after 10 minutes")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_single_run()
