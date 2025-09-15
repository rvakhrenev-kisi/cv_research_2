#!/usr/bin/env python3
"""
Step-by-step parameter tuning for people counting.
Tests one parameter at a time to find optimal values.
"""

import json
import subprocess
import os
from datetime import datetime

class StepByStepTuner:
    def __init__(self):
        self.ground_truth = self.load_ground_truth()
        self.current_params = {
            "confidence": 0.1,
            "iou": 0.3,
            "imgsz": 640
        }
        self.best_params = self.current_params.copy()
        self.best_accuracy = 0.0
        
    def load_ground_truth(self):
        """Load expected results from video_content.txt"""
        try:
            with open("video_content.txt", 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading ground truth: {e}")
            return {}
    
    def run_test(self, params=None):
        """Run a single test with current or given parameters"""
        if params is None:
            params = self.current_params
            
        print(f"\n🧪 Testing parameters: {params}")
        
        # Run batch detection
        cmd = [
            "python", "batch_tailgating_detection.py",
            "--model-size", "x",
            "--confidence", str(params["confidence"])
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            # Parse results (simplified - you'd need to implement proper parsing)
            accuracy = self.parse_and_calculate_accuracy(result.stdout)
            
            print(f"   📊 Accuracy: {accuracy:.3f}")
            
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_params = params.copy()
                print(f"   🏆 New best accuracy!")
            
            return accuracy
            
        except Exception as e:
            print(f"❌ Error running test: {e}")
            return 0.0
    
    def parse_and_calculate_accuracy(self, output):
        """Parse output and calculate accuracy (simplified)"""
        # This is a simplified version - you'd need to implement proper parsing
        # For now, return a mock accuracy based on some heuristics
        
        # Look for success indicators in output
        success_indicators = [
            "✅ Success:",
            "People count - Up:",
            "Total:"
        ]
        
        success_count = sum(1 for indicator in success_indicators if indicator in output)
        
        # Mock accuracy calculation (replace with real implementation)
        accuracy = min(1.0, success_count / 10.0)
        
        return accuracy
    
    def tune_confidence(self):
        """Tune confidence parameter"""
        print("\n🎯 Tuning CONFIDENCE parameter")
        print("=" * 40)
        
        confidence_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        best_confidence = self.current_params["confidence"]
        best_accuracy = 0.0
        
        for conf in confidence_values:
            params = self.current_params.copy()
            params["confidence"] = conf
            
            accuracy = self.run_test(params)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_confidence = conf
        
        self.current_params["confidence"] = best_confidence
        print(f"🏆 Best confidence: {best_confidence} (accuracy: {best_accuracy:.3f})")
    
    def tune_iou(self):
        """Tune IoU parameter"""
        print("\n🎯 Tuning IoU parameter")
        print("=" * 40)
        
        iou_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        best_iou = self.current_params["iou"]
        best_accuracy = 0.0
        
        for iou in iou_values:
            params = self.current_params.copy()
            params["iou"] = iou
            
            accuracy = self.run_test(params)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_iou = iou
        
        self.current_params["iou"] = best_iou
        print(f"🏆 Best IoU: {best_iou} (accuracy: {best_accuracy:.3f})")
    
    def tune_imgsz(self):
        """Tune image size parameter"""
        print("\n🎯 Tuning IMAGE SIZE parameter")
        print("=" * 40)
        
        imgsz_values = [320, 640, 1280]
        best_imgsz = self.current_params["imgsz"]
        best_accuracy = 0.0
        
        for imgsz in imgsz_values:
            params = self.current_params.copy()
            params["imgsz"] = imgsz
            
            accuracy = self.run_test(params)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_imgsz = imgsz
        
        self.current_params["imgsz"] = best_imgsz
        print(f"🏆 Best image size: {best_imgsz} (accuracy: {best_accuracy:.3f})")
    
    def run_full_tuning(self):
        """Run complete parameter tuning process"""
        print("🎯 Starting Step-by-Step Parameter Tuning")
        print("=" * 60)
        print(f"📋 Ground truth loaded: {len(self.ground_truth)} datasets")
        print(f"🔧 Starting parameters: {self.current_params}")
        
        # Step 1: Tune confidence
        self.tune_confidence()
        
        # Step 2: Tune IoU
        self.tune_iou()
        
        # Step 3: Tune image size
        self.tune_imgsz()
        
        # Final test with best parameters
        print(f"\n🏆 FINAL TEST with best parameters: {self.best_params}")
        final_accuracy = self.run_test(self.best_params)
        
        print(f"\n✅ TUNING COMPLETED!")
        print(f"🏆 Best parameters: {self.best_params}")
        print(f"🏆 Best accuracy: {self.best_accuracy:.3f}")
        
        # Save results
        results = {
            "best_params": self.best_params,
            "best_accuracy": self.best_accuracy,
            "final_accuracy": final_accuracy,
            "timestamp": datetime.now().isoformat()
        }
        
        with open("best_parameters.json", 'w') as f:
            json.dump(results, f, indent=2)
        print("💾 Results saved to best_parameters.json")

def main():
    tuner = StepByStepTuner()
    tuner.run_full_tuning()

if __name__ == "__main__":
    main()
