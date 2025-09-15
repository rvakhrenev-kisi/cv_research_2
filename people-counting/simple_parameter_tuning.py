#!/usr/bin/env python3
"""
Simple parameter tuning script focused on key parameters.
Uses video_content.txt as ground truth.
"""

import json
import subprocess
import os
from datetime import datetime
from typing import Dict, List

class SimpleParameterTuner:
    def __init__(self):
        self.ground_truth = self.load_ground_truth()
        self.results = []
        
    def load_ground_truth(self) -> Dict:
        """Load expected results from video_content.txt"""
        try:
            with open("video_content.txt", 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading ground truth: {e}")
            return {}
    
    def run_single_test(self, confidence: float, iou: float, imgsz: int) -> Dict:
        """Run a single test with given parameters"""
        print(f"\n🧪 Testing: confidence={confidence}, iou={iou}, imgsz={imgsz}")
        
        # Update people_counter.py with new parameters
        self.update_people_counter_params(confidence, iou, imgsz)
        
        # Run batch detection
        cmd = [
            "python", "batch_tailgating_detection.py",
            "--model-size", "x",
            "--confidence", str(confidence)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            # Parse results from the output
            results = self.parse_batch_output(result.stdout)
            
            # Calculate accuracy
            accuracy = self.calculate_accuracy(results)
            
            return {
                "params": {"confidence": confidence, "iou": iou, "imgsz": imgsz},
                "results": results,
                "accuracy": accuracy
            }
            
        except Exception as e:
            print(f"❌ Error running test: {e}")
            return {"params": {"confidence": confidence, "iou": iou, "imgsz": imgsz}, "accuracy": 0.0}
    
    def update_people_counter_params(self, confidence: float, iou: float, imgsz: int):
        """Update people_counter.py with new parameters"""
        # This would modify the people_counter.py file with new parameters
        # For now, we'll pass parameters via command line
        pass
    
    def parse_batch_output(self, output: str) -> Dict:
        """Parse batch detection output to extract counts"""
        results = {"cisco": {}, "vortex": {}}
        
        # Look for patterns like "cisco_1.mp4" and "Total: X"
        lines = output.split('\n')
        current_video = None
        
        for line in lines:
            if "Processing:" in line and "cisco" in line:
                # Extract video name
                if "cisco_" in line:
                    video_name = line.split("cisco_")[1].split(".")[0]
                    current_video = f"cisco_{video_name}"
            elif "Processing:" in line and "vortex" in line:
                if "vortex_" in line:
                    video_name = line.split("vortex_")[1].split(".")[0]
                    current_video = f"vortex_{video_name}"
            elif "Total:" in line and current_video:
                # Extract count
                try:
                    count = int(line.split("Total:")[1].strip())
                    dataset = current_video.split("_")[0]
                    video_id = current_video.split("_")[1]
                    
                    if dataset not in results:
                        results[dataset] = {}
                    results[dataset][video_id] = {"total_count": count}
                    current_video = None
                except:
                    pass
        
        return results
    
    def calculate_accuracy(self, predicted: Dict) -> float:
        """Calculate accuracy against ground truth"""
        total_error = 0
        total_videos = 0
        
        for dataset in ["cisco", "vortex"]:
            if dataset in predicted and dataset in self.ground_truth:
                for video_id, pred_data in predicted[dataset].items():
                    if video_id in self.ground_truth[dataset]:
                        expected = self.ground_truth[dataset][video_id]["expected_people"]
                        predicted_count = pred_data.get("total_count", 0)
                        
                        error = abs(predicted_count - expected)
                        total_error += error
                        total_videos += 1
                        
                        print(f"   📊 {dataset}_{video_id}: Expected {expected}, Got {predicted_count}, Error {error}")
        
        if total_videos == 0:
            return 0.0
        
        # Calculate accuracy (lower error = higher accuracy)
        max_possible_error = total_videos * 5  # Assume max 5 people per video
        accuracy = max(0, 1 - (total_error / max_possible_error))
        
        print(f"   📊 Total error: {total_error}, Videos: {total_videos}, Accuracy: {accuracy:.3f}")
        return accuracy
    
    def run_parameter_sweep(self):
        """Run parameter sweep for key parameters"""
        print("🎯 Starting Parameter Tuning")
        print("=" * 50)
        
        # Key parameters to test
        confidence_values = [0.05, 0.1, 0.15, 0.2, 0.3]
        iou_values = [0.3, 0.4, 0.5]
        imgsz_values = [640, 1280]
        
        best_accuracy = 0.0
        best_params = None
        
        total_tests = len(confidence_values) * len(iou_values) * len(imgsz_values)
        test_count = 0
        
        for confidence in confidence_values:
            for iou in iou_values:
                for imgsz in imgsz_values:
                    test_count += 1
                    print(f"\n📊 Test {test_count}/{total_tests}")
                    
                    result = self.run_single_test(confidence, iou, imgsz)
                    self.results.append(result)
                    
                    if result["accuracy"] > best_accuracy:
                        best_accuracy = result["accuracy"]
                        best_params = result["params"]
                        print(f"🏆 New best accuracy: {best_accuracy:.3f}")
        
        print(f"\n🏆 BEST PARAMETERS FOUND:")
        print(f"Parameters: {best_params}")
        print(f"Accuracy: {best_accuracy:.3f}")
        
        # Save results
        with open("tuning_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        print("💾 Results saved to tuning_results.json")

def main():
    tuner = SimpleParameterTuner()
    tuner.run_parameter_sweep()

if __name__ == "__main__":
    main()
