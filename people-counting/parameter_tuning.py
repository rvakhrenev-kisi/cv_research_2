#!/usr/bin/env python3
"""
Parameter tuning script for people counting detection.
Uses video_content.txt as ground truth to find optimal parameters.
"""

import json
import subprocess
import os
import time
from datetime import datetime
from pathlib import Path
import itertools
from typing import Dict, List, Tuple, Any

class ParameterTuner:
    def __init__(self, ground_truth_file="video_content.txt"):
        self.ground_truth_file = ground_truth_file
        self.ground_truth = self.load_ground_truth()
        self.results = []
        
    def load_ground_truth(self) -> Dict:
        """Load expected results from video_content.txt"""
        try:
            with open(self.ground_truth_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading ground truth: {e}")
            return {}
    
    def run_detection(self, params: Dict) -> Dict:
        """Run detection with given parameters"""
        cmd = [
            "python", "batch_tailgating_detection.py",
            "--model-size", params.get("model_size", "x"),
            "--confidence", str(params.get("confidence", 0.1))
        ]
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"tuning_run_{timestamp}"
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            # Parse results from output
            results = self.parse_results(result.stdout, run_dir)
            return results
        except Exception as e:
            print(f"❌ Error running detection: {e}")
            return {}
    
    def parse_results(self, output: str, run_dir: str) -> Dict:
        """Parse detection results from output"""
        results = {"cisco": {}, "vortex": {}}
        
        # Extract counts from output (this would need to be implemented based on actual output format)
        # For now, return placeholder structure
        return results
    
    def calculate_accuracy(self, predicted: Dict, expected: Dict) -> Dict:
        """Calculate accuracy metrics"""
        metrics = {
            "cisco": {"total_error": 0, "videos": 0, "accuracy": 0.0},
            "vortex": {"total_error": 0, "videos": 0, "accuracy": 0.0},
            "overall": {"total_error": 0, "videos": 0, "accuracy": 0.0}
        }
        
        for dataset in ["cisco", "vortex"]:
            if dataset in predicted and dataset in expected:
                for video_id, pred_data in predicted[dataset].items():
                    if video_id in expected[dataset]:
                        expected_count = expected[dataset][video_id]["expected_people"]
                        predicted_count = pred_data.get("total_count", 0)
                        
                        error = abs(predicted_count - expected_count)
                        metrics[dataset]["total_error"] += error
                        metrics[dataset]["videos"] += 1
                        
                        metrics["overall"]["total_error"] += error
                        metrics["overall"]["videos"] += 1
        
        # Calculate accuracy (lower error = higher accuracy)
        for dataset in metrics:
            if metrics[dataset]["videos"] > 0:
                max_possible_error = metrics[dataset]["videos"] * 10  # Assume max 10 people per video
                accuracy = max(0, 1 - (metrics[dataset]["total_error"] / max_possible_error))
                metrics[dataset]["accuracy"] = accuracy
        
        return metrics
    
    def grid_search(self, param_ranges: Dict) -> List[Dict]:
        """Perform grid search over parameter ranges"""
        print("🔍 Starting grid search for optimal parameters...")
        
        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        combinations = list(itertools.product(*param_values))
        
        print(f"📊 Testing {len(combinations)} parameter combinations...")
        
        best_results = []
        
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            print(f"\n🧪 Test {i+1}/{len(combinations)}: {params}")
            
            # Run detection
            results = self.run_detection(params)
            
            # Calculate accuracy
            accuracy = self.calculate_accuracy(results, self.ground_truth)
            
            # Store results
            result_entry = {
                "params": params,
                "accuracy": accuracy,
                "results": results
            }
            best_results.append(result_entry)
            
            print(f"   📊 Overall accuracy: {accuracy['overall']['accuracy']:.3f}")
            print(f"   📊 Cisco accuracy: {accuracy['cisco']['accuracy']:.3f}")
            print(f"   📊 Vortex accuracy: {accuracy['vortex']['accuracy']:.3f}")
            
            # Save intermediate results
            self.save_results(best_results, f"tuning_progress_{i+1}.json")
        
        return best_results
    
    def save_results(self, results: List[Dict], filename: str):
        """Save tuning results to file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {filename}")
    
    def find_best_parameters(self, results: List[Dict]) -> Dict:
        """Find the best parameter combination"""
        best_result = max(results, key=lambda x: x["accuracy"]["overall"]["accuracy"])
        return best_result

def main():
    """Main tuning function"""
    print("🎯 Parameter Tuning for People Counting Detection")
    print("=" * 60)
    
    # Initialize tuner
    tuner = ParameterTuner()
    
    # Define parameter ranges to test
    param_ranges = {
        "confidence": [0.05, 0.1, 0.15, 0.2, 0.3],
        "iou": [0.3, 0.4, 0.5, 0.6],
        "imgsz": [640, 1280],
        "model_size": ["x"]  # Focus on YOLOv10x
    }
    
    print(f"📋 Ground truth loaded: {len(tuner.ground_truth)} datasets")
    print(f"🔧 Parameter ranges: {param_ranges}")
    
    # Run grid search
    results = tuner.grid_search(param_ranges)
    
    # Find best parameters
    best = tuner.find_best_parameters(results)
    
    print("\n🏆 BEST PARAMETERS FOUND:")
    print("=" * 40)
    print(f"Parameters: {best['params']}")
    print(f"Overall accuracy: {best['accuracy']['overall']['accuracy']:.3f}")
    print(f"Cisco accuracy: {best['accuracy']['cisco']['accuracy']:.3f}")
    print(f"Vortex accuracy: {best['accuracy']['vortex']['accuracy']:.3f}")
    
    # Save final results
    tuner.save_results(results, "final_tuning_results.json")
    tuner.save_results([best], "best_parameters.json")
    
    print("\n✅ Parameter tuning completed!")

if __name__ == "__main__":
    main()
