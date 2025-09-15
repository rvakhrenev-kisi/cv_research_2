#!/usr/bin/env python3
"""
Quick visual parameter tuning - focuses on most important parameters.
"""

import json
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class QuickVisualTuner:
    def __init__(self):
        self.ground_truth = self.load_ground_truth()
        self.results = []
        self.output_dir = "quick_tuning_results"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📋 Ground truth loaded: {len(self.ground_truth)} datasets")
        
    def load_ground_truth(self):
        """Load expected results from video_content.txt"""
        try:
            with open("video_content.txt", 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading ground truth: {e}")
            return {}
    
    def run_quick_test(self, confidence=0.1):
        """Run quick test on single video"""
        print(f"\n🧪 Quick test: confidence={confidence}")
        
        cmd = [
            "python", "people_counter.py",
            "--video", "../cisco/1.mp4",
            "--model", "models/yolov10x.pt",
            "--model-type", "yolo12",
            "--line-start", "521", "898",
            "--line-end", "737", "622",
            "--confidence", str(confidence),
            "--output", f"quick_test_{confidence}.mp4",
            "--output-height", "0",
            "--verbose"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), timeout=300)
            
            # Parse results
            counts = self.parse_counts(result.stdout)
            expected = self.ground_truth["cisco"]["1"]["expected_people"]
            predicted = counts.get("total", 0)
            error = abs(predicted - expected)
            accuracy = max(0, 1 - (error / expected)) if expected > 0 else 0.0
            
            print(f"   📊 Expected: {expected}, Got: {predicted}, Accuracy: {accuracy:.3f}")
            
            # Clean up
            test_file = f"quick_test_{confidence}.mp4"
            if os.path.exists(test_file):
                os.remove(test_file)
            
            return accuracy, counts
            
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Timeout for confidence {confidence}")
            return 0.0, {}
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return 0.0, {}
    
    def parse_counts(self, output):
        """Parse people counts from output"""
        counts = {"up": 0, "down": 0, "total": 0}
        
        for line in output.split('\n'):
            if "People count - Up:" in line:
                try:
                    counts["up"] = int(line.split("Up:")[1].strip())
                except:
                    pass
            elif "People count - Down:" in line:
                try:
                    counts["down"] = int(line.split("Down:")[1].strip())
                except:
                    pass
            elif "Total:" in line:
                try:
                    counts["total"] = int(line.split("Total:")[1].strip())
                except:
                    pass
        
        return counts
    
    def test_confidence_range(self):
        """Test confidence range with visualization"""
        print("🎯 Testing Confidence Range (Quick)")
        print("=" * 50)
        
        # Focus on most promising range
        confidence_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        results = []
        
        for conf in confidence_values:
            accuracy, counts = self.run_quick_test(conf)
            results.append({
                "confidence": conf,
                "accuracy": accuracy,
                "counts": counts
            })
        
        # Find best
        best_result = max(results, key=lambda x: x["accuracy"])
        
        print(f"\n🏆 BEST CONFIDENCE: {best_result['confidence']}")
        print(f"🏆 Best accuracy: {best_result['accuracy']:.3f}")
        
        # Create visualization
        self.create_confidence_plot(results)
        
        return best_result
    
    def create_confidence_plot(self, results):
        """Create confidence tuning visualization"""
        confidences = [r["confidence"] for r in results]
        accuracies = [r["accuracy"] for r in results]
        
        plt.figure(figsize=(12, 8))
        
        # Main plot
        plt.subplot(2, 1, 1)
        plt.plot(confidences, accuracies, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Accuracy')
        plt.title('Confidence Tuning Results')
        plt.grid(True, alpha=0.3)
        
        # Highlight best
        best_idx = np.argmax(accuracies)
        plt.plot(confidences[best_idx], accuracies[best_idx], 'ro', markersize=12, 
                label=f'Best: {confidences[best_idx]} (acc: {accuracies[best_idx]:.3f})')
        plt.legend()
        
        # Detailed view of low confidence range
        plt.subplot(2, 1, 2)
        low_conf_mask = np.array(confidences) <= 0.2
        if np.any(low_conf_mask):
            low_conf = np.array(confidences)[low_conf_mask]
            low_acc = np.array(accuracies)[low_conf_mask]
            plt.plot(low_conf, low_acc, 'go-', linewidth=2, markersize=8)
            plt.xlabel('Confidence Threshold (Low Range)')
            plt.ylabel('Accuracy')
            plt.title('Detailed View: Low Confidence Range')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.output_dir, "confidence_tuning_plot.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plot saved to {plot_file}")
        
        # Save data
        data_file = os.path.join(self.output_dir, "confidence_results.json")
        with open(data_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Data saved to {data_file}")

def main():
    print("🎯 Quick Visual Parameter Tuning")
    print("=" * 50)
    
    tuner = QuickVisualTuner()
    
    # Test confidence range
    best_confidence = tuner.test_confidence_range()
    
    print(f"\n✅ Quick tuning completed!")
    print(f"📁 Results saved to: {tuner.output_dir}/")
    print(f"🏆 Best confidence: {best_confidence['confidence']} (accuracy: {best_confidence['accuracy']:.3f})")

if __name__ == "__main__":
    main()
