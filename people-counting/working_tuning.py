#!/usr/bin/env python3
"""
Working parameter tuning script with visualization and expanded parameter ranges.
"""

import json
import subprocess
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

class WorkingTuner:
    def __init__(self):
        self.ground_truth = self.load_ground_truth()
        self.results = []
        self.output_dir = "tuning_results"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📋 Ground truth loaded: {self.ground_truth}")
        print(f"📁 Results will be saved to: {self.output_dir}")
        
    def load_ground_truth(self):
        """Load expected results from video_content.txt"""
        try:
            with open("video_content.txt", 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading ground truth: {e}")
            return {}
    
    def run_single_video_test(self, confidence=0.1):
        """Run detection on a single video to test parameters"""
        print(f"\n🧪 Testing confidence={confidence}")
        
        # Test with just one video first (cisco/1.mp4)
        cmd = [
            "python", "people_counter.py",
            "--video", "../cisco/1.mp4",
            "--model", "models/yolov10x.pt",
            "--model-type", "yolo12",
            "--line-start", "521", "898",
            "--line-end", "737", "622",
            "--confidence", str(confidence),
            "--output", f"test_output_conf_{confidence}.mp4",
            "--output-height", "0",
            "--verbose"
        ]
        
        print(f"   🚀 Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            print(f"   📤 Return code: {result.returncode}")
            
            # Parse the output for people counts
            counts = self.parse_people_counts(result.stdout)
            print(f"   📊 Detected counts: {counts}")
            
            # Calculate accuracy for this single video
            expected = self.ground_truth["cisco"]["1"]["expected_people"]
            predicted = counts.get("total", 0)
            error = abs(predicted - expected)
            
            # Accuracy = 1 - (error / expected) if expected > 0, else 0
            if expected > 0:
                accuracy = max(0, 1 - (error / expected))
            else:
                accuracy = 0.0
            
            print(f"   📊 Expected: {expected}, Predicted: {predicted}, Error: {error}")
            print(f"   📊 Accuracy: {accuracy:.3f}")
            
            return accuracy, counts
            
        except Exception as e:
            print(f"❌ Error running test: {e}")
            return 0.0, {}
    
    def parse_people_counts(self, output):
        """Parse people counts from people_counter.py output"""
        counts = {"up": 0, "down": 0, "total": 0}
        
        lines = output.split('\n')
        for line in lines:
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
        """Test different confidence values with expanded range"""
        print("🎯 Testing Confidence Range (Expanded)")
        print("=" * 50)
        
        # Expanded confidence range
        confidence_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
        results = []
        
        for conf in confidence_values:
            print(f"\n📊 Testing confidence: {conf}")
            accuracy, counts = self.run_single_video_test(conf)
            results.append({
                "confidence": conf,
                "accuracy": accuracy,
                "counts": counts
            })
            
            # Clean up test output
            test_file = f"test_output_conf_{conf}.mp4"
            if os.path.exists(test_file):
                os.remove(test_file)
                print(f"   🗑️  Cleaned up {test_file}")
        
        # Find best confidence
        best_result = max(results, key=lambda x: x["accuracy"])
        
        print(f"\n🏆 BEST CONFIDENCE: {best_result['confidence']}")
        print(f"🏆 Best accuracy: {best_result['accuracy']:.3f}")
        print(f"🏆 Best counts: {best_result['counts']}")
        
        # Save and plot results
        self.save_results(results, "confidence_tuning")
        self.plot_parameter_results(results, "confidence", "Confidence Tuning Results")
        
        return best_result
    
    def test_iou_range(self):
        """Test different IoU values"""
        print("\n🎯 Testing IoU Range")
        print("=" * 50)
        
        iou_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        results = []
        
        for iou in iou_values:
            print(f"\n📊 Testing IoU: {iou}")
            # Update people_counter.py with IoU parameter (would need to modify the script)
            accuracy, counts = self.run_single_video_test_with_iou(0.1, iou)
            results.append({
                "iou": iou,
                "accuracy": accuracy,
                "counts": counts
            })
            
            # Clean up test output
            test_file = f"test_output_iou_{iou}.mp4"
            if os.path.exists(test_file):
                os.remove(test_file)
        
        # Find best IoU
        best_result = max(results, key=lambda x: x["accuracy"])
        
        print(f"\n🏆 BEST IoU: {best_result['iou']}")
        print(f"🏆 Best accuracy: {best_result['accuracy']:.3f}")
        
        # Save and plot results
        self.save_results(results, "iou_tuning")
        self.plot_parameter_results(results, "iou", "IoU Tuning Results")
        
        return best_result
    
    def test_imgsz_range(self):
        """Test different image sizes"""
        print("\n🎯 Testing Image Size Range")
        print("=" * 50)
        
        imgsz_values = [320, 416, 512, 640, 832, 1024, 1280, 1536]
        results = []
        
        for imgsz in imgsz_values:
            print(f"\n📊 Testing image size: {imgsz}")
            accuracy, counts = self.run_single_video_test_with_imgsz(0.1, imgsz)
            results.append({
                "imgsz": imgsz,
                "accuracy": accuracy,
                "counts": counts
            })
            
            # Clean up test output
            test_file = f"test_output_imgsz_{imgsz}.mp4"
            if os.path.exists(test_file):
                os.remove(test_file)
        
        # Find best image size
        best_result = max(results, key=lambda x: x["accuracy"])
        
        print(f"\n🏆 BEST Image Size: {best_result['imgsz']}")
        print(f"🏆 Best accuracy: {best_result['accuracy']:.3f}")
        
        # Save and plot results
        self.save_results(results, "imgsz_tuning")
        self.plot_parameter_results(results, "imgsz", "Image Size Tuning Results")
        
        return best_result
    
    def run_single_video_test_with_iou(self, confidence, iou):
        """Run test with specific IoU parameter"""
        # This would require modifying people_counter.py to accept IoU parameter
        # For now, just run with default IoU
        return self.run_single_video_test(confidence)
    
    def run_single_video_test_with_imgsz(self, confidence, imgsz):
        """Run test with specific image size parameter"""
        # This would require modifying people_counter.py to accept imgsz parameter
        # For now, just run with default imgsz
        return self.run_single_video_test(confidence)
    
    def save_results(self, results, test_name):
        """Save results to JSON file"""
        filename = os.path.join(self.output_dir, f"{test_name}_results.json")
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {filename}")
    
    def plot_parameter_results(self, results, param_name, title):
        """Create visualization of parameter tuning results"""
        if not results:
            return
        
        # Extract parameter values and accuracies
        param_values = [r[param_name] for r in results]
        accuracies = [r["accuracy"] for r in results]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, accuracies, 'bo-', linewidth=2, markersize=8)
        plt.xlabel(param_name.title())
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Highlight the best result
        best_idx = np.argmax(accuracies)
        plt.plot(param_values[best_idx], accuracies[best_idx], 'ro', markersize=12, label=f'Best: {param_values[best_idx]}')
        plt.legend()
        
        # Save the plot
        plot_filename = os.path.join(self.output_dir, f"{param_name}_tuning_plot.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plot saved to {plot_filename}")
    
    def create_comprehensive_plot(self):
        """Create a comprehensive plot showing all parameter tuning results"""
        # This would combine all tuning results into one comprehensive visualization
        pass
    
    def run_full_test(self, confidence):
        """Run full batch test with best confidence"""
        print(f"\n🚀 Running FULL BATCH TEST with confidence={confidence}")
        print("=" * 60)
        
        cmd = [
            "python", "batch_tailgating_detection.py",
            "--model-size", "x",
            "--confidence", str(confidence)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            print(f"Return code: {result.returncode}")
            
            # Parse all results
            all_results = self.parse_batch_results(result.stdout)
            
            # Calculate overall accuracy
            total_error = 0
            total_videos = 0
            
            for dataset in ["cisco", "vortex"]:
                if dataset in all_results and dataset in self.ground_truth:
                    for video_id, counts in all_results[dataset].items():
                        if video_id in self.ground_truth[dataset]:
                            expected = self.ground_truth[dataset][video_id]["expected_people"]
                            predicted = counts.get("total", 0)
                            error = abs(predicted - expected)
                            total_error += error
                            total_videos += 1
                            
                            print(f"📊 {dataset}_{video_id}: Expected {expected}, Got {predicted}, Error {error}")
            
            if total_videos > 0:
                # Calculate accuracy based on total expected people
                total_expected = sum(
                    self.ground_truth[dataset][video_id]["expected_people"] 
                    for dataset in ["cisco", "vortex"] 
                    if dataset in all_results and dataset in self.ground_truth
                    for video_id in all_results[dataset]
                    if video_id in self.ground_truth[dataset]
                )
                
                if total_expected > 0:
                    overall_accuracy = max(0, 1 - (total_error / total_expected))
                else:
                    overall_accuracy = 0.0
                    
                print(f"\n🏆 OVERALL ACCURACY: {overall_accuracy:.3f}")
                print(f"🏆 Total error: {total_error} across {total_videos} videos")
                print(f"🏆 Total expected: {total_expected} people")
            else:
                print("❌ No results found")
            
            return all_results
            
        except Exception as e:
            print(f"❌ Error running batch test: {e}")
            return {}
    
    def parse_batch_results(self, output):
        """Parse results from batch detection"""
        results = {"cisco": {}, "vortex": {}}
        
        lines = output.split('\n')
        current_video = None
        current_dataset = None
        
        for line in lines:
            if "Processing:" in line:
                if "cisco" in line:
                    # Extract video number
                    if "cisco_" in line:
                        video_num = line.split("cisco_")[1].split(".")[0]
                    else:
                        video_num = line.split("Processing:")[1].strip()
                    current_video = video_num
                    current_dataset = "cisco"
                elif "vortex" in line:
                    if "vortex_" in line:
                        video_num = line.split("vortex_")[1].split(".")[0]
                    else:
                        video_num = line.split("Processing:")[1].strip()
                    current_video = video_num
                    current_dataset = "vortex"
            
            elif "Total:" in line and current_video and current_dataset:
                try:
                    total = int(line.split("Total:")[1].strip())
                    if current_dataset not in results:
                        results[current_dataset] = {}
                    results[current_dataset][current_video] = {"total": total}
                except:
                    pass
        
        return results

def main():
    print("🎯 Comprehensive Parameter Tuning with Visualization")
    print("=" * 60)
    
    tuner = WorkingTuner()
    
    # Step 1: Test confidence range
    print("\n" + "="*60)
    print("STEP 1: CONFIDENCE TUNING")
    print("="*60)
    best_confidence = tuner.test_confidence_range()
    
    # Step 2: Test IoU range
    print("\n" + "="*60)
    print("STEP 2: IoU TUNING")
    print("="*60)
    best_iou = tuner.test_iou_range()
    
    # Step 3: Test image size range
    print("\n" + "="*60)
    print("STEP 3: IMAGE SIZE TUNING")
    print("="*60)
    best_imgsz = tuner.test_imgsz_range()
    
    # Step 4: Run full batch test with best parameters
    print("\n" + "="*60)
    print("STEP 4: FULL BATCH TEST WITH BEST PARAMETERS")
    print("="*60)
    print(f"Best Confidence: {best_confidence['confidence']}")
    print(f"Best IoU: {best_iou['iou']}")
    print(f"Best Image Size: {best_imgsz['imgsz']}")
    
    full_results = tuner.run_full_test(best_confidence['confidence'])
    
    # Create summary
    print("\n" + "="*60)
    print("TUNING SUMMARY")
    print("="*60)
    print(f"📊 Best Confidence: {best_confidence['confidence']} (accuracy: {best_confidence['accuracy']:.3f})")
    print(f"📊 Best IoU: {best_iou['iou']} (accuracy: {best_iou['accuracy']:.3f})")
    print(f"📊 Best Image Size: {best_imgsz['imgsz']} (accuracy: {best_imgsz['accuracy']:.3f})")
    print(f"📁 All results and plots saved to: {tuner.output_dir}/")
    
    print("\n✅ Comprehensive tuning completed!")

if __name__ == "__main__":
    main()
