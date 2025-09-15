#!/usr/bin/env python3
"""
YOLOv11-specific parameter tuning script.
Based on Medium article recommendations for YOLOv11 + ByteTrack.
"""

import json
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

class YOLOv11Tuner:
    def __init__(self):
        self.ground_truth = self.load_ground_truth()
        self.results = []
        self.output_dir = "yolov11_tuning_results"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📋 Ground truth loaded: {len(self.ground_truth)} datasets")
        print(f"📁 Results will be saved to: {self.output_dir}")
        
    def load_ground_truth(self):
        """Load expected results from video_content.txt"""
        try:
            with open("video_content.txt", 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading ground truth: {e}")
            return {}
    
    def get_available_models(self):
        """Get list of available YOLOv11 models"""
        models_dir = Path("models")
        yolov11_models = []
        
        if models_dir.exists():
            for model_file in models_dir.glob("yolo11*.pt"):
                yolov11_models.append(model_file.name)
        
        if not yolov11_models:
            print("❌ No YOLOv11 models found. Run download_yolov11.py first")
            return []
        
        print(f"✅ Found YOLOv11 models: {yolov11_models}")
        return sorted(yolov11_models)
    
    def run_yolov11_test(self, model_name, confidence=0.1):
        """Run test with YOLOv11 model"""
        print(f"\n🧪 Testing YOLOv11: {model_name} (confidence={confidence})")
        
        cmd = [
            "python", "people_counter.py",
            "--video", "../cisco/1.mp4",
            "--model", f"models/{model_name}",
            "--model-type", "yolo12",  # people_counter.py uses yolo12 type
            "--line-start", "521", "898",
            "--line-end", "737", "622",
            "--confidence", str(confidence),
            "--output", f"yolov11_test_{model_name}_{confidence}.mp4",
            "--output-height", "0",
            "--verbose"
        ]
        
        print(f"   🚀 Running: {' '.join(cmd)}")
        
        try:
            print(f"   ⏱️  Processing with YOLOv11 (this may take 2-3 minutes)...")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), timeout=600)
            
            print(f"   📤 Return code: {result.returncode}")
            
            if result.returncode != 0:
                print(f"   ❌ Command failed with return code {result.returncode}")
                if result.stderr:
                    print(f"   📤 STDERR: {result.stderr[:200]}...")
                return 0.0, {}
            
            # Parse results
            counts = self.parse_counts(result.stdout)
            expected = self.ground_truth["cisco"]["1"]["expected_people"]
            predicted = counts.get("total", 0)
            error = abs(predicted - expected)
            accuracy = max(0, 1 - (error / expected)) if expected > 0 else 0.0
            
            print(f"   📊 Expected: {expected}, Got: {predicted}, Accuracy: {accuracy:.3f}")
            
            # Clean up
            test_file = f"yolov11_test_{model_name}_{confidence}.mp4"
            if os.path.exists(test_file):
                os.remove(test_file)
            
            return accuracy, counts
            
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Timeout for {model_name}")
            return 0.0, {}
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return 0.0, {}
    
    def parse_counts(self, output):
        """Parse people counts from output"""
        counts = {"up": 0, "down": 0, "total": 0}
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            
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
            elif "Total:" in line and "people" in line.lower():
                try:
                    counts["total"] = int(line.split("Total:")[1].strip())
                except:
                    pass
            elif "Total people" in line:
                try:
                    counts["total"] = int(line.split("Total people")[1].strip().split()[0])
                except:
                    pass
        
        return counts
    
    def test_yolov11_models(self):
        """Test different YOLOv11 model variants"""
        print("🎯 Testing YOLOv11 Model Variants")
        print("=" * 50)
        print("Based on Medium article: YOLOv11 + ByteTrack recommendations")
        
        models = self.get_available_models()
        if not models:
            return None
        
        # Test with optimal confidence for CCTV
        confidence = 0.1
        results = []
        
        for model in models:
            accuracy, counts = self.run_yolov11_test(model, confidence)
            results.append({
                "model": model,
                "confidence": confidence,
                "accuracy": accuracy,
                "counts": counts
            })
        
        # Find best model
        best_result = max(results, key=lambda x: x["accuracy"])
        
        print(f"\n🏆 BEST YOLOv11 MODEL: {best_result['model']}")
        print(f"🏆 Best accuracy: {best_result['accuracy']:.3f}")
        print(f"🏆 Best counts: {best_result['counts']}")
        
        # Create visualization
        self.create_model_comparison_plot(results)
        
        return best_result
    
    def test_yolov11_confidence_range(self, model_name):
        """Test confidence range with best YOLOv11 model"""
        print(f"\n🎯 Testing Confidence Range with {model_name}")
        print("=" * 50)
        
        # Enhanced confidence range for YOLOv11
        confidence_values = [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        results = []
        
        for conf in confidence_values:
            accuracy, counts = self.run_yolov11_test(model_name, conf)
            results.append({
                "model": model_name,
                "confidence": conf,
                "accuracy": accuracy,
                "counts": counts
            })
        
        # Find best confidence
        best_result = max(results, key=lambda x: x["accuracy"])
        
        print(f"\n🏆 BEST CONFIDENCE for {model_name}: {best_result['confidence']}")
        print(f"🏆 Best accuracy: {best_result['accuracy']:.3f}")
        
        # Create visualization
        self.create_confidence_plot(results, model_name)
        
        return best_result
    
    def create_model_comparison_plot(self, results):
        """Create model comparison visualization"""
        if not results:
            return
        
        models = [r["model"] for r in results]
        accuracies = [r["accuracy"] for r in results]
        
        plt.figure(figsize=(12, 8))
        
        # Model comparison bar chart
        plt.subplot(2, 2, 1)
        bars = plt.bar(range(len(models)), accuracies, color='skyblue', edgecolor='black')
        plt.xlabel('YOLOv11 Model')
        plt.ylabel('Accuracy')
        plt.title('YOLOv11 Model Comparison')
        plt.xticks(range(len(models)), [m.replace('yolo11', 'Y11').replace('.pt', '') for m in models], rotation=45)
        
        # Highlight best model
        best_idx = np.argmax(accuracies)
        bars[best_idx].set_color('red')
        bars[best_idx].set_label(f'Best: {models[best_idx]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy distribution
        plt.subplot(2, 2, 2)
        plt.hist(accuracies, bins=8, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.title('Accuracy Distribution Across Models')
        plt.grid(True, alpha=0.3)
        
        # Model size vs accuracy (approximate)
        plt.subplot(2, 2, 3)
        model_sizes = {
            'yolo11n.pt': 1, 'yolo11s.pt': 2, 'yolo11m.pt': 3, 
            'yolo11l.pt': 4, 'yolo11x.pt': 5
        }
        sizes = [model_sizes.get(m, 0) for m in models]
        plt.scatter(sizes, accuracies, s=100, alpha=0.7, c=accuracies, cmap='viridis')
        plt.xlabel('Model Size (Relative)')
        plt.ylabel('Accuracy')
        plt.title('Model Size vs Accuracy')
        plt.colorbar(label='Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Performance summary
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, f"Best Model: {models[best_idx]}", fontsize=12, weight='bold')
        plt.text(0.1, 0.7, f"Best Accuracy: {accuracies[best_idx]:.3f}", fontsize=12)
        plt.text(0.1, 0.6, f"Total Models: {len(models)}", fontsize=12)
        plt.text(0.1, 0.5, f"Avg Accuracy: {np.mean(accuracies):.3f}", fontsize=12)
        plt.text(0.1, 0.4, f"Std Accuracy: {np.std(accuracies):.3f}", fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Performance Summary')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.output_dir, "yolov11_model_comparison.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Model comparison plot saved to {plot_file}")
        
        # Save data
        data_file = os.path.join(self.output_dir, "yolov11_model_results.json")
        with open(data_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Model data saved to {data_file}")
    
    def create_confidence_plot(self, results, model_name):
        """Create confidence tuning plot for specific model"""
        if not results:
            return
        
        confidences = [r["confidence"] for r in results]
        accuracies = [r["accuracy"] for r in results]
        
        plt.figure(figsize=(12, 6))
        
        # Main plot
        plt.subplot(1, 2, 1)
        plt.plot(confidences, accuracies, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Accuracy')
        plt.title(f'Confidence Tuning - {model_name}')
        plt.grid(True, alpha=0.3)
        
        # Highlight best
        best_idx = np.argmax(accuracies)
        plt.plot(confidences[best_idx], accuracies[best_idx], 'ro', markersize=12, 
                label=f'Best: {confidences[best_idx]} (acc: {accuracies[best_idx]:.3f})')
        plt.legend()
        
        # Low confidence detailed view
        plt.subplot(1, 2, 2)
        low_conf_mask = np.array(confidences) <= 0.2
        if np.any(low_conf_mask):
            low_conf = np.array(confidences)[low_conf_mask]
            low_acc = np.array(accuracies)[low_conf_mask]
            plt.plot(low_conf, low_acc, 'go-', linewidth=2, markersize=8)
            plt.xlabel('Confidence Threshold (Low Range)')
            plt.ylabel('Accuracy')
            plt.title(f'Detailed View - {model_name}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.output_dir, f"yolov11_confidence_{model_name.replace('.pt', '')}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confidence plot saved to {plot_file}")

def main():
    print("🎯 YOLOv11 Parameter Tuning")
    print("=" * 60)
    print("Based on Medium article: YOLOv11 + ByteTrack recommendations")
    print("=" * 60)
    
    tuner = YOLOv11Tuner()
    
    # Step 1: Test different YOLOv11 models
    print("\n" + "="*60)
    print("STEP 1: YOLOv11 MODEL COMPARISON")
    print("="*60)
    best_model_result = tuner.test_yolov11_models()
    
    if best_model_result:
        # Step 2: Test confidence range with best model
        print("\n" + "="*60)
        print("STEP 2: CONFIDENCE TUNING WITH BEST MODEL")
        print("="*60)
        best_confidence_result = tuner.test_yolov11_confidence_range(best_model_result["model"])
        
        # Summary
        print("\n" + "="*60)
        print("YOLOv11 TUNING SUMMARY")
        print("="*60)
        print(f"🏆 Best Model: {best_model_result['model']}")
        print(f"🏆 Best Confidence: {best_confidence_result['confidence']}")
        print(f"🏆 Best Accuracy: {best_confidence_result['accuracy']:.3f}")
        print(f"📁 Results saved to: {tuner.output_dir}/")
        
        print(f"\n💡 Recommendations:")
        print(f"   - Use {best_model_result['model']} for best performance")
        print(f"   - Set confidence to {best_confidence_result['confidence']} for CCTV footage")
        print(f"   - YOLOv11 should provide better accuracy than YOLOv10")
    
    print(f"\n✅ YOLOv11 tuning completed!")

if __name__ == "__main__":
    main()
