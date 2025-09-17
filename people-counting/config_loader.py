#!/usr/bin/env python3
"""
Configuration loader for people counting parameters.
Loads settings from config.yaml file.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_file="configs/global.yaml"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuration loaded from {self.config_file}")
            return config
        except FileNotFoundError:
            print(f"❌ Config file {self.config_file} not found. Using defaults.")
            return self.get_default_config()
        except yaml.YAMLError as e:
            print(f"❌ Error parsing {self.config_file}: {e}. Using defaults.")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file not found"""
        return {
            "model": {
                "name": "yolo11n.pt",
                "type": "yolo12"
            },
            "tracker": {
                "use_ultralytics": False,
                "backend": "botsort",
                "yaml": "trackers/botsort.yaml",
                "overrides": {}
            },
            "detection": {
                "confidence": 0.1,
                "iou": 0.3,
                "imgsz": 640,
                "agnostic_nms": False
            },
            "tracking": {
                "track_high_thresh": 0.6,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.7,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "frame_rate": 30
            },
            "video": {
                "output_height": 0,
                "verbose": True
            },
            "lines": {
                "cisco": {
                    "start": [521, 898],
                    "end": [737, 622],
                    "direction": "right_to_left"
                },
                "vortex": {
                    "start": [521, 898],
                    "end": [737, 622],
                    "direction": "left_to_right"
                }
            },
            "performance": {
                "gpu_enabled": True,
                "batch_size": 1,
                "num_workers": 4
            },
            "output": {
                "base_dir": "outputs",
                "create_timestamped_dir": True,
                "save_individual_results": True,
                "save_summary": True,
                "save_plots": True
            }
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration
        
        Note: The 'type' parameter is a legacy parameter from the original
        people_counter.py script and is not actually used in detection logic.
        It's kept for compatibility but doesn't affect YOLO model functionality.
        """
        return self.config.get("model", {})
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration"""
        return self.config.get("detection", {})
    
    def get_tracking_config(self) -> Dict[str, Any]:
        """Get tracking configuration"""
        return self.config.get("tracking", {})
    
    def get_video_config(self) -> Dict[str, Any]:
        """Get video processing configuration"""
        return self.config.get("video", {})
    
    def get_tracker_config(self) -> Dict[str, Any]:
        """Get tracker configuration (Ultralytics trackers)."""
        return self.config.get("tracker", {})
    
    def get_line_config(self, dataset: str) -> Dict[str, Any]:
        """Get line configuration for specific dataset.
        Priority:
          1) configs/datasets/<dataset>/line.json
          2) legacy configs/<dataset>_line_config.json
          3) legacy config.yaml 'lines'
        """
        try:
            import json
            # 1) New per-dataset location
            ds_path = Path("configs") / "datasets" / dataset / "line.json"
            if ds_path.exists():
                with open(ds_path, 'r') as f:
                    cfg = json.load(f)
                if "line" in cfg:
                    line = cfg["line"]
                else:
                    line = cfg
                if all(k in line for k in ("x1","y1","x2","y2")):
                    direction = cfg.get("direction", {})
                    return {
                        "start": [line["x1"], line["y1"]],
                        "end": [line["x2"], line["y2"]],
                        "direction": direction.get("direction", "unknown") if isinstance(direction, dict) else direction,
                        "direction_point": [direction.get("x", 0), direction.get("y", 0)] if isinstance(direction, dict) else None
                    }

            # 2) Legacy JSON
            legacy_path = Path("configs") / f"{dataset}_line_config.json"
            if legacy_path.exists():
                with open(legacy_path, 'r') as f:
                    cfg = json.load(f)
                line = cfg.get("line", cfg)
                if all(k in line for k in ("x1","y1","x2","y2")):
                    direction = cfg.get("direction", {})
                    return {
                        "start": [line["x1"], line["y1"]],
                        "end": [line["x2"], line["y2"]],
                        "direction": direction.get("direction", "unknown") if isinstance(direction, dict) else direction,
                        "direction_point": [direction.get("x", 0), direction.get("y", 0)] if isinstance(direction, dict) else None
                    }
        except Exception:
            pass
        # 3) Fallback to legacy config.yaml 'lines'
        lines = self.config.get("lines", {})
        return lines.get(dataset, {})

    def get_dataset_detection_config(self, dataset: str) -> Dict[str, Any]:
        """Load detection.yaml for dataset ONLY (no global merge)."""
        import yaml as _yaml
        ds_file = Path("configs") / "datasets" / dataset / "detection.yaml"
        if ds_file.exists():
            try:
                with open(ds_file, 'r') as f:
                    ds_cfg = _yaml.safe_load(f) or {}
                if isinstance(ds_cfg, dict):
                    return ds_cfg
            except Exception:
                return {}
        return {}

    def get_dataset_tracker_yaml(self, dataset: str) -> str:
        """Return tracker yaml path for dataset if exists, else default template."""
        ds_yaml = Path("configs") / "datasets" / dataset / "tracker.yaml"
        if ds_yaml.exists():
            return str(ds_yaml)
        # Fallback to defaults template
        return str(Path("configs") / "defaults" / "tracker" / "botsort.yaml")
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.config.get("performance", {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self.config.get("output", {})
    
    def get_cctv_config(self) -> Dict[str, Any]:
        """Get CCTV optimization configuration"""
        return self.config.get("cctv", {})

    def get_dataset_video_config(self, dataset: str) -> Dict[str, Any]:
        """Load video.yaml for dataset ONLY (per-dataset video processing params)."""
        import yaml as _yaml
        ds_file = Path("configs") / "datasets" / dataset / "video.yaml"
        if ds_file.exists():
            try:
                with open(ds_file, 'r') as f:
                    ds_cfg = _yaml.safe_load(f) or {}
                if isinstance(ds_cfg, dict):
                    return ds_cfg
            except Exception:
                return {}
        return {}
    
    def print_config_summary(self):
        """Print a summary of current configuration"""
        print("\n📋 Current Configuration Summary:")
        print("=" * 50)
        
        model = self.get_model_config()
        detection = self.get_detection_config()
        tracking = self.get_tracking_config()
        tracker = self.get_tracker_config()
        
        print(f"🤖 Model: {model.get('name', 'N/A')} ({model.get('type', 'N/A')})")
        print(f"🎯 Detection: conf={detection.get('confidence', 'N/A')}, iou={detection.get('iou', 'N/A')}, imgsz={detection.get('imgsz', 'N/A')}")
        print(f"🔄 Tracking: high={tracking.get('track_high_thresh', 'N/A')}, low={tracking.get('track_low_thresh', 'N/A')}")
        if tracker.get('use_ultralytics', False):
            print(f"🧭 Tracker: {tracker.get('backend', 'botsort')} via {tracker.get('yaml', '')}")
        
        cctv = self.get_cctv_config()
        if cctv.get('optimized', False):
            print(f"📹 CCTV Optimized: Yes")
        
        print("=" * 50)

def main():
    """Test configuration loading"""
    loader = ConfigLoader()
    loader.print_config_summary()

if __name__ == "__main__":
    main()
