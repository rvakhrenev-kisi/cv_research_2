#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO tracking with BoT-SORT and save ID-labeled video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", default="models/yolo11x.pt", help="YOLO model path")
    parser.add_argument("--tracker", default="trackers/botsort.yaml", help="Tracker YAML path")
    parser.add_argument("--output", default="", help="Output video path (optional)")
    parser.add_argument("--conf", type=float, default=0.1, help="Detection confidence")
    parser.add_argument("--iou", type=float, default=0.3, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Input size")
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    # Determine output path
    in_name = Path(args.video).stem
    out_dir = Path("outputs/trackers")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else out_dir / f"{in_name}_botsort.mp4"

    # Run tracking. Ultralytics will save a labeled video when save=True.
    results = model.track(
        source=args.video,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        persist=True,
        save=True,
        project=str(out_dir),
        name=in_name,
        exist_ok=True,
        verbose=True,
    )

    print(f"✅ Tracking complete. Check outputs under: {out_dir}")


if __name__ == "__main__":
    main()
