#!/usr/bin/env python3
"""
Initialize per-dataset configuration folders based on folders found in input/.

For each dataset directory under input/<dataset>/, creates:
  configs/datasets/<dataset>/
    - detection.yaml  (copied from configs/defaults/detection.yaml)  [no overwrite]
    - tracker.yaml    (copied from configs/defaults/tracker/botsort.yaml)  [no overwrite]
    - line.json       (copied from configs/defaults/line.json)  [no overwrite]

Existing files are left untouched.
"""

import shutil
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_if_missing(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Default template not found: {src}")
    if dst.exists():
        return
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    input_dir = project_root / "input"
    configs_root = project_root / "configs"
    defaults_dir = configs_root / "defaults"
    datasets_dir = configs_root / "datasets"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Default templates
    default_detection = defaults_dir / "detection.yaml"
    default_tracker = defaults_dir / "tracker" / "botsort.yaml"
    default_line = defaults_dir / "line.json"
    default_video = defaults_dir / "video.yaml"

    datasets = [p.name for p in input_dir.iterdir() if p.is_dir()]
    if not datasets:
        print("No datasets found in input/. Nothing to initialize.")
        return

    print(f"Found datasets: {', '.join(datasets)}")
    for ds in datasets:
        ds_cfg_dir = datasets_dir / ds
        ensure_dir(ds_cfg_dir)

        # detection.yaml
        copy_if_missing(default_detection, ds_cfg_dir / "detection.yaml")
        # tracker.yaml
        copy_if_missing(default_tracker, ds_cfg_dir / "tracker.yaml")
        # line.json
        copy_if_missing(default_line, ds_cfg_dir / "line.json")

        # video.yaml (per-dataset video processing params)
        copy_if_missing(default_video, ds_cfg_dir / "video.yaml")

        print(f"Initialized configs for dataset: {ds}")


if __name__ == "__main__":
    main()


