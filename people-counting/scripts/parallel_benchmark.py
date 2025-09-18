#!/usr/bin/env python3
"""
Run multiple people_counter.py instances in parallel and measure overall GPU utilization.

Designed for Google Colab / Linux with nvidia-smi available.

Example:
  python scripts/parallel_benchmark.py \
    --videos-glob "input/cisco/*.mp4" \
    --workers 3 \
    --model models/yolo11m.pt \
    --dataset cisco \
    --imgsz 1536 --conf 0.12 --iou 0.4

Outputs:
  - prints aggregate runtime, per-process runtimes
  - saves gpu_utilization.csv with timestamp,utilization(%),memory_used(MB)
"""

import argparse
import glob
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime


def run_cmd(cmd: str) -> int:
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    sys.stdout.write(out or "")
    sys.stderr.write(err or "")
    return proc.returncode


def nvidia_smi_sampler(stop_event: threading.Event, out_path: str, interval_s: float = 1.0) -> None:
    """Sample GPU utilization and memory used every interval and write CSV."""
    header_written = False
    with open(out_path, "w", encoding="utf-8") as f:
        while not stop_event.is_set():
            try:
                # Query first GPU (index 0). Colab often has a single GPU.
                cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,nounits,noheader"
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if res.returncode == 0 and res.stdout.strip():
                    util_mem = res.stdout.strip().splitlines()[0].split(",")
                    util = util_mem[0].strip()
                    mem = util_mem[1].strip() if len(util_mem) > 1 else ""
                    ts = datetime.utcnow().isoformat()
                    if not header_written:
                        f.write("timestamp,utilization_percent,memory_used_mb\n")
                        header_written = True
                    f.write(f"{ts},{util},{mem}\n")
                    f.flush()
            except Exception:
                pass
            stop_event.wait(interval_s)


def worker_loop(idx: int, jobs_q: "queue.Queue[str]", base_args: dict, results: dict) -> None:
    start = time.time()
    rcodes = []
    while True:
        try:
            video = jobs_q.get_nowait()
        except queue.Empty:
            break
        out_path = base_args["output_template"].format(worker=idx, name=os.path.splitext(os.path.basename(video))[0])
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cmd = [
            sys.executable, "people_counter.py",
            "--video", video,
            "--model", base_args["model"],
            "--model-type", base_args["model_type"],
            "--dataset", base_args["dataset"],
            "--line-start", str(base_args["line_start"][0]), str(base_args["line_start"][1]),
            "--line-end", str(base_args["line_end"][0]), str(base_args["line_end"][1]),
            "--confidence", str(base_args["conf"]),
            "--iou", str(base_args["iou"]),
            "--imgsz", str(base_args["imgsz"]),
            "--output", out_path,
            "--output-height", str(base_args["output_height"]),
        ]
        if base_args["verbose"]:
            cmd.append("--verbose")
        if base_args["tracker_yaml"]:
            cmd.extend(["--tracker-yaml", base_args["tracker_yaml"]])
        if base_args["tracker_type"]:
            cmd.extend(["--tracker-type", base_args["tracker_type"]])

        rc = run_cmd(" ".join([shlex.quote(c) for c in cmd]))
        rcodes.append(rc)
    dur = time.time() - start
    results[idx] = {"seconds": round(dur, 3), "return_codes": rcodes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-glob", required=True, help="Glob for input videos (e.g., input/cisco/*.mp4)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dataset", type=str, default="cisco")
    ap.add_argument("--model", type=str, default="models/yolo11m.pt")
    ap.add_argument("--model-type", type=str, default="yolo12")
    ap.add_argument("--conf", type=float, default=0.12)
    ap.add_argument("--iou", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--output-height", type=int, default=0)
    ap.add_argument("--line-start", type=int, nargs=2, default=[0, 0])
    ap.add_argument("--line-end", type=int, nargs=2, default=[100, 100])
    ap.add_argument("--tracker-yaml", type=str, default="")
    ap.add_argument("--tracker-type", type=str, default="")
    ap.add_argument("--output-dir", type=str, default="outputs/parallel")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--sample-interval", type=float, default=1.0, help="nvidia-smi sample interval seconds")
    args = ap.parse_args()

    videos = sorted(glob.glob(args.videos_glob))
    if not videos:
        print(f"No videos matched glob: {args.videos_glob}")
        sys.exit(1)

    # Prepare job queue
    jobs_q: "queue.Queue[str]" = queue.Queue()
    for v in videos:
        jobs_q.put(v)

    # Output template
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_base = os.path.join(args.output_dir, f"parallel_{ts}")
    os.makedirs(out_base, exist_ok=True)
    output_template = os.path.join(out_base, "worker{worker}_{name}.mp4")

    base_args = {
        "model": args.model,
        "model_type": args.model_type,
        "dataset": args.dataset,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "output_height": args.output_height,
        "line_start": args.line_start,
        "line_end": args.line_end,
        "verbose": args.verbose,
        "tracker_yaml": args.tracker_yaml,
        "tracker_type": args.tracker_type,
        "output_template": output_template,
    }

    # Start GPU sampler
    stop_event = threading.Event()
    sampler_thread = threading.Thread(target=nvidia_smi_sampler, args=(stop_event, os.path.join(out_base, "gpu_utilization.csv"), args.sample_interval), daemon=True)
    sampler_thread.start()

    # Launch workers
    threads = []
    results = {}
    t0 = time.time()
    for wi in range(max(1, args.workers)):
        th = threading.Thread(target=worker_loop, args=(wi, jobs_q, base_args, results), daemon=True)
        th.start()
        threads.append(th)
        time.sleep(0.5)  # small stagger to smooth VRAM spikes

    for th in threads:
        th.join()
    total_sec = time.time() - t0

    # Stop sampler
    stop_event.set()
    sampler_thread.join(timeout=2)

    # Print summary
    done = sum(len(v["return_codes"]) for v in results.values())
    print("\n=== Parallel Benchmark Summary ===")
    print(f"Videos processed: {done}/{len(videos)}")
    print(f"Workers: {args.workers}")
    print(f"Total wall time: {round(total_sec, 3)} sec")
    for wi in sorted(results.keys()):
        print(f"  Worker {wi}: {results[wi]['seconds']} sec, rcodes={results[wi]['return_codes']}")
    print(f"GPU utilization log: {os.path.join(out_base, 'gpu_utilization.csv')}")


if __name__ == "__main__":
    main()


