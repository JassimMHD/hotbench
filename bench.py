r"""Offline benchmark: the same frames through every image size x thread count.

    python bench.py                                    (the .env source and model)
    python bench.py --source clips\street.mp4 --imgsz 960,768,640 --threads 1,2,4 --cores 0-3
    python bench.py --cores 0,1 --threads 1,2          (act like a 2-core server)

Prints a markdown table and writes runs/bench-<time>.csv. "cores used" is the
process's CPU time / wall time while predicting: what one stream at full speed
costs. "count" is the mean over the frames, to compare sizes on the same frames.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(exist_ok=True)  # else it falls back to \tmp

import cv2  # noqa: E402
import psutil  # noqa: E402

from lab import config  # noqa: E402
from lab.infer import _ort_threads, class_ids, onnx_input_size  # noqa: E402


def read_frames(source: str, n: int) -> list:
    """n frames spread over a clip, or the next n frames of a stream."""
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        sys.exit(f"cannot open {config.mask(source)}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    if Path(source).is_file() and total > n:
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / n))
            ok, img = cap.read()
            if ok:
                frames.append(img)
    else:
        while len(frames) < n:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
    cap.release()
    if not frames:
        sys.exit("no frames read")
    return frames


def main() -> None:
    cfg = config.load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=cfg.source)
    ap.add_argument("--weights", default=cfg.weights)
    ap.add_argument("--imgsz", default="960,768,640")
    ap.add_argument("--threads", default="1,2")
    ap.add_argument("--cores", default=cfg.cores, help='"0,1" (default from .env), "0-3", "" = all')
    ap.add_argument("--conf", type=float, default=cfg.conf)
    ap.add_argument("--classes", default=cfg.classes, help='"person, car"; "" = every class')
    ap.add_argument("--frames", type=int, default=30)
    args = ap.parse_args()

    proc = psutil.Process()
    cores = config.parse_cores(args.cores)
    proc.cpu_affinity(cores)

    import torch
    from ultralytics import YOLO

    source = config.resolve(args.source)
    frames = read_frames(source, args.frames)
    h, w = frames[0].shape[:2]
    onnx = args.weights.endswith(".onnx")
    model = YOLO(config.resolve(args.weights), task="detect")
    try:
        classes = class_ids(model.names, args.classes)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"{Path(args.weights).name} on {config.mask(args.source)} ({w}x{h}), "
          f"{len(frames)} frames, cores {cores} of {psutil.cpu_count()}\n")

    sizes = [int(s) for s in args.imgsz.split(",")]
    fixed = onnx_input_size(Path(config.resolve(args.weights))) if onnx else None
    if fixed:
        print(f"static ONNX: only {fixed}px, ignoring --imgsz {args.imgsz}\n")
        sizes = [fixed]
    rows = []
    for imgsz in sizes:
        for threads in [int(t) for t in args.threads.split(",")]:
            # With ONNX, torch only pre/post-processes: one thread, ORT gets `threads`.
            torch.set_num_threads(1 if onnx else threads)
            if onnx:  # ORT fixes its thread count per session: a fresh one
                _ort_threads(threads)
                model = YOLO(config.resolve(args.weights), task="detect")
            for img in frames[:2]:  # warm-up at this size
                model.predict(img, imgsz=imgsz, conf=args.conf, max_det=cfg.max_det,
                              classes=classes, device="cpu", verbose=False)
            times, counts = [], []
            cpu0, wall0 = sum(proc.cpu_times()[:2]), perf_counter()
            for img in frames:
                t = perf_counter()
                r = model.predict(img, imgsz=imgsz, conf=args.conf, max_det=cfg.max_det,
                                  classes=classes, device="cpu", verbose=False)[0]
                times.append((perf_counter() - t) * 1000)
                counts.append(len(r.boxes))
            wall = perf_counter() - wall0
            used = (sum(proc.cpu_times()[:2]) - cpu0) / wall
            times.sort()
            rows.append({
                "imgsz": imgsz,
                "threads": threads,
                "ms_p50": round(statistics.median(times)),
                "ms_p95": round(times[min(len(times) - 1, int(0.95 * len(times)))]),
                "inferences_per_s": round(len(frames) / wall, 2),
                "cores_used": round(used, 2),
                "core_s_per_inference": round(used * wall / len(frames), 2),
                "count_mean": round(statistics.mean(counts), 1),
                "rss_mb": round(proc.memory_info().rss / 2**20),
            })
            print(rows[-1], flush=True)

    print()
    keys = list(rows[0])
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for row in rows:
        print("| " + " | ".join(str(row[k]) for k in keys) + " |")

    out = ROOT / "runs" / f"bench-{datetime.now():%Y%m%d-%H%M%S}.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwritten {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
