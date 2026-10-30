"""Once a second: one snapshot of everything the page shows, and a CSV row
while recording (runs/<time>.csv)."""

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime
from pathlib import Path
from time import monotonic

import psutil

from lab.config import ROOT, Config, mask, parse_cores
from lab.infer import Inferer
from lab.metrics import ThreadGroups, median, pct
from lab.reader import Reader
from lab.render import Renderer

STAGES = ("decode", "inference", "display", "other")


def _r(value, digits: int = 1):
    return None if value is None else round(value, digits)


class Stats:
    def __init__(self, cfg: Config, reader: Reader, inferer: Inferer, renderer: Renderer,
                 groups: ThreadGroups):
        self.cfg = cfg
        self.reader = reader
        self.inferer = inferer
        self.renderer = renderer
        self.groups = groups
        self.proc = psutil.Process()
        self.latest: dict = {}
        self.recording: Path | None = None
        self._writer = None
        self._file = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, name="stats", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        self.record(False)

    def record(self, on: bool) -> Path | None:
        with self._lock:
            if self._file:
                self._file.close()
                self._file, self._writer, self.recording = None, None, None
            if on:
                path = ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}.csv"
                path.parent.mkdir(exist_ok=True)
                self._file = path.open("w", newline="", encoding="utf-8")
                self._writer = None  # header from the first row
                self.recording = path
            return self.recording

    def _run(self) -> None:
        self.groups.cpu_seconds_since_last()
        self.proc.cpu_percent(None)
        psutil.cpu_percent(None)
        last = monotonic()
        while not self._stop.wait(1.0):
            now = monotonic()
            wall, last = now - last, now
            try:
                self.latest = self._snapshot(wall)
                self._write()
            except Exception:  # a stats glitch must not stop the stats
                import logging

                logging.getLogger("lab.stats").exception("stats failed")

    def _snapshot(self, wall: float) -> dict:
        cfg, rd, inf, rn = self.cfg, self.reader, self.inferer, self.renderer
        used = self.groups.cpu_seconds_since_last()
        cores = {s: round(used.get(s, 0.0) / wall, 2) for s in STAGES}
        total_cores = round(self.proc.cpu_percent(None) / 100, 2)
        allowed = parse_cores(cfg.cores)
        timings = list(inf.timings)
        ms = [r.ms for r in timings]
        latency = [(r.t_done - r.t_frame) * 1000 for r in timings]
        res = inf.latest
        p10, p90 = inf.spread()
        smoothed = inf.smoothed()
        return {
            "time": time.time(),
            "source": mask(cfg.source),
            "stream": {
                "status": rd.status,
                "error": rd.error,
                "width": rd.width,
                "height": rd.height,
                "declared_fps": rd.declared_fps,
                "decode_fps": _r(rd.decoded.rate()),
                "reconnects": rd.reconnects,
            },
            "inference": {
                "status": inf.status,
                "error": inf.error,
                "model": inf.model_name,
                "imgsz": inf.imgsz(),
                "fps": _r(inf.done.rate(), 2),
                "total": inf.total,
                "ms_p50": _r(median(ms), 0),
                "ms_p95": _r(pct(ms, 95), 0),
                "pre_ms": _r(median([r.pre_ms for r in timings])),
                "infer_ms": _r(median([r.infer_ms for r in timings])),
                "post_ms": _r(median([r.post_ms for r in timings])),
                "latency_ms_p50": _r(median(latency), 0),
                "skipped_per_inference": _r(median([r.skipped for r in timings])),
                "count": res.count if res else None,
                "confidence": res.confidence if res else None,
                "smoothed": _r(smoothed),
                "spread_p10": p10,
                "spread_p90": p90,
                "box_age_s": _r(monotonic() - res.t_frame, 2) if res else None,
            },
            "display": {
                "viewers": rn.viewers,
                "fps": _r(rn.rendered.rate()),
                "ms": _r(rn.last_ms, 0),
            },
            "cost": {
                "cores_total": total_cores,
                **{f"cores_{s}": cores[s] for s in STAGES},
                "rss_mb": round(self.proc.memory_info().rss / 2**20),
                "system_cpu_pct": psutil.cpu_percent(None),
                "cores_allowed": len(allowed),
                "cpus": psutil.cpu_count(),
                "oversubscribed": cfg.threads > len(allowed),
            },
            "config": cfg.public(),
            "recording": self.recording.name if self.recording else None,
        }

    def _write(self) -> None:
        with self._lock:
            if not self._file:
                return
            s = self.latest
            row = {"time": datetime.fromtimestamp(s["time"]).isoformat(timespec="seconds")}
            for group in ("stream", "inference", "display", "cost"):
                for key, value in s[group].items():
                    if key != "error":
                        row[f"{group}.{key}"] = value
            for key in ("imgsz", "conf", "threads", "cores", "infer_max_fps", "display_fps"):
                row[f"config.{key}"] = s["config"][key]
            if self._writer is None:
                self._writer = csv.DictWriter(self._file, fieldnames=list(row))
                self._writer.writeheader()
            self._writer.writerow(row)
            self._file.flush()
