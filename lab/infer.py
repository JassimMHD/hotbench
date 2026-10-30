"""Inference thread: always the newest frame, never a queue.

While the model works on one frame the reader keeps overwriting its slot, so
the next predict takes the live frame and the frames in between are skipped.
The delay therefore stays at about one inference instead of growing.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter, sleep

import numpy as np

from lab.config import Config, resolve
from lab.metrics import Rolling, ThreadGroups, median, pct
from lab.reader import Reader

log = logging.getLogger("lab.infer")


def _ort_threads(threads: int) -> None:
    """Make every ONNX Runtime session use `threads` threads.

    ultralytics creates its session without SessionOptions, so ORT starts one
    thread per logical CPU (16 here). Pinned to 2 cores that is 10x slower
    (1337 ms vs 137 ms a frame). ORT fixes the count when the session is made,
    so a THREADS change reloads an .onnx model."""
    import onnxruntime as ort

    real = getattr(ort, "_lab_real_session", ort.InferenceSession)
    ort._lab_real_session = real

    def session(path, sess_options=None, *args, **kwargs):
        if sess_options is None:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = threads
            sess_options.inter_op_num_threads = 1
        return real(path, sess_options, *args, **kwargs)

    ort.InferenceSession = session


def onnx_input_size(path: Path) -> int | None:
    """The square input size of a static ONNX model; None when it is dynamic."""
    import onnxruntime as ort

    real = getattr(ort, "_lab_real_session", ort.InferenceSession)
    shape = real(str(path), providers=["CPUExecutionProvider"]).get_inputs()[0].shape
    return shape[-1] if isinstance(shape[-1], int) else None


def class_ids(names: dict[int, str], text: str) -> list[int] | None:
    """'person, car' or '0,2' -> the model's class ids; '' -> None (every class).

    ValueError names the classes the model doesn't have."""
    wanted = [part.strip() for part in text.split(",") if part.strip()]
    if not wanted:
        return None
    by_name = {name.lower(): i for i, name in names.items()}
    ids, unknown = [], []
    for item in wanted:
        if item.isdigit() and int(item) in names:
            ids.append(int(item))
        elif item.lower() in by_name:
            ids.append(by_name[item.lower()])
        else:
            unknown.append(item)
    if unknown:
        raise ValueError(f"the model has no class {', '.join(unknown)} "
                         f"(it knows: {', '.join(list(names.values())[:12])}…)")
    return ids


@dataclass(frozen=True)
class Result:
    seq: int  # the frame's
    t_frame: float  # when that frame was decoded (monotonic)
    t_done: float
    count: int
    confidence: float  # mean score of the counted boxes
    boxes: np.ndarray  # (N, 4) xyxy in the frame's pixels
    labels: tuple[str, ...]  # the class name of each box
    frame_w: int
    frame_h: int
    ms: float  # the whole predict call
    pre_ms: float  # ultralytics' own split
    infer_ms: float
    post_ms: float
    skipped: int  # frames decoded since the previous predict and never counted


class Inferer:
    def __init__(self, cfg: Config, reader: Reader, groups: ThreadGroups):
        self.cfg = cfg
        self.reader = reader
        self.groups = groups
        self.model = None
        self.model_name = ""
        self.status = "loading"
        self.error = ""
        self.latest: Result | None = None
        self.latest_frame = None
        self.fixed_imgsz: int | None = None
        self.total = 0
        self.done = Rolling(10)  # for inferences/s
        self.timings: deque[Result] = deque(maxlen=60)
        self.counts = Rolling(600)  # (t, count) for the median; window read from cfg
        self._threads = 0
        self._reload = threading.Event()
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, name="inference", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def reload(self) -> None:
        """Load the configured weights again (after a change from the page)."""
        self._reload.set()

    def imgsz(self) -> int:
        """IMGSZ, unless the model only takes one size (a static ONNX export)."""
        return self.fixed_imgsz or self.cfg.imgsz

    def smoothed(self) -> float | None:
        return median(self.counts.values(self.cfg.smooth_seconds))

    def spread(self) -> tuple[float | None, float | None]:
        """p10 / p90 of the raw counts in the smoothing window."""
        values = self.counts.values(self.cfg.smooth_seconds)
        return pct(values, 10), pct(values, 90)

    def _set_threads(self) -> None:
        import torch

        if self._threads != self.cfg.threads:
            with self.groups.claim("inference"):
                # With ONNX, torch only does the pre/post-processing: more torch
                # threads just spin against ORT's (542 ms vs 229 ms on 2 cores).
                torch.set_num_threads(1 if self.cfg.weights.endswith(".onnx") else self.cfg.threads)
            if self._threads and self.model_name and self.cfg.weights.endswith(".onnx"):
                self._reload.set()  # ORT's thread count is fixed per session
            self._threads = self.cfg.threads

    def _load(self) -> None:
        self._reload.clear()
        self.status, self.error, self.model = "loading", "", None
        path = Path(resolve(self.cfg.weights))
        try:
            with self.groups.claim("inference"):
                from ultralytics import YOLO  # heavy: import here, in this thread

                self._threads = 0  # apply the thread split for this model type
                self._set_threads()
                self.fixed_imgsz = None
                if path.suffix == ".onnx":
                    _ort_threads(self.cfg.threads)
                    self.fixed_imgsz = onnx_input_size(path)
                    if self.fixed_imgsz and self.fixed_imgsz != self.cfg.imgsz:
                        log.warning("%s takes only %dpx (exported without dynamic=True): "
                                    "using that, not IMGSZ=%d", path.name, self.fixed_imgsz,
                                    self.cfg.imgsz)
                model = YOLO(str(path), task="detect")
                # Warm-up at the real size: the first predict is slow.
                h = max(self.reader.height, 576)
                w = max(self.reader.width, 720)
                model.predict(np.zeros((h, w, 3), np.uint8), imgsz=self.imgsz(),
                              conf=self.cfg.conf, max_det=self.cfg.max_det,
                              device="cpu", verbose=False)
        except Exception as exc:
            self.status, self.error = "down", str(exc)[:300] or type(exc).__name__
            log.exception("model failed to load")
            return
        self.model, self.model_name = model, path.stem
        self.status = "ready"
        log.info("model ready: %s", path.name)

    def _run(self) -> None:
        self.groups.own("inference")
        self._load()
        last_seq = 0
        last_start = 0.0
        while not self._stop.is_set():
            if self._reload.is_set():
                self._load()
            if self.model is None:
                self._reload.wait(1)
                continue
            if not self.cfg.infer_enabled:
                self.status = "paused"
                sleep(0.2)
                continue
            self.status = "ready"
            if self.cfg.infer_max_fps > 0:
                wait = last_start + 1 / self.cfg.infer_max_fps - monotonic()
                if wait > 0:
                    sleep(min(wait, 0.2))
                    continue
            try:
                classes = class_ids(self.model.names, self.cfg.classes)
            except ValueError as exc:  # a class name the model doesn't know
                self.error = str(exc)
                sleep(0.5)
                continue
            frame = self.reader.wait_newer(last_seq, 0.5)
            if frame is None:
                continue
            skipped = frame.seq - last_seq - 1 if last_seq else 0
            last_seq = frame.seq
            last_start = monotonic()
            try:
                self._set_threads()
                self._predict(frame, skipped, classes)
            except Exception as exc:
                self.error = str(exc)[:300]
                log.exception("predict failed")
                sleep(0.5)

    def _predict(self, frame, skipped: int, classes: list[int] | None) -> None:
        cfg = self.cfg
        started = perf_counter()
        r = self.model.predict(frame.image, imgsz=self.imgsz(), conf=cfg.conf,
                               max_det=cfg.max_det, classes=classes, device="cpu",
                               verbose=False)[0]
        ms = (perf_counter() - started) * 1000
        scores = r.boxes.conf.numpy()
        result = Result(
            seq=frame.seq,
            t_frame=frame.t,
            t_done=monotonic(),
            count=len(scores),
            confidence=round(float(scores.mean()), 3) if len(scores) else 0.0,
            boxes=r.boxes.xyxy.numpy(),
            labels=tuple(r.names[int(c)] for c in r.boxes.cls.numpy()),
            frame_w=frame.image.shape[1],
            frame_h=frame.image.shape[0],
            ms=ms,
            pre_ms=r.speed.get("preprocess") or 0.0,
            infer_ms=r.speed.get("inference") or 0.0,
            post_ms=r.speed.get("postprocess") or 0.0,
            skipped=max(0, skipped),
        )
        self.latest_frame = frame  # the frame `latest` was counted on (for save-frame)
        self.latest = result
        self.total += 1
        self.done.add(t=result.t_done)
        self.timings.append(result)
        self.counts.add(result.count, t=result.t_done)
        self.error = ""
