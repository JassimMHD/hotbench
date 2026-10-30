"""Display: the newest frame + the newest boxes + a HUD, as JPEG for /video.mjpg.

Runs only while someone watches, at DISPLAY_FPS. The boxes update at the
inference rate; between two inferences they are drawn on newer frames, and the
HUD says how old they are.
"""

from __future__ import annotations

import threading
from time import monotonic, sleep

import cv2
import numpy as np

from lab.config import Config
from lab.infer import Inferer, Result
from lab.metrics import Rolling, ThreadGroups

BOX = (90, 220, 110)  # BGR
TEXT = (255, 255, 255)
LABEL_MAX = 40  # more boxes than this: no class names, they would cover the image


def draw(image: np.ndarray, result: Result | None, width: int, hud: str) -> np.ndarray:
    """A resized copy with the boxes and one line of text on it."""
    h, w = image.shape[:2]
    scale = min(1.0, width / w)
    out = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) \
        if scale < 1 else image.copy()
    if result is not None and len(result.boxes):
        # Boxes are in the pixels of the frame they came from (same stream size).
        sx = out.shape[1] / result.frame_w
        sy = out.shape[0] / result.frame_h
        thick = 1 if out.shape[1] < 1600 else 2
        labels = result.labels if len(result.boxes) <= LABEL_MAX else ()
        font_scale = max(0.45, out.shape[1] / 2400)
        for i, (x1, y1, x2, y2) in enumerate(result.boxes):
            x, y = int(x1 * sx), int(y1 * sy)
            cv2.rectangle(out, (x, y), (int(x2 * sx), int(y2 * sy)), BOX, thick)
            if labels:  # a filled tag on the box's top edge, readable on any background
                (tw, th), _ = cv2.getTextSize(labels[i], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                y = max(y, th + 6)
                cv2.rectangle(out, (x, y - th - 6), (x + tw + 6, y), BOX, -1)
                cv2.putText(out, labels[i], (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    bar = max(28, out.shape[0] // 22)
    cv2.rectangle(out, (0, 0), (out.shape[1], bar), (0, 0, 0), -1)
    cv2.putText(out, hud, (10, int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                bar / 42, TEXT, 1, cv2.LINE_AA)
    return out


class Renderer:
    def __init__(self, cfg: Config, reader, inferer: Inferer, groups: ThreadGroups):
        self.cfg = cfg
        self.reader = reader
        self.inferer = inferer
        self.groups = groups
        self.jpeg: bytes | None = None
        self.seq = 0
        self.viewers = 0
        self.rendered = Rolling(5)
        self.last_ms = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, name="display", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def watch(self, delta: int) -> None:
        with self._lock:
            self.viewers += delta

    def hud(self, result: Result | None) -> str:
        inf = self.inferer
        if result is None:
            return f"model {inf.status}"
        smoothed = inf.smoothed()
        age = monotonic() - result.t_frame
        return (f"count {result.count}   median {smoothed:.0f}   "
                f"{inf.done.rate():.1f} inf/s   {result.ms:.0f} ms   boxes {age:.1f}s old"
                if smoothed is not None else f"count {result.count}")

    def _run(self) -> None:
        self.groups.own("display")
        last_frame = 0
        while not self._stop.is_set():
            if self.viewers <= 0:
                sleep(0.2)
                continue
            started = monotonic()
            frame = self.reader.latest
            if frame is None or frame.seq == last_frame:
                sleep(0.02)
                continue
            last_frame = frame.seq
            result = self.inferer.latest
            image = draw(frame.image, result, self.cfg.display_width, self.hud(result))
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.jpeg = buf.tobytes()
                self.seq += 1
                self.rendered.add()
            self.last_ms = (monotonic() - started) * 1000
            sleep(max(0.0, 1 / self.cfg.display_fps - (monotonic() - started)))
