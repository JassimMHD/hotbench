"""Stream reader: decodes EVERY frame and keeps only the newest.

Reading every frame keeps the RTSP buffer empty, so the frame the model takes is
always the live one (a reader that falls behind shows up as decode fps below the
camera's fps). A file source is played at its own fps and loops.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep

import cv2
import numpy as np

from lab.config import Config, mask, resolve
from lab.metrics import Rolling, ThreadGroups

# RTSP over TCP: UDP drops packets on a busy link and the frames come out smeared.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

log = logging.getLogger("lab.reader")
TIMEOUT_MS = 10_000


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR
    seq: int
    t: float  # monotonic, when it was decoded


class Reader:
    def __init__(self, cfg: Config, groups: ThreadGroups):
        self.cfg = cfg
        self.groups = groups
        self.latest: Frame | None = None
        self.status = "starting"
        self.error = ""
        self.width = self.height = 0
        self.declared_fps = 0.0
        self.reconnects = 0
        self.decoded = Rolling(5)
        self._cond = threading.Condition()
        self._seq = 0
        self._reopen = threading.Event()
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, name="reader", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def reopen(self) -> None:
        """Switch to the configured source (after a change from the page)."""
        self._reopen.set()

    def wait_newer(self, seq: int, timeout: float) -> Frame | None:
        """The newest frame once it is newer than `seq`, or None after `timeout`."""
        with self._cond:
            self._cond.wait_for(
                lambda: self.latest is not None and self.latest.seq > seq, timeout
            )
            frame = self.latest
        return frame if frame is not None and frame.seq > seq else None

    def _open(self, source: str) -> cv2.VideoCapture:
        if source.isdigit():
            return cv2.VideoCapture(int(source))
        return cv2.VideoCapture(
            source, cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT_MS],
        )

    def _run(self) -> None:
        self.groups.own("decode")
        backoff = 1.0
        while not self._stop.is_set():
            self._reopen.clear()
            source = resolve(self.cfg.source)
            if not source:
                self.status, self.error = "no source", "set SOURCE in .env or on the page"
                self._reopen.wait(1)
                continue
            self.status, self.error = "connecting", ""
            with self.groups.claim("decode"):
                cap = self._open(source)
            if not cap.isOpened():
                cap.release()
                self.status, self.error = "offline", f"cannot open {mask(source)}"
                log.warning("%s; retry in %.0fs", self.error, backoff)
                self.reconnects += 1
                self._reopen.wait(backoff)
                backoff = min(backoff * 2, 30)
                continue
            backoff = 1.0
            try:
                self._read(cap, Path(source).is_file())
            except Exception as exc:  # never let the thread die
                log.exception("reader crashed")
                self.error = mask(str(exc))[:200]
            finally:
                cap.release()
            if not self._stop.is_set() and not self._reopen.is_set():
                self.status = "offline"
                self.reconnects += 1
                sleep(1)

    def _read(self, cap: cv2.VideoCapture, is_file: bool) -> None:
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.declared_fps = round(cap.get(cv2.CAP_PROP_FPS) or 0.0, 2)
        self.status = "live"
        log.info("reading %s: %dx%d @ %s fps", mask(self.cfg.source), self.width,
                 self.height, self.declared_fps or "?")
        period = 1 / self.declared_fps if is_file and self.declared_fps > 0 else 0.0
        next_t = monotonic()
        rewound = False
        while not self._stop.is_set() and not self._reopen.is_set():
            ok, image = cap.read()
            if not ok:
                if is_file and not rewound:  # loop the clip
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    rewound = True
                    continue
                self.error = "stream ended or no frame within 10 s"
                return
            rewound = False
            now = monotonic()
            self._seq += 1
            with self._cond:
                self.latest = Frame(image, self._seq, now)
                self._cond.notify_all()
            self.decoded.add(t=now)
            if period:  # a file plays at its own speed, like a camera
                next_t += period
                wait = next_t - monotonic()
                if wait > 0:
                    sleep(wait)
                elif wait < -1:  # fell far behind: do not race to catch up
                    next_t = monotonic()
