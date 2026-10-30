"""Rolling numbers and CPU per stage.

CPU per stage: torch and ffmpeg run their work on threads of their own (the
OpenMP pool, the decoder's frame threads), so a stage's own thread_time() misses
most of it. Instead every OS thread of the process is given to the stage that
created it: `claim(stage)` records the threads that appear while the stage opens
its stream or loads its model, and `own(stage)` the stage's own thread. Threads
nobody claimed (the web server, Python's main thread) are "other".
"""

from __future__ import annotations

import statistics
import threading
from collections import deque
from contextlib import contextmanager
from time import monotonic

import psutil


class Rolling:
    """(time, value) pairs from the last `seconds`."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._items: deque[tuple[float, float]] = deque()
        self._lock = threading.Lock()

    def add(self, value: float = 1.0, t: float | None = None) -> None:
        t = monotonic() if t is None else t
        with self._lock:
            self._items.append((t, value))
            self._trim(t)

    def _trim(self, now: float) -> None:
        while self._items and self._items[0][0] < now - self.seconds:
            self._items.popleft()

    def values(self, seconds: float | None = None) -> list[float]:
        now = monotonic()
        since = now - (seconds if seconds is not None else self.seconds)
        with self._lock:
            self._trim(now)
            return [v for t, v in self._items if t >= since]

    def rate(self) -> float:
        """Events per second over the window (0 when fewer than 2)."""
        with self._lock:
            self._trim(monotonic())
            if len(self._items) < 2:
                return 0.0
            span = self._items[-1][0] - self._items[0][0]
            return (len(self._items) - 1) / span if span > 0 else 0.0


def pct(values: list[float], q: float) -> float | None:
    """The q-th percentile (0-100), nearest rank; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q / 100 * (len(ordered) - 1))))]


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


class ThreadGroups:
    def __init__(self) -> None:
        self.proc = psutil.Process()
        self._stage: dict[int, str] = {}
        self._prev: dict[int, float] = {}
        self._lock = threading.Lock()

    def _ids(self) -> set[int]:
        return {t.id for t in self.proc.threads()}

    def own(self, stage: str) -> None:
        with self._lock:
            self._stage[threading.get_native_id()] = stage

    @contextmanager
    def claim(self, stage: str):
        before = self._ids()
        try:
            yield
        finally:
            new = self._ids() - before
            with self._lock:
                for tid in new:
                    self._stage.setdefault(tid, stage)

    def cpu_seconds_since_last(self) -> dict[str, float]:
        """CPU seconds used per stage since the previous call."""
        used: dict[str, float] = {}
        with self._lock:
            first = not self._prev
            seen = {}
            for t in self.proc.threads():
                total = t.user_time + t.system_time
                seen[t.id] = total
                # A thread born since the last call counts from 0.
                delta = total - self._prev.get(t.id, total if first else 0.0)
                stage = self._stage.get(t.id, "other")
                used[stage] = used.get(stage, 0.0) + max(0.0, delta)
            self._prev = seen
        return used
