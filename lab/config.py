"""Settings: .env (next to app.py) or the environment, changeable live from the page."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psutil

ROOT = Path(__file__).resolve().parent.parent
# Official Ultralytics weights (yolo26n.pt, yolo11s-seg.pt, …) download on first use.
OFFICIAL_WEIGHTS = re.compile(r"^yolo[\w.-]*\.pt$")


def load_dotenv() -> None:
    """KEY=VALUE lines from ./.env, without overriding the real environment."""
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        value = value.split(" #", 1)[0].strip()
        if sep and key.strip() and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value)


def mask(url: str) -> str:
    """The URL with its password replaced by ***, safe for the page and logs."""
    parts = urlsplit(url)
    userinfo, at, hostport = parts.netloc.rpartition("@")
    if not at or ":" not in userinfo:
        return url
    return urlunsplit(parts._replace(netloc=f"{userinfo.split(':', 1)[0]}:***@{hostport}"))


def resolve(path: str) -> str:
    """A relative file path is under the project folder; URLs and camera indexes stay as they are."""
    if not path or "://" in path or path.isdigit() or Path(path).is_absolute():
        return path
    return str((ROOT / path).resolve())


def parse_cores(text: str) -> list[int]:
    """'0,1' or '0-3' -> [0, 1] / [0, 1, 2, 3]; '' -> every core."""
    total = psutil.cpu_count() or 1
    if not text.strip():
        return list(range(total))
    cores: set[int] = set()
    for part in text.split(","):
        lo, dash, hi = part.strip().partition("-")
        cores.update(range(int(lo), int(hi) + 1) if dash else [int(lo)])
    if not cores or min(cores) < 0 or max(cores) >= total:
        raise ValueError(f"cores must be between 0 and {total - 1}")
    return sorted(cores)


@dataclass
class Config:
    source: str = "0"  # the first webcam
    weights: str = "weights/yolo26n.pt"  # COCO, 80 classes; downloaded on first run
    classes: str = ""  # what to detect and count: "person, car" or "0,2"; empty = every class
    imgsz: int = 640
    conf: float = 0.25
    max_det: int = 300
    threads: int = field(default_factory=lambda: min(4, psutil.cpu_count() or 1))
    cores: str = ""  # empty = every core
    infer_enabled: bool = True
    infer_max_fps: float = 0.0  # 0 = as fast as the CPU allows
    display_fps: float = 10.0
    display_width: int = 1280
    smooth_seconds: float = 30.0
    host: str = "127.0.0.1"
    port: int = 8300

    def public(self) -> dict:
        data = asdict(self)
        data["source"] = mask(self.source)
        return data

    def update(self, changes: dict) -> set[str]:
        """Apply a partial update from the page; ValueError on a bad value.
        Returns the names that changed."""
        new = Config(**asdict(self))
        types = {f.name: f.type for f in fields(Config)}
        for name, value in changes.items():
            if name not in types or name in ("host", "port"):
                raise ValueError(f"unknown or fixed setting: {name}")
            if name == "source" and isinstance(value, str) and "***" in value:
                continue  # the page sent the masked URL back unchanged
            setattr(new, name, _convert(types[name], value))
        new.validate()
        changed = {n for n in types if getattr(new, n) != getattr(self, n)}
        for name in changed:
            setattr(self, name, getattr(new, name))
        return changed

    def validate(self) -> None:
        if not 320 <= self.imgsz <= 1920 or self.imgsz % 32:
            raise ValueError("imgsz must be a multiple of 32 between 320 and 1920")
        if not 0.01 <= self.conf <= 0.99:
            raise ValueError("conf must be between 0.01 and 0.99")
        if not 1 <= self.max_det <= 3000:
            raise ValueError("max_det must be between 1 and 3000")
        if not 1 <= self.threads <= (psutil.cpu_count() or 1):
            raise ValueError("threads must be between 1 and the number of CPUs")
        parse_cores(self.cores)
        if not 0 <= self.infer_max_fps <= 60:
            raise ValueError("infer_max_fps must be between 0 and 60")
        if not 1 <= self.display_fps <= 30:
            raise ValueError("display_fps must be between 1 and 30")
        if not 320 <= self.display_width <= 3840:
            raise ValueError("display_width must be between 320 and 3840")
        if not 1 <= self.smooth_seconds <= 600:
            raise ValueError("smooth_seconds must be between 1 and 600")
        path = Path(resolve(self.weights))
        if not path.exists() and not OFFICIAL_WEIGHTS.match(path.name):
            raise ValueError(f"weights not found: {self.weights} (give a .pt or .onnx file, "
                             "or an official Ultralytics name such as weights/yolo26n.pt)")


def _convert(kind: str, value):
    if kind == "bool":
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    return str(value).strip()


def load() -> Config:
    load_dotenv()
    cfg = Config()
    for f in fields(Config):
        value = os.environ.get(f.name.upper())
        # An empty number keeps its default (THREADS= ...); an empty text is
        # meaningful (CORES= means every core).
        if value is None or (not value.strip() and f.type != "str"):
            continue
        setattr(cfg, f.name, _convert(f.type, value))
    cfg.validate()
    return cfg
