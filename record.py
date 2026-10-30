r"""Record N seconds of a stream to clips/, unchanged (-c copy), for repeatable tests.

    python record.py --seconds 60 --name entrance-cam-25fps
    python record.py --source rtsp://user:pass@<camera-ip>:554/... --seconds 120

Then point SOURCE (or bench.py --source) at clips/<name>.mp4. Needs ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from lab import config

ROOT = Path(__file__).resolve().parent


def main() -> None:
    cfg = config.load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=cfg.source)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--name", default=f"clip-{datetime.now():%Y%m%d-%H%M%S}")
    args = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH")

    source = config.resolve(args.source)
    out = ROOT / "clips" / f"{args.name}.mp4"
    out.parent.mkdir(exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if source.startswith("rtsp://"):
        cmd += ["-rtsp_transport", "tcp"]
    cmd += ["-i", source, "-t", str(args.seconds), "-an", "-c", "copy", str(out)]
    print(f"recording {args.seconds}s of {config.mask(args.source)} -> clips/{out.name}")
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if proc.returncode != 0:
        # ffmpeg can echo the URL with its password.
        sys.exit(proc.stderr.replace(source, config.mask(source)).strip()[-500:])
    print(f"done: {out.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
