r"""Hotbench (prototype): live object detection on one stream, and what it costs.

    run.bat     (or: python app.py)
    open        http://localhost:8300

    GET  /                the page
    GET  /video.mjpg      the stream with boxes (MJPEG)
    GET  /stats           one JSON snapshot a second (Server-Sent Events)
    GET  /api/config      current settings;  POST: change some of them live
    POST /api/record      {"on": true|false}: CSV of the stats, one row a second
    POST /api/save-frame  the current frame (raw + boxes + count) to runs/frames/

Settings: .env (see .env.example). It is a test tool: it listens on 127.0.0.1 only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Keep ultralytics' settings file in this folder, not the user's shared one.
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(exist_ok=True)  # else it falls back to \tmp

import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from datetime import datetime  # noqa: E402

import cv2  # noqa: E402
import psutil  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import Body, FastAPI, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402

from lab import config  # noqa: E402
from lab.infer import Inferer  # noqa: E402
from lab.metrics import ThreadGroups  # noqa: E402
from lab.reader import Reader  # noqa: E402
from lab.render import Renderer, draw  # noqa: E402
from lab.stats import Stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("lab")

cfg = config.load()
groups = ThreadGroups()
reader = Reader(cfg, groups)
inferer = Inferer(cfg, reader, groups)
renderer = Renderer(cfg, reader, inferer, groups)
stats = Stats(cfg, reader, inferer, renderer, groups)


def apply_cores() -> None:
    """Pin the whole process (every thread) to CORES, e.g. "0,1" to act like a 2-core server."""
    psutil.Process().cpu_affinity(config.parse_cores(cfg.cores))


@asynccontextmanager
async def lifespan(app: FastAPI):
    apply_cores()
    log.info("cores %s, torch threads %d, source %s", cfg.cores or "all", cfg.threads,
             config.mask(cfg.source))
    for part in (reader, inferer, renderer, stats):
        part.start()
    yield
    for part in (stats, renderer, inferer, reader):
        part.stop()


app = FastAPI(title="Hotbench", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.get("/")
async def page() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/video.mjpg")
async def video() -> StreamingResponse:
    async def frames():
        renderer.watch(+1)
        try:
            last = 0
            while True:
                jpeg, seq = renderer.jpeg, renderer.seq
                if jpeg and seq != last:
                    last = seq
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(0.02)
        finally:
            renderer.watch(-1)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@app.get("/stats")
async def stats_stream(request: Request) -> StreamingResponse:
    async def events():
        last = None
        while not await request.is_disconnected():
            snap = stats.latest
            if snap and snap is not last:
                last = snap
                yield f"data: {json.dumps(snap)}\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store"})


@app.get("/api/config")
async def get_config() -> dict:
    return cfg.public()


@app.post("/api/config")
async def set_config(changes: dict = Body(...)) -> JSONResponse:
    try:
        changed = cfg.update(changes)
    except (ValueError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if "cores" in changed:
        apply_cores()
    if "source" in changed:
        reader.reopen()
    if "weights" in changed:
        inferer.reload()
    if changed:
        log.info("settings changed: %s", ", ".join(sorted(changed)))
    return JSONResponse({"changed": sorted(changed), "config": cfg.public()})


@app.post("/api/record")
async def record(body: dict = Body(...)) -> dict:
    path = stats.record(bool(body.get("on")))
    return {"recording": path.name if path else None}


@app.post("/api/save-frame")
async def save_frame() -> JSONResponse:
    # The frame the newest count was made on, with exactly its boxes: for
    # checking the count by hand.
    frame, result = inferer.latest_frame, inferer.latest
    if frame is None or result is None:
        return JSONResponse({"error": "no counted frame yet"}, status_code=409)
    folder = ROOT / "runs" / "frames"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}-count{result.count}"
    cv2.imwrite(str(folder / f"{stem}-raw.jpg"), frame.image)
    cv2.imwrite(str(folder / f"{stem}-boxes.jpg"),
                draw(frame.image, result, frame.image.shape[1], f"count {result.count}"))
    meta = {"count": result.count, "labels": list(result.labels), "smoothed": inferer.smoothed(),
            "confidence": result.confidence, "frame_seq": frame.seq,
            "config": cfg.public()}
    (folder / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return JSONResponse({"saved": f"runs/frames/{stem}-raw.jpg"})


if __name__ == "__main__":
    print(f"Hotbench on http://localhost:{cfg.port}", file=sys.stderr)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")
