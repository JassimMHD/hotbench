# Hotbench

**Live-test any local model. Measure what it really costs.**

Hotbench puts a model on a live input (a webcam, an IP camera or a video file)
and shows two things side by side in your browser: **what the model sees**
(boxes, labels, a live count) and **what it costs on this machine** (CPU cores
per stage, RAM, latency, inferences per second, frames it had to skip).

It is for researchers, students and engineers who have a model and need
answers in minutes, not after writing a demo app:

- Does it work on *live* input, not just on my test images?
- How fast is it on *this* machine: a laptop, a 2-core server, an edge box?
- What does one more camera cost? Which image size, thread count or export is
  worth it?

> **Status: pre-alpha.** This is the prototype Hotbench grows from: live object
> detection with Ultralytics models (`.pt` and `.onnx`). Speech models, chat
> with local LLMs and a plugin system for any model are on the
> [roadmap](#roadmap). Interfaces will change before 1.0.

---

## Contents

- [What you get](#what-you-get)
- [Quick start](#quick-start)
- [Use your own model, camera or video](#use-your-own-model-camera-or-video)
- [Reading the page](#reading-the-page)
- [Settings](#settings)
- [Benchmark offline](#benchmark-offline)
- [Record a stream](#record-a-stream)
- [How it works](#how-it-works)
- [Privacy and security](#privacy-and-security)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Licence](#licence)

## What you get

| | |
|---|---|
| **Live view** | the stream with boxes and class labels, plus a HUD: count, median, inferences/s, ms, box age |
| **Count** | the objects in the last frame, a rolling median, and its spread (how stable the count is) |
| **Inference** | inferences/s, time p50/p95, split into pre-process / inference / post-process, frame-to-result delay, frames skipped |
| **Cost** | CPU cores used by **decode**, **inference**, **display** and **other**; process RAM; whole-machine CPU |
| **Charts** | the last 2 minutes of count, cores and inference time |
| **Live settings** | change the source, model, classes, image size, threads, cores and rate without a restart |
| **Evidence** | a CSV of every stat (one row a second), and "Save frame" (raw image + boxes + JSON) to check a count by hand |
| **Offline benchmark** | the same frames through every image size × thread count → a table |

## Quick start

You need **Python 3.10+** and a webcam (or a video file; see below).

**Windows**

```bat
git clone https://github.com/JassimMHD/hotbench
cd hotbench
py -3.12 -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
run.bat
```

**Linux / macOS**

```bash
git clone https://github.com/JassimMHD/hotbench
cd hotbench
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # macOS: drop --index-url
.venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/python app.py
```

Open **http://localhost:8300**. On the first run Hotbench downloads
`yolo26n.pt`, a small detector for the 80 [COCO](https://cocodataset.org)
classes (people, cars, animals, everyday objects), into `weights/`, then starts
on your first webcam.

> Have an NVIDIA GPU? Install the CUDA build of torch instead of the CPU one.
> The prototype runs inference on the CPU, which is what it measures. GPU
> support comes in v0.2.

## Use your own model, camera or video

Set these in `.env`, or type them into **Settings** on the page and press **Apply**.

**A video file** (plays at its own speed and loops, like a camera):

```ini
SOURCE=clips/street.mp4
```

**An IP camera** (RTSP, read over TCP). The password is masked on the page, in
the logs and in the CSV:

```ini
SOURCE=rtsp://user:PASSWORD@192.168.1.10:554/stream1
```

**Another webcam:** `SOURCE=1`, `SOURCE=2`, …

**Your own model:** any Ultralytics detection model, as `.pt` or as an `.onnx`
export:

```ini
WEIGHTS=weights/my-detector.pt
WEIGHTS=weights/my-detector.onnx      # needs: pip install onnxruntime
```

Official names download by themselves: `weights/yolo26s.pt`,
`weights/yolo11n.pt`, …

**Count only some classes:** names or ids, comma-separated. Empty = every class.

```ini
CLASSES=person
CLASSES=car, truck, bus
```

**Imitate a smaller machine:** pin the whole process to some cores, to see
what a 2-core server would do:

```ini
CORES=0,1
THREADS=2
```

## Reading the page

- **Count:** *this frame* is the raw count of the last inference. *Median*
  is over the last `SMOOTH_SECONDS`, the number you would show a user. A wide
  *spread* (p10–p90) means the count jumps from frame to frame.
- **Inference → Frame → count:** how old a frame is when its result is ready.
  It stays at about one inference time, because Hotbench never queues frames
  (see [How it works](#how-it-works)).
- **Inference → Frames skipped each:** how many camera frames went by
  uncounted between two inferences. At 25 fps and 4 inferences/s, that's about 5.
- **Stream → Camera fps / decoded:** if *decoded* is below the camera's fps,
  the machine can't even decode the stream in real time.
- **Cost:** cores used by each stage. `1.00` = one full core. The bar is out of
  the cores the process is allowed to use. *Other* is mostly the web server.
- **Box age** (HUD): how old the boxes on screen are. The video runs at
  `DISPLAY_FPS`, and the boxes update at the inference rate.

**Buttons:** **Start CSV** writes `runs/<time>.csv`, one row a second.
**Save frame** writes the last inferred frame raw, with its boxes, and as JSON
to `runs/frames/`, so you can count it by hand.

## Settings

All of them are in `.env` (see `.env.example`). All except `PORT` can also be
changed live on the page.

| Setting | Default | What it does |
|---|---|---|
| `SOURCE` | `0` | camera index, video file, or RTSP/HTTP URL |
| `WEIGHTS` | `weights/yolo26n.pt` | `.pt` / `.onnx` file, or an official Ultralytics name |
| `CLASSES` | *(all)* | which classes to detect and count |
| `IMGSZ` | `640` | inference image size (multiple of 32). Bigger finds smaller objects and costs more |
| `CONF` | `0.25` | minimum confidence for a box |
| `MAX_DET` | `300` | maximum boxes per frame. Raise it for crowded scenes, or the count is capped |
| `THREADS` | `min(4, CPUs)` | inference threads |
| `CORES` | *(all)* | CPU cores the whole process may use: `0,1` or `0-3` |
| `INFER_MAX_FPS` | `0` | cap on inferences/s (0 = as fast as possible). `1` shows the cost of one count a second |
| `INFER_ENABLED` | `true` | `false` measures decoding and display alone |
| `DISPLAY_FPS` | `10` | browser video rate. It costs CPU and shows up as "display" |
| `DISPLAY_WIDTH` | `1280` | browser video width |
| `SMOOTH_SECONDS` | `30` | window of the median count |
| `PORT` | `8300` | the page is at `http://localhost:PORT` |

## Benchmark offline

The live page shows one setting at a time. `bench.py` sends the **same
frames** through every combination, so the results compare fairly:

```bash
python bench.py --source clips/street.mp4 --imgsz 960,768,640 --threads 1,2,4
python bench.py --cores 0,1 --threads 1,2          # act like a 2-core server
```

It prints a markdown table (ms p50/p95, inferences/s, cores used, core-seconds
per inference, mean count, RAM) and writes `runs/bench-<time>.csv`. The
*mean count* column shows what a smaller image size does to accuracy.

## Record a stream

A camera never shows the same scene twice. Record one, then test against the
recording:

```bash
python record.py --source rtsp://user:PASSWORD@192.168.1.10:554/stream1 --seconds 60 --name entrance
python bench.py --source clips/entrance.mp4
```

`record.py` copies the stream without re-encoding (`-c copy`), so it costs
almost no CPU. It needs [ffmpeg](https://ffmpeg.org/download.html) on your PATH.

## How it works

```
source ──► Reader thread ──► latest-frame slot (1 frame, always overwritten)
           decodes EVERY frame        │
           (keeps the stream live)    ├──► Inference thread: takes the NEWEST frame,
                                      │    runs the model, skips the rest
                                      │    → count, boxes, timings
                                      ▼
                         Display thread: newest frame + newest boxes + HUD
                                      → JPEG → /video.mjpg
Browser:  <img src="/video.mjpg">  +  EventSource("/stats")  →  panels and charts
```

- **Latest frame wins.** A demo that queues frames falls further behind every
  second when the model is slower than the camera. Hotbench always takes the
  newest frame and drops the rest, so the delay stays at about one inference,
  as in a real deployment. The dropped frames are counted and shown.
- **CPU per stage.** Each OS thread is credited to the stage that started it.
  Torch's and ONNX Runtime's worker threads count as *inference*, and ffmpeg's
  decoder threads as *decode*. So the cost split is real, not estimated.
- **Warm-up is not measured.** The first inference runs on a blank frame
  before the numbers start.
- **ONNX thread fix.** ONNX Runtime starts one thread per logical CPU by
  default, which is up to 10× slower when the process is pinned to a few cores.
  Hotbench caps it at `THREADS`.

| File | Role |
|---|---|
| `app.py` | the web server: page, MJPEG video, stats stream (SSE), settings API |
| `lab/reader.py` | reads the source: reconnects, plays files at their own fps |
| `lab/infer.py` | the inference loop, class filter, timings |
| `lab/render.py` | boxes, labels and HUD on the video |
| `lab/stats.py` | one snapshot a second, CSV recording |
| `lab/metrics.py` | rolling rates and percentiles, CPU per stage |
| `lab/config.py` | settings from `.env`, validation, password masking |
| `static/index.html` | the page (plain JS, no build step) |
| `bench.py`, `record.py` | offline benchmark, stream recorder |

## Privacy and security

- The server listens on **127.0.0.1 only**: other machines can't open the
  page or the video.
- Passwords in camera URLs are replaced by `***` on the page, in logs, in CSV
  files and in saved-frame JSON.
- Nothing is uploaded anywhere. The only network access is the one-time model
  download.
- `.env`, `weights/`, `clips/` and `runs/` are git-ignored, so your camera
  URLs, models and recordings stay out of commits.
- `.pt` files can run code when loaded. Only load models from sources you trust.

Found a vulnerability? See [SECURITY.md](SECURITY.md).

## Troubleshooting

| Problem | Fix |
|---|---|
| `stream offline: cannot open 0` | no webcam at index 0: try `SOURCE=1`, or use a video file |
| An IP camera refuses the connection | cameras allow only 2–4 sessions: close other viewers (NVR, apps) or use the camera's sub-stream |
| Smeared or grey frames from RTSP | already read over TCP. Check the camera's bitrate, or use the sub-stream |
| `weights not found` | check the path in `WEIGHTS`; it is relative to this folder |
| `the model has no class …` | `CLASSES` must use the model's own class names. The error lists them |
| Very slow `.onnx` model | set `THREADS` no higher than the cores in `CORES` |
| *Torch threads > cores allowed* warning | lower `THREADS` or allow more `CORES`: the threads fight over the cores |
| The page says "server offline" | the server stopped: check the terminal for the error |

## Roadmap

| Version | Scope |
|---|---|
| **v0.1** | `pip install hotbench`, a `hotbench` CLI, ONNX Runtime as the default backend, Ultralytics as an optional extra |
| **v0.2** | any vision model through adapters (OpenVINO, Hugging Face, timm), plugins, GPU metrics, saved runs + comparisons |
| **v0.3** | live speech: mic input, voice activity detection, speech-to-text, text-to-speech, real-time factor |
| **v0.4** | live chat with local LLMs (llama.cpp, Ollama, OpenAI-compatible servers), time to first token, tokens/s; vision-language models on the camera |
| **v0.5** | the voice loop (mic → speech-to-text → LLM → speech) with per-stage latency; two models side by side |
| **1.0** | a stable adapter API and run format |

Ideas and requests are welcome in [Discussions](https://github.com/JassimMHD/hotbench/discussions).

## Contributing

Bug reports, adapter requests and benchmark results from your hardware all
help. Read [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md) first.

## Licence

Hotbench is licensed under [Apache-2.0](LICENSE). Copyright 2026 Mohamed Jassim.

The prototype uses [Ultralytics](https://github.com/ultralytics/ultralytics),
which is licensed **AGPL-3.0**, as are its pretrained weights. If you distribute
a program that includes Ultralytics, or offer it as a network service, the AGPL
applies to that program. From v0.1, Ultralytics becomes an optional extra and
the default backend is ONNX Runtime (MIT).

If you use Hotbench in research, please cite it ([CITATION.cff](CITATION.cff)).
