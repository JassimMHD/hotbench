# Changelog

All notable changes to Hotbench are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Until 1.0, minor versions may change
the adapter API; such changes are listed under **Changed** with a migration note.

## [Unreleased]

### Added

- Prototype: live object detection on one stream (webcam, video file or RTSP
  camera) with Ultralytics models (`.pt` / `.onnx`). Latest-frame-wins
  inference, so latency stays at about one inference.
- Live cost in the browser: CPU cores per stage (decode, inference, display,
  other), RAM, latency p50/p95, inferences/s, frames skipped, with 2-minute charts.
- Count with a rolling median and spread; `CLASSES` to count only some classes;
  class labels on the boxes.
- Live settings (source, model, classes, image size, threads, CPU cores, rate),
  CSV recording, "Save frame" for checking a count by hand.
- `bench.py` (image size × threads on the same frames) and `record.py` (save a
  stream to a clip).
- The official `yolo26n.pt` (COCO) is downloaded on first run, so a fresh clone
  works with a webcam and no other setup.
