# Module 2 — Pose + Tracking Benchmark Notes

## Environment

- CPU: AMD64, 8 physical / 16 logical cores (AuthenticAMD Family 25 Model 117)
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
- RAM: 15.3 GB total
- OS: Windows 11
- Model: `yolov8n-pose.pt` (Ultralytics 8.4.126)
- Tracker: ByteTrack (`bytetrack.yaml`), confidence 0.35, IoU 0.5, imgsz 640
- `configs/default.yaml: pose_tracking.device` now defaults to `"0"` (GPU);
  pass `--device cpu` to `extract_keypoints`/`benchmark`/`batch_extract` to
  force CPU on a machine without a CUDA GPU.

## Method

`python -m pipeline.benchmark <clip> ... --output artifacts/debug/benchmark_results.json`
times each `PersonPoseTracker.track_frame` call (model inference + ByteTrack
association) on 15-FPS-sampled, 640x360 frames, and samples process RSS after
every frame via `psutil`. Per-clip numbers below include the first clip's
one-time CUDA/cuDNN warm-up cost; a hot GPU (as in the full-dataset batch
run) is faster than the isolated per-clip numbers suggest.

## Results — per-clip, isolated process

| Clip | Device | Sampled frames | Mean FPS | p95 frame latency | Peak RSS | Max active tracks |
|---|---|---|---|---|---|---|
| `data/normal/normal-9.mp4` (2 people) | CPU | 158 | 14.7 | 44.3 ms | 378 MB | 2 |
| `data/shoplifting/shoplifting-1.mp4` (2 people, interaction) | CPU | 156 | 23.5 | 44.3 ms | 399 MB | 2 |
| `data/normal/normal-9.mp4` (2 people) | GPU (cold) | 158 | 15.5 | 33.4 ms | 1409 MB | 2 |
| `data/shoplifting/shoplifting-1.mp4` (2 people, interaction) | GPU (warm) | 156 | 40.5 | 29.8 ms | 1420 MB | 2 |

Raw JSON: `artifacts/debug/benchmark_results.json` /
`benchmark_results_gpu.json` (git-ignored, regenerate on demand).

## Results — full-dataset batch run (GPU, model loaded once)

`python -m pipeline.batch_extract` processed all 182 clips (179 newly
extracted; 3 already present from earlier validation), reusing one loaded
model/tracker across the whole run instead of reloading per clip:

- 179 clips extracted, **0 failures**
- 29,925 total sampled frames, 41,455 total person-records
- 886.1s total processing time → **~33.8 FPS average**, ~4.95s/clip

## Reading

- On CPU, mean throughput (~15-24 FPS) is close to the 15 FPS target
  sampling rate — enough for one live stream, not much headroom for more.
- On GPU, an isolated cold-start clip is only modestly faster than CPU (CUDA
  context + cuDNN kernel selection overhead dominates a single short clip),
  but once warm, per-clip throughput roughly doubles to triples CPU, and the
  full 182-clip batch (model loaded once) averaged **~34 FPS** — over 2x the
  15 FPS target, with real headroom for multiple camera streams or for the
  classifier stage layered on top.
- p95 latency stays well under the ~67 ms budget for 15 FPS on both devices,
  so occasional slow frames do not by themselves cause the pipeline to fall
  behind the target rate.
- Peak RSS is much higher on GPU (~1.4 GB vs ~380-400 MB on CPU) — this is
  host memory pinned by the CUDA runtime/allocator, not GPU VRAM, and is a
  fixed process-level cost rather than something that grows with clip count.
- These numbers describe the raw pose+tracking stage only. Later modules
  (feature extraction, classifier) add their own cost and should be
  benchmarked separately once implemented.

## Known issues / follow-ups

- `torch`/`torchvision` must be matching CUDA builds (both `+cu126` here) —
  upgrading only `torch` leaves a CPU-only `torchvision` that raises
  `NotImplementedError: Could not run 'torchvision::nms' with ... 'CUDA'`.
  If re-installing on a new machine, install both from the same
  `--index-url https://download.pytorch.org/whl/cuXXX`.
- The GPU numbers above still describe single-clip/single-stream throughput;
  concurrent multi-camera inference would need explicit batching to make
  full use of the GPU.
- Five `normal-*` clips (`normal-1`, `-13`, `-25`, `-29`, `-39`) legitimately
  produced 0 person-records — visually confirmed as empty-room/no-person
  shots, not a pipeline bug.
