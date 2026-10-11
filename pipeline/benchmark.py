"""Measure Module 2 pose+tracking throughput and memory on the local CPU."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

from detection.person_pose import PersonPoseTracker
from detection.tracker import TrackerSettings
from pipeline.config import load_config
from pipeline.video_reader import VideoReader


@dataclass(frozen=True)
class BenchmarkResult:
    clip: str
    sampled_frames: int
    mean_fps: float
    p95_latency_ms: float
    peak_rss_mb: float
    max_active_tracks: int


def benchmark_clip(
    video_path: str | Path,
    target_fps: float = 15.0,
    resize: tuple[int, int] = (640, 360),
    model_path: str = "yolov8n-pose.pt",
    device: str = "cpu",
    tracker_settings: TrackerSettings | None = None,
    image_size: int = 640,
) -> BenchmarkResult:
    tracker = PersonPoseTracker(
        model_path=model_path,
        tracker=tracker_settings or TrackerSettings(),
        image_size=image_size,
        device=device,
    )
    tracker.reset()
    process = psutil.Process()

    frame_latencies_ms: list[float] = []
    max_active_tracks = 0
    peak_rss_mb = process.memory_info().rss / (1024 * 1024)

    with VideoReader(video_path, target_fps=target_fps, resize=resize) as reader:
        for sampled in reader.iter_frames():
            start = time.perf_counter()
            people = tracker.track_frame(sampled.frame_bgr)
            frame_latencies_ms.append((time.perf_counter() - start) * 1000.0)
            max_active_tracks = max(max_active_tracks, len(people))
            peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024 * 1024))

    mean_latency_ms = statistics.fmean(frame_latencies_ms)
    return BenchmarkResult(
        clip=str(video_path),
        sampled_frames=len(frame_latencies_ms),
        mean_fps=1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0,
        p95_latency_ms=statistics.quantiles(frame_latencies_ms, n=100)[94],
        peak_rss_mb=peak_rss_mb,
        max_active_tracks=max_active_tracks,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8n-Pose + ByteTrack on one or more clips.")
    parser.add_argument("clips", nargs="+", help="Video paths to benchmark")
    parser.add_argument("--output", default=None, help="Optional JSON path to record results")
    parser.add_argument("--config", default="configs/default.yaml", help="Pipeline config YAML path")
    parser.add_argument("--device", default=None, help="Override pose_tracking.device from the config (e.g. cpu, 0)")
    args = parser.parse_args()

    config = load_config(args.config)
    video_cfg = config.get("video", {})
    pose_cfg = config.get("pose_tracking", {})
    resize = (video_cfg.get("resize_width", 640), video_cfg.get("resize_height", 360))
    target_fps = video_cfg.get("target_fps", 15.0)
    device = args.device if args.device is not None else pose_cfg.get("device", "cpu")
    tracker_settings = TrackerSettings(
        config=pose_cfg.get("tracker", "bytetrack.yaml"),
        confidence=pose_cfg.get("confidence", 0.35),
        iou_threshold=pose_cfg.get("iou_threshold", 0.5),
    )

    results = [
        benchmark_clip(
            clip,
            target_fps=target_fps,
            resize=resize,
            model_path=pose_cfg.get("model_path", "yolov8n-pose.pt"),
            device=device,
            tracker_settings=tracker_settings,
            image_size=pose_cfg.get("image_size", 640),
        )
        for clip in args.clips
    ]
    for result in results:
        print(result)

    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")


if __name__ == "__main__":
    _cli()
