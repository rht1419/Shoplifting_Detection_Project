"""Extract tracked YOLOv8 pose records and an optional debug video from a clip."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from detection.person_pose import PersonPoseTracker, draw_people
from detection.tracker import TrackerSettings
from pipeline.config import load_config
from pipeline.video_reader import VideoReader


@dataclass(frozen=True)
class ExtractionSummary:
    source_path: str
    output_path: str
    sampled_frames: int
    person_records: int


def extract_video(
    video_path: str | Path,
    output_path: str | Path,
    camera_id: str | None = None,
    target_fps: float = 15.0,
    resize: tuple[int, int] = (640, 360),
    debug_video_path: str | Path | None = None,
    model_path: str = "yolov8n-pose.pt",
    device: str = "cpu",
    tracker_settings: TrackerSettings | None = None,
    image_size: int = 640,
    tracker: PersonPoseTracker | None = None,
) -> ExtractionSummary:
    """Write one JSON object per tracked person per sampled frame.

    Pass a pre-built ``tracker`` to reuse one loaded model across many clips
    (e.g. a batch run); it is reset before each call so track IDs never leak
    between clips. Without one, a fresh tracker is built and torn down here.
    """

    source = Path(video_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tracker = tracker or PersonPoseTracker(
        model_path=model_path,
        tracker=tracker_settings or TrackerSettings(),
        image_size=image_size,
        device=device,
    )
    tracker.reset()
    records = 0
    sampled_frames = 0
    debug_writer: cv2.VideoWriter | None = None

    try:
        with VideoReader(source, target_fps=target_fps, resize=resize) as reader, output.open("w", encoding="utf-8") as stream:
            if debug_video_path is not None:
                debug_path = Path(debug_video_path)
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                debug_writer = cv2.VideoWriter(str(debug_path), cv2.VideoWriter_fourcc(*"mp4v"), target_fps, resize)
                if not debug_writer.isOpened():
                    raise ValueError(f"Could not open debug-video destination: {debug_path}")

            for sampled in reader.iter_frames():
                people = tracker.track_frame(sampled.frame_bgr)
                sampled_frames += 1
                for person in people:
                    record = {
                        "camera_id": camera_id or source.stem,
                        "frame_index": sampled.source_frame_index,
                        "sampled_frame_index": sampled.sampled_frame_index,
                        "timestamp_ms": sampled.timestamp_ms,
                        **asdict(person),
                    }
                    stream.write(json.dumps(record) + "\n")
                    records += 1
                if debug_writer is not None:
                    debug_writer.write(draw_people(sampled.frame_bgr, people))
    finally:
        if debug_writer is not None:
            debug_writer.release()

    return ExtractionSummary(str(source), str(output), sampled_frames, records)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Extract tracked people and pose keypoints from one video.")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--camera-id", default=None, help="Logical camera identifier")
    parser.add_argument("--debug-video", default=None, help="Optional annotated MP4 path")
    parser.add_argument("--config", default="configs/default.yaml", help="Pipeline config YAML path")
    parser.add_argument("--device", default=None, help="Override pose_tracking.device from the config (e.g. cpu, 0)")
    parser.add_argument("--target-fps", type=float, default=None, help="Override video.target_fps from the config")
    args = parser.parse_args()

    config = load_config(args.config)
    video_cfg = config.get("video", {})
    pose_cfg = config.get("pose_tracking", {})
    resize = (video_cfg.get("resize_width", 640), video_cfg.get("resize_height", 360))
    target_fps = args.target_fps if args.target_fps is not None else video_cfg.get("target_fps", 15.0)
    device = args.device if args.device is not None else pose_cfg.get("device", "cpu")
    tracker_settings = TrackerSettings(
        config=pose_cfg.get("tracker", "bytetrack.yaml"),
        confidence=pose_cfg.get("confidence", 0.35),
        iou_threshold=pose_cfg.get("iou_threshold", 0.5),
    )

    summary = extract_video(
        args.video,
        args.output,
        args.camera_id,
        target_fps,
        resize,
        debug_video_path=args.debug_video,
        model_path=pose_cfg.get("model_path", "yolov8n-pose.pt"),
        device=device,
        tracker_settings=tracker_settings,
        image_size=pose_cfg.get("image_size", 640),
    )
    print(summary)


if __name__ == "__main__":
    _cli()
