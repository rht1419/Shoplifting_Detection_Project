"""Real-time demo window: play a clip (or webcam) with live pose+tracking overlays."""

from __future__ import annotations

import argparse
import time

import cv2

from detection.person_pose import PersonPoseTracker, draw_people
from detection.tracker import TrackerSettings
from pipeline.config import load_config
from pipeline.video_reader import VideoReader

QUIT_KEYS = (ord("q"), 27)  # q or Esc


def _overlay_hud(frame, people_count: int, processing_fps: float):
    cv2.putText(
        frame,
        f"proc {processing_fps:4.1f} FPS | people {people_count} | q to quit",
        (10, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def _run_on_file(source: str, tracker: PersonPoseTracker, target_fps: float, resize: tuple[int, int], window_name: str, loop: bool) -> None:
    frame_budget_s = 1.0 / target_fps
    while True:
        tracker.reset()
        with VideoReader(source, target_fps=target_fps, resize=resize) as reader:
            for sampled in reader.iter_frames():
                start = time.perf_counter()
                people = tracker.track_frame(sampled.frame_bgr)
                annotated = draw_people(sampled.frame_bgr, people)
                elapsed = time.perf_counter() - start
                processing_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                _overlay_hud(annotated, len(people), processing_fps)
                cv2.imshow(window_name, annotated)
                wait_s = max(frame_budget_s - elapsed, 0.001)
                if cv2.waitKey(max(1, int(wait_s * 1000))) & 0xFF in QUIT_KEYS:
                    return
        if not loop:
            return


def _run_on_webcam(index: int, tracker: PersonPoseTracker, resize: tuple[int, int], window_name: str) -> None:
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise ValueError(f"Could not open webcam index {index}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            start = time.perf_counter()
            people = tracker.track_frame(frame)
            annotated = draw_people(frame, people)
            elapsed = time.perf_counter() - start
            processing_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            _overlay_hud(annotated, len(people), processing_fps)
            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                break
    finally:
        capture.release()


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Live pose+tracking demo window.")
    parser.add_argument("--source", default="data/shoplifting/shoplifting-1.mp4", help="Video file path, or webcam index (e.g. 0)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None, help="Override pose_tracking.device from the config (e.g. cpu, 0)")
    parser.add_argument("--no-loop", action="store_true", help="Play a file source once instead of looping")
    args = parser.parse_args()

    config = load_config(args.config)
    video_cfg = config.get("video", {})
    pose_cfg = config.get("pose_tracking", {})
    resize = (video_cfg.get("resize_width", 640), video_cfg.get("resize_height", 360))
    target_fps = video_cfg.get("target_fps", 15.0)
    device = args.device if args.device is not None else pose_cfg.get("device", "cpu")

    tracker = PersonPoseTracker(
        model_path=pose_cfg.get("model_path", "yolov8n-pose.pt"),
        tracker=TrackerSettings(
            config=pose_cfg.get("tracker", "bytetrack.yaml"),
            confidence=pose_cfg.get("confidence", 0.35),
            iou_threshold=pose_cfg.get("iou_threshold", 0.5),
        ),
        image_size=pose_cfg.get("image_size", 640),
        device=device,
    )

    window_name = "Shoplifting Detection - Module 2 Demo (q to quit)"
    try:
        if args.source.isdigit():
            _run_on_webcam(int(args.source), tracker, resize, window_name)
        else:
            _run_on_file(args.source, tracker, target_fps, resize, window_name, loop=not args.no_loop)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _cli()
