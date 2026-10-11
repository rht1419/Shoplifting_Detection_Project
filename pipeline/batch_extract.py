"""Run pose+tracking extraction for every clip listed in the dataset manifest.

Reuses one loaded model across all clips (resetting tracker state between
each) so a full-dataset run does not pay repeated model-load/GPU-transfer
cost per clip.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from detection.person_pose import PersonPoseTracker
from detection.tracker import TrackerSettings
from pipeline.config import load_config
from pipeline.extract_keypoints import extract_video


def run_batch(
    manifest_path: str | Path,
    keypoints_root: str | Path,
    config_path: str | Path = "configs/default.yaml",
    device: str | None = None,
    skip_existing: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    config = load_config(config_path)
    video_cfg = config.get("video", {})
    pose_cfg = config.get("pose_tracking", {})
    resize = (video_cfg.get("resize_width", 640), video_cfg.get("resize_height", 360))
    target_fps = video_cfg.get("target_fps", 15.0)
    resolved_device = device if device is not None else pose_cfg.get("device", "cpu")

    tracker = PersonPoseTracker(
        model_path=pose_cfg.get("model_path", "yolov8n-pose.pt"),
        tracker=TrackerSettings(
            config=pose_cfg.get("tracker", "bytetrack.yaml"),
            confidence=pose_cfg.get("confidence", 0.35),
            iou_threshold=pose_cfg.get("iou_threshold", 0.5),
        ),
        image_size=pose_cfg.get("image_size", 640),
        device=resolved_device,
    )

    keypoints_dir = Path(keypoints_root)
    keypoints_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        clip_id = row["clip_id"]
        output_path = keypoints_dir / f"{clip_id}.jsonl"
        if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
            print(f"[{index}/{len(rows)}] skip {clip_id} (already extracted)", flush=True)
            continue

        start = time.perf_counter()
        try:
            summary = extract_video(
                row["source_path"],
                output_path,
                camera_id=clip_id,
                target_fps=target_fps,
                resize=resize,
                tracker=tracker,
            )
        except Exception as error:  # noqa: BLE001 - one bad clip must not kill a 180-clip batch
            elapsed = time.perf_counter() - start
            print(f"[{index}/{len(rows)}] FAILED {clip_id} after {elapsed:.1f}s: {error}", flush=True)
            failures.append({"clip_id": clip_id, "error": str(error)})
            continue

        elapsed = time.perf_counter() - start
        print(
            f"[{index}/{len(rows)}] {clip_id}: {summary.sampled_frames} frames, "
            f"{summary.person_records} person-records, {elapsed:.1f}s",
            flush=True,
        )
        results.append({"clip_id": clip_id, "elapsed_s": elapsed, **vars(summary)})

    return results, failures


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Batch-extract pose+tracking artifacts for every manifest clip.")
    parser.add_argument("--manifest", default="data/manifests/clips.csv")
    parser.add_argument("--keypoints-root", default="data/keypoints")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None, help="Override pose_tracking.device from the config (e.g. cpu, 0)")
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-extract clips that already have output")
    args = parser.parse_args()

    results, failures = run_batch(
        args.manifest,
        args.keypoints_root,
        args.config,
        args.device,
        skip_existing=not args.no_skip_existing,
    )

    total_frames = sum(int(r["sampled_frames"]) for r in results)
    total_records = sum(int(r["person_records"]) for r in results)
    print(f"\nDone: {len(results)} clips extracted, {len(failures)} failed.")
    print(f"Total sampled frames: {total_frames}, total person-records: {total_records}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  {failure['clip_id']}: {failure['error']}")


if __name__ == "__main__":
    _cli()
