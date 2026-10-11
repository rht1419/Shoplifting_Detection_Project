# Modules 1–2 Implementation Plan

## Goal

Deliver a reproducible offline foundation that reads CCTV/video files, standardizes frames, runs one lightweight YOLOv8n-Pose model, and returns stable person tracks with pose keypoints. This milestone does not include the behaviour classifier, alerts, backend, or dashboard.

## Scope

| Module | Deliverable |
|---|---|
| 1. Video ingestion and preprocessing | Validate video metadata, sample at 15 FPS, resize frames, and emit timestamps. |
| 2. Person pose and tracking | Run YOLOv8n-Pose and ByteTrack, emitting person ID, box, canonical keypoints, confidence and track state. |

## Output contract

Each tracked person in a processed frame must provide camera ID, frame index, timestamp in milliseconds, track ID, `bbox_xyxy`, `keypoints_xyc`, detection confidence, and track age. Save one generated pose artifact per input clip plus a metadata row in `data/manifests/clips.csv`.

## Implementation tasks

1. Create a pinned Python environment and `requirements.txt`.
2. Add `configs/default.yaml` for input size, target FPS, detector confidence and tracker parameters.
3. Implement `pipeline/video_reader.py` with OpenCV metadata validation and deterministic 15-FPS sampling.
4. Implement `detection/person_pose.py` around `yolov8n-pose.pt`, restricted to people.
5. Implement `detection/tracker.py` with ByteTrack; reset state for every clip.
6. Implement `pipeline/extract_keypoints.py` to connect reader, detector and tracker and serialize the output.
7. Add an annotated debug-video mode with box, skeleton, track ID and effective FPS.
8. Add a manifest generator and tests for frame sampling, output schema and tracker reset.

## Acceptance criteria

- A 60-second video produces a readable pose artifact and one valid manifest entry.
- Visible people retain stable IDs; brief occlusions do not create excessive new IDs.
- Each sequence contains the expected canonical joint count and valid x/y/confidence values.
- Timestamps are monotonic and the effective rate is 15 FPS within one frame over one minute.
- The debug video visibly overlays skeletons, boxes and track IDs.
- Benchmark notes record mean FPS, p95 frame latency, peak RAM and number of active tracks on the laptop CPU.
- Raw videos, downloaded datasets, weights and generated artifacts are excluded from Git.

## Test set for engineering validation

Use three short non-training videos: one person walking/standing, two people crossing/occluding, and one retail-style shelf interaction. These validate the pipeline only and must not be included in model evaluation.

## Handoff to the next milestone

Hand over source code, pinned requirements, default config, generated pose artifacts, debug videos, benchmark notes and known issues. Start feature extraction only after this output schema is stable.
