"""Deterministic video ingestion for the shoplifting-alert pipeline.

The reader preserves source timestamps, samples input videos at a fixed target
rate, and optionally resizes frames. It deliberately returns BGR frames because
OpenCV and the later YOLO pipeline consume BGR NumPy arrays.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2


SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata read directly from a source video."""

    source_path: str
    frame_count: int
    source_fps: float
    width: int
    height: int
    duration_ms: float


@dataclass(frozen=True)
class SampledFrame:
    """One frame emitted by :class:`VideoReader`."""

    source_frame_index: int
    sampled_frame_index: int
    timestamp_ms: float
    frame_bgr: object


class VideoReader:
    """Read a video and emit monotonically ordered frames at ``target_fps``."""

    def __init__(
        self,
        video_path: str | Path,
        target_fps: float = 15.0,
        resize: tuple[int, int] | None = (640, 360),
    ) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be greater than zero")
        if resize is not None and (resize[0] <= 0 or resize[1] <= 0):
            raise ValueError("resize dimensions must be greater than zero")

        self.path = Path(video_path)
        self.target_fps = float(target_fps)
        self.resize = resize
        self._capture: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            raise RuntimeError("Call open() before requesting metadata")
        return self._metadata

    def open(self) -> VideoMetadata:
        if not self.path.is_file():
            raise FileNotFoundError(f"Video does not exist: {self.path}")

        self.close()
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"OpenCV could not open video: {self.path}")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(source_fps) or source_fps <= 0:
            capture.release()
            raise ValueError(f"Video has invalid FPS ({source_fps}): {self.path}")
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError(f"Video has invalid dimensions ({width}x{height}): {self.path}")

        duration_ms = (frame_count / source_fps * 1000.0) if frame_count > 0 else 0.0
        self._capture = capture
        self._metadata = VideoMetadata(
            source_path=str(self.path),
            frame_count=frame_count,
            source_fps=source_fps,
            width=width,
            height=height,
            duration_ms=duration_ms,
        )
        return self._metadata

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._metadata = None

    def iter_frames(self) -> Iterator[SampledFrame]:
        """Yield sampled, resized BGR frames in timestamp order.

        Timestamp-based selection avoids a rounding drift that can occur when
        the source FPS is not an exact multiple of the target FPS.
        """

        if self._capture is None or self._metadata is None:
            self.open()
        assert self._capture is not None
        assert self._metadata is not None

        interval_ms = 1000.0 / self.target_fps
        next_sample_ms = 0.0
        source_index = 0
        sampled_index = 0
        last_timestamp_ms = -1.0

        while True:
            ok, frame = self._capture.read()
            if not ok:
                break

            timestamp_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
            if not math.isfinite(timestamp_ms) or timestamp_ms <= last_timestamp_ms:
                timestamp_ms = source_index / self._metadata.source_fps * 1000.0
            last_timestamp_ms = timestamp_ms

            if timestamp_ms + 0.01 < next_sample_ms:
                source_index += 1
                continue

            if self.resize is not None:
                frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_AREA)

            yield SampledFrame(
                source_frame_index=source_index,
                sampled_frame_index=sampled_index,
                timestamp_ms=timestamp_ms,
                frame_bgr=frame,
            )
            sampled_index += 1
            next_sample_ms += interval_ms
            while next_sample_ms <= timestamp_ms:
                next_sample_ms += interval_ms
            source_index += 1


def infer_label(video_path: Path, raw_root: Path) -> str:
    """Infer the label from the first directory below ``raw_root`` when known."""

    relative = video_path.relative_to(raw_root)
    first_directory = relative.parts[0].lower() if len(relative.parts) > 1 else ""
    return {"normal": "normal", "shoplifting": "shoplifting"}.get(first_directory, "unknown")


def discover_videos(root: str | Path, extensions: Sequence[str] = SUPPORTED_EXTENSIONS) -> list[Path]:
    root_path = Path(root)
    normalized_extensions = {extension.lower() for extension in extensions}
    return sorted(path for path in root_path.rglob("*") if path.is_file() and path.suffix.lower() in normalized_extensions)


def build_manifest(
    raw_root: str | Path,
    manifest_path: str | Path,
    target_fps: float = 15.0,
) -> list[dict[str, object]]:
    """Inspect every supported video and write a reproducible clip manifest."""

    root = Path(raw_root)
    rows: list[dict[str, object]] = []
    for video_path in discover_videos(root):
        with VideoReader(video_path, target_fps=target_fps, resize=None) as reader:
            row = asdict(reader.metadata)
        row["clip_id"] = video_path.stem
        row["label"] = infer_label(video_path, root)
        row["target_fps"] = target_fps
        rows.append(row)

    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "label",
        "source_path",
        "frame_count",
        "source_fps",
        "width",
        "height",
        "duration_ms",
        "target_fps",
    ]
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Inspect videos and build a clip manifest.")
    parser.add_argument("--raw-root", default="data", help="Directory containing raw video folders.")
    parser.add_argument("--manifest", default="data/manifests/clips.csv", help="CSV path to create.")
    parser.add_argument("--target-fps", type=float, default=15.0, help="Sampling FPS for downstream processing.")
    args = parser.parse_args()

    rows = build_manifest(args.raw_root, args.manifest, args.target_fps)
    labels = {label: sum(row["label"] == label for row in rows) for label in sorted({str(row["label"]) for row in rows})}
    print(f"Wrote {len(rows)} clips to {args.manifest}: {labels}")


if __name__ == "__main__":
    _cli()
