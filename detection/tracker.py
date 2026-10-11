"""ByteTrack configuration used by the Ultralytics tracking runtime."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackerSettings:
    """Settings passed to Ultralytics' built-in ByteTrack tracker."""

    config: str = "bytetrack.yaml"
    confidence: float = 0.35
    iou_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
