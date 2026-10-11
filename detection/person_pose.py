"""YOLOv8n-Pose + ByteTrack wrapper for per-frame people and keypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from detection.tracker import TrackerSettings


# COCO-17 limb connections used only for debugging overlays.
COCO_SKELETON = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


@dataclass(frozen=True)
class PersonPose:
    """A tracked person and its COCO-17 keypoints for one video frame."""

    track_id: int | None
    bbox_xyxy: tuple[float, float, float, float]
    detection_confidence: float
    keypoints_xyc: tuple[tuple[float, float, float], ...]
    track_age: int


class PersonPoseTracker:
    """Run YOLOv8n-Pose and associate detections with ByteTrack IDs.

    The YOLO model is imported lazily so data/manifest operations do not require
    PyTorch or downloaded weights. Ultralytics downloads ``yolov8n-pose.pt`` on
    its first use when it is not already cached.
    """

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        tracker: TrackerSettings | None = None,
        image_size: int = 640,
        device: str = "cpu",
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be greater than zero")
        self.model_path = model_path
        self.tracker = tracker or TrackerSettings()
        self.image_size = image_size
        self.device = device
        self._model: Any | None = None
        self._track_ages: dict[int, int] = {}

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError("Install dependencies with: python -m pip install -r requirements.txt") from error
            self._model = YOLO(self.model_path)
        return self._model

    def reset(self) -> None:
        """Clear ByteTrack state before processing a new, independent clip."""

        self._track_ages.clear()
        if self._model is None or not hasattr(self._model, "predictor"):
            return
        for tracker in getattr(self._model.predictor, "trackers", []):
            if hasattr(tracker, "reset"):
                tracker.reset()

    def track_frame(self, frame_bgr: np.ndarray) -> list[PersonPose]:
        """Return only COCO-person detections with ByteTrack IDs for one frame."""

        model = self._load_model()
        results = model.track(
            source=frame_bgr,
            persist=True,
            tracker=self.tracker.config,
            classes=[0],  # COCO person class only
            conf=self.tracker.confidence,
            iou=self.tracker.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [None] * len(boxes)
        keypoints = result.keypoints.data.cpu().numpy()

        people: list[PersonPose] = []
        for bbox, confidence, track_id, points in zip(boxes, confidences, ids, keypoints, strict=True):
            track_id_int = None if track_id is None else int(track_id)
            if track_id_int is None:
                age = 1
            else:
                age = self._track_ages.get(track_id_int, 0) + 1
                self._track_ages[track_id_int] = age
            people.append(
                PersonPose(
                    track_id=track_id_int,
                    bbox_xyxy=tuple(float(value) for value in bbox),
                    detection_confidence=float(confidence),
                    keypoints_xyc=tuple(tuple(float(value) for value in point) for point in points),
                    track_age=age,
                )
            )
        return people


def draw_people(frame_bgr: np.ndarray, people: list[PersonPose]) -> np.ndarray:
    """Draw lightweight boxes, IDs and skeletons for a debug video."""

    image = frame_bgr.copy()
    for person in people:
        x1, y1, x2, y2 = (round(value) for value in person.bbox_xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"ID {person.track_id}" if person.track_id is not None else "ID pending"
        cv2.putText(image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
        for start, end in COCO_SKELETON:
            first, second = person.keypoints_xyc[start], person.keypoints_xyc[end]
            if first[2] >= 0.25 and second[2] >= 0.25:
                cv2.line(image, (round(first[0]), round(first[1])), (round(second[0]), round(second[1])), (255, 170, 0), 2)
        for x, y, confidence in person.keypoints_xyc:
            if confidence >= 0.25:
                cv2.circle(image, (round(x), round(y)), 3, (0, 80, 255), -1)
    return image
