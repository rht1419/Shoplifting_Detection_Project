import numpy as np
import pytest

from detection.person_pose import PersonPose, PersonPoseTracker, draw_people
from detection.tracker import TrackerSettings


class _FakeArray:
    """Mimics the torch-tensor `.cpu().numpy()` chain ultralytics results expose."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> "_FakeArray":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, ids: np.ndarray | None) -> None:
        self.xyxy = _FakeArray(xyxy)
        self.conf = _FakeArray(conf)
        self.id = None if ids is None else _FakeArray(ids)

    def __len__(self) -> int:
        return len(self.conf.numpy())


class _FakeKeypoints:
    def __init__(self, data: np.ndarray) -> None:
        self.data = _FakeArray(data)


class _FakeResult:
    def __init__(self, ids: np.ndarray | None) -> None:
        count = 0 if ids is None else len(ids)
        self.boxes = _FakeBoxes(
            xyxy=np.tile(np.array([0.0, 0.0, 10.0, 10.0]), (count, 1)),
            conf=np.full(count, 0.9),
            ids=ids,
        )
        self.keypoints = _FakeKeypoints(np.zeros((count, 17, 3)))


class _FakeModel:
    def __init__(self, ids_per_call: list[np.ndarray | None]) -> None:
        self._ids_per_call = iter(ids_per_call)

    def track(self, **_: object) -> list[_FakeResult]:
        return [_FakeResult(next(self._ids_per_call))]


def test_tracker_settings_validate_thresholds() -> None:
    with pytest.raises(ValueError):
        TrackerSettings(confidence=0)


def test_track_age_increments_while_id_persists_and_resets() -> None:
    tracker = PersonPoseTracker()
    tracker._model = _FakeModel(
        [
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0, 2.0]),
        ]
    )
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    first = tracker.track_frame(frame)
    second = tracker.track_frame(frame)
    third = tracker.track_frame(frame)

    assert [person.track_age for person in first] == [1]
    assert [person.track_age for person in second] == [2]
    assert {person.track_id: person.track_age for person in third} == {1: 3, 2: 1}

    tracker.reset()
    tracker._model = _FakeModel([np.array([1.0])])
    after_reset = tracker.track_frame(frame)
    assert [person.track_age for person in after_reset] == [1]


def test_draw_people_returns_debug_frame() -> None:
    person = PersonPose(
        track_id=5,
        bbox_xyxy=(10, 10, 60, 80),
        detection_confidence=0.9,
        keypoints_xyc=tuple((20.0, 20.0, 0.9) for _ in range(17)),
        track_age=3,
    )
    output = draw_people(np.zeros((100, 100, 3), dtype=np.uint8), [person])
    assert output.shape == (100, 100, 3)
    assert output.any()
