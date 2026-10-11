from pathlib import Path

from pipeline.video_reader import discover_videos, infer_label


def test_infer_label_from_dataset_folder() -> None:
    root = Path("data")
    assert infer_label(root / "normal" / "normal-1.mp4", root) == "normal"
    assert infer_label(root / "shoplifting" / "shoplifting-1.mp4", root) == "shoplifting"


def test_discover_videos_finds_dataset_clips(tmp_path: Path) -> None:
    normal_dir = tmp_path / "normal"
    normal_dir.mkdir(parents=True)
    (normal_dir / "sample1.mp4").write_text("dummy")

    shop_dir = tmp_path / "shoplifting"
    shop_dir.mkdir(parents=True)
    (shop_dir / "sample2.mp4").write_text("dummy")

    videos = discover_videos(tmp_path)
    assert any(path.parent.name == "normal" for path in videos)
    assert any(path.parent.name == "shoplifting" for path in videos)

