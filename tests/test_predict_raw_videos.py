from pathlib import Path

import pytest

from scripts.predict_raw_videos import discover_examples, to_video_example


def touch_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_discover_celebdf_examples_uses_directory_labels(tmp_path: Path) -> None:
    touch_video(tmp_path / "Celeb-real" / "id0_0000.mp4")
    touch_video(tmp_path / "YouTube-real" / "00001.mp4")
    touch_video(tmp_path / "Celeb-synthesis" / "id0_id1_0000.mp4")
    (tmp_path / "Celeb-real" / "notes.txt").write_text("skip", encoding="utf-8")

    rows = discover_examples(tmp_path)

    assert [str(row["relative_path"]) for row in rows] == [
        "Celeb-real/id0_0000.mp4",
        "YouTube-real/00001.mp4",
        "Celeb-synthesis/id0_id1_0000.mp4",
    ]
    assert [row["class_name"] for row in rows] == ["real", "real", "fake"]
    assert [row["true_label"] for row in rows] == [0, 0, 1]
    assert rows[-1]["generator_id"] == "Celeb-synthesis"

    example = to_video_example(rows[-1])

    assert example.source_id_kind == "celebdf"
    assert example.generator_id == "Celeb-synthesis"


def test_discover_ffpp_examples_sets_source_id_kind(tmp_path: Path) -> None:
    touch_video(tmp_path / "original" / "000.mp4")
    touch_video(tmp_path / "Deepfakes" / "000_001.mp4")

    rows = discover_examples(tmp_path)

    assert [row["source_id_kind"] for row in rows] == ["ffpp", "ffpp"]
    assert [row["class_name"] for row in rows] == ["real", "fake"]
    assert rows[-1]["generator_id"] == "Deepfakes"

    example = to_video_example(rows[-1])

    assert example.source_id_kind == "ffpp"


def test_discover_examples_rejects_unknown_layout(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Unsupported raw prediction dataset layout"):
        discover_examples(tmp_path)
