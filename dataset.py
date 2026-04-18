from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class VideoExample:
    path: Path
    label: int
    class_name: str
    source_id: str
    split: str
    identity_id: str | None = None


def infer_real_source_id(path: Path) -> str:
    return path.name.split("_", 1)[0]


def infer_fake_source_id(path: Path) -> str:
    stem = path.stem
    if "_clip_" in stem:
        return stem.split("_clip_", 1)[0]
    if stem.endswith("_swapped"):
        return stem[: -len("_swapped")]
    return stem


def discover_real_fake_video_paths(
    real_dir: str | Path,
    fake_dir: str | Path,
) -> tuple[list[Path], list[Path]]:
    real_root = Path(real_dir)
    fake_root = Path(fake_dir)
    real_paths = sorted(path for path in real_root.glob("*.mp4") if path.is_file())
    fake_paths = sorted(path for path in fake_root.glob("*/*.mp4") if path.is_file())
    if not real_paths:
        raise FileNotFoundError(f"No real videos found in {real_root}")
    if not fake_paths:
        raise FileNotFoundError(f"No fake videos found in {fake_root}")
    return real_paths, fake_paths


def _split_groups(
    source_ids: Iterable[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    unique_ids = sorted(set(source_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)

    total = len(unique_ids)
    train_cut = int(total * train_ratio)
    val_cut = train_cut + int(total * val_ratio)
    if total >= 3:
        train_cut = max(1, min(train_cut, total - 2))
        val_cut = max(train_cut + 1, min(val_cut, total - 1))
    elif total == 2:
        train_cut = 1
        val_cut = 1
    else:
        train_cut = 1
        val_cut = 1

    split_by_source: dict[str, str] = {}
    for index, source_id in enumerate(unique_ids):
        if index < train_cut:
            split_by_source[source_id] = "train"
        elif index < val_cut:
            split_by_source[source_id] = "val"
        else:
            split_by_source[source_id] = "test"
    return split_by_source


def build_real_fake_examples(
    real_dir: str | Path,
    fake_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> list[VideoExample]:
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("`train_ratio` must be in (0.0, 1.0).")
    if val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("`val_ratio` must be in (0.0, 1.0) and leave room for test split.")

    real_paths, fake_paths = discover_real_fake_video_paths(real_dir=real_dir, fake_dir=fake_dir)
    real_split_by_source = _split_groups(
        (infer_real_source_id(path) for path in real_paths),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    fake_split_by_source = _split_groups(
        (infer_fake_source_id(path) for path in fake_paths),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    examples: list[VideoExample] = []
    for path in real_paths:
        source_id = infer_real_source_id(path)
        examples.append(
            VideoExample(
                path=path,
                label=0,
                class_name="real",
                source_id=source_id,
                split=real_split_by_source[source_id],
            )
        )
    for path in fake_paths:
        source_id = infer_fake_source_id(path)
        examples.append(
            VideoExample(
                path=path,
                label=1,
                class_name="fake",
                source_id=source_id,
                split=fake_split_by_source[source_id],
                identity_id=path.parent.name,
            )
        )
    return examples


def summarize_examples(examples: Sequence[VideoExample]) -> dict[str, dict[str, int]]:
    summary = {split: {"real": 0, "fake": 0, "total": 0} for split in VALID_SPLITS}
    for example in examples:
        split_summary = summary[example.split]
        split_summary[example.class_name] += 1
        split_summary["total"] += 1
    return summary


def write_dataset_manifest(
    examples: Sequence[VideoExample],
    output_path: str | Path,
) -> Path:
    manifest_path = Path(output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "label", "class_name", "source_id", "split", "identity_id"),
        )
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "path": str(example.path),
                    "label": example.label,
                    "class_name": example.class_name,
                    "source_id": example.source_id,
                    "split": example.split,
                    "identity_id": example.identity_id or "",
                }
            )
    return manifest_path


def load_dataset_manifest(
    manifest_path: str | Path,
    split: str | None = None,
) -> list[VideoExample]:
    if split is not None and split not in VALID_SPLITS:
        raise ValueError(f"`split` must be one of {VALID_SPLITS}, got {split!r}")

    examples: list[VideoExample] = []
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_split = row["split"]
            if split is not None and row_split != split:
                continue
            examples.append(
                VideoExample(
                    path=Path(row["path"]),
                    label=int(row["label"]),
                    class_name=row["class_name"],
                    source_id=row["source_id"],
                    split=row_split,
                    identity_id=row["identity_id"] or None,
                )
            )
    return examples


def load_video_clip(
    path: str | Path,
    num_frames: int,
    image_size: int = 224,
) -> dict[str, Any]:
    clip_path = Path(path)
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {clip_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        cap.release()
        raise RuntimeError(f"Video has only {total_frames} frames, need at least {num_frames}")

    indices = torch.linspace(0, total_frames - 1, steps=num_frames).round().to(dtype=torch.int64).tolist()
    clip_frames: list[torch.Tensor] = []
    rgb_frames: list[np.ndarray] = []

    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to read frame {frame_index} from {clip_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frames.append(frame)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frame_tensor = (frame_tensor - IMAGENET_MEAN) / IMAGENET_STD
        clip_frames.append(frame_tensor)

    cap.release()
    return {
        "video": torch.stack(clip_frames, dim=1),
        "video_rgb_frames": rgb_frames,
    }


class LabeledVideoDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        examples: Sequence[VideoExample],
        num_frames: int,
        image_size: int = 224,
    ) -> None:
        self.examples = list(examples)
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        clip = load_video_clip(
            path=example.path,
            num_frames=self.num_frames,
            image_size=self.image_size,
        )
        return {
            "video": clip["video"],
            "video_rgb_frames": clip["video_rgb_frames"],
            "label": torch.tensor([float(example.label)], dtype=torch.float32),
            "path": str(example.path),
            "source_id": example.source_id,
            "split": example.split,
            "class_name": example.class_name,
            "identity_id": example.identity_id,
        }


def collate_labeled_video_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch.")
    return {
        "video": torch.stack([item["video"] for item in items], dim=0),
        "video_rgb_frames": [item["video_rgb_frames"] for item in items],
        "label": torch.stack([item["label"] for item in items], dim=0),
        "path": [item["path"] for item in items],
        "source_id": [item["source_id"] for item in items],
        "split": [item["split"] for item in items],
        "class_name": [item["class_name"] for item in items],
        "identity_id": [item["identity_id"] for item in items],
    }
