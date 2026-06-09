from __future__ import annotations

import csv
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from face_crop_config import (
    FACE_CROP_STATUS_DETECTED,
    FACE_CROP_STATUS_DISABLED,
    FACE_CROP_STATUS_FALLBACK,
    RPPG_CACHE_VARIANT,
    face_crop_cache_variant,
    merged_face_crop_config,
)
from frame_sampling import (
    frame_sampling_cache_variant,
    resolve_frame_sampling_config,
    sample_frame_indices,
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".webm")
VIDEO_DECODE_MODES: tuple[str, ...] = ("seek", "scan")
VIDEO_CLIP_CACHE_VERSION = 1
RPPG_CLIP_CACHE_VERSION = 1
RPPG_DEFAULT_FPS = 30.0
RPPG_FACE_CROP_STATUS_DETECTED = FACE_CROP_STATUS_DETECTED
RPPG_FACE_CROP_STATUS_FALLBACK = FACE_CROP_STATUS_FALLBACK
RPPG_FACE_CROP_STATUS_DISABLED = FACE_CROP_STATUS_DISABLED
RPPG_HAAR_CASCADE = "haarcascade_frontalface_default.xml"


@dataclass(frozen=True)
class VideoExample:
    path: Path
    label: int
    class_name: str
    source_id: str
    split: str
    metadata_filename: str | None = None
    identity_id: str | None = None
    generator_id: str | None = None
    source_id_kind: str | None = None
    age_bin: str | None = None
    gender: str | None = None
    ethnicity: str | None = None
    emotion: str | None = None


METADATA_COLUMNS: tuple[str, ...] = (
    "filename",
    "age",
    "gender",
    "ethnicity",
    "emotion",
    "aus_summary",
)
AUDIT_MANIFEST_COLUMNS: tuple[str, ...] = (
    "generator_id",
    "source_id_kind",
    "age_bin",
    "gender",
    "ethnicity",
    "emotion",
)
SPLIT_AUDIT_FIELDS: tuple[str, ...] = ("gender", "ethnicity", "emotion", "age_bin")
SOURCE_VIDEO_TIME_RE = re.compile(r"^(?P<source>.+?)_\d{2}_\d{2}_\d{1,2}-\d{2}_\d{2}_\d{1,2}")


@dataclass(frozen=True)
class MetadataRow:
    filename: str
    age: str
    gender: str
    ethnicity: str
    emotion: str
    aus_summary: str


@dataclass(frozen=True)
class _SplitGroup:
    group_id: str
    examples: tuple[VideoExample, ...]
    counts: Counter[str]
    generator_counts: Counter[str]
    audit_counts: dict[str, Counter[str]]


def infer_real_source_id(path: Path) -> str:
    return path.name.split("_", 1)[0]


def infer_fake_source_id(path: Path) -> str:
    stem = path.stem
    if "_clip_" in stem:
        return stem.split("_clip_", 1)[0]
    if stem.endswith("_swapped"):
        return stem[: -len("_swapped")]
    return stem


def strip_generated_suffix(stem: str) -> str:
    for suffix in ("_swapped", "_background_augmented"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def source_payload_from_stem(stem: str) -> str:
    stripped = strip_generated_suffix(stem)
    if "____" in stripped:
        return stripped.rsplit("____", 1)[1]
    return stripped


def infer_metadata_source_id(filename: str, class_name: str) -> tuple[str, str]:
    relative_path = Path(filename)
    stem = relative_path.stem
    payload = source_payload_from_stem(stem)
    match = SOURCE_VIDEO_TIME_RE.match(payload)
    if match is not None:
        return f"source_video:{match.group('source')}", "source_video"
    if "_clip_" in payload:
        return f"clip:{payload.split('_clip_', 1)[0]}", "clip"
    if class_name == "fake":
        return f"synthetic_file:{filename}", "synthetic_file"
    return f"source_video:{payload}", "source_video"


def infer_generator_id(filename: str, class_name: str) -> str:
    if class_name == "real":
        return "real"
    parts = Path(filename).parts
    if len(parts) < 2:
        raise ValueError(f"Fake metadata filename must include generator folder: {filename!r}")
    return parts[0]


def age_to_bin(value: str) -> str:
    try:
        age = float(value)
    except ValueError:
        return "unknown"
    if not math.isfinite(age) or age < 0.0:
        return "unknown"
    decade = int(age // 10) * 10
    return f"{decade}s"


def load_video_metadata(meta_path: str | Path) -> list[MetadataRow]:
    path = Path(meta_path)
    rows: list[MetadataRow] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in METADATA_COLUMNS if column not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"Metadata file {path} is missing columns: {', '.join(missing)}")
        for row in reader:
            filename = str(row["filename"]).strip()
            if not filename:
                raise ValueError(f"Metadata file {path} contains an empty filename.")
            if filename in seen:
                raise ValueError(f"Metadata file {path} contains duplicate filename: {filename}")
            seen.add(filename)
            rows.append(
                MetadataRow(
                    filename=filename,
                    age=str(row["age"]).strip(),
                    gender=str(row["gender"]).strip(),
                    ethnicity=str(row["ethnicity"]).strip(),
                    emotion=str(row["emotion"]).strip(),
                    aus_summary=str(row["aus_summary"]),
                )
            )
    return rows


def _metadata_video_paths(root: Path, class_name: str) -> set[str]:
    if class_name == "real":
        return {
            entry.name
            for entry in os.scandir(root)
            if entry.is_file() and Path(entry.name).suffix.lower() in VIDEO_EXTENSIONS
        }
    paths: set[str] = set()
    for directory in os.scandir(root):
        if not directory.is_dir():
            continue
        for entry in os.scandir(directory.path):
            if entry.is_file() and Path(entry.name).suffix.lower() in VIDEO_EXTENSIONS:
                paths.add(f"{directory.name}/{entry.name}")
    return paths


def validate_metadata_file_coverage(
    root: str | Path,
    rows: Sequence[MetadataRow],
    class_name: str,
) -> None:
    root_path = Path(root)
    metadata_files = {row.filename for row in rows}
    video_files = _metadata_video_paths(root_path, class_name)
    missing_metadata = sorted(video_files - metadata_files)
    missing_files = sorted(metadata_files - video_files)
    if missing_metadata or missing_files:
        details: list[str] = []
        if missing_metadata:
            details.append(
                f"{len(missing_metadata)} videos missing metadata, first={missing_metadata[0]}"
            )
        if missing_files:
            details.append(
                f"{len(missing_files)} metadata rows missing files, first={missing_files[0]}"
            )
        raise FileNotFoundError(f"Metadata coverage mismatch in {root_path}: {'; '.join(details)}")


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


def _discover_flat_video_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def build_labeled_folder_examples(
    root_dir: str | Path,
    split: str,
) -> list[VideoExample]:
    if split not in VALID_SPLITS:
        raise ValueError(f"`split` must be one of {VALID_SPLITS}, got {split!r}")

    root = Path(root_dir)
    real_dir = root / "real"
    fake_dir = root / "fake"
    if not real_dir.is_dir():
        raise FileNotFoundError(f"Missing real video folder: {real_dir}")
    if not fake_dir.is_dir():
        raise FileNotFoundError(f"Missing fake video folder: {fake_dir}")

    real_paths = _discover_flat_video_paths(real_dir)
    fake_paths = _discover_flat_video_paths(fake_dir)
    if not real_paths:
        raise FileNotFoundError(f"No supported real videos found in {real_dir}")
    if not fake_paths:
        raise FileNotFoundError(f"No supported fake videos found in {fake_dir}")

    examples: list[VideoExample] = []
    for path in real_paths:
        examples.append(
            VideoExample(
                path=path,
                label=0,
                class_name="real",
                source_id=path.stem,
                split=split,
                metadata_filename=path.name,
            )
        )
    for path in fake_paths:
        examples.append(
            VideoExample(
                path=path,
                label=1,
                class_name="fake",
                source_id=path.stem,
                split=split,
                metadata_filename=path.name,
            )
        )
    return examples


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


def _build_metadata_example(
    root: Path,
    row: MetadataRow,
    class_name: str,
) -> VideoExample:
    label = 0 if class_name == "real" else 1
    source_id, source_id_kind = infer_metadata_source_id(row.filename, class_name)
    generator_id = infer_generator_id(row.filename, class_name)
    return VideoExample(
        path=root / row.filename,
        label=label,
        class_name=class_name,
        source_id=source_id,
        split="train",
        metadata_filename=row.filename,
        identity_id=None if class_name == "real" else generator_id,
        generator_id=generator_id,
        source_id_kind=source_id_kind,
        age_bin=age_to_bin(row.age),
        gender=row.gender or None,
        ethnicity=row.ethnicity or None,
        emotion=row.emotion or None,
    )


def build_metadata_real_fake_examples(
    real_dir: str | Path,
    fake_dir: str | Path,
) -> list[VideoExample]:
    real_root = Path(real_dir)
    fake_root = Path(fake_dir)
    real_rows = load_video_metadata(real_root / "meta.csv")
    fake_rows = load_video_metadata(fake_root / "meta.csv")
    validate_metadata_file_coverage(real_root, real_rows, "real")
    validate_metadata_file_coverage(fake_root, fake_rows, "fake")
    examples = [_build_metadata_example(real_root, row, "real") for row in real_rows]
    examples.extend(_build_metadata_example(fake_root, row, "fake") for row in fake_rows)
    return examples


def _replace_example_split(example: VideoExample, split: str) -> VideoExample:
    return VideoExample(
        path=example.path,
        label=example.label,
        class_name=example.class_name,
        source_id=example.source_id,
        split=split,
        metadata_filename=example.metadata_filename,
        identity_id=example.identity_id,
        generator_id=example.generator_id,
        source_id_kind=example.source_id_kind,
        age_bin=example.age_bin,
        gender=example.gender,
        ethnicity=example.ethnicity,
        emotion=example.emotion,
    )


def _audit_value(example: VideoExample, field: str) -> str:
    value = getattr(example, field)
    if value is None or value == "":
        return "unknown"
    return str(value)


def _build_split_groups(examples: Sequence[VideoExample]) -> list[_SplitGroup]:
    by_source: dict[str, list[VideoExample]] = defaultdict(list)
    for example in examples:
        by_source[example.source_id].append(example)

    groups: list[_SplitGroup] = []
    for source_id, group_examples in sorted(by_source.items()):
        counts = Counter(example.class_name for example in group_examples)
        generator_counts = Counter(
            example.generator_id or example.identity_id or "unknown"
            for example in group_examples
            if example.class_name == "fake"
        )
        audit_counts = {
            field: Counter(_audit_value(example, field) for example in group_examples)
            for field in SPLIT_AUDIT_FIELDS
        }
        groups.append(
            _SplitGroup(
                group_id=source_id,
                examples=tuple(group_examples),
                counts=counts,
                generator_counts=generator_counts,
                audit_counts=audit_counts,
            )
        )
    return groups


def _hamilton_quotas(capacities: Mapping[str, int], total: int) -> dict[str, int]:
    positive = {key: value for key, value in capacities.items() if value > 0}
    if total <= 0 or not positive:
        return dict.fromkeys(capacities, 0)
    weights = {key: math.sqrt(value) for key, value in positive.items()}
    weight_total = sum(weights.values())
    raw = {key: (weights[key] / weight_total) * total for key in positive}
    quotas = {key: min(math.floor(value), positive[key]) for key, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(
        positive,
        key=lambda key: (raw[key] - math.floor(raw[key]), positive[key], key),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for key in order:
            if quotas[key] >= positive[key]:
                continue
            quotas[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return {key: quotas.get(key, 0) for key in capacities}


def _fake_generator_eval_quotas(
    examples: Sequence[VideoExample],
    fake_target_per_split: int,
) -> dict[str, int]:
    capacities = Counter(
        example.generator_id or example.identity_id or "unknown"
        for example in examples
        if example.class_name == "fake"
    )
    per_split_capacities = {
        generator: count // 3 if count >= 3 else count // 2
        for generator, count in capacities.items()
    }
    quotas = _hamilton_quotas(per_split_capacities, fake_target_per_split)
    if sum(quotas.values()) < fake_target_per_split:
        raise ValueError(
            "Cannot allocate fake generator quotas for eval splits: "
            f"target={fake_target_per_split} available={sum(per_split_capacities.values())}"
        )
    return quotas


def _counter_add(left: Mapping[str, int], right: Mapping[str, int]) -> Counter[str]:
    result = Counter(left)
    result.update(right)
    return result


def _distribution_l1(counts: Mapping[str, int], target: Mapping[str, float]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    keys = set(counts) | set(target)
    return sum(abs((counts.get(key, 0) / total) - target.get(key, 0.0)) for key in keys)


def _target_distributions(examples: Sequence[VideoExample]) -> dict[str, dict[str, float]]:
    targets: dict[str, dict[str, float]] = {}
    for field in SPLIT_AUDIT_FIELDS:
        counts = Counter(_audit_value(example, field) for example in examples)
        total = sum(counts.values())
        targets[field] = {key: value / total for key, value in counts.items()} if total > 0 else {}
    return targets


def _metadata_drift_score(
    split_audit_counts: Mapping[str, Counter[str]],
    group: _SplitGroup,
    targets: Mapping[str, Mapping[str, float]],
) -> float:
    score = 0.0
    for field in SPLIT_AUDIT_FIELDS:
        combined = _counter_add(split_audit_counts.get(field, Counter()), group.audit_counts[field])
        score += _distribution_l1(combined, targets.get(field, {}))
    return score


def _split_targets_met(
    counts: Mapping[str, int],
    generator_counts: Mapping[str, int],
    real_target: int,
    fake_generator_targets: Mapping[str, int],
) -> bool:
    if counts.get("real", 0) < real_target:
        return False
    for generator, target in fake_generator_targets.items():
        if generator_counts.get(generator, 0) < target:
            return False
    return True


def _group_deficit_gain(
    group: _SplitGroup,
    counts: Mapping[str, int],
    generator_counts: Mapping[str, int],
    real_target: int,
    fake_generator_targets: Mapping[str, int],
) -> int:
    real_gain = min(group.counts.get("real", 0), max(0, real_target - counts.get("real", 0)))
    fake_gain = 0
    for generator, target in fake_generator_targets.items():
        fake_gain += min(
            group.generator_counts.get(generator, 0),
            max(0, target - generator_counts.get(generator, 0)),
        )
    return real_gain + fake_gain


def _group_overshoot(
    group: _SplitGroup,
    counts: Mapping[str, int],
    generator_counts: Mapping[str, int],
    real_target: int,
    fake_generator_targets: Mapping[str, int],
) -> int:
    real_deficit = max(0, real_target - counts.get("real", 0))
    overshoot = max(0, group.counts.get("real", 0) - real_deficit)
    for generator, value in group.generator_counts.items():
        target = fake_generator_targets.get(generator, 0)
        deficit = max(0, target - generator_counts.get(generator, 0))
        overshoot += max(0, value - deficit)
    return overshoot


def _allocate_eval_split(
    split: str,
    remaining: dict[str, _SplitGroup],
    counts: Counter[str],
    generator_counts: Counter[str],
    audit_counts: dict[str, Counter[str]],
    real_target: int,
    fake_generator_targets: Mapping[str, int],
    target_distributions: Mapping[str, Mapping[str, float]],
    ranks: Mapping[str, float],
) -> list[_SplitGroup]:
    selected: list[_SplitGroup] = []
    while not _split_targets_met(counts, generator_counts, real_target, fake_generator_targets):
        primary_candidates = []
        for group in remaining.values():
            gain = _group_deficit_gain(
                group,
                counts,
                generator_counts,
                real_target,
                fake_generator_targets,
            )
            if gain <= 0:
                continue
            overshoot = _group_overshoot(
                group,
                counts,
                generator_counts,
                real_target,
                fake_generator_targets,
            )
            size = len(group.examples)
            primary_candidates.append(
                (-gain, overshoot, size, ranks[group.group_id], group.group_id)
            )
        if not primary_candidates:
            raise ValueError(
                f"Cannot satisfy {split} split targets: real={counts.get('real', 0)}/"
                f"{real_target} fake_generators={dict(generator_counts)} targets="
                f"{dict(fake_generator_targets)}"
            )
        shortlisted = sorted(primary_candidates)[:32]
        candidates = []
        for gain, overshoot, size, rank, group_id in shortlisted:
            group = remaining[group_id]
            drift = _metadata_drift_score(audit_counts, group, target_distributions)
            candidates.append((gain, overshoot, drift, size, rank, group_id))
        _, _, _, _, _, group_id = min(candidates)
        group = remaining.pop(group_id)
        selected.append(group)
        counts.update(group.counts)
        generator_counts.update(group.generator_counts)
        for field in SPLIT_AUDIT_FIELDS:
            audit_counts[field].update(group.audit_counts[field])
    return selected


def split_metadata_examples(
    examples: Sequence[VideoExample],
    eval_real_count: int = 500,
    eval_fake_count: int = 500,
    seed: int = 0,
) -> list[VideoExample]:
    if eval_real_count <= 0 or eval_fake_count <= 0:
        raise ValueError("Eval split class counts must be positive.")
    total_counts = Counter(example.class_name for example in examples)
    if total_counts["real"] < eval_real_count * 2:
        raise ValueError(
            f"Not enough real videos for val/test eval pools: have={total_counts['real']} "
            f"need={eval_real_count * 2}"
        )
    if total_counts["fake"] < eval_fake_count * 2:
        raise ValueError(
            f"Not enough fake videos for val/test eval pools: have={total_counts['fake']} "
            f"need={eval_fake_count * 2}"
        )

    fake_generator_targets = _fake_generator_eval_quotas(examples, eval_fake_count)
    groups = _build_split_groups(examples)
    rng = random.Random(seed)
    ranks = {group.group_id: rng.random() for group in groups}
    remaining = {group.group_id: group for group in groups}
    target_distributions = _target_distributions(examples)

    split_groups: dict[str, list[_SplitGroup]] = {"test": [], "val": [], "train": []}
    split_counts: dict[str, Counter[str]] = {split: Counter() for split in VALID_SPLITS}
    split_generator_counts: dict[str, Counter[str]] = {split: Counter() for split in VALID_SPLITS}
    split_audit_counts: dict[str, dict[str, Counter[str]]] = {
        split: {field: Counter() for field in SPLIT_AUDIT_FIELDS} for split in VALID_SPLITS
    }

    for split in ("test", "val"):
        split_groups[split] = _allocate_eval_split(
            split=split,
            remaining=remaining,
            counts=split_counts[split],
            generator_counts=split_generator_counts[split],
            audit_counts=split_audit_counts[split],
            real_target=eval_real_count,
            fake_generator_targets=fake_generator_targets,
            target_distributions=target_distributions,
            ranks=ranks,
        )

    split_groups["train"] = [remaining[key] for key in sorted(remaining)]

    split_by_group: dict[str, str] = {}
    for split, groups_for_split in split_groups.items():
        for group in groups_for_split:
            split_by_group[group.group_id] = split

    return [
        _replace_example_split(example, split_by_group[example.source_id])
        for group in groups
        for example in group.examples
    ]


def _build_legacy_real_fake_examples(
    real_dir: str | Path,
    fake_dir: str | Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> list[VideoExample]:
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
                metadata_filename=path.name,
                generator_id="real",
                source_id_kind="source_video",
            )
        )
    for path in fake_paths:
        source_id = infer_fake_source_id(path)
        generator_id = path.parent.name
        examples.append(
            VideoExample(
                path=path,
                label=1,
                class_name="fake",
                source_id=source_id,
                split=fake_split_by_source[source_id],
                metadata_filename=f"{generator_id}/{path.name}",
                identity_id=generator_id,
                generator_id=generator_id,
                source_id_kind="source_video",
            )
        )
    return examples


def build_real_fake_examples(
    real_dir: str | Path,
    fake_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 0,
    eval_real_count: int = 500,
    eval_fake_count: int = 500,
) -> list[VideoExample]:
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("`train_ratio` must be in (0.0, 1.0).")
    if val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("`val_ratio` must be in (0.0, 1.0) and leave room for test split.")

    real_root = Path(real_dir)
    fake_root = Path(fake_dir)
    if (real_root / "meta.csv").is_file() and (fake_root / "meta.csv").is_file():
        examples = build_metadata_real_fake_examples(real_root, fake_root)
        return split_metadata_examples(
            examples,
            eval_real_count=eval_real_count,
            eval_fake_count=eval_fake_count,
            seed=seed,
        )
    return _build_legacy_real_fake_examples(
        real_dir=real_dir,
        fake_dir=fake_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )


def summarize_examples(examples: Sequence[VideoExample]) -> dict[str, dict[str, int]]:
    summary = {split: {"real": 0, "fake": 0, "total": 0} for split in VALID_SPLITS}
    for example in examples:
        split_summary = summary[example.split]
        split_summary[example.class_name] += 1
        split_summary["total"] += 1
    return summary


def class_generator_counts(examples: Sequence[VideoExample]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {split: Counter() for split in VALID_SPLITS}
    for example in examples:
        if example.class_name == "fake":
            counts[example.split][example.generator_id or example.identity_id or "unknown"] += 1
    return {split: dict(counter) for split, counter in counts.items()}


def _field_distribution_by_split(
    examples: Sequence[VideoExample],
    field: str,
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {split: Counter() for split in VALID_SPLITS}
    for example in examples:
        counts[example.split][_audit_value(example, field)] += 1
    return {split: dict(counter) for split, counter in counts.items()}


def _source_leakage(examples: Sequence[VideoExample]) -> dict[str, list[str]]:
    split_by_source: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        if example.source_id_kind == "synthetic_file":
            continue
        split_by_source[example.source_id].add(example.split)
    leaked = {
        source_id: sorted(splits)
        for source_id, splits in sorted(split_by_source.items())
        if len(splits) > 1
    }
    return leaked


def summarize_split_audit(examples: Sequence[VideoExample]) -> dict[str, Any]:
    summary = summarize_examples(examples)
    balanced_capacity = {
        split: 2 * min(split_summary["real"], split_summary["fake"])
        for split, split_summary in summary.items()
    }
    source_groups = {split: set() for split in VALID_SPLITS}
    for example in examples:
        source_groups[example.split].add(example.source_id)
    metadata = {
        field: _field_distribution_by_split(examples, field) for field in SPLIT_AUDIT_FIELDS
    }
    leakage = _source_leakage(examples)
    return {
        "summary": summary,
        "fake_generators": class_generator_counts(examples),
        "metadata": metadata,
        "balanced_eval_capacity": balanced_capacity,
        "source_group_counts": {split: len(source_groups[split]) for split in VALID_SPLITS},
        "source_leakage_count": len(leakage),
        "source_leakage": leakage,
    }


def format_split_audit(examples: Sequence[VideoExample]) -> list[str]:
    audit = summarize_split_audit(examples)
    lines = [
        f"split_audit summary={audit['summary']}",
        f"split_audit fake_generators={audit['fake_generators']}",
        f"split_audit balanced_eval_capacity={audit['balanced_eval_capacity']}",
        f"split_audit source_group_counts={audit['source_group_counts']}",
        f"split_audit source_leakage_count={audit['source_leakage_count']}",
    ]
    for field, payload in audit["metadata"].items():
        lines.append(f"split_audit {field}={payload}")
    return lines


def write_dataset_manifest(
    examples: Sequence[VideoExample],
    output_path: str | Path,
) -> Path:
    manifest_path = Path(output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "path",
            "label",
            "class_name",
            "source_id",
            "split",
            "metadata_filename",
            "identity_id",
            *AUDIT_MANIFEST_COLUMNS,
        )
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
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
                    "metadata_filename": example.metadata_filename or "",
                    "identity_id": example.identity_id or "",
                    "generator_id": example.generator_id or "",
                    "source_id_kind": example.source_id_kind or "",
                    "age_bin": example.age_bin or "",
                    "gender": example.gender or "",
                    "ethnicity": example.ethnicity or "",
                    "emotion": example.emotion or "",
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
                    metadata_filename=row.get("metadata_filename") or None,
                    identity_id=row["identity_id"] or None,
                    generator_id=row.get("generator_id") or None,
                    source_id_kind=row.get("source_id_kind") or None,
                    age_bin=row.get("age_bin") or None,
                    gender=row.get("gender") or None,
                    ethnicity=row.get("ethnicity") or None,
                    emotion=row.get("emotion") or None,
                )
            )
    return examples


def load_video_clip(
    path: str | Path,
    num_frames: int,
    image_size: int = 224,
    decode_mode: str = "scan",
    face_crop_config: Mapping[str, Any] | None = None,
    frame_sampling: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if decode_mode not in VIDEO_DECODE_MODES:
        raise ValueError(f"`decode_mode` must be one of {VIDEO_DECODE_MODES}, got {decode_mode!r}")
    clip_path = Path(path)
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {clip_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        fps = RPPG_DEFAULT_FPS
    indices = sample_frame_indices(
        total_frames=total_frames,
        num_frames=num_frames,
        frame_sampling=frame_sampling,
    )
    rgb_frames: list[np.ndarray] = []

    if decode_mode == "seek":
        for frame_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"Failed to read frame {frame_index} from {clip_path}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames.append(frame)
    else:
        target_by_frame: dict[int, list[int]] = defaultdict(list)
        for target, frame_index in enumerate(indices):
            target_by_frame[int(frame_index)].append(target)
        selected_frames: list[np.ndarray | None] = [None] * len(indices)
        max_frame_index = int(indices[-1])
        for frame_index in range(max_frame_index + 1):
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"Failed to read frame {frame_index} from {clip_path}")
            targets = target_by_frame.get(frame_index, ())
            for target in targets:
                selected_frames[target] = frame
        for target, frame in enumerate(selected_frames):
            if frame is None:
                cap.release()
                raise RuntimeError(
                    f"Failed to collect sampled frame {indices[target]} from {clip_path}"
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames.append(frame)

    cap.release()
    box = None
    face_crop_status = FACE_CROP_STATUS_DISABLED
    if face_crop_config is not None:
        box, face_crop_status = _resolve_clip_face_crop(
            rgb_frames,
            face_crop_config=face_crop_config,
            field_name="video",
        )
    resized_frames = crop_resize_rgb_frames(rgb_frames, box, image_size)
    return {
        "video": _imagenet_video_tensor(resized_frames),
        "video_rgb_frames": resized_frames,
        "video_fps": fps,
        "face_crop_status": face_crop_status,
    }


def sample_contiguous_center_indices(total_frames: int, num_frames: int) -> list[int]:
    if total_frames < num_frames:
        raise RuntimeError(f"Video has only {total_frames} frames, need at least {num_frames}")
    start = (total_frames - num_frames) // 2
    return list(range(start, start + num_frames))


def diff_normalized_video_tensor(video: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if video.ndim != 4:
        raise ValueError(f"`video` must have shape [3, T, H, W], got {tuple(video.shape)}")
    if video.shape[0] != 3:
        raise ValueError(f"`video` channel dimension must be 3, got {video.shape[0]}")
    video = video.float()
    diff = torch.zeros_like(video)
    diff[:, :-1] = (video[:, 1:] - video[:, :-1]) / (video[:, 1:] + video[:, :-1] + eps)
    std = diff.std(correction=0)
    if torch.isfinite(std) and float(std) > eps:
        diff = diff / std
    return torch.nan_to_num(diff)


def largest_face_box(boxes: Sequence[Sequence[int]]) -> tuple[int, int, int, int] | None:
    if len(boxes) == 0:
        return None
    x, y, w, h = max(boxes, key=lambda box: int(box[2]) * int(box[3]))
    return int(x), int(y), int(w), int(h)


def median_face_box(boxes: Sequence[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    values = np.asarray(boxes, dtype=np.float32)
    x, y, w, h = np.median(values, axis=0).round().astype(np.int64).tolist()
    return int(x), int(y), int(w), int(h)


def enlarge_box(
    box: tuple[int, int, int, int],
    coef: float,
    frame_shape: tuple[int, int, int] | tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    size_w = max(1.0, float(w) * float(coef))
    size_h = max(1.0, float(h) * float(coef))
    center_x = float(x) + float(w) / 2.0
    center_y = float(y) + float(h) / 2.0
    x1 = max(0, round(center_x - size_w / 2.0))
    y1 = max(0, round(center_y - size_h / 2.0))
    x2 = min(frame_w, round(center_x + size_w / 2.0))
    y2 = min(frame_h, round(center_y + size_h / 2.0))
    if x2 <= x1:
        x2 = min(frame_w, x1 + 1)
    if y2 <= y1:
        y2 = min(frame_h, y1 + 1)
    return x1, y1, x2 - x1, y2 - y1


def _opencv_haar_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / RPPG_HAAR_CASCADE
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load OpenCV Haar cascade: {cascade_path}")
    return detector


def _detect_largest_face(
    frame: np.ndarray,
    detector: cv2.CascadeClassifier,
) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    return largest_face_box(boxes)


def resolve_face_crop_box(
    frames: Sequence[np.ndarray],
    detection_frequency: int = 16,
    large_box_coef: float = 1.5,
    detector: cv2.CascadeClassifier | None = None,
) -> tuple[tuple[int, int, int, int] | None, str]:
    if not frames:
        raise ValueError("Cannot crop empty frame sequence.")
    detector = detector or _opencv_haar_detector()
    stride = max(1, int(detection_frequency))
    detected: list[tuple[int, int, int, int]] = []
    for frame in frames[::stride]:
        box = _detect_largest_face(frame, detector)
        if box is not None:
            detected.append(box)
    median_box = median_face_box(detected)
    if median_box is None:
        return None, FACE_CROP_STATUS_FALLBACK
    return enlarge_box(median_box, large_box_coef, frames[0].shape), FACE_CROP_STATUS_DETECTED


def resolve_rppg_face_crop_box(
    frames: Sequence[np.ndarray],
    detection_frequency: int = 16,
    large_box_coef: float = 1.5,
    detector: cv2.CascadeClassifier | None = None,
) -> tuple[tuple[int, int, int, int] | None, str]:
    return resolve_face_crop_box(
        frames=frames,
        detection_frequency=detection_frequency,
        large_box_coef=large_box_coef,
        detector=detector,
    )


def crop_resize_rgb_frames(
    frames: Sequence[np.ndarray],
    box: tuple[int, int, int, int] | None,
    image_size: int,
) -> list[np.ndarray]:
    resized: list[np.ndarray] = []
    for frame in frames:
        crop = frame
        if box is not None:
            x, y, w, h = box
            crop = frame[y : y + h, x : x + w]
        resized.append(cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA))
    return resized


def _rppg_config_value(config: Mapping[str, Any] | None, key: str, default: Any) -> Any:
    if config is None:
        return default
    return config.get(key, default)


def _resolve_clip_face_crop(
    frames: Sequence[np.ndarray],
    face_crop_config: Mapping[str, Any],
    field_name: str,
) -> tuple[tuple[int, int, int, int] | None, str]:
    if not bool(face_crop_config["enabled"]):
        return None, FACE_CROP_STATUS_DISABLED
    backend = str(face_crop_config["backend"])
    if backend != "opencv_haar":
        raise ValueError(f"Unsupported {field_name} face crop backend: {backend!r}")
    box, status = resolve_face_crop_box(
        frames,
        detection_frequency=int(face_crop_config["detection_frequency"]),
        large_box_coef=float(face_crop_config["large_box_coef"]),
    )
    fallback = str(face_crop_config["fallback"])
    if box is None and fallback != "full_frame":
        raise RuntimeError(f"{field_name} face crop failed and fallback={fallback!r}")
    return box, status


def _imagenet_video_tensor(frames: Sequence[np.ndarray]) -> torch.Tensor:
    tensors = []
    for frame in frames:
        frame_tensor = (
            torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0
        )
        tensors.append((frame_tensor - IMAGENET_MEAN) / IMAGENET_STD)
    return torch.stack(tensors, dim=1)


def load_rppg_video_clip(
    path: str | Path,
    num_frames: int,
    image_size: int = 128,
    rppg_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    clip_path = Path(path)
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {clip_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        fps = RPPG_DEFAULT_FPS
    indices = sample_contiguous_center_indices(total_frames, num_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
    frames: list[np.ndarray] = []
    for frame_index in indices:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to read frame {frame_index} from {clip_path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    face_crop_config = merged_face_crop_config(
        global_config=None,
        modality_config=rppg_config,
        default_enabled=True,
    )
    box, status = _resolve_clip_face_crop(
        frames,
        face_crop_config=face_crop_config,
        field_name="rPPG",
    )

    cropped_frames = crop_resize_rgb_frames(frames, box, image_size)
    video = torch.stack(
        [
            torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0
            for frame in cropped_frames
        ],
        dim=1,
    )
    input_type = str(_rppg_config_value(rppg_config, "input_type", "diff_normalized"))
    if input_type == "diff_normalized":
        video = diff_normalized_video_tensor(video)
    elif input_type != "rgb":
        raise ValueError(f"Unsupported rPPG input_type: {input_type!r}")
    return {
        "video": video,
        "video_rgb_frames": cropped_frames,
        "video_fps": fps,
        "rppg_face_crop_status": status,
    }


def normalize_video_dataset_root(dataset_root: str | Path | None) -> Path | None:
    if dataset_root is None:
        return None
    root = Path(dataset_root)
    videos_root = root / "videos"
    if videos_root.is_dir():
        return videos_root
    return root


def metadata_filename_for_example(
    example: VideoExample,
    dataset_root: str | Path | None = None,
) -> str:
    def validate(filename: str) -> str:
        path = PurePosixPath(filename)
        if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
            raise ValueError(f"Invalid metadata filename: {filename!r}")
        return path.as_posix()

    if example.metadata_filename:
        return validate(example.metadata_filename)
    root = normalize_video_dataset_root(dataset_root)
    if root is None:
        return validate(example.path.name)
    class_root = (root / example.class_name).resolve()
    try:
        return validate(example.path.resolve().relative_to(class_root).as_posix())
    except ValueError:
        return validate(example.path.name)


def video_clip_cache_item_path(
    cache_dir: str | Path,
    example: VideoExample,
    num_frames: int,
    image_size: int,
    cache_variant: str | None = None,
    frame_sampling_variant: str | None = None,
    dataset_root: str | Path | None = None,
) -> Path:
    filename = metadata_filename_for_example(example, dataset_root)
    root = Path(cache_dir)
    if frame_sampling_variant is not None:
        root = root / frame_sampling_variant
    if cache_variant is None:
        return (
            root
            / f"frames_{int(num_frames)}_size_{int(image_size)}"
            / example.class_name
            / f"{filename}.pt"
        )
    return (
        root
        / cache_variant
        / f"frames_{int(num_frames)}_size_{int(image_size)}"
        / example.class_name
        / f"{filename}.pt"
    )


def rppg_clip_cache_item_path(
    cache_dir: str | Path,
    example: VideoExample,
    num_frames: int,
    image_size: int,
    dataset_root: str | Path | None = None,
) -> Path:
    filename = metadata_filename_for_example(example, dataset_root)
    return (
        Path(cache_dir)
        / RPPG_CACHE_VARIANT
        / f"frames_{int(num_frames)}_size_{int(image_size)}"
        / example.class_name
        / f"{filename}.pt"
    )


def _video_clip_cache_header(
    example: VideoExample,
    num_frames: int,
    image_size: int,
    cache_variant: str | None = None,
    frame_sampling_variant: str | None = None,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    header = {
        "version": VIDEO_CLIP_CACHE_VERSION,
        "class_name": example.class_name,
        "filename": metadata_filename_for_example(example, dataset_root),
        "num_frames": int(num_frames),
        "image_size": int(image_size),
    }
    if cache_variant is not None:
        header["cache_variant"] = cache_variant
    if frame_sampling_variant is not None:
        header["frame_sampling_variant"] = frame_sampling_variant
    return header


def _video_clip_cache_matches(
    payload: Mapping[str, Any],
    example: VideoExample,
    num_frames: int,
    image_size: int,
    cache_variant: str | None = None,
    frame_sampling_variant: str | None = None,
    dataset_root: str | Path | None = None,
) -> bool:
    expected = _video_clip_cache_header(
        example,
        num_frames,
        image_size,
        cache_variant,
        frame_sampling_variant,
        dataset_root,
    )
    return all(payload.get(key) == value for key, value in expected.items())


def _rppg_clip_cache_header(
    example: VideoExample,
    num_frames: int,
    image_size: int,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "version": RPPG_CLIP_CACHE_VERSION,
        "cache_variant": RPPG_CACHE_VARIANT,
        "class_name": example.class_name,
        "filename": metadata_filename_for_example(example, dataset_root),
        "num_frames": int(num_frames),
        "image_size": int(image_size),
    }


def _rppg_clip_cache_matches(
    payload: Mapping[str, Any],
    example: VideoExample,
    num_frames: int,
    image_size: int,
    dataset_root: str | Path | None = None,
) -> bool:
    expected = _rppg_clip_cache_header(example, num_frames, image_size, dataset_root)
    return all(payload.get(key) == value for key, value in expected.items())


def load_video_clip_for_example(
    example: VideoExample,
    num_frames: int,
    image_size: int = 224,
    decode_mode: str = "scan",
    face_crop_config: Mapping[str, Any] | None = None,
    cache_variant: str | None = None,
    frame_sampling: Mapping[str, Any] | None = None,
    frame_sampling_variant: str | None = None,
    clip_cache_dir: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    if clip_cache_dir is None:
        kwargs: dict[str, Any] = {}
        if frame_sampling is not None:
            kwargs["frame_sampling"] = frame_sampling
        return load_video_clip(
            path=example.path,
            num_frames=num_frames,
            image_size=image_size,
            decode_mode=decode_mode,
            face_crop_config=face_crop_config,
            **kwargs,
        )

    cache_path = video_clip_cache_item_path(
        clip_cache_dir,
        example,
        num_frames=num_frames,
        image_size=image_size,
        cache_variant=cache_variant,
        frame_sampling_variant=frame_sampling_variant,
        dataset_root=dataset_root,
    )
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(payload, Mapping) and _video_clip_cache_matches(
            payload,
            example,
            num_frames=num_frames,
            image_size=image_size,
            cache_variant=cache_variant,
            frame_sampling_variant=frame_sampling_variant,
            dataset_root=dataset_root,
        ):
            video = payload.get("video")
            frames = payload.get("video_rgb_frames")
            if isinstance(video, torch.Tensor) and isinstance(frames, list):
                fps = float(payload.get("video_fps", RPPG_DEFAULT_FPS))
                return {
                    "video": video,
                    "video_rgb_frames": frames,
                    "video_fps": fps,
                    "face_crop_status": str(
                        payload.get("face_crop_status", FACE_CROP_STATUS_DISABLED)
                    ),
                }

    clip = load_video_clip(
        path=example.path,
        num_frames=num_frames,
        image_size=image_size,
        decode_mode=decode_mode,
        face_crop_config=face_crop_config,
        frame_sampling=frame_sampling,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")
    torch.save(
        {
            **_video_clip_cache_header(
                example,
                num_frames,
                image_size,
                cache_variant,
                frame_sampling_variant,
                dataset_root,
            ),
            "video": clip["video"].detach().cpu(),
            "video_rgb_frames": clip["video_rgb_frames"],
            "video_fps": float(clip.get("video_fps", RPPG_DEFAULT_FPS)),
            "face_crop_status": clip.get("face_crop_status", FACE_CROP_STATUS_DISABLED),
        },
        tmp_path,
    )
    tmp_path.replace(cache_path)
    return clip


def load_rppg_video_clip_for_example(
    example: VideoExample,
    num_frames: int,
    image_size: int = 128,
    rppg_config: Mapping[str, Any] | None = None,
    clip_cache_dir: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    if clip_cache_dir is None:
        return load_rppg_video_clip(
            path=example.path,
            num_frames=num_frames,
            image_size=image_size,
            rppg_config=rppg_config,
        )

    cache_path = rppg_clip_cache_item_path(
        clip_cache_dir,
        example,
        num_frames=num_frames,
        image_size=image_size,
        dataset_root=dataset_root,
    )
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(payload, Mapping) and _rppg_clip_cache_matches(
            payload,
            example,
            num_frames=num_frames,
            image_size=image_size,
            dataset_root=dataset_root,
        ):
            video = payload.get("video")
            frames = payload.get("video_rgb_frames")
            if isinstance(video, torch.Tensor) and isinstance(frames, list):
                return {
                    "video": video,
                    "video_rgb_frames": frames,
                    "video_fps": float(payload.get("video_fps", RPPG_DEFAULT_FPS)),
                    "rppg_face_crop_status": str(
                        payload.get("rppg_face_crop_status", RPPG_FACE_CROP_STATUS_FALLBACK)
                    ),
                }

    clip = load_rppg_video_clip(
        path=example.path,
        num_frames=num_frames,
        image_size=image_size,
        rppg_config=rppg_config,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")
    torch.save(
        {
            **_rppg_clip_cache_header(example, num_frames, image_size, dataset_root),
            "video": clip["video"].detach().cpu(),
            "video_rgb_frames": clip["video_rgb_frames"],
            "video_fps": float(clip.get("video_fps", RPPG_DEFAULT_FPS)),
            "rppg_face_crop_status": clip.get(
                "rppg_face_crop_status", RPPG_FACE_CROP_STATUS_FALLBACK
            ),
        },
        tmp_path,
    )
    tmp_path.replace(cache_path)
    return clip


class LabeledVideoDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        examples: Sequence[VideoExample],
        num_frames: int | Mapping[str, int],
        image_size: int = 224,
        decode_mode: str = "scan",
        clip_cache_dir: str | Path | None = None,
        dataset_root: str | Path | None = None,
        image_size_by_modality: Mapping[str, int] | None = None,
        rppg_config: Mapping[str, Any] | None = None,
        modality_configs: Mapping[str, Mapping[str, Any]] | None = None,
        global_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.examples = list(examples)
        self.num_frames = num_frames
        self.image_size = image_size
        self.image_size_by_modality = dict(image_size_by_modality or {})
        self.rppg_config = dict(rppg_config or {})
        self.modality_configs = {
            str(key): dict(value) for key, value in (modality_configs or {}).items()
        }
        self.global_config = dict(global_config or {})
        self.clip_cache_dir = Path(clip_cache_dir) if clip_cache_dir is not None else None
        self.dataset_root = normalize_video_dataset_root(dataset_root)
        if decode_mode not in VIDEO_DECODE_MODES:
            raise ValueError(f"`decode_mode` must be one of {VIDEO_DECODE_MODES}.")
        self.decode_mode = decode_mode
        if isinstance(num_frames, Mapping):
            if not num_frames:
                raise ValueError("`num_frames` mapping must not be empty.")
            self.frame_counts_by_modality = dict(num_frames)
        else:
            self.frame_counts_by_modality = None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        if self.frame_counts_by_modality is None:
            load_start = time.perf_counter()
            frame_sampling = resolve_frame_sampling_config(self.global_config)
            frame_sampling_variant = frame_sampling_cache_variant(self.global_config)
            clip = load_video_clip_for_example(
                example=example,
                num_frames=int(self.num_frames),
                image_size=self.image_size,
                decode_mode=self.decode_mode,
                frame_sampling=frame_sampling,
                frame_sampling_variant=frame_sampling_variant,
                clip_cache_dir=self.clip_cache_dir,
                dataset_root=self.dataset_root,
            )
            load_seconds = time.perf_counter() - load_start
            video = clip["video"]
            video_rgb_frames = clip["video_rgb_frames"]
            video_fps = float(clip.get("video_fps", RPPG_DEFAULT_FPS))
            video_by_modality = None
            video_rgb_frames_by_modality = None
            video_fps_by_modality = None
            face_crop_status_by_modality = None
            rppg_face_crop_status_by_modality = None
            load_timings_by_modality = {"default": load_seconds}
        else:
            clips_by_spec: dict[tuple[str, int, int, str | None, str | None], dict[str, Any]] = {}
            load_seconds_by_spec: dict[tuple[str, int, int, str | None, str | None], float] = {}
            video_by_modality = {}
            video_rgb_frames_by_modality = {}
            video_fps_by_modality = {}
            face_crop_status_by_modality = {}
            rppg_face_crop_status_by_modality = {}
            for modality_name, frame_count in self.frame_counts_by_modality.items():
                modality_image_size = int(
                    self.image_size_by_modality.get(modality_name, self.image_size)
                )
                if modality_name == "rppg":
                    loader_kind = "rppg"
                    cache_variant = RPPG_CACHE_VARIANT
                    face_crop_config = None
                    frame_sampling = None
                    frame_sampling_variant = None
                else:
                    loader_kind = "default"
                    modality_config = self.modality_configs.get(modality_name, {})
                    frame_sampling = resolve_frame_sampling_config(
                        self.global_config,
                        modality_config,
                    )
                    frame_sampling_variant = frame_sampling_cache_variant(
                        self.global_config,
                        modality_config,
                    )
                    face_crop_config = merged_face_crop_config(
                        global_config=self.global_config,
                        modality_config=modality_config,
                        default_enabled=False,
                    )
                    cache_variant = face_crop_cache_variant(
                        global_config=self.global_config,
                        modality_config=modality_config,
                        default_enabled=False,
                    )
                clip_key = (
                    loader_kind,
                    int(frame_count),
                    modality_image_size,
                    cache_variant,
                    frame_sampling_variant,
                )
                if clip_key not in clips_by_spec:
                    load_start = time.perf_counter()
                    if modality_name == "rppg":
                        clips_by_spec[clip_key] = load_rppg_video_clip_for_example(
                            example=example,
                            num_frames=frame_count,
                            image_size=modality_image_size,
                            rppg_config=self.rppg_config,
                            clip_cache_dir=self.clip_cache_dir,
                            dataset_root=self.dataset_root,
                        )
                    else:
                        clips_by_spec[clip_key] = load_video_clip_for_example(
                            example=example,
                            num_frames=frame_count,
                            image_size=modality_image_size,
                            decode_mode=self.decode_mode,
                            face_crop_config=face_crop_config
                            if cache_variant is not None
                            else None,
                            cache_variant=cache_variant,
                            frame_sampling=frame_sampling,
                            frame_sampling_variant=frame_sampling_variant,
                            clip_cache_dir=self.clip_cache_dir,
                            dataset_root=self.dataset_root,
                        )
                    load_seconds_by_spec[clip_key] = time.perf_counter() - load_start
                clip = clips_by_spec[clip_key]
                video_by_modality[modality_name] = clip["video"]
                video_rgb_frames_by_modality[modality_name] = clip["video_rgb_frames"]
                video_fps_by_modality[modality_name] = float(
                    clip.get("video_fps", RPPG_DEFAULT_FPS)
                )
                face_crop_status_by_modality[modality_name] = str(
                    clip.get(
                        "rppg_face_crop_status" if modality_name == "rppg" else "face_crop_status",
                        FACE_CROP_STATUS_DISABLED,
                    )
                )
                if modality_name == "rppg":
                    rppg_face_crop_status_by_modality[modality_name] = str(
                        clip.get("rppg_face_crop_status", RPPG_FACE_CROP_STATUS_FALLBACK)
                    )

            count_usage = {
                clip_key: sum(
                    1
                    for modality_name, value in self.frame_counts_by_modality.items()
                    if (
                        "rppg" if modality_name == "rppg" else "default",
                        int(value),
                        int(self.image_size_by_modality.get(modality_name, self.image_size)),
                        RPPG_CACHE_VARIANT
                        if modality_name == "rppg"
                        else face_crop_cache_variant(
                            global_config=self.global_config,
                            modality_config=self.modality_configs.get(modality_name, {}),
                            default_enabled=False,
                        ),
                        None
                        if modality_name == "rppg"
                        else frame_sampling_cache_variant(
                            global_config=self.global_config,
                            modality_config=self.modality_configs.get(modality_name, {}),
                        ),
                    )
                    == clip_key
                )
                for clip_key in load_seconds_by_spec
            }
            load_timings_by_modality = {
                modality_name: load_seconds_by_spec[
                    (
                        "rppg" if modality_name == "rppg" else "default",
                        int(frame_count),
                        int(self.image_size_by_modality.get(modality_name, self.image_size)),
                        RPPG_CACHE_VARIANT
                        if modality_name == "rppg"
                        else face_crop_cache_variant(
                            global_config=self.global_config,
                            modality_config=self.modality_configs.get(modality_name, {}),
                            default_enabled=False,
                        ),
                        None
                        if modality_name == "rppg"
                        else frame_sampling_cache_variant(
                            global_config=self.global_config,
                            modality_config=self.modality_configs.get(modality_name, {}),
                        ),
                    )
                ]
                / count_usage[
                    (
                        "rppg" if modality_name == "rppg" else "default",
                        int(frame_count),
                        int(self.image_size_by_modality.get(modality_name, self.image_size)),
                        RPPG_CACHE_VARIANT
                        if modality_name == "rppg"
                        else face_crop_cache_variant(
                            global_config=self.global_config,
                            modality_config=self.modality_configs.get(modality_name, {}),
                            default_enabled=False,
                        ),
                        None
                        if modality_name == "rppg"
                        else frame_sampling_cache_variant(
                            global_config=self.global_config,
                            modality_config=self.modality_configs.get(modality_name, {}),
                        ),
                    )
                ]
                for modality_name, frame_count in self.frame_counts_by_modality.items()
            }

            first_modality = next(iter(self.frame_counts_by_modality))
            video = video_by_modality[first_modality]
            video_rgb_frames = video_rgb_frames_by_modality[first_modality]
            video_fps = video_fps_by_modality[first_modality]

        item = {
            "video": video,
            "video_rgb_frames": video_rgb_frames,
            "video_fps": video_fps,
            "label": torch.tensor([float(example.label)], dtype=torch.float32),
            "path": str(example.path),
            "source_id": example.source_id,
            "split": example.split,
            "class_name": example.class_name,
            "identity_id": example.identity_id,
            "metadata_filename": example.metadata_filename,
            "generator_id": example.generator_id,
            "source_id_kind": example.source_id_kind,
            "age_bin": example.age_bin,
            "gender": example.gender,
            "ethnicity": example.ethnicity,
            "emotion": example.emotion,
            "load_timings_by_modality": load_timings_by_modality,
        }
        if video_by_modality is not None and video_rgb_frames_by_modality is not None:
            item["video_by_modality"] = video_by_modality
            item["video_rgb_frames_by_modality"] = video_rgb_frames_by_modality
            item["video_fps_by_modality"] = video_fps_by_modality
            item["face_crop_status_by_modality"] = face_crop_status_by_modality
            item["rppg_face_crop_status_by_modality"] = rppg_face_crop_status_by_modality
        return {
            **item,
        }


def collate_labeled_video_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch.")
    batch = {
        "video": torch.stack([item["video"] for item in items], dim=0),
        "video_rgb_frames": [item["video_rgb_frames"] for item in items],
        "video_fps": torch.tensor(
            [float(item.get("video_fps", RPPG_DEFAULT_FPS)) for item in items],
            dtype=torch.float32,
        ),
        "label": torch.stack([item["label"] for item in items], dim=0),
        "path": [item["path"] for item in items],
        "source_id": [item["source_id"] for item in items],
        "split": [item["split"] for item in items],
        "class_name": [item["class_name"] for item in items],
        "identity_id": [item["identity_id"] for item in items],
        "metadata_filename": [item.get("metadata_filename") for item in items],
        "generator_id": [item.get("generator_id") for item in items],
        "source_id_kind": [item.get("source_id_kind") for item in items],
        "age_bin": [item.get("age_bin") for item in items],
        "gender": [item.get("gender") for item in items],
        "ethnicity": [item.get("ethnicity") for item in items],
        "emotion": [item.get("emotion") for item in items],
    }
    if "load_timings_by_modality" in items[0]:
        batch["load_timings_by_modality"] = {
            modality_name: [item["load_timings_by_modality"][modality_name] for item in items]
            for modality_name in items[0]["load_timings_by_modality"]
        }
    if "video_by_modality" in items[0]:
        modality_names = tuple(items[0]["video_by_modality"].keys())
        batch["video_by_modality"] = {
            modality_name: torch.stack(
                [item["video_by_modality"][modality_name] for item in items],
                dim=0,
            )
            for modality_name in modality_names
        }
        batch["video_rgb_frames_by_modality"] = {
            modality_name: [item["video_rgb_frames_by_modality"][modality_name] for item in items]
            for modality_name in modality_names
        }
        batch["video_fps_by_modality"] = {
            modality_name: torch.tensor(
                [
                    float(
                        item.get("video_fps_by_modality", {}).get(modality_name, RPPG_DEFAULT_FPS)
                    )
                    for item in items
                ],
                dtype=torch.float32,
            )
            for modality_name in modality_names
        }
        batch["face_crop_status_by_modality"] = {
            modality_name: [
                item.get("face_crop_status_by_modality", {}).get(modality_name) for item in items
            ]
            for modality_name in modality_names
        }
        batch["rppg_face_crop_status_by_modality"] = {
            modality_name: [
                item.get("rppg_face_crop_status_by_modality", {}).get(modality_name)
                for item in items
            ]
            for modality_name in modality_names
        }
    return batch
