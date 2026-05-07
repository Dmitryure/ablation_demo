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
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".webm")
VIDEO_DECODE_MODES: tuple[str, ...] = ("seek", "scan")


@dataclass(frozen=True)
class VideoExample:
    path: Path
    label: int
    class_name: str
    source_id: str
    split: str
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
    decode_mode: str = "seek",
) -> dict[str, Any]:
    if decode_mode not in VIDEO_DECODE_MODES:
        raise ValueError(f"`decode_mode` must be one of {VIDEO_DECODE_MODES}, got {decode_mode!r}")
    clip_path = Path(path)
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {clip_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        cap.release()
        raise RuntimeError(f"Video has only {total_frames} frames, need at least {num_frames}")

    indices = (
        torch.linspace(0, total_frames - 1, steps=num_frames).round().to(dtype=torch.int64).tolist()
    )
    clip_frames: list[torch.Tensor] = []
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
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame_tensor = (frame_tensor - IMAGENET_MEAN) / IMAGENET_STD
            clip_frames.append(frame_tensor)
    else:
        target_by_frame = {int(frame_index): target for target, frame_index in enumerate(indices)}
        selected_frames: list[np.ndarray | None] = [None] * len(indices)
        max_frame_index = int(indices[-1])
        for frame_index in range(max_frame_index + 1):
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"Failed to read frame {frame_index} from {clip_path}")
            target = target_by_frame.get(frame_index)
            if target is not None:
                selected_frames[target] = frame
        for target, frame in enumerate(selected_frames):
            if frame is None:
                cap.release()
                raise RuntimeError(
                    f"Failed to collect sampled frame {indices[target]} from {clip_path}"
                )
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
        num_frames: int | Mapping[str, int],
        image_size: int = 224,
        decode_mode: str = "seek",
    ) -> None:
        self.examples = list(examples)
        self.num_frames = num_frames
        self.image_size = image_size
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
            clip = load_video_clip(
                path=example.path,
                num_frames=int(self.num_frames),
                image_size=self.image_size,
                decode_mode=self.decode_mode,
            )
            load_seconds = time.perf_counter() - load_start
            video = clip["video"]
            video_rgb_frames = clip["video_rgb_frames"]
            video_by_modality = None
            video_rgb_frames_by_modality = None
            load_timings_by_modality = {"default": load_seconds}
        else:
            clips_by_count: dict[int, dict[str, Any]] = {}
            load_seconds_by_count: dict[int, float] = {}
            video_by_modality = {}
            video_rgb_frames_by_modality = {}
            for modality_name, frame_count in self.frame_counts_by_modality.items():
                if frame_count not in clips_by_count:
                    load_start = time.perf_counter()
                    clips_by_count[frame_count] = load_video_clip(
                        path=example.path,
                        num_frames=frame_count,
                        image_size=self.image_size,
                        decode_mode=self.decode_mode,
                    )
                    load_seconds_by_count[frame_count] = time.perf_counter() - load_start
                clip = clips_by_count[frame_count]
                video_by_modality[modality_name] = clip["video"]
                video_rgb_frames_by_modality[modality_name] = clip["video_rgb_frames"]

            count_usage = {
                frame_count: sum(
                    1 for value in self.frame_counts_by_modality.values() if value == frame_count
                )
                for frame_count in load_seconds_by_count
            }
            load_timings_by_modality = {
                modality_name: load_seconds_by_count[frame_count] / count_usage[frame_count]
                for modality_name, frame_count in self.frame_counts_by_modality.items()
            }

            first_modality = next(iter(self.frame_counts_by_modality))
            video = video_by_modality[first_modality]
            video_rgb_frames = video_rgb_frames_by_modality[first_modality]

        item = {
            "video": video,
            "video_rgb_frames": video_rgb_frames,
            "label": torch.tensor([float(example.label)], dtype=torch.float32),
            "path": str(example.path),
            "source_id": example.source_id,
            "split": example.split,
            "class_name": example.class_name,
            "identity_id": example.identity_id,
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
        return {
            **item,
        }


def collate_labeled_video_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch.")
    batch = {
        "video": torch.stack([item["video"] for item in items], dim=0),
        "video_rgb_frames": [item["video_rgb_frames"] for item in items],
        "label": torch.stack([item["label"] for item in items], dim=0),
        "path": [item["path"] for item in items],
        "source_id": [item["source_id"] for item in items],
        "split": [item["split"] for item in items],
        "class_name": [item["class_name"] for item in items],
        "identity_id": [item["identity_id"] for item in items],
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
    return batch
