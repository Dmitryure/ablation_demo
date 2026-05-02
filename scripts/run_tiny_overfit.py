from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (
    LabeledVideoDataset,
    VideoExample,
    build_labeled_folder_examples,
    collate_labeled_video_batch,
)
from frame_config import describe_frame_counts, resolve_modality_frame_counts
from pipeline import build_fusion_pipeline, load_pipeline_yaml
from task_models import BinaryFusionClassifier, build_binary_fusion_classifier

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "registry_fusion.yaml"
DEFAULT_OVERFIT_DIR = PROJECT_ROOT / "tests" / "overfit_videos"
DEFAULT_PREDICT_DIR = PROJECT_ROOT / "tests" / "predict_videos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests" / "overfit_runs"
RUN_DIR_PREFIX = "run_"
FEATURE_CACHE_VERSION = 1
FEATURE_CACHE_DIRNAME = "_feature_cache"
HEAD_TYPES = ("cls_linear", "cls_mlp", "attention_mil", "modality_gated_mil")


@dataclass(frozen=True)
class PredictionRow:
    path: str
    class_name: str
    label: int
    probability: float
    prediction: int
    split: str
    seen_in_train: bool = False


@dataclass(frozen=True)
class PrecomputedRunData:
    train_examples: list[VideoExample]
    predict_examples: list[VideoExample]
    train_items: list[dict[str, Any]]
    predict_items: list[dict[str, Any]]
    train_timing_rows: list[dict[str, Any]]
    predict_timing_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class PrecomputedFeatureResult:
    items: list[dict[str, Any]]
    timing_rows: list[dict[str, Any]]
    cache_status: str
    cache_path: Path | None
    elapsed_seconds: float


@dataclass(frozen=True)
class EpochTrainResult:
    loss: float
    accuracy: float
    elapsed_seconds: float
    videos_per_second: float
    seconds_per_video: float


class PrecomputedFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(self, items: Sequence[Mapping[str, Any]]) -> None:
        self.items = [dict(item) for item in items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit binary fake/real classifier on tiny local videos."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overfit-dir", type=Path, default=DEFAULT_OVERFIT_DIR)
    parser.add_argument("--predict-dir", type=Path, default=DEFAULT_PREDICT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument(
        "--modality-permutations",
        choices=("none", "singletons", "singletons-plus-all", "all"),
        default="none",
        help=(
            "Run modality subset combinations. Order does not matter for this model; "
            "`all` runs every non-empty subset and can be slow."
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--target-train-accuracy", type=float, default=1.0)
    parser.add_argument("--head-type", choices=HEAD_TYPES, default=None)
    parser.add_argument("--head-hidden-dim", type=int, default=None)
    parser.add_argument("--head-dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_config(config_path: Path, device: str | None) -> dict[str, Any]:
    config = load_pipeline_yaml(config_path)
    if device is not None:
        config["device"] = device
    return config


def build_head_config(config: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    raw_config = config.get("head")
    if raw_config is None:
        head_config: dict[str, Any] = {}
    elif isinstance(raw_config, Mapping):
        head_config = dict(raw_config)
    else:
        raise ValueError("Config `head` must be a mapping when provided.")

    head_type = getattr(args, "head_type", None)
    hidden_dim = getattr(args, "head_hidden_dim", None)
    dropout = getattr(args, "head_dropout", None)
    if head_type is not None:
        head_config["type"] = head_type
    if hidden_dim is not None:
        head_config["hidden_dim"] = hidden_dim
    if dropout is not None:
        head_config["dropout"] = dropout
    return head_config or None


def resolve_base_modalities(
    config: Mapping[str, Any], requested: Sequence[str] | None
) -> tuple[str, ...]:
    if requested is not None and len(requested) > 0:
        return tuple(requested)
    modalities = config.get("modalities")
    if not isinstance(modalities, list) or not all(isinstance(item, str) for item in modalities):
        raise ValueError("Config `modalities` must be a list of strings.")
    return tuple(modalities)


def build_modality_sets(
    base_modalities: Sequence[str],
    mode: str,
) -> list[tuple[str, ...]]:
    base = tuple(base_modalities)
    if not base:
        raise ValueError("At least one modality is required.")
    if mode == "none":
        return [base]
    if mode == "singletons":
        return [(modality,) for modality in base]
    if mode == "singletons-plus-all":
        sets = [(modality,) for modality in base]
        if len(base) > 1:
            sets.append(base)
        return sets
    if mode == "all":
        return [
            tuple(combo)
            for size in range(1, len(base) + 1)
            for combo in itertools.combinations(base, size)
        ]
    raise ValueError(f"Unsupported modality permutation mode: {mode}")


def modality_set_name(modalities: Sequence[str]) -> str:
    return "__".join(modalities)


def parse_run_index(path: Path) -> int | None:
    name = path.name
    if not name.startswith(RUN_DIR_PREFIX):
        return None
    suffix = name[len(RUN_DIR_PREFIX) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def next_run_index(output_dir: Path) -> int:
    if not output_dir.exists():
        return 1
    indices = [
        index
        for child in output_dir.iterdir()
        if child.is_dir() and (index := parse_run_index(child)) is not None
    ]
    return max(indices, default=0) + 1


def allocate_indexed_run_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_index = next_run_index(output_dir)
    while True:
        run_dir = output_dir / f"{RUN_DIR_PREFIX}{run_index:03d}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            run_index += 1


def should_log_epoch(epoch: int, total_epochs: int) -> bool:
    return epoch % 10 == 0 or epoch == total_epochs


def freeze_encoder_modules(model: BinaryFusionClassifier) -> None:
    for parameter in model.pipeline.encoder_modules.parameters():
        parameter.requires_grad = False
    for extractor in model.pipeline.extractors.values():
        encoder = getattr(extractor, "encoder", None)
        if isinstance(encoder, torch.nn.Module):
            for parameter in encoder.parameters():
                parameter.requires_grad = False


def model_device(model: torch.nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is None:
        return torch.device("cpu")
    return parameter.device


def class_name_from_label(label: int) -> str:
    return "fake" if label == 1 else "real"


def example_label_name_keys(examples: Sequence[VideoExample]) -> set[tuple[str, str]]:
    return {(example.class_name, example.path.name) for example in examples}


def find_label_name_overlaps(
    train_examples: Sequence[VideoExample],
    predict_examples: Sequence[VideoExample],
) -> list[tuple[str, str]]:
    train_keys = example_label_name_keys(train_examples)
    predict_keys = example_label_name_keys(predict_examples)
    return sorted(train_keys & predict_keys)


def move_tensor_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def split_feature_batch(
    feature_batch: Mapping[str, Any],
    raw_batch: Mapping[str, Any],
) -> list[dict[str, Any]]:
    labels = raw_batch["label"]
    paths = raw_batch["path"]
    class_names = raw_batch["class_name"]
    splits = raw_batch["split"]
    batch_size = int(labels.shape[0])
    items: list[dict[str, Any]] = []
    for index in range(batch_size):
        item: dict[str, Any] = {
            "label": labels[index].detach().cpu(),
            "path": paths[index],
            "class_name": class_names[index],
            "split": splits[index],
        }
        for key, value in feature_batch.items():
            if isinstance(value, torch.Tensor):
                item[key] = value[index].detach().cpu()
        items.append(item)
    return items


def normalize_frame_counts_for_cache(frame_counts: int | Mapping[str, int]) -> dict[str, int]:
    if isinstance(frame_counts, Mapping):
        return {str(key): int(value) for key, value in sorted(frame_counts.items())}
    return {"default": int(frame_counts)}


def jsonable_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(config, sort_keys=True, default=str))


def example_cache_record(example: VideoExample) -> dict[str, Any]:
    stat = example.path.stat()
    return {
        "path": str(example.path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "label": example.label,
        "class_name": example.class_name,
        "source_id": example.source_id,
        "split": example.split,
        "identity_id": example.identity_id,
    }


def feature_cache_fingerprint(
    *,
    config: Mapping[str, Any],
    examples: Sequence[VideoExample],
    modalities: Sequence[str],
    frame_counts: int | Mapping[str, int],
    image_size: int,
    cache_label: str,
) -> str:
    payload = {
        "version": FEATURE_CACHE_VERSION,
        "cache_label": cache_label,
        "modalities": list(modalities),
        "frame_counts": normalize_frame_counts_for_cache(frame_counts),
        "image_size": int(image_size),
        "config": jsonable_config(config),
        "examples": [example_cache_record(example) for example in examples],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_cache_path(cache_dir: Path, fingerprint: str) -> Path:
    return cache_dir / f"{fingerprint}.pt"


def load_feature_items_from_cache(
    cache_path: Path, fingerprint: str
) -> list[dict[str, Any]] | None:
    if not cache_path.exists():
        return None
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("fingerprint") != fingerprint:
        return None
    items = payload.get("items")
    if not isinstance(items, list):
        return None
    return [dict(item) for item in items]


def write_feature_items_cache(
    cache_path: Path,
    fingerprint: str,
    items: Sequence[Mapping[str, Any]],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fingerprint": fingerprint,
            "version": FEATURE_CACHE_VERSION,
            "items": [dict(item) for item in items],
        },
        cache_path,
    )


def append_timing_rows(
    timing_rows: list[dict[str, Any]],
    raw_batch: Mapping[str, Any],
    feature_timings: Mapping[str, float],
    cache_status: str,
    cache_path: Path | None,
) -> None:
    paths = raw_batch["path"]
    class_names = raw_batch["class_name"]
    splits = raw_batch["split"]
    labels = raw_batch["label"].view(-1).to(dtype=torch.long)
    load_timings = raw_batch.get("load_timings_by_modality", {})
    batch_size = len(paths)
    for modality_name, elapsed_seconds in feature_timings.items():
        load_values = load_timings.get(modality_name) if isinstance(load_timings, Mapping) else None
        if load_values is None and isinstance(load_timings, Mapping):
            load_values = load_timings.get("default")
        extract_seconds = float(elapsed_seconds) / batch_size
        for index in range(batch_size):
            load_seconds = float(load_values[index]) if isinstance(load_values, list) else 0.0
            total_seconds = load_seconds + extract_seconds
            timing_rows.append(
                {
                    "path": paths[index],
                    "class_name": class_names[index],
                    "label": int(labels[index].item()),
                    "split": splits[index],
                    "modality": modality_name,
                    "elapsed_seconds": f"{total_seconds:.6f}",
                    "load_seconds": f"{load_seconds:.6f}",
                    "extract_seconds": f"{extract_seconds:.6f}",
                    "batch_elapsed_seconds": f"{float(elapsed_seconds):.6f}",
                    "batch_extract_seconds": f"{float(elapsed_seconds):.6f}",
                    "batch_size": batch_size,
                    "cache_status": cache_status,
                    "cache_path": "" if cache_path is None else str(cache_path),
                }
            )


def cached_timing_rows(
    items: Sequence[Mapping[str, Any]],
    modalities: Sequence[str],
    cache_path: Path,
    elapsed_seconds: float,
) -> list[dict[str, Any]]:
    per_row_seconds = (
        0.0 if not items or not modalities else elapsed_seconds / (len(items) * len(modalities))
    )
    rows: list[dict[str, Any]] = []
    for item in items:
        label = item["label"]
        label_value = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
        for modality_name in modalities:
            rows.append(
                {
                    "path": item["path"],
                    "class_name": item["class_name"],
                    "label": label_value,
                    "split": item["split"],
                    "modality": modality_name,
                    "elapsed_seconds": f"{per_row_seconds:.6f}",
                    "load_seconds": "0.000000",
                    "extract_seconds": "0.000000",
                    "batch_elapsed_seconds": f"{elapsed_seconds:.6f}",
                    "batch_extract_seconds": "0.000000",
                    "batch_size": len(items),
                    "cache_status": "hit",
                    "cache_path": str(cache_path),
                }
            )
    return rows


def precompute_feature_items(
    pipeline: torch.nn.Module,
    examples: Sequence[VideoExample],
    frame_counts: int | Mapping[str, int],
    image_size: int,
    batch_size: int,
    config: Mapping[str, Any] | None = None,
    modalities: Sequence[str] = (),
    cache_dir: Path | None = None,
    cache_label: str = "",
) -> PrecomputedFeatureResult:
    start_time = time.perf_counter()
    if cache_dir is not None and config is not None and modalities:
        fingerprint = feature_cache_fingerprint(
            config=config,
            examples=examples,
            modalities=modalities,
            frame_counts=frame_counts,
            image_size=image_size,
            cache_label=cache_label,
        )
        cache_path = feature_cache_path(cache_dir, fingerprint)
        cached_items = load_feature_items_from_cache(cache_path, fingerprint)
        if cached_items is not None:
            elapsed_seconds = time.perf_counter() - start_time
            print(f"feature cache hit: {cache_path}")
            return PrecomputedFeatureResult(
                items=cached_items,
                timing_rows=cached_timing_rows(
                    cached_items,
                    modalities=modalities,
                    cache_path=cache_path,
                    elapsed_seconds=elapsed_seconds,
                ),
                cache_status="hit",
                cache_path=cache_path,
                elapsed_seconds=elapsed_seconds,
            )
        print(f"feature cache miss: {cache_path}")
    else:
        fingerprint = None
        cache_path = None

    dataset = LabeledVideoDataset(examples=examples, num_frames=frame_counts, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_labeled_video_batch,
    )
    items: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    pipeline.eval()
    with torch.no_grad():
        for raw_batch in loader:
            feature_batch = pipeline.prepare_features(raw_batch)
            items.extend(split_feature_batch(feature_batch, raw_batch))
            append_timing_rows(
                timing_rows,
                raw_batch=raw_batch,
                feature_timings=getattr(pipeline, "last_feature_timings", {}),
                cache_status="miss" if cache_path is not None else "disabled",
                cache_path=cache_path,
            )
    if cache_path is not None and fingerprint is not None:
        write_feature_items_cache(cache_path, fingerprint, items)
        print(f"feature cache wrote: {cache_path}")
    elapsed_seconds = time.perf_counter() - start_time
    return PrecomputedFeatureResult(
        items=items,
        timing_rows=timing_rows,
        cache_status="miss" if cache_path is not None else "disabled",
        cache_path=cache_path,
        elapsed_seconds=elapsed_seconds,
    )


def precompute_run_data(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    modalities: Sequence[str],
) -> PrecomputedRunData:
    build_result = build_fusion_pipeline(config=config, modalities=modalities)
    pipeline = build_result.pipeline
    train_examples = build_labeled_folder_examples(args.overfit_dir, split="train")
    predict_examples = build_labeled_folder_examples(args.predict_dir, split="test")
    frame_counts = resolve_modality_frame_counts(config, modalities)

    print(
        "precompute train features: "
        f"{len(train_examples)} videos, frames: {describe_frame_counts(frame_counts)}"
    )
    train_result = precompute_feature_items(
        pipeline=pipeline,
        examples=train_examples,
        frame_counts=frame_counts,
        image_size=int(config["image_size"]),
        batch_size=args.batch_size,
        config=config,
        modalities=modalities,
        cache_dir=None
        if getattr(args, "no_feature_cache", False)
        else getattr(args, "feature_cache_dir", None),
        cache_label="train",
    )
    print_precompute_timing_summary("precompute train timing", train_result.timing_rows)
    print(
        "precompute predict features: "
        f"{len(predict_examples)} videos, frames: {describe_frame_counts(frame_counts)}"
    )
    predict_result = precompute_feature_items(
        pipeline=pipeline,
        examples=predict_examples,
        frame_counts=frame_counts,
        image_size=int(config["image_size"]),
        batch_size=args.batch_size,
        config=config,
        modalities=modalities,
        cache_dir=None
        if getattr(args, "no_feature_cache", False)
        else getattr(args, "feature_cache_dir", None),
        cache_label="predict",
    )
    print_precompute_timing_summary("precompute predict timing", predict_result.timing_rows)
    pipeline.close()
    if build_result.device.type == "cuda":
        torch.cuda.empty_cache()
    return PrecomputedRunData(
        train_examples=train_examples,
        predict_examples=predict_examples,
        train_items=train_result.items,
        predict_items=predict_result.items,
        train_timing_rows=train_result.timing_rows,
        predict_timing_rows=predict_result.timing_rows,
    )


def collate_precomputed_feature_batch(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch.")
    batch: dict[str, Any] = {}
    keys = items[0].keys()
    for key in keys:
        values = [item[key] for item in items]
        if all(isinstance(value, torch.Tensor) for value in values):
            batch[key] = torch.stack(values, dim=0)
        else:
            batch[key] = list(values)
    return batch


def build_feature_loader(
    items: Sequence[Mapping[str, Any]],
    batch_size: int,
    shuffle: bool,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        PrecomputedFeatureDataset(items),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_precomputed_feature_batch,
    )


def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = (torch.sigmoid(logits) >= 0.5).to(dtype=torch.float32)
    return float((predictions == labels).to(dtype=torch.float32).mean().item())


def train_one_epoch(
    model: BinaryFusionClassifier,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
) -> EpochTrainResult:
    device = model_device(model)
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    start = time.perf_counter()
    for batch in loader:
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(move_tensor_batch_to_device(batch, device))
        loss = loss_fn(output.logits, labels)
        loss.backward()
        optimizer.step()

        batch_count = int(labels.numel())
        total_loss += float(loss.item()) * batch_count
        total_correct += binary_accuracy(output.logits.detach(), labels) * batch_count
        total_count += batch_count
    elapsed_seconds = time.perf_counter() - start
    videos_per_second = 0.0 if elapsed_seconds <= 0.0 else total_count / elapsed_seconds
    seconds_per_video = 0.0 if total_count == 0 else elapsed_seconds / total_count
    return EpochTrainResult(
        loss=total_loss / total_count,
        accuracy=total_correct / total_count,
        elapsed_seconds=elapsed_seconds,
        videos_per_second=videos_per_second,
        seconds_per_video=seconds_per_video,
    )


def predict_rows(
    model: BinaryFusionClassifier,
    loader: DataLoader[dict[str, Any]],
    seen_keys: set[tuple[str, str]] | None = None,
) -> tuple[float, list[PredictionRow]]:
    device = model_device(model)
    seen = seen_keys or set()
    rows: list[PredictionRow] = []
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(move_tensor_batch_to_device(batch, device))
            probabilities = output.probabilities.detach().cpu().view(-1)
            labels = batch["label"].view(-1).to(dtype=torch.long)
            predictions = (probabilities >= 0.5).to(dtype=torch.long)
            paths = batch["path"]
            class_names = batch["class_name"]
            splits = batch["split"]
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
            for index, probability in enumerate(probabilities.tolist()):
                rows.append(
                    PredictionRow(
                        path=paths[index],
                        class_name=class_names[index],
                        label=int(labels[index].item()),
                        probability=float(probability),
                        prediction=int(predictions[index].item()),
                        split=splits[index],
                        seen_in_train=(class_names[index], Path(paths[index]).name) in seen,
                    )
                )
    return correct / total, rows


def summarize_seen_predict_rows(rows: Sequence[PredictionRow]) -> dict[str, float | int | None]:
    seen_predict_rows = [row for row in rows if row.split == "test" and row.seen_in_train]
    misses = sum(1 for row in seen_predict_rows if row.prediction != row.label)
    count = len(seen_predict_rows)
    accuracy = None if count == 0 else (count - misses) / count
    return {
        "predict_seen_count": count,
        "predict_seen_misses": misses,
        "predict_seen_accuracy": accuracy,
    }


def summarize_unseen_predict_rows(rows: Sequence[PredictionRow]) -> dict[str, float | int | None]:
    unseen_predict_rows = [row for row in rows if row.split == "test" and not row.seen_in_train]
    misses = sum(1 for row in unseen_predict_rows if row.prediction != row.label)
    count = len(unseen_predict_rows)
    accuracy = None if count == 0 else (count - misses) / count
    return {
        "predict_unseen_count": count,
        "predict_unseen_misses": misses,
        "predict_unseen_accuracy": accuracy,
    }


def miss_rate_percent(misses: int | float | None, count: int | float | None) -> float | None:
    if count in (None, 0):
        return None
    if misses is None:
        return None
    return float(misses) / float(count) * 100.0


def format_miss_line(name: str, misses: int | float | None, count: int | float | None) -> str:
    percent = miss_rate_percent(misses, count)
    if percent is None:
        return f"{name}={misses} of {count} (n/a)"
    return f"{name}={misses} of {count} ({percent:.2f}%)"


def write_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "train_loss",
                "train_accuracy",
                "train_elapsed_seconds",
                "train_videos_per_second",
                "train_seconds_per_video",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_predictions(path: Path, rows: Sequence[PredictionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "path",
                "class_name",
                "label",
                "prediction",
                "probability",
                "split",
                "seen_in_train",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row.path,
                    "class_name": row.class_name,
                    "label": row.label,
                    "prediction": row.prediction,
                    "probability": f"{row.probability:.8f}",
                    "split": row.split,
                    "seen_in_train": int(row.seen_in_train),
                }
            )


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_precompute_timings(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "path",
        "class_name",
        "label",
        "split",
        "modality",
        "elapsed_seconds",
        "load_seconds",
        "extract_seconds",
        "batch_elapsed_seconds",
        "batch_extract_seconds",
        "batch_size",
        "cache_status",
        "cache_path",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_precompute_timings(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_modality: dict[str, dict[str, Any]] = {}
    total = 0.0
    for row in rows:
        modality = str(row["modality"])
        elapsed = float(row["elapsed_seconds"])
        total += elapsed
        current = by_modality.setdefault(
            modality,
            {
                "count": 0,
                "total_seconds": 0.0,
                "load_seconds": 0.0,
                "extract_seconds": 0.0,
                "max_seconds": 0.0,
            },
        )
        current["count"] += 1
        current["total_seconds"] += elapsed
        current["load_seconds"] += float(row.get("load_seconds", 0.0) or 0.0)
        current["extract_seconds"] += float(row.get("extract_seconds", 0.0) or 0.0)
        current["max_seconds"] = max(current["max_seconds"], elapsed)

    formatted: dict[str, dict[str, float | int]] = {}
    for modality, values in by_modality.items():
        count = int(values["count"])
        total_seconds = float(values["total_seconds"])
        formatted[modality] = {
            "count": count,
            "total_seconds": total_seconds,
            "load_seconds": float(values["load_seconds"]),
            "extract_seconds": float(values["extract_seconds"]),
            "mean_seconds": 0.0 if count == 0 else total_seconds / count,
            "max_seconds": float(values["max_seconds"]),
        }
    return {
        "total_seconds": total,
        "by_modality": formatted,
    }


def print_precompute_timing_summary(
    title: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    summary = summarize_precompute_timings(rows)
    print(f"{title}: total={summary['total_seconds']:.3f}s")
    by_modality = summary["by_modality"]
    if not isinstance(by_modality, Mapping) or not by_modality:
        print(f"{title}: no timing rows")
        return
    for modality_name, values in sorted(by_modality.items()):
        if not isinstance(values, Mapping):
            continue
        print(
            f"  {modality_name}: "
            f"count={values['count']} "
            f"total={values['total_seconds']:.3f}s "
            f"load={values['load_seconds']:.3f}s "
            f"extract={values['extract_seconds']:.3f}s "
            f"mean={values['mean_seconds']:.3f}s/video "
            f"max={values['max_seconds']:.3f}s/video"
        )


def write_train_accuracy_plot(path: Path, rows: Sequence[Mapping[str, Any]], title: str) -> bool:
    if not rows:
        return False
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: could not import matplotlib; skipped train accuracy plot: {exc}")
        return False

    epochs = [int(row["epoch"]) for row in rows]
    train_accuracy = [float(row["train_accuracy"]) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_accuracy, marker="o", linewidth=1.8, markersize=3.0)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def write_modality_accuracy_plot(path: Path, summaries: Sequence[Mapping[str, Any]]) -> bool:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: could not import matplotlib; skipped modality accuracy plot: {exc}")
        return False

    series: list[tuple[str, list[int], list[float]]] = []
    for summary in summaries:
        metrics_path = summary.get("metrics_csv")
        modalities = summary.get("modalities")
        if not isinstance(metrics_path, str) or not isinstance(modalities, list):
            continue
        rows = read_metrics(Path(metrics_path))
        if not rows:
            continue
        epochs = [int(row["epoch"]) for row in rows]
        train_accuracy = [float(row["train_accuracy"]) for row in rows]
        series.append(("+".join(str(modality) for modality in modalities), epochs, train_accuracy))

    if not series:
        return False

    columns = 1 if len(series) == 1 else 2 if len(series) <= 8 else 3
    rows_count = (len(series) + columns - 1) // columns
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(6.0 * columns, 3.6 * rows_count),
        squeeze=False,
    )
    for index, (label, epochs, train_accuracy) in enumerate(series):
        ax = axes[index // columns][index % columns]
        ax.plot(epochs, train_accuracy, linewidth=1.8, marker="o", markersize=2.8)
        ax.set_title(label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("train accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)

    for index in range(len(series), rows_count * columns):
        axes[index // columns][index % columns].axis("off")

    fig.suptitle("Train accuracy by modality subset", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def run_tiny_overfit_experiment(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    modalities: Sequence[str],
    output_dir: Path,
    precomputed: PrecomputedRunData | None = None,
) -> dict[str, Any]:
    build_result = build_fusion_pipeline(config=config, modalities=modalities)
    model = build_binary_fusion_classifier(
        build_result.pipeline,
        dim=int(config["dim"]),
        head_config=build_head_config(config, args),
    )
    freeze_encoder_modules(model)
    device = build_result.device
    model = model.to(device)

    if precomputed is None:
        train_examples = build_labeled_folder_examples(args.overfit_dir, split="train")
        predict_examples = build_labeled_folder_examples(args.predict_dir, split="test")
    else:
        train_examples = precomputed.train_examples
        predict_examples = precomputed.predict_examples
    train_seen_keys = example_label_name_keys(train_examples)
    overlaps = find_label_name_overlaps(train_examples, predict_examples)
    if overlaps:
        print(f"warning: {len(overlaps)} predict videos overlap overfit videos by label/name.")

    if precomputed is None:
        frame_counts = resolve_modality_frame_counts(config, modalities)
        print(f"precompute train features: {len(train_examples)} videos")
        train_result = precompute_feature_items(
            pipeline=model.pipeline,
            examples=train_examples,
            frame_counts=frame_counts,
            image_size=int(config["image_size"]),
            batch_size=args.batch_size,
            config=config,
            modalities=modalities,
            cache_dir=None
            if getattr(args, "no_feature_cache", False)
            else getattr(args, "feature_cache_dir", None),
            cache_label="train",
        )
        print_precompute_timing_summary("precompute train timing", train_result.timing_rows)
        print(f"precompute predict features: {len(predict_examples)} videos")
        predict_result = precompute_feature_items(
            pipeline=model.pipeline,
            examples=predict_examples,
            frame_counts=frame_counts,
            image_size=int(config["image_size"]),
            batch_size=args.batch_size,
            config=config,
            modalities=modalities,
            cache_dir=None
            if getattr(args, "no_feature_cache", False)
            else getattr(args, "feature_cache_dir", None),
            cache_label="predict",
        )
        print_precompute_timing_summary("precompute predict timing", predict_result.timing_rows)
        train_items = train_result.items
        predict_items = predict_result.items
        timing_rows = [*train_result.timing_rows, *predict_result.timing_rows]
    else:
        train_items = precomputed.train_items
        predict_items = precomputed.predict_items
        timing_rows = [*precomputed.train_timing_rows, *precomputed.predict_timing_rows]

    train_loader = build_feature_loader(train_items, batch_size=args.batch_size, shuffle=True)
    predict_loader = build_feature_loader(predict_items, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    metrics: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_path = output_dir / "best.pt"
    print(
        "training: "
        f"epochs={args.epochs} train_videos={len(train_items)} batch_size={args.batch_size}"
    )
    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(model, train_loader, optimizer, loss_fn)
        metrics.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_result.loss:.8f}",
                "train_accuracy": f"{train_result.accuracy:.8f}",
                "train_elapsed_seconds": f"{train_result.elapsed_seconds:.6f}",
                "train_videos_per_second": f"{train_result.videos_per_second:.6f}",
                "train_seconds_per_video": f"{train_result.seconds_per_video:.6f}",
            }
        )
        if train_result.accuracy > best_accuracy:
            best_accuracy = train_result.accuracy
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={train_result.loss:.6f} "
            f"train_accuracy={train_result.accuracy:.4f} "
            f"elapsed={train_result.elapsed_seconds:.3f}s "
            f"per_video={train_result.seconds_per_video:.4f}s "
            f"videos_per_s={train_result.videos_per_second:.2f}",
            flush=True,
        )

    train_accuracy, train_rows = predict_rows(
        model,
        build_feature_loader(train_items, args.batch_size, False),
        seen_keys=train_seen_keys,
    )
    predict_accuracy, predict_rows_output = predict_rows(
        model, predict_loader, seen_keys=train_seen_keys
    )
    seen_predict_summary = summarize_seen_predict_rows(predict_rows_output)
    unseen_predict_summary = summarize_unseen_predict_rows(predict_rows_output)
    if train_accuracy < args.target_train_accuracy:
        print(
            "warning: train accuracy did not reach target "
            f"{args.target_train_accuracy:.4f}; got {train_accuracy:.4f}."
        )
    print(
        format_miss_line(
            "predict_seen_misses",
            seen_predict_summary["predict_seen_misses"],
            seen_predict_summary["predict_seen_count"],
        )
    )
    print(
        format_miss_line(
            "predict_unseen_misses",
            unseen_predict_summary["predict_unseen_misses"],
            unseen_predict_summary["predict_unseen_count"],
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    write_metrics(metrics_path, metrics)
    precompute_timings_path = output_dir / "precompute_timings.csv"
    write_precompute_timings(precompute_timings_path, timing_rows)
    train_accuracy_plot = output_dir / "train_accuracy.png"
    plot_written = write_train_accuracy_plot(
        train_accuracy_plot,
        metrics,
        title=f"Train accuracy: {','.join(modalities)}",
    )
    write_predictions(output_dir / "predictions.csv", [*train_rows, *predict_rows_output])
    summary = {
        "output_dir": str(output_dir),
        "best_checkpoint": str(best_path),
        "metrics_csv": str(metrics_path),
        "precompute_timings_csv": str(precompute_timings_path),
        "precompute_timing": summarize_precompute_timings(timing_rows),
        "train_timing": {
            "total_seconds": sum(float(row["train_elapsed_seconds"]) for row in metrics),
            "mean_epoch_seconds": 0.0
            if not metrics
            else sum(float(row["train_elapsed_seconds"]) for row in metrics) / len(metrics),
            "mean_seconds_per_video": 0.0
            if not metrics
            else sum(float(row["train_seconds_per_video"]) for row in metrics) / len(metrics),
        },
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "target_train_accuracy": args.target_train_accuracy,
        "final_train_accuracy": train_accuracy,
        "predict_accuracy": predict_accuracy,
        "overlap_count": len(overlaps),
        **seen_predict_summary,
        **unseen_predict_summary,
        "train_count": len(train_examples),
        "predict_count": len(predict_examples),
        "device": str(device),
        "modalities": list(model.pipeline.enabled_modalities),
        "train_accuracy_plot": str(train_accuracy_plot) if plot_written else None,
    }
    write_summary(output_dir / "summary.json", summary)
    model.pipeline.close()
    print(f"wrote: {output_dir}")
    if plot_written:
        print(f"wrote: {train_accuracy_plot}")
    return summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    config = build_config(args.config, args.device)
    archive_output_dir = args.output_dir
    run_output_dir = allocate_indexed_run_dir(archive_output_dir)
    if args.feature_cache_dir is None:
        args.feature_cache_dir = archive_output_dir / FEATURE_CACHE_DIRNAME
    print(f"archive_output_dir={archive_output_dir}")
    print(f"run_output_dir={run_output_dir}")
    if args.no_feature_cache:
        print("feature_cache=disabled")
    else:
        print(f"feature_cache_dir={args.feature_cache_dir}")
    base_modalities = resolve_base_modalities(config, args.modalities)
    modality_sets = build_modality_sets(base_modalities, args.modality_permutations)
    if len(modality_sets) > 1:
        print(f"running {len(modality_sets)} modality subset experiments")

    summaries: list[dict[str, Any]] = []
    precomputed = None
    if len(modality_sets) > 1:
        print(f"caching frozen extractor features for base modalities: {','.join(base_modalities)}")
        precomputed = precompute_run_data(args=args, config=config, modalities=base_modalities)

    for modalities in modality_sets:
        if len(modality_sets) == 1 and args.modality_permutations == "none":
            output_dir = run_output_dir
        else:
            output_dir = run_output_dir / "modality_permutations" / modality_set_name(modalities)
        print(f"modalities={','.join(modalities)}")
        summary = run_tiny_overfit_experiment(
            args=args,
            config=config,
            modalities=modalities,
            output_dir=output_dir,
            precomputed=precomputed,
        )
        summaries.append(summary)

    if len(modality_sets) > 1:
        aggregate_plot_path = run_output_dir / "modality_permutations_train_accuracy.png"
        aggregate_plot_written = write_modality_accuracy_plot(aggregate_plot_path, summaries)
        aggregate_path = run_output_dir / "modality_permutations_summary.json"
        aggregate_timing_rows = []
        if precomputed is not None:
            aggregate_timing_rows = [
                *precomputed.train_timing_rows,
                *precomputed.predict_timing_rows,
            ]
        aggregate_timings_path = run_output_dir / "precompute_timings.csv"
        write_precompute_timings(aggregate_timings_path, aggregate_timing_rows)
        write_summary(
            aggregate_path,
            {
                "archive_output_dir": str(archive_output_dir),
                "run_dir": str(run_output_dir),
                "run_index": parse_run_index(run_output_dir),
                "mode": args.modality_permutations,
                "base_modalities": list(base_modalities),
                "precompute_timings_csv": str(aggregate_timings_path),
                "precompute_timing": summarize_precompute_timings(aggregate_timing_rows),
                "train_accuracy_plot": str(aggregate_plot_path) if aggregate_plot_written else None,
                "runs": summaries,
            },
        )
        print(f"wrote: {aggregate_timings_path}")
        print(f"wrote: {aggregate_path}")
        if aggregate_plot_written:
            print(f"wrote: {aggregate_plot_path}")


if __name__ == "__main__":
    main()
