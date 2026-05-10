from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (
    VideoExample,
    build_metadata_real_fake_examples,
    build_real_fake_examples,
    split_metadata_examples,
)
from feature_cache import build_feature_cache_specs
from pipeline import build_fusion_pipeline, load_pipeline_yaml
from scripts.run_iterative_cached_ablation import (
    build_cached_loader,
    build_sharded_cached_loader,
    filter_examples_with_shards,
    freeze_encoder_modules,
    read_sharded_cache_keys,
    resolve_cached_loader_config,
    resolve_sharded_loader_config,
    resolve_video_root,
    select_fully_cached_examples,
    training_run_section,
)
from task_models.generator_multitask_classifier import (
    GeneratorMultitaskClassifier,
    build_generator_multitask_classifier,
)
from training_losses import generator_loss_fn, multitask_loss
from training_metrics import (
    PredictionRecord,
    binary_metrics,
    composite_checkpoint_score,
    generator_confusion,
    generator_metrics,
    known_generator_precision_at_coverage,
    macro_generator_recall,
    prediction_summary,
    worst_generator_recall,
    write_dict_rows,
    write_predictions,
)
from training_samplers import MultitaskGeneratorBatchSampler
from training_targets import (
    GeneratorTargetSpec,
    binary_labels_for_batch,
    build_generator_target_spec,
    fake_generator_counts,
    filter_excluded_generators,
    generator_labels_for_batch,
    grouped_generator_name,
    real_fake_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a read-only cached real/fake + fake-generator multitask model."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def run_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    run = training_run_section(config)
    return run if isinstance(run, Mapping) else {}


def path_value(run: Mapping[str, Any], key: str, default: str | Path | None = None) -> Path | None:
    value = run.get(key, default)
    if value is None:
        return None
    return Path(str(value))


def sequence_value(run: Mapping[str, Any], key: str, default: Sequence[str]) -> tuple[str, ...]:
    value = run.get(key, default)
    if value is None:
        return tuple(default)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"`training.run.{key}` must be a list.")
    return tuple(str(item) for item in value)


def mapping_sequence_value(run: Mapping[str, Any], key: str) -> dict[str, tuple[str, ...]]:
    value = run.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"`training.run.{key}` must be a mapping.")
    result: dict[str, tuple[str, ...]] = {}
    for group_name, raw_names in value.items():
        if isinstance(raw_names, str) or not isinstance(raw_names, Sequence):
            raise ValueError(f"`training.run.{key}.{group_name}` must be a list.")
        result[str(group_name)] = tuple(str(name) for name in raw_names)
    return result


def int_value(run: Mapping[str, Any], key: str, default: int) -> int:
    value = run.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"`training.run.{key}` must be an integer.")
    return value


def float_value(run: Mapping[str, Any], key: str, default: float) -> float:
    value = run.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"`training.run.{key}` must be a number.")
    return float(value)


def resolve_head_config(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("head", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("Config `head` must be a mapping.")
    return dict(raw)


def load_filtered_examples(
    dataset_root: Path,
    excluded_generators: Sequence[str],
    eval_count_per_split: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> list[VideoExample]:
    video_root = resolve_video_root(dataset_root)
    real_dir = video_root / "real"
    fake_dir = video_root / "fake"
    if (real_dir / "meta.csv").is_file() and (fake_dir / "meta.csv").is_file():
        examples = build_metadata_real_fake_examples(real_dir, fake_dir)
        examples = filter_excluded_generators(examples, excluded_generators)
        return split_metadata_examples(
            examples,
            eval_real_count=eval_count_per_split,
            eval_fake_count=eval_count_per_split,
            seed=seed,
        )
    examples = build_real_fake_examples(
        real_dir=real_dir,
        fake_dir=fake_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        eval_real_count=eval_count_per_split,
        eval_fake_count=eval_count_per_split,
    )
    return filter_excluded_generators(examples, excluded_generators)


def split_examples(examples: Sequence[VideoExample]) -> dict[str, list[VideoExample]]:
    return {
        split: [example for example in examples if example.split == split]
        for split in ("train", "val", "test")
    }


def count_fake_groups(
    examples: Sequence[VideoExample],
    target: GeneratorTargetSpec,
) -> dict[str, int]:
    counts: Counter[str] = Counter(
        grouped_generator_name(example, target)
        for example in examples
        if example.class_name == "fake"
    )
    return dict(sorted(counts.items()))


def split_train_fake_group_cap(
    train_examples: Sequence[VideoExample],
    target: GeneratorTargetSpec,
    cap: int,
    seed: int,
) -> tuple[list[VideoExample], list[VideoExample]]:
    if cap <= 0:
        raise ValueError("`training.run.train_fake_group_cap` must be positive.")
    real_examples = [example for example in train_examples if example.class_name == "real"]
    fake_by_group: dict[str, list[VideoExample]] = defaultdict(list)
    for example in train_examples:
        if example.class_name == "fake":
            fake_by_group[grouped_generator_name(example, target)].append(example)

    selected_fakes: list[VideoExample] = []
    holdout_fakes: list[VideoExample] = []
    rng = random.Random(seed)
    for group_name in sorted(fake_by_group):
        group = list(fake_by_group[group_name])
        rng.shuffle(group)
        selected_fakes.extend(group[:cap])
        holdout_fakes.extend(group[cap:])

    selected = [*real_examples, *selected_fakes]
    rng.shuffle(selected)
    rng.shuffle(holdout_fakes)
    selected_paths = {example.path for example in selected}
    overlap = selected_paths.intersection(example.path for example in holdout_fakes)
    if overlap:
        raise ValueError(f"Train/extra fake holdout overlap detected: {sorted(overlap)[0]}")
    return selected, holdout_fakes


def build_readonly_loaders(
    config: Mapping[str, Any],
    examples_by_split: Mapping[str, Sequence[VideoExample]],
    extra_fake_holdout: Sequence[VideoExample],
    all_examples: Sequence[VideoExample],
    dataset_root: Path,
    cache_dir: Path | None,
    sharded_cache_dir: Path | None,
    modalities: Sequence[str],
    train_sampler: MultitaskGeneratorBatchSampler,
    batch_size: int,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    loader_config = resolve_cached_loader_config(config)
    if sharded_cache_dir is not None:
        sharded_config = resolve_sharded_loader_config(config)
        loaders = {
            "train": build_sharded_cached_loader(
                examples_by_split["train"],
                all_cache_examples=all_examples,
                sharded_cache_dir=sharded_cache_dir,
                batch_size=batch_size,
                shuffle=True,
                dataset_root=dataset_root,
                loader_config=loader_config,
                sharded_loader_config=sharded_config,
                batch_sampler=train_sampler,
            ),
            "train_eval": build_sharded_cached_loader(
                examples_by_split["train"],
                all_cache_examples=all_examples,
                sharded_cache_dir=sharded_cache_dir,
                batch_size=batch_size,
                shuffle=False,
                dataset_root=dataset_root,
                loader_config=loader_config,
                sharded_loader_config=sharded_config,
            ),
            "val": build_sharded_cached_loader(
                examples_by_split["val"],
                all_cache_examples=all_examples,
                sharded_cache_dir=sharded_cache_dir,
                batch_size=batch_size,
                shuffle=False,
                dataset_root=dataset_root,
                loader_config=loader_config,
                sharded_loader_config=sharded_config,
            ),
            "test": build_sharded_cached_loader(
                examples_by_split["test"],
                all_cache_examples=all_examples,
                sharded_cache_dir=sharded_cache_dir,
                batch_size=batch_size,
                shuffle=False,
                dataset_root=dataset_root,
                loader_config=loader_config,
                sharded_loader_config=sharded_config,
            ),
        }
        if extra_fake_holdout:
            loaders["extra_fake_holdout"] = build_sharded_cached_loader(
                extra_fake_holdout,
                all_cache_examples=all_examples,
                sharded_cache_dir=sharded_cache_dir,
                batch_size=batch_size,
                shuffle=False,
                dataset_root=dataset_root,
                loader_config=loader_config,
                sharded_loader_config=sharded_config,
            )
        return loaders, {
            "cached_loader": asdict(loader_config),
            "sharded_loader": asdict(sharded_config),
            "sharded_cache_dir": str(sharded_cache_dir),
        }

    if cache_dir is None:
        raise ValueError("Config requires `training.run.cache_dir` or `sharded_cache_dir`.")
    specs = build_feature_cache_specs(config, modalities)
    loaders = {
        "train": build_cached_loader(
            examples_by_split["train"],
            cache_dir,
            specs,
            modalities,
            batch_size=batch_size,
            shuffle=True,
            dataset_root=dataset_root,
            loader_config=loader_config,
            batch_sampler=train_sampler,
        ),
        "train_eval": build_cached_loader(
            examples_by_split["train"],
            cache_dir,
            specs,
            modalities,
            batch_size=batch_size,
            shuffle=False,
            dataset_root=dataset_root,
            loader_config=loader_config,
        ),
        "val": build_cached_loader(
            examples_by_split["val"],
            cache_dir,
            specs,
            modalities,
            batch_size=batch_size,
            shuffle=False,
            dataset_root=dataset_root,
            loader_config=loader_config,
        ),
        "test": build_cached_loader(
            examples_by_split["test"],
            cache_dir,
            specs,
            modalities,
            batch_size=batch_size,
            shuffle=False,
            dataset_root=dataset_root,
            loader_config=loader_config,
        ),
    }
    if extra_fake_holdout:
        loaders["extra_fake_holdout"] = build_cached_loader(
            extra_fake_holdout,
            cache_dir,
            specs,
            modalities,
            batch_size=batch_size,
            shuffle=False,
            dataset_root=dataset_root,
            loader_config=loader_config,
        )
    return loaders, {"cached_loader": asdict(loader_config), "cache_dir": str(cache_dir)}


def filter_cached_examples(
    examples_by_split: Mapping[str, Sequence[VideoExample]],
    all_examples: Sequence[VideoExample],
    dataset_root: Path,
    cache_dir: Path | None,
    sharded_cache_dir: Path | None,
    config: Mapping[str, Any],
    modalities: Sequence[str],
) -> tuple[dict[str, list[VideoExample]], list[VideoExample]]:
    if sharded_cache_dir is not None:
        shard_keys = read_sharded_cache_keys(sharded_cache_dir)
        filtered = {
            split: filter_examples_with_shards(
                examples,
                shard_keys,
                dataset_root,
                label=split,
            )
            for split, examples in examples_by_split.items()
        }
        all_filtered = filter_examples_with_shards(
            all_examples,
            shard_keys,
            dataset_root,
            label="all",
        )
        return filtered, all_filtered
    if cache_dir is None:
        raise ValueError("Config requires `training.run.cache_dir` or `sharded_cache_dir`.")
    specs = build_feature_cache_specs(config, modalities)
    filtered = {
        split: select_fully_cached_examples(
            examples,
            cache_dir=cache_dir,
            specs=specs,
            modalities=modalities,
            dataset_root=dataset_root,
        )[0]
        for split, examples in examples_by_split.items()
    }
    all_filtered = select_fully_cached_examples(
        all_examples,
        cache_dir=cache_dir,
        specs=specs,
        modalities=modalities,
        dataset_root=dataset_root,
    )[0]
    return filtered, all_filtered


def move_tensor_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def train_epoch(
    model: GeneratorMultitaskClassifier,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    generator_loss: torch.nn.Module,
    target: GeneratorTargetSpec,
    binary_weight: float,
    generator_weight: float,
) -> dict[str, float]:
    device = model_device(model)
    model.train()
    total_loss = 0.0
    total_binary_loss = 0.0
    total_generator_loss = 0.0
    total_count = 0
    start = time.perf_counter()
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        labels = binary_labels_for_batch(batch, device)
        fake_indices, generator_labels = generator_labels_for_batch(batch, target, device)
        output = model(move_tensor_batch_to_device(batch, device))
        loss, parts = multitask_loss(
            binary_logits=output.binary_logits,
            binary_labels=labels,
            generator_logits=output.generator_logits.index_select(0, fake_indices),
            generator_labels=generator_labels,
            binary_weight=binary_weight,
            generator_weight=generator_weight,
            generator_loss=generator_loss,
        )
        loss.backward()
        optimizer.step()
        count = int(labels.numel())
        total_count += count
        total_loss += parts["loss"] * count
        total_binary_loss += parts["binary_loss"] * count
        total_generator_loss += parts["generator_loss"] * count
    elapsed = time.perf_counter() - start
    return {
        "loss": total_loss / max(1, total_count),
        "binary_loss": total_binary_loss / max(1, total_count),
        "generator_loss": total_generator_loss / max(1, total_count),
        "elapsed_seconds": elapsed,
    }


def predict(
    model: GeneratorMultitaskClassifier,
    loader: Any,
    target: GeneratorTargetSpec,
    split: str,
) -> list[PredictionRecord]:
    device = model_device(model)
    model.eval()
    rows: list[PredictionRecord] = []
    with torch.no_grad():
        for batch in loader:
            output = model(move_tensor_batch_to_device(batch, device))
            binary_probabilities = output.binary_probabilities.detach().cpu().view(-1)
            binary_predictions = (binary_probabilities >= 0.5).to(dtype=torch.long)
            generator_probabilities = output.generator_probabilities.detach().cpu()
            topk_count = min(2, generator_probabilities.shape[1])
            top_probabilities, top_indices = torch.topk(
                generator_probabilities, k=topk_count, dim=1
            )
            binary_labels = batch["label"].view(-1).to(dtype=torch.long)
            paths = [str(value) for value in batch["path"]]
            class_names = [str(value) for value in batch["class_name"]]
            generator_ids = [str(value or "") for value in batch.get("generator_id", [])]
            if len(generator_ids) != len(paths):
                generator_ids = ["" for _ in paths]
            for index, path in enumerate(paths):
                generator_label = (
                    target.generator_name_for(generator_ids[index])
                    if class_names[index] == "fake"
                    else ""
                )
                prediction_index = int(top_indices[index, 0].item())
                probability = float(top_probabilities[index, 0].item())
                top2_index = (
                    int(top_indices[index, 1].item()) if topk_count > 1 else prediction_index
                )
                top2_probability = (
                    float(top_probabilities[index, 1].item()) if topk_count > 1 else 0.0
                )
                binary_probability = float(binary_probabilities[index].item())
                generator_margin = probability - top2_probability
                rows.append(
                    PredictionRecord(
                        path=path,
                        split=split,
                        class_name=class_names[index],
                        generator_id=generator_ids[index]
                        or ("real" if class_names[index] == "real" else "unknown"),
                        binary_label=int(binary_labels[index].item()),
                        binary_probability=binary_probability,
                        binary_prediction=int(binary_predictions[index].item()),
                        generator_label=generator_label,
                        generator_prediction=target.generator_names[prediction_index],
                        generator_probability=probability,
                        generator_top2_prediction=target.generator_names[top2_index],
                        generator_top2_probability=top2_probability,
                        generator_margin=generator_margin,
                        binary_confidence=max(binary_probability, 1.0 - binary_probability),
                        binary_confidence_label=binary_confidence_label(binary_probability),
                        generator_confidence_label=generator_confidence_label(
                            probability,
                            generator_margin,
                        ),
                    )
                )
    return rows


def write_epoch_metrics(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def binary_confidence_label(probability: float) -> str:
    if probability >= 0.95:
        return "confident_fake"
    if probability >= 0.80:
        return "likely_fake"
    if probability > 0.20:
        return "uncertain"
    if probability > 0.05:
        return "likely_real"
    return "confident_real"


def generator_confidence_label(probability: float, margin: float) -> str:
    if probability >= 0.95 and margin >= 0.50:
        return "confident"
    if probability >= 0.80 and margin >= 0.30:
        return "likely"
    return "uncertain"


def calibrated_output(
    record: PredictionRecord,
    unknown_group_name: str,
    binary_threshold: float,
    generator_confidence_threshold: float,
    generator_margin_threshold: float,
) -> str:
    if record.binary_probability < binary_threshold:
        return "real"
    if record.generator_prediction == unknown_group_name:
        return "fake_unknown_or_other"
    if record.generator_probability < generator_confidence_threshold:
        return "fake_unknown_or_other"
    if record.generator_margin < generator_margin_threshold:
        return "fake_unknown_or_other"
    return f"fake_{record.generator_prediction}"


def apply_calibration(
    records: Sequence[PredictionRecord],
    unknown_group_name: str,
    binary_threshold: float,
    generator_confidence_threshold: float,
    generator_margin_threshold: float,
) -> list[PredictionRecord]:
    return [
        replace(
            record,
            calibrated_output=calibrated_output(
                record=record,
                unknown_group_name=unknown_group_name,
                binary_threshold=binary_threshold,
                generator_confidence_threshold=generator_confidence_threshold,
                generator_margin_threshold=generator_margin_threshold,
            ),
        )
        for record in records
    ]


def calibration_metrics(
    records: Sequence[PredictionRecord],
    unknown_group_name: str,
    binary_threshold: float,
    generator_confidence_threshold: float,
    generator_margin_threshold: float,
) -> dict[str, float]:
    fake_records = [record for record in records if record.binary_label == 1]
    known_records = [
        record for record in fake_records if record.generator_label != unknown_group_name
    ]
    named_known = []
    correct_known = []
    unknown_outputs = 0
    for record in fake_records:
        output = calibrated_output(
            record=record,
            unknown_group_name=unknown_group_name,
            binary_threshold=binary_threshold,
            generator_confidence_threshold=generator_confidence_threshold,
            generator_margin_threshold=generator_margin_threshold,
        )
        if output == "fake_unknown_or_other":
            unknown_outputs += 1
        if record.generator_label == unknown_group_name:
            continue
        expected = f"fake_{record.generator_label}"
        if output.startswith("fake_") and output != "fake_unknown_or_other":
            named_known.append(record)
            if output == expected:
                correct_known.append(record)
    fake_detected = sum(
        1 for record in fake_records if record.binary_probability >= binary_threshold
    )
    precision = len(correct_known) / len(named_known) if named_known else 0.0
    coverage = len(named_known) / len(known_records) if known_records else 0.0
    fake_recall = len(fake_records) and fake_detected / len(fake_records)
    return {
        "binary_threshold": binary_threshold,
        "generator_confidence_threshold": generator_confidence_threshold,
        "generator_margin_threshold": generator_margin_threshold,
        "fake_count": float(len(fake_records)),
        "known_fake_count": float(len(known_records)),
        "fake_recall": float(fake_recall),
        "known_generator_precision": precision,
        "known_generator_coverage": coverage,
        "unknown_or_low_confidence_rate": (
            unknown_outputs / len(fake_records) if fake_records else 0.0
        ),
        "selection_score": precision * min(1.0, coverage / 0.30),
    }


def choose_calibration_policy(
    val_records: Sequence[PredictionRecord],
    unknown_group_name: str,
) -> dict[str, float]:
    candidates: list[dict[str, float]] = []
    for confidence_step in range(50, 96, 5):
        for margin_step in range(5, 51, 5):
            candidates.append(
                calibration_metrics(
                    records=val_records,
                    unknown_group_name=unknown_group_name,
                    binary_threshold=0.5,
                    generator_confidence_threshold=confidence_step / 100.0,
                    generator_margin_threshold=margin_step / 100.0,
                )
            )
    return max(
        candidates,
        key=lambda row: (
            row["selection_score"],
            row["known_generator_precision"],
            row["known_generator_coverage"],
            row["generator_confidence_threshold"],
            row["generator_margin_threshold"],
        ),
    )


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def extra_fake_holdout_summary(
    records: Sequence[PredictionRecord],
    unknown_group_name: str,
    calibration: Mapping[str, float],
) -> dict[str, object]:
    probabilities = [record.binary_probability for record in records]
    fake_detected = [
        record
        for record in records
        if record.binary_probability >= float(calibration["binary_threshold"])
    ]
    by_group: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        by_group[record.generator_label].append(record)
    calibrated_records = apply_calibration(
        records,
        unknown_group_name=unknown_group_name,
        binary_threshold=float(calibration["binary_threshold"]),
        generator_confidence_threshold=float(calibration["generator_confidence_threshold"]),
        generator_margin_threshold=float(calibration["generator_margin_threshold"]),
    )
    unknown_outputs = [
        record
        for record in calibrated_records
        if record.calibrated_output == "fake_unknown_or_other"
    ]
    known_named = [
        record
        for record in calibrated_records
        if record.generator_label != unknown_group_name
        and record.calibrated_output.startswith("fake_")
        and record.calibrated_output != "fake_unknown_or_other"
    ]
    known_correct = [
        record
        for record in known_named
        if record.calibrated_output == f"fake_{record.generator_label}"
    ]
    return {
        "count": len(records),
        "fake_recall": len(fake_detected) / len(records) if records else 0.0,
        "false_negative_rate": 1.0 - (len(fake_detected) / len(records) if records else 0.0),
        "mean_fake_probability": sum(probabilities) / len(probabilities) if probabilities else 0.0,
        "median_fake_probability": median(probabilities),
        "low_confidence_or_unknown_rate": (len(unknown_outputs) / len(records) if records else 0.0),
        "known_generator_precision": (
            len(known_correct) / len(known_named) if known_named else 0.0
        ),
        "known_generator_coverage": (
            len(known_named)
            / max(1, sum(1 for record in records if record.generator_label != unknown_group_name))
        ),
        "fake_recall_by_group": {
            group: sum(
                1
                for record in group_records
                if record.binary_probability >= float(calibration["binary_threshold"])
            )
            / len(group_records)
            for group, group_records in sorted(by_group.items())
        },
    }


def main() -> None:
    args = parse_args()
    config = load_pipeline_yaml(args.config)
    if args.device is not None:
        config["device"] = args.device
    run = run_section(config)

    dataset_root = path_value(run, "dataset_root", "/mnt/d/final_dataset")
    if dataset_root is None:
        raise ValueError("Missing dataset_root.")
    cache_dir = path_value(run, "cache_dir")
    sharded_cache_dir = path_value(run, "sharded_cache_dir")
    output_dir = path_value(run, "output_dir", "runs/generator_multitask")
    if output_dir is None:
        raise ValueError("Missing output_dir.")
    modalities = sequence_value(run, "modalities", config.get("modalities", ()))
    excluded_generators = sequence_value(run, "excluded_generators", ("dreamidv",))
    generator_groups = mapping_sequence_value(run, "generator_groups")
    eval_count_per_split = int_value(run, "eval_count_per_split", 500)
    seed = int_value(run, "seed", 0)
    batch_size = int_value(run, "batch_size", 16)
    real_per_batch = int_value(run, "real_per_batch", batch_size // 2)
    fake_per_batch = int_value(run, "fake_per_batch", batch_size - real_per_batch)
    epochs = int_value(run, "epochs", 20)
    lr = float_value(run, "lr", 1e-3)
    weight_decay = float_value(run, "weight_decay", 1e-2)
    target_quota = int_value(run, "fake_generator_target_quota", 500)
    max_repeat = int_value(run, "fake_generator_max_repeat", 4)
    train_fake_group_cap = int_value(run, "train_fake_group_cap", 500)
    binary_weight = float_value(run, "binary_loss_weight", 1.0)
    generator_weight = float_value(run, "generator_loss_weight", 0.5)
    binary_warmup_epochs = int_value(run, "binary_warmup_epochs", 0)
    generator_loss_type = str(run.get("generator_loss", "class_balanced_focal"))
    focal_beta = float_value(run, "focal_beta", 0.999)
    focal_gamma = float_value(run, "focal_gamma", 2.0)

    torch.manual_seed(seed)
    all_examples = load_filtered_examples(
        dataset_root=dataset_root,
        excluded_generators=excluded_generators,
        eval_count_per_split=eval_count_per_split,
        train_ratio=float_value(run, "train_ratio", 0.8),
        val_ratio=float_value(run, "val_ratio", 0.1),
        seed=seed,
    )
    examples_by_split = split_examples(all_examples)
    examples_by_split, all_cache_examples = filter_cached_examples(
        examples_by_split=examples_by_split,
        all_examples=all_examples,
        dataset_root=dataset_root,
        cache_dir=cache_dir,
        sharded_cache_dir=sharded_cache_dir,
        config=config,
        modalities=modalities,
    )
    target = build_generator_target_spec(all_cache_examples, generator_groups=generator_groups)
    capped_train, extra_fake_holdout = split_train_fake_group_cap(
        examples_by_split["train"],
        target=target,
        cap=train_fake_group_cap,
        seed=seed,
    )
    examples_by_split = {**examples_by_split, "train": capped_train}
    train_sampler = MultitaskGeneratorBatchSampler(
        examples_by_split["train"],
        batch_size=batch_size,
        real_per_batch=real_per_batch,
        fake_per_batch=fake_per_batch,
        target_quota=target_quota,
        max_repeat=max_repeat,
        seed=seed,
        target=target,
    )

    run_config = {
        "config_path": str(args.config),
        "dataset_root": str(dataset_root),
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "sharded_cache_dir": None if sharded_cache_dir is None else str(sharded_cache_dir),
        "output_dir": str(output_dir),
        "modalities": list(modalities),
        "excluded_generators": list(excluded_generators),
        "generator_groups": {key: list(value) for key, value in generator_groups.items()},
        "train_fake_group_cap": train_fake_group_cap,
        "target": target.to_json(),
        "counts": {
            split: {
                "real_fake": real_fake_counts(examples),
                "fake_generators": fake_generator_counts(examples),
                "fake_generator_groups": count_fake_groups(examples, target),
            }
            for split, examples in examples_by_split.items()
        },
        "extra_fake_holdout": {
            "count": len(extra_fake_holdout),
            "raw_fake_generators": fake_generator_counts(extra_fake_holdout),
            "fake_generator_groups": count_fake_groups(extra_fake_holdout, target),
        },
        "sampler": train_sampler.summary(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", run_config)
    if args.dry_run or bool(run.get("dry_run", False)):
        print(json.dumps(run_config, indent=2, sort_keys=True), flush=True)
        return

    loaders, loader_summary = build_readonly_loaders(
        config=config,
        examples_by_split=examples_by_split,
        extra_fake_holdout=extra_fake_holdout,
        all_examples=all_cache_examples,
        dataset_root=dataset_root,
        cache_dir=cache_dir,
        sharded_cache_dir=sharded_cache_dir,
        modalities=modalities,
        train_sampler=train_sampler,
        batch_size=batch_size,
    )
    build_result = build_fusion_pipeline(config=config, modalities=modalities)
    model = build_generator_multitask_classifier(
        build_result.pipeline,
        dim=int(config["dim"]),
        num_generators=target.num_generators,
        head_config=resolve_head_config(config),
    ).to(build_result.device)
    freeze_encoder_modules(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen_loss = generator_loss_fn(
        generator_loss_type,
        counts=fake_generator_counts(examples_by_split["train"], target=target),
        target=target,
        beta=focal_beta,
        gamma=focal_gamma,
    )

    best_score: float | None = None
    best_path = output_dir / "best.pt"
    epoch_rows: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        current_generator_weight = 0.0 if epoch <= binary_warmup_epochs else generator_weight
        train_result = train_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            generator_loss=gen_loss,
            target=target,
            binary_weight=binary_weight,
            generator_weight=current_generator_weight,
        )
        val_records = predict(model, loaders["val"], target, split="val")
        val_binary = binary_metrics(val_records)
        val_generator = generator_metrics(val_records)
        val_prediction_summary = prediction_summary(val_records)
        macro_recall = macro_generator_recall(val_generator)
        worst_recall = worst_generator_recall(val_generator)
        known_quality = known_generator_precision_at_coverage(
            val_records,
            unknown_group_name=target.unknown_group_name,
        )
        score = composite_checkpoint_score(
            val_binary,
            macro_recall,
            float(known_quality["known_generator_precision_at_coverage"]),
        )
        improved = best_score is None or score > best_score
        if improved:
            best_score = score
            torch.save(model.state_dict(), best_path)
        epoch_rows.append(
            {
                "epoch": epoch,
                **{key: f"{value:.8f}" for key, value in train_result.items()},
                "val_binary_balanced_accuracy": f"{val_binary.balanced_accuracy:.8f}",
                "val_binary_recall": f"{val_binary.recall:.8f}",
                "val_binary_predicted_fake_rate": (
                    f"{float(val_prediction_summary['binary_predicted_fake_rate']):.8f}"
                ),
                "val_real_binary_probability_mean": (
                    f"{float(val_prediction_summary['real_binary_probability_mean']):.8f}"
                ),
                "val_fake_binary_probability_mean": (
                    f"{float(val_prediction_summary['fake_binary_probability_mean']):.8f}"
                ),
                "val_macro_generator_recall": f"{macro_recall:.8f}",
                "val_worst_generator_recall": f"{worst_recall:.8f}",
                "val_known_generator_precision": (
                    f"{known_quality['known_generator_precision']:.8f}"
                ),
                "val_known_generator_coverage": (
                    f"{known_quality['known_generator_coverage']:.8f}"
                ),
                "val_known_generator_precision_at_coverage": (
                    f"{known_quality['known_generator_precision_at_coverage']:.8f}"
                ),
                "generator_loss_weight": f"{current_generator_weight:.8f}",
                "checkpoint_score": f"{score:.8f}",
                "best_checkpoint": int(improved),
            }
        )
        write_epoch_metrics(output_dir / "metrics.csv", epoch_rows)
        print(
            f"epoch={epoch}/{epochs} loss={train_result['loss']:.4f} "
            f"val_binary_bal_acc={val_binary.balanced_accuracy:.4f} "
            f"val_pred_fake_rate={float(val_prediction_summary['binary_predicted_fake_rate']):.4f} "
            f"val_real_p={float(val_prediction_summary['real_binary_probability_mean']):.4f} "
            f"val_fake_p={float(val_prediction_summary['fake_binary_probability_mean']):.4f} "
            f"val_macro_gen_recall={macro_recall:.4f} "
            f"val_worst_gen_recall={worst_recall:.4f} score={score:.4f} "
            f"known_gen_pac={known_quality['known_generator_precision_at_coverage']:.4f} "
            f"best={int(improved)}",
            flush=True,
        )

    model.load_state_dict(
        torch.load(best_path, map_location=build_result.device, weights_only=False)
    )
    records = [
        *predict(model, loaders["train_eval"], target, split="train"),
        *predict(model, loaders["val"], target, split="val"),
        *predict(model, loaders["test"], target, split="test"),
    ]
    val_records_for_calibration = [record for record in records if record.split == "val"]
    calibration = choose_calibration_policy(
        val_records_for_calibration,
        unknown_group_name=target.unknown_group_name,
    )
    records = apply_calibration(
        records,
        unknown_group_name=target.unknown_group_name,
        binary_threshold=float(calibration["binary_threshold"]),
        generator_confidence_threshold=float(calibration["generator_confidence_threshold"]),
        generator_margin_threshold=float(calibration["generator_margin_threshold"]),
    )
    extra_fake_holdout_records: list[PredictionRecord] = []
    if extra_fake_holdout:
        extra_fake_holdout_records = apply_calibration(
            predict(
                model,
                loaders["extra_fake_holdout"],
                target,
                split="extra_fake_holdout",
            ),
            unknown_group_name=target.unknown_group_name,
            binary_threshold=float(calibration["binary_threshold"]),
            generator_confidence_threshold=float(calibration["generator_confidence_threshold"]),
            generator_margin_threshold=float(calibration["generator_margin_threshold"]),
        )
    split_summaries: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_records = [record for record in records if record.split == split]
        gen_rows = generator_metrics(split_records)
        known_quality = known_generator_precision_at_coverage(
            split_records,
            unknown_group_name=target.unknown_group_name,
        )
        split_summaries[split] = {
            "binary_metrics": asdict(binary_metrics(split_records)),
            "generator_metrics": gen_rows,
            "macro_generator_recall": macro_generator_recall(gen_rows),
            "worst_generator_recall": worst_generator_recall(gen_rows),
            "known_generator_quality": known_quality,
        }
    write_predictions(output_dir / "predictions.csv", records)
    if extra_fake_holdout_records:
        write_predictions(
            output_dir / "extra_fake_holdout_predictions.csv",
            extra_fake_holdout_records,
        )
        write_json(
            output_dir / "extra_fake_holdout_summary.json",
            {
                "calibration": calibration,
                **extra_fake_holdout_summary(
                    extra_fake_holdout_records,
                    unknown_group_name=target.unknown_group_name,
                    calibration=calibration,
                ),
            },
        )
    write_dict_rows(
        output_dir / "generator_metrics.csv",
        [
            {"split": split, **row}
            for split in ("train", "val", "test")
            for row in split_summaries[split]["generator_metrics"]
        ],
    )
    write_dict_rows(output_dir / "generator_confusion.csv", generator_confusion(records))
    write_json(
        output_dir / "summary.json",
        {
            **run_config,
            **loader_summary,
            "epochs": epochs,
            "best_checkpoint": str(best_path),
            "best_checkpoint_score": best_score,
            "calibration": calibration,
            "extra_fake_holdout_summary": (
                extra_fake_holdout_summary(
                    extra_fake_holdout_records,
                    unknown_group_name=target.unknown_group_name,
                    calibration=calibration,
                )
                if extra_fake_holdout_records
                else None
            ),
            "split_summaries": split_summaries,
        },
    )
    build_result.pipeline.close()
    print(f"wrote: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
