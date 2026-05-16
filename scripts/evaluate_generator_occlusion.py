from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import build_fusion_pipeline
from task_models.generator_multitask_classifier import build_generator_multitask_classifier
from training_metrics import (
    binary_metrics,
    generator_metrics,
    macro_generator_recall,
    worst_generator_recall,
)
from training_samplers import MultitaskGeneratorBatchSampler
from training_targets import build_generator_target_spec

from scripts.run_generator_multitask_training import (
    build_readonly_loaders,
    filter_cached_examples,
    float_value,
    int_value,
    load_filtered_examples,
    load_pipeline_yaml,
    mapping_sequence_value,
    path_value,
    predict,
    resolve_head_config,
    run_section,
    sequence_value,
    split_examples,
    split_train_fake_group_cap,
)


DEFAULT_SPLITS = ("val", "test")
ROW_FIELDS = (
    "split",
    "occluded_modality",
    "accuracy",
    "balanced_accuracy",
    "fake_recall",
    "real_specificity",
    "precision",
    "f1",
    "false_positive",
    "false_negative",
    "macro_generator_recall",
    "worst_generator_recall",
)


class OccludedLoader:
    def __init__(self, loader: Any, dropped_modality: str | None) -> None:
        self.loader = loader
        self.dropped_modality = dropped_modality

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        for batch in self.loader:
            if self.dropped_modality is None:
                yield batch
            else:
                yield {**batch, "dropped_modalities": (self.dropped_modality,)}

    def __len__(self) -> int:
        return len(self.loader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a generator-multitask checkpoint while dropping each configured modality."
        )
    )
    parser.add_argument("config", type=Path, help="Training YAML used for the checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Model weights to evaluate.")
    parser.add_argument("output_csv", type=Path, help="CSV file to write occlusion metrics into.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test", "extra_fake_holdout"),
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to evaluate. `train` uses the deterministic train_eval loader.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        help="Override modalities from training.run.modalities/config.modalities.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def split_loader_key(split: str) -> str:
    return "train_eval" if split == "train" else split


def build_rows(
    model: torch.nn.Module,
    loaders: Mapping[str, Any],
    target: Any,
    modalities: Sequence[str],
    splits: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in splits:
        loader_key = split_loader_key(split)
        if loader_key not in loaders:
            raise ValueError(f"Requested split is unavailable: {split}")
        for dropped_modality in (None, *modalities):
            records = predict(
                model,
                OccludedLoader(loaders[loader_key], dropped_modality),
                target,
                split=split,
            )
            binary = binary_metrics(records)
            generator = generator_metrics(records)
            rows.append(
                {
                    "split": split,
                    "occluded_modality": "<none>"
                    if dropped_modality is None
                    else dropped_modality,
                    "accuracy": binary.accuracy,
                    "balanced_accuracy": binary.balanced_accuracy,
                    "fake_recall": binary.recall,
                    "real_specificity": binary.specificity,
                    "precision": binary.precision,
                    "f1": binary.f1,
                    "false_positive": binary.false_positive,
                    "false_negative": binary.false_negative,
                    "macro_generator_recall": macro_generator_recall(generator),
                    "worst_generator_recall": worst_generator_recall(generator),
                }
            )
    return rows


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        print(
            "split={split} occluded={occluded_modality} "
            "balanced_accuracy={balanced_accuracy:.6f} "
            "fake_recall={fake_recall:.6f} "
            "real_specificity={real_specificity:.6f} "
            "macro_generator_recall={macro_generator_recall:.6f}".format(**row),
            flush=True,
        )


def main() -> None:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    output_csv = resolve_repo_path(args.output_csv)

    config = load_pipeline_yaml(config_path)
    if args.device is not None:
        config["device"] = args.device
    run = run_section(config)

    dataset_root = path_value(run, "dataset_root", "/mnt/d/final_dataset")
    if dataset_root is None:
        raise ValueError("Missing training.run.dataset_root.")
    cache_dir = path_value(run, "cache_dir")
    sharded_cache_dir = path_value(run, "sharded_cache_dir")
    modalities = tuple(
        args.modalities
        if args.modalities is not None
        else sequence_value(run, "modalities", config.get("modalities", ()))
    )
    if not modalities:
        raise ValueError("No modalities configured.")

    seed = int_value(run, "seed", 0)
    batch_size = args.batch_size or int_value(run, "batch_size", 16)
    real_per_batch = int_value(run, "real_per_batch", batch_size // 2)
    fake_per_batch = int_value(run, "fake_per_batch", batch_size - real_per_batch)
    target_quota = int_value(run, "fake_generator_target_quota", 500)
    max_repeat = int_value(run, "fake_generator_max_repeat", 4)
    train_fake_group_cap = int_value(run, "train_fake_group_cap", 500)

    torch.manual_seed(seed)
    all_examples = load_filtered_examples(
        dataset_root=dataset_root,
        excluded_generators=sequence_value(run, "excluded_generators", ("dreamidv",)),
        eval_count_per_split=int_value(run, "eval_count_per_split", 500),
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
    target = build_generator_target_spec(
        all_cache_examples,
        generator_groups=mapping_sequence_value(run, "generator_groups"),
    )
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

    loaders, _loader_summary = build_readonly_loaders(
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
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=build_result.device, weights_only=False)
    )
    model.eval()

    try:
        rows = build_rows(
            model=model,
            loaders=loaders,
            target=target,
            modalities=modalities,
            splits=tuple(args.splits),
        )
        write_rows(output_csv, rows)
        print_rows(rows)
        print(f"wrote: {output_csv}", flush=True)
    finally:
        build_result.pipeline.close()


if __name__ == "__main__":
    main()
