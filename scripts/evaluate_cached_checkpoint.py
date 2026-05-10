from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import VideoExample, build_real_fake_examples, load_dataset_manifest, summarize_examples
from feature_cache import cache_example_key
from scripts.run_iterative_cached_ablation import (
    build_sharded_cached_loader,
    class_counts,
    filter_examples_with_shards,
    prediction_generator_metrics,
    prediction_rows_metrics,
    predict_rows,
    read_sharded_cache_keys,
    resolve_cached_loader_config,
    resolve_sharded_loader_config,
    resolve_video_root,
    video_metadata_summary,
    write_generator_metrics,
    write_json,
    write_predictions,
)
from pipeline import build_fusion_pipeline
from task_models import build_binary_fusion_classifier


SUBSETS = ("leftover-val-test", "leftover-all", "leftover-train", "all-val-test")
CLASS_FILTERS = ("fake", "real", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a cached ablation checkpoint on examples outside its run manifest."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--subset", choices=SUBSETS, default="leftover-val-test")
    parser.add_argument("--class-filter", choices=CLASS_FILTERS, default="fake")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser.parse_args()


def load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed run config: {path}")
    return payload


def default_checkpoint(run_dir: Path) -> Path:
    summaries = sorted(run_dir.glob("train_*/**/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No round summary found under {run_dir}")
    with summaries[-1].open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    checkpoint = payload.get("best_checkpoint")
    if not checkpoint:
        raise ValueError(f"Round summary missing best_checkpoint: {summaries[-1]}")
    return Path(str(checkpoint))


def selected_keys(run_dir: Path, dataset_root: Path) -> set[str]:
    manifest_path = run_dir / "manifest.csv"
    return {
        cache_example_key(example, dataset_root)
        for example in load_dataset_manifest(manifest_path)
    }


def split_matches_subset(example: VideoExample, subset: str) -> bool:
    if subset in {"leftover-val-test", "all-val-test"}:
        return example.split in {"val", "test"}
    if subset == "leftover-train":
        return example.split == "train"
    if subset == "leftover-all":
        return True
    raise ValueError(f"Unsupported subset: {subset}")


def class_matches_filter(example: VideoExample, class_filter: str) -> bool:
    return class_filter == "all" or example.class_name == class_filter


def select_eval_examples(
    examples: list[VideoExample],
    run_dir: Path,
    dataset_root: Path,
    subset: str,
    class_filter: str,
) -> list[VideoExample]:
    seen_keys = selected_keys(run_dir, dataset_root)
    selected: list[VideoExample] = []
    for example in examples:
        key = cache_example_key(example, dataset_root)
        if subset.startswith("leftover-") and key in seen_keys:
            continue
        if not split_matches_subset(example, subset):
            continue
        if not class_matches_filter(example, class_filter):
            continue
        selected.append(example)
    return selected


def resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_all_cached_examples(
    run_config: dict[str, Any],
    dataset_root: Path,
    sharded_cache_dir: Path,
) -> list[VideoExample]:
    resolved_run = run_config.get("resolved_training_run", {})
    if not isinstance(resolved_run, dict):
        resolved_run = {}
    video_root = resolve_video_root(dataset_root)
    examples = build_real_fake_examples(
        real_dir=video_root / "real",
        fake_dir=video_root / "fake",
        train_ratio=float(resolved_run.get("train_ratio", 0.8)),
        val_ratio=float(resolved_run.get("val_ratio", 0.1)),
        seed=int(resolved_run.get("seed", run_config.get("seed", 0))),
    )
    sharded_keys = read_sharded_cache_keys(sharded_cache_dir)
    return filter_examples_with_shards(
        examples,
        sharded_keys,
        dataset_root,
        label="eval_all_cached",
    )


def load_model(
    run_config: dict[str, Any],
    checkpoint: Path,
    modalities: tuple[str, ...],
    device_override: str | None,
) -> tuple[Any, torch.device]:
    config = dict(run_config["pipeline_config"])
    if device_override is not None:
        config["device"] = device_override
    build_result = build_fusion_pipeline(config=config, modalities=modalities)
    model = build_binary_fusion_classifier(
        build_result.pipeline,
        dim=int(config["dim"]),
        head_config=config.get("head"),
    )
    model = model.to(build_result.device)
    state = torch.load(checkpoint, map_location=build_result.device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model, build_result.device


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    run_config = load_run_config(run_dir)
    resolved_run = run_config.get("resolved_training_run", {})
    if not isinstance(resolved_run, dict):
        resolved_run = {}

    dataset_root = Path(str(run_config["dataset_root"]))
    sharded_cache_dir = resolve_path(str(run_config.get("sharded_cache_dir") or "")) or resolve_path(
        str(resolved_run.get("sharded_cache_dir") or "")
    )
    if sharded_cache_dir is None:
        raise ValueError("This evaluator currently requires a sharded cache run.")

    modalities = tuple(str(item) for item in run_config["base_modalities"])
    checkpoint = args.checkpoint or default_checkpoint(run_dir)
    output_dir = args.output_dir or (
        run_dir / "leftover_eval" / f"{args.subset}_{args.class_filter}"
    )

    all_cached_examples = build_all_cached_examples(run_config, dataset_root, sharded_cache_dir)
    eval_examples = select_eval_examples(
        all_cached_examples,
        run_dir=run_dir,
        dataset_root=dataset_root,
        subset=args.subset,
        class_filter=args.class_filter,
    )
    if not eval_examples:
        raise ValueError(
            f"No examples selected for subset={args.subset} class_filter={args.class_filter}."
        )

    model, _device = load_model(
        run_config,
        checkpoint=checkpoint,
        modalities=modalities,
        device_override=args.device,
    )
    loader = build_sharded_cached_loader(
        examples=eval_examples,
        all_cache_examples=all_cached_examples,
        sharded_cache_dir=sharded_cache_dir,
        batch_size=args.batch_size,
        shuffle=False,
        dataset_root=dataset_root,
        loader_config=resolve_cached_loader_config(run_config["pipeline_config"]),
        sharded_loader_config=resolve_sharded_loader_config(run_config["pipeline_config"]),
        seed=int(resolved_run.get("seed", run_config.get("seed", 0))),
    )
    accuracy, prediction_rows = predict_rows(
        model,
        loader,
        progress_label=f"eval {args.subset}/{args.class_filter}",
        progress_every=args.progress_every,
    )
    metrics = prediction_rows_metrics(prediction_rows)
    generator_metrics = prediction_generator_metrics(prediction_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir / "predictions.csv", prediction_rows)
    write_generator_metrics(output_dir / "generator_metrics.csv", generator_metrics)
    write_json(
        output_dir / "summary.json",
        {
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "subset": args.subset,
            "class_filter": args.class_filter,
            "modalities": list(modalities),
            "count": len(eval_examples),
            "class_counts": class_counts(eval_examples),
            "accuracy": accuracy,
            "metrics": asdict(metrics),
            "generator_metrics": generator_metrics,
            "video_metadata": video_metadata_summary(eval_examples),
            "all_cached_count": len(all_cached_examples),
            "selected_manifest_count": len(selected_keys(run_dir, dataset_root)),
            "predictions_csv": str(output_dir / "predictions.csv"),
            "generator_metrics_csv": str(output_dir / "generator_metrics.csv"),
        },
    )
    model.pipeline.close()
    print(
        f"eval: subset={args.subset} class_filter={args.class_filter} "
        f"count={len(eval_examples)} accuracy={accuracy:.4f} "
        f"balanced_accuracy={metrics.balanced_accuracy:.4f} f1={metrics.f1:.4f}",
        flush=True,
    )
    print(f"wrote: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
