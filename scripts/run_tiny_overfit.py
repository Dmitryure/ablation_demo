from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (  # noqa: E402
    LabeledVideoDataset,
    VideoExample,
    build_labeled_folder_examples,
    collate_labeled_video_batch,
)
from pipeline import build_fusion_pipeline, load_pipeline_yaml  # noqa: E402
from task_model import BinaryFusionClassifier, build_binary_fusion_classifier  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "registry_fusion.yaml"
DEFAULT_OVERFIT_DIR = PROJECT_ROOT / "tests" / "overfit_videos"
DEFAULT_PREDICT_DIR = PROJECT_ROOT / "tests" / "predict_videos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests" / "overfit_runs"


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


class PrecomputedFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(self, items: Sequence[Mapping[str, Any]]) -> None:
        self.items = [dict(item) for item in items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit binary fake/real classifier on tiny local videos.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overfit-dir", type=Path, default=DEFAULT_OVERFIT_DIR)
    parser.add_argument("--predict-dir", type=Path, default=DEFAULT_PREDICT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_config(config_path: Path, device: str | None) -> dict[str, Any]:
    config = load_pipeline_yaml(config_path)
    if device is not None:
        config["device"] = device
    return config


def resolve_base_modalities(config: Mapping[str, Any], requested: Sequence[str] | None) -> tuple[str, ...]:
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


def precompute_feature_items(
    pipeline: torch.nn.Module,
    examples: Sequence[VideoExample],
    num_frames: int,
    image_size: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    dataset = LabeledVideoDataset(examples=examples, num_frames=num_frames, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_labeled_video_batch,
    )
    items: list[dict[str, Any]] = []
    pipeline.eval()
    with torch.no_grad():
        for raw_batch in loader:
            feature_batch = pipeline.prepare_features(raw_batch)
            items.extend(split_feature_batch(feature_batch, raw_batch))
    return items


def precompute_run_data(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    modalities: Sequence[str],
) -> PrecomputedRunData:
    build_result = build_fusion_pipeline(config=config, modalities=modalities)
    pipeline = build_result.pipeline
    train_examples = build_labeled_folder_examples(args.overfit_dir, split="train")
    predict_examples = build_labeled_folder_examples(args.predict_dir, split="test")

    print(f"precompute train features: {len(train_examples)} videos")
    train_items = precompute_feature_items(
        pipeline=pipeline,
        examples=train_examples,
        num_frames=int(config["frames"]),
        image_size=int(config["image_size"]),
        batch_size=args.batch_size,
    )
    print(f"precompute predict features: {len(predict_examples)} videos")
    predict_items = precompute_feature_items(
        pipeline=pipeline,
        examples=predict_examples,
        num_frames=int(config["frames"]),
        image_size=int(config["image_size"]),
        batch_size=args.batch_size,
    )
    pipeline.close()
    if build_result.device.type == "cuda":
        torch.cuda.empty_cache()
    return PrecomputedRunData(
        train_examples=train_examples,
        predict_examples=predict_examples,
        train_items=train_items,
        predict_items=predict_items,
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
) -> tuple[float, float]:
    device = model_device(model)
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
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
    return total_loss / total_count, total_correct / total_count


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
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "train_accuracy"))
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
    model = build_binary_fusion_classifier(build_result.pipeline, dim=int(config["dim"]))
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
        print(f"precompute train features: {len(train_examples)} videos")
        train_items = precompute_feature_items(
            pipeline=model.pipeline,
            examples=train_examples,
            num_frames=int(config["frames"]),
            image_size=int(config["image_size"]),
            batch_size=args.batch_size,
        )
        print(f"precompute predict features: {len(predict_examples)} videos")
        predict_items = precompute_feature_items(
            pipeline=model.pipeline,
            examples=predict_examples,
            num_frames=int(config["frames"]),
            image_size=int(config["image_size"]),
            batch_size=args.batch_size,
        )
    else:
        train_items = precomputed.train_items
        predict_items = precomputed.predict_items

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
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, loss_fn)
        metrics.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.8f}",
                "train_accuracy": f"{train_accuracy:.8f}",
            }
        )
        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
        if should_log_epoch(epoch, args.epochs):
            print(f"epoch={epoch} train_loss={train_loss:.6f} train_accuracy={train_accuracy:.4f}")

    train_accuracy, train_rows = predict_rows(
        model,
        build_feature_loader(train_items, args.batch_size, False),
        seen_keys=train_seen_keys,
    )
    predict_accuracy, predict_rows_output = predict_rows(model, predict_loader, seen_keys=train_seen_keys)
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
    train_accuracy_plot = output_dir / "train_accuracy.png"
    plot_written = write_train_accuracy_plot(
        train_accuracy_plot,
        metrics,
        title=f"Train accuracy: {','.join(modalities)}",
    )
    write_predictions(output_dir / "predictions.csv", [*train_rows, *predict_rows_output])
    summary = {
        "best_checkpoint": str(best_path),
        "metrics_csv": str(metrics_path),
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
            output_dir = args.output_dir
        else:
            output_dir = args.output_dir / "modality_permutations" / modality_set_name(modalities)
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
        aggregate_plot_path = args.output_dir / "modality_permutations_train_accuracy.png"
        aggregate_plot_written = write_modality_accuracy_plot(aggregate_plot_path, summaries)
        aggregate_path = args.output_dir / "modality_permutations_summary.json"
        write_summary(
            aggregate_path,
            {
                "mode": args.modality_permutations,
                "base_modalities": list(base_modalities),
                "train_accuracy_plot": str(aggregate_plot_path) if aggregate_plot_written else None,
                "runs": summaries,
            },
        )
        print(f"wrote: {aggregate_path}")
        if aggregate_plot_written:
            print(f"wrote: {aggregate_plot_path}")


if __name__ == "__main__":
    main()
