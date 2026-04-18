from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import LabeledVideoDataset, collate_labeled_video_batch, load_dataset_manifest
from prediction import (
    ClipRealFakePredictor,
    build_binary_classification_loss,
    build_prediction_model,
    load_prediction_yaml,
)


@dataclass(frozen=True)
class RunConfig:
    epochs: int
    batch_size: int
    num_workers: int
    weight_decay: float
    output_dir: Path
    resume_from: Path | None = None
    save_every_epoch: bool = False
    log_interval: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("`epochs` must be positive.")
        if self.batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")
        if self.num_workers < 0:
            raise ValueError("`num_workers` must be non-negative.")
        if self.weight_decay < 0.0:
            raise ValueError("`weight_decay` must be non-negative.")
        if self.log_interval < 0:
            raise ValueError("`log_interval` must be non-negative.")


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    num_examples: int


@dataclass(frozen=True)
class TrainingResult:
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    history: tuple[dict[str, Any], ...]
    best_epoch: int


@dataclass(frozen=True)
class LoadedTrainingResult:
    model: ClipRealFakePredictor
    config: dict[str, Any]
    enabled_modalities: tuple[str, ...]
    optimizer: torch.optim.Optimizer | None
    scheduler: Any | None
    epoch: int
    best_metrics: EpochMetrics | None


def _log(message: str) -> None:
    print(message, flush=True)


def _labels_from_batch(batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    labels = batch.get("label")
    if not isinstance(labels, torch.Tensor):
        raise ValueError("Batch is missing tensor `label`.")
    return labels.to(device=device, dtype=torch.float32)


def _binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = labels == 1.0
    negative = labels == 0.0
    num_positive = int(positive.sum())
    num_negative = int(negative.sum())
    if num_positive == 0 or num_negative == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)

    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[start:end] = average_rank
        start = end

    original_order_ranks = np.empty_like(ranks)
    original_order_ranks[order] = ranks
    positive_rank_sum = float(original_order_ranks[positive].sum())
    auc = (positive_rank_sum - num_positive * (num_positive + 1) / 2.0) / (num_positive * num_negative)
    return float(auc)


def _compute_epoch_metrics(
    total_loss: float,
    labels: list[torch.Tensor],
    probs: list[torch.Tensor],
) -> EpochMetrics:
    if not labels or not probs:
        return EpochMetrics(
            loss=0.0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            roc_auc=None,
            num_examples=0,
        )

    all_labels = torch.cat(labels, dim=0).reshape(-1).float()
    all_probs = torch.cat(probs, dim=0).reshape(-1).float()
    predictions = (all_probs >= 0.5).float()
    num_examples = int(all_labels.numel())
    mean_loss = float(total_loss / max(num_examples, 1))

    true_positive = int(((predictions == 1.0) & (all_labels == 1.0)).sum().item())
    true_negative = int(((predictions == 0.0) & (all_labels == 0.0)).sum().item())
    false_positive = int(((predictions == 1.0) & (all_labels == 0.0)).sum().item())
    false_negative = int(((predictions == 0.0) & (all_labels == 1.0)).sum().item())

    accuracy = float((true_positive + true_negative) / max(num_examples, 1))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = float(true_positive / precision_denominator) if precision_denominator else 0.0
    recall = float(true_positive / recall_denominator) if recall_denominator else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if precision + recall else 0.0
    roc_auc = _binary_roc_auc(
        labels=all_labels.numpy(),
        scores=all_probs.numpy(),
    )
    return EpochMetrics(
        loss=mean_loss,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        num_examples=num_examples,
    )


def _serialize_epoch_metrics(metrics: EpochMetrics | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return {
        "loss": float(metrics.loss),
        "accuracy": float(metrics.accuracy),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1": float(metrics.f1),
        "roc_auc": None if metrics.roc_auc is None else float(metrics.roc_auc),
        "num_examples": int(metrics.num_examples),
    }


def _deserialize_epoch_metrics(payload: Mapping[str, Any] | None) -> EpochMetrics | None:
    if payload is None:
        return None
    return EpochMetrics(
        loss=float(payload["loss"]),
        accuracy=float(payload["accuracy"]),
        precision=float(payload["precision"]),
        recall=float(payload["recall"]),
        f1=float(payload["f1"]),
        roc_auc=None if payload["roc_auc"] is None else float(payload["roc_auc"]),
        num_examples=int(payload["num_examples"]),
    )


def _serialize_run_config(run_config: RunConfig) -> dict[str, Any]:
    payload = asdict(run_config)
    payload["output_dir"] = str(run_config.output_dir)
    payload["resume_from"] = None if run_config.resume_from is None else str(run_config.resume_from)
    return payload


def _deserialize_run_config(payload: Mapping[str, Any]) -> RunConfig:
    return RunConfig(
        epochs=int(payload["epochs"]),
        batch_size=int(payload["batch_size"]),
        num_workers=int(payload["num_workers"]),
        weight_decay=float(payload["weight_decay"]),
        output_dir=Path(payload["output_dir"]),
        resume_from=None if payload.get("resume_from") is None else Path(str(payload["resume_from"])),
        save_every_epoch=bool(payload.get("save_every_epoch", False)),
        log_interval=int(payload.get("log_interval", 0)),
    )


def _history_entry(epoch: int, train_metrics: EpochMetrics, val_metrics: EpochMetrics) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "train": _serialize_epoch_metrics(train_metrics),
        "val": _serialize_epoch_metrics(val_metrics),
    }


def _checkpoint_payload(
    model: ClipRealFakePredictor,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    enabled_modalities: Sequence[str],
    run_config: RunConfig,
    history: Sequence[dict[str, Any]],
    epoch: int,
    best_epoch: int,
    best_metrics: EpochMetrics | None,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "config": copy.deepcopy(dict(config)),
        "enabled_modalities": list(enabled_modalities),
        "run_config": _serialize_run_config(run_config),
        "history": list(history),
        "best_metrics": _serialize_epoch_metrics(best_metrics),
    }


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)
    return path


def _load_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Training checkpoint must deserialize to a mapping.")
    return payload


def _build_dataloaders(
    config: Mapping[str, Any],
    manifest_path: str | Path,
    run_config: RunConfig,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_examples = load_dataset_manifest(manifest_path, split="train")
    val_examples = load_dataset_manifest(manifest_path, split="val")
    if not train_examples:
        raise ValueError("Manifest must contain at least one `train` example.")
    if not val_examples:
        raise ValueError("Manifest must contain at least one `val` example.")

    generator = torch.Generator().manual_seed(int(config.get("seed", 0)))
    num_frames = int(config["frames"])
    image_size = int(config.get("image_size", 224))

    train_dataset = LabeledVideoDataset(
        examples=train_examples,
        num_frames=num_frames,
        image_size=image_size,
    )
    val_dataset = LabeledVideoDataset(
        examples=val_examples,
        num_frames=num_frames,
        image_size=image_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=run_config.batch_size,
        shuffle=True,
        num_workers=run_config.num_workers,
        collate_fn=collate_labeled_video_batch,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=run_config.batch_size,
        shuffle=False,
        num_workers=run_config.num_workers,
        collate_fn=collate_labeled_video_batch,
    )
    return train_loader, val_loader


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _batch_count(dataloader: DataLoader[Any]) -> int | None:
    try:
        return len(dataloader)
    except TypeError:
        return None


def _batch_summary(batch: Mapping[str, Any]) -> str:
    summary_parts: list[str] = []
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            summary_parts.append(f"{key}=tensor{tuple(value.shape)}")
        elif isinstance(value, list):
            summary_parts.append(f"{key}=list[{len(value)}]")
        else:
            summary_parts.append(f"{key}={type(value).__name__}")
    return ", ".join(summary_parts)


def _format_feature_timings(model: ClipRealFakePredictor) -> str:
    if not model.last_feature_timings:
        return ""
    return " ".join(
        f"{name}_extract={duration:.3f}s" for name, duration in model.last_feature_timings.items()
    )


def _train_one_epoch(
    model: ClipRealFakePredictor,
    dataloader: DataLoader[Any],
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int | None = None,
    log_interval: int = 0,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    labels: list[torch.Tensor] = []
    probs: list[torch.Tensor] = []
    total_batches = _batch_count(dataloader)
    iterator = iter(dataloader)
    batch_index = 0

    while True:
        data_wait_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        data_wait_time = time.perf_counter() - data_wait_start
        batch_index += 1
        batch_labels = _labels_from_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        _sync_device(device)
        forward_start = time.perf_counter()
        output = model(batch)
        loss = criterion(output.logits, batch_labels)
        _sync_device(device)
        forward_time = time.perf_counter() - forward_start

        _sync_device(device)
        backward_start = time.perf_counter()
        loss.backward()
        _sync_device(device)
        backward_time = time.perf_counter() - backward_start

        _sync_device(device)
        step_start = time.perf_counter()
        optimizer.step()
        _sync_device(device)
        step_time = time.perf_counter() - step_start

        batch_size = int(batch_labels.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        labels.append(batch_labels.detach().cpu())
        probs.append(output.probs.detach().cpu())

        if batch_index == 1:
            _log(f"[train] epoch={epoch or '?'} first_batch { _batch_summary(batch) }")
        if log_interval > 0 and batch_index % log_interval == 0:
            batch_total_time = data_wait_time + forward_time + backward_time + step_time
            batch_position = (
                f"{batch_index}/{total_batches}" if total_batches is not None else str(batch_index)
            )
            feature_timings = _format_feature_timings(model)
            _log(
                "[train] "
                f"epoch={epoch or '?'} batch={batch_position} "
                f"loss={loss.detach().item():.4f} "
                f"data={data_wait_time:.3f}s "
                f"forward={forward_time:.3f}s "
                f"backward={backward_time:.3f}s "
                f"step={step_time:.3f}s "
                f"total={batch_total_time:.3f}s"
                + (f" {feature_timings}" if feature_timings else "")
            )

    return _compute_epoch_metrics(total_loss=total_loss, labels=labels, probs=probs)


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _device_for_build(target_device: torch.device | None, config: Mapping[str, Any]) -> torch.device:
    if target_device is not None:
        return target_device
    device_spec = str(config.get("device", "cpu")).lower()
    if device_spec == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def validate_model(
    model: ClipRealFakePredictor,
    dataloader: DataLoader[Any],
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int | None = None,
    log_interval: int = 0,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    labels: list[torch.Tensor] = []
    probs: list[torch.Tensor] = []
    total_batches = _batch_count(dataloader)
    iterator = iter(dataloader)
    batch_index = 0

    with torch.no_grad():
        while True:
            data_wait_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            data_wait_time = time.perf_counter() - data_wait_start
            batch_index += 1
            batch_labels = _labels_from_batch(batch, device)

            _sync_device(device)
            forward_start = time.perf_counter()
            output = model(batch)
            loss = criterion(output.logits, batch_labels)
            _sync_device(device)
            forward_time = time.perf_counter() - forward_start

            batch_size = int(batch_labels.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            labels.append(batch_labels.detach().cpu())
            probs.append(output.probs.detach().cpu())

            if batch_index == 1:
                _log(f"[val] epoch={epoch or '?'} first_batch { _batch_summary(batch) }")
            if log_interval > 0 and batch_index % log_interval == 0:
                batch_total_time = data_wait_time + forward_time
                batch_position = (
                    f"{batch_index}/{total_batches}" if total_batches is not None else str(batch_index)
                )
                feature_timings = _format_feature_timings(model)
                _log(
                    "[val] "
                    f"epoch={epoch or '?'} batch={batch_position} "
                    f"loss={loss.detach().item():.4f} "
                    f"data={data_wait_time:.3f}s "
                    f"forward={forward_time:.3f}s "
                    f"total={batch_total_time:.3f}s"
                    + (f" {feature_timings}" if feature_timings else "")
                )

    return _compute_epoch_metrics(total_loss=total_loss, labels=labels, probs=probs)


def train_model(
    config: Mapping[str, Any],
    manifest_path: str | Path,
    run_config: RunConfig,
    modalities: Sequence[str] | None = None,
) -> TrainingResult:
    train_loader, val_loader = _build_dataloaders(config=config, manifest_path=manifest_path, run_config=run_config)
    build_result = build_prediction_model(config, modalities=modalities)
    model = build_result.model
    criterion = build_binary_classification_loss(build_result.training_config, device=build_result.device)
    optimizer = model.build_optimizer(
        build_result.training_config,
        weight_decay=run_config.weight_decay,
    )

    output_dir = run_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_epoch = 0
    best_metrics: EpochMetrics | None = None
    resolved_modalities = tuple(modalities or model.enabled_modalities)

    _log(
        "[run] "
        f"device={build_result.device} "
        f"modalities={resolved_modalities} "
        f"output_dir={output_dir}"
    )
    _log(
        "[data] "
        f"train_examples={len(train_loader.dataset)} "
        f"val_examples={len(val_loader.dataset)} "
        f"batch_size={run_config.batch_size} "
        f"num_workers={run_config.num_workers}"
    )
    _log(
        "[train_cfg] "
        f"freeze_encoders={build_result.training_config.freeze_encoders} "
        f"lr_head={build_result.training_config.lr_head} "
        f"lr_fusion={build_result.training_config.lr_fusion} "
        f"weight_decay={run_config.weight_decay}"
    )
    for warning in build_result.warnings:
        _log(f"[warning] {warning}")

    try:
        if run_config.resume_from is not None:
            payload = _load_checkpoint_payload(run_config.resume_from)
            model.load_state_dict(payload["model_state_dict"])
            optimizer_state = payload.get("optimizer_state_dict")
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
                _move_optimizer_state_to_device(optimizer, build_result.device)
            start_epoch = int(payload.get("epoch", 0))
            history = list(payload.get("history", []))
            best_epoch = int(payload.get("best_epoch", 0))
            best_metrics = _deserialize_epoch_metrics(payload.get("best_metrics"))
            _log(
                "[resume] "
                f"checkpoint={run_config.resume_from} "
                f"start_epoch={start_epoch} "
                f"best_epoch={best_epoch}"
            )

        if start_epoch >= run_config.epochs:
            raise ValueError(
                f"`run_config.epochs` ({run_config.epochs}) must exceed resumed epoch ({start_epoch})."
            )

        last_checkpoint_path = output_dir / "last.pt"
        best_checkpoint_path = output_dir / "best.pt"

        for epoch_index in range(start_epoch, run_config.epochs):
            epoch = epoch_index + 1
            epoch_start = time.perf_counter()
            _log(f"[epoch] start epoch={epoch}/{run_config.epochs}")
            train_metrics = _train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=build_result.device,
                epoch=epoch,
                log_interval=run_config.log_interval,
            )
            val_metrics = validate_model(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=build_result.device,
                epoch=epoch,
                log_interval=run_config.log_interval,
            )
            history.append(_history_entry(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics))

            if best_metrics is None or val_metrics.loss < best_metrics.loss:
                best_epoch = epoch
                best_metrics = val_metrics

            checkpoint = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                config=config,
                enabled_modalities=modalities or model.enabled_modalities,
                run_config=run_config,
                history=history,
                epoch=epoch,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
            )
            last_checkpoint_path = _save_checkpoint(last_checkpoint_path, checkpoint)
            _log(f"[checkpoint] saved last={last_checkpoint_path}")
            if run_config.save_every_epoch:
                epoch_checkpoint_path = _save_checkpoint(output_dir / f"epoch_{epoch:04d}.pt", checkpoint)
                _log(f"[checkpoint] saved epoch={epoch_checkpoint_path}")

            if best_epoch == epoch:
                checkpoint = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    enabled_modalities=modalities or model.enabled_modalities,
                    run_config=run_config,
                    history=history,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_metrics=best_metrics,
                )
                best_checkpoint_path = _save_checkpoint(best_checkpoint_path, checkpoint)
                _log(f"[checkpoint] saved best={best_checkpoint_path}")

            epoch_time = time.perf_counter() - epoch_start
            _log(
                "[epoch] "
                f"done epoch={epoch}/{run_config.epochs} "
                f"train_loss={train_metrics.loss:.4f} "
                f"train_acc={train_metrics.accuracy:.4f} "
                f"val_loss={val_metrics.loss:.4f} "
                f"val_acc={val_metrics.accuracy:.4f} "
                f"val_f1={val_metrics.f1:.4f} "
                f"val_roc_auc={'nan' if val_metrics.roc_auc is None else f'{val_metrics.roc_auc:.4f}'} "
                f"best_epoch={best_epoch} "
                f"time={epoch_time:.2f}s"
            )

        return TrainingResult(
            best_checkpoint_path=best_checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            history=tuple(history),
            best_epoch=best_epoch,
        )
    finally:
        model.close()


def train_from_yaml(
    config_path: str | Path,
    manifest_path: str | Path,
    run_config: RunConfig,
    modalities: Sequence[str] | None = None,
) -> TrainingResult:
    return train_model(
        config=load_prediction_yaml(config_path),
        manifest_path=manifest_path,
        run_config=run_config,
        modalities=modalities,
    )


def load_training_result(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    with_optimizer: bool = False,
) -> LoadedTrainingResult:
    payload = _load_checkpoint_payload(checkpoint_path)
    config = copy.deepcopy(dict(payload["config"]))
    target_device = _device_for_build(
        target_device=None if device is None else torch.device(device),
        config=config,
    )
    config["device"] = "cuda" if target_device.type == "cuda" else "cpu"
    enabled_modalities = tuple(str(name) for name in payload["enabled_modalities"])

    build_result = build_prediction_model(config, modalities=enabled_modalities)
    model = build_result.model.to(target_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    optimizer: torch.optim.Optimizer | None = None
    if with_optimizer:
        run_config = _deserialize_run_config(payload["run_config"])
        optimizer = model.build_optimizer(
            build_result.training_config,
            weight_decay=run_config.weight_decay,
        )
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            _move_optimizer_state_to_device(optimizer, target_device)

    return LoadedTrainingResult(
        model=model,
        config=config,
        enabled_modalities=enabled_modalities,
        optimizer=optimizer,
        scheduler=None,
        epoch=int(payload.get("epoch", 0)),
        best_metrics=_deserialize_epoch_metrics(payload.get("best_metrics")),
    )
