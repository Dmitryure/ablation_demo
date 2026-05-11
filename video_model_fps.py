from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import torch

from dataset import (
    RPPG_DEFAULT_FPS,
    LabeledVideoDataset,
    VideoExample,
    collate_labeled_video_batch,
)
from feature_cache import build_feature_cache_specs
from pipeline import build_fusion_pipeline, load_pipeline_yaml
from scripts.run_iterative_cached_ablation import move_tensor_batch_to_device
from task_models import build_binary_fusion_classifier
from task_models.generator_multitask_classifier import build_generator_multitask_classifier

PROJECT_ROOT = Path(__file__).resolve().parent
ModelKind = Literal["binary", "generator_multitask"]


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float
    duration_seconds: float


@dataclass(frozen=True)
class ModelSelection:
    summary_path: Path
    run_config_path: Path
    checkpoint_path: Path
    score_key: str
    score: float
    model_kind: ModelKind


@dataclass(frozen=True)
class InferenceModel:
    model: torch.nn.Module
    device: torch.device
    config: Mapping[str, Any]
    modalities: tuple[str, ...]
    model_kind: ModelKind
    checkpoint_path: Path
    summary_path: Path
    run_config_path: Path
    generator_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class FpsPrediction:
    video_path: Path
    label: str
    fake_probability: float
    threshold: float
    model_kind: ModelKind
    modalities: tuple[str, ...]
    checkpoint_path: Path
    summary_path: Path
    score_key: str
    score: float
    device: str
    video_frame_count: int
    video_fps: float
    video_duration_seconds: float
    sampled_frame_count: int
    decode_seconds: float
    forward_seconds: float
    forward_seconds_mean: float
    forward_seconds_median: float
    end_to_end_seconds: float
    end_to_end_video_fps: float
    forward_sampled_frame_fps: float
    repeats: int
    warmup_runs: int
    extractor_seconds: Mapping[str, float]
    load_seconds_by_modality: Mapping[str, float]
    generator_label: str | None = None
    generator_probability: float | None = None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed JSON object: {path}")
    return payload


def resolve_project_path(value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    if base_dir is not None:
        return base_dir / path
    return candidate


def infer_model_kind(summary: Mapping[str, Any], run_config: Mapping[str, Any]) -> ModelKind:
    if "target" in run_config or "target" in summary:
        return "generator_multitask"
    return "binary"


def summary_score(summary: Mapping[str, Any], score_key: str | None = None) -> tuple[str, float]:
    keys = (score_key,) if score_key is not None else (
        "best_checkpoint_score",
        "best_checkpoint_metric_value",
    )
    for key in keys:
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return key, float(value)
    raise ValueError("Summary has no numeric best checkpoint score.")


def find_run_config(summary_path: Path) -> Path:
    for directory in summary_path.parents:
        candidate = directory / "run_config.json"
        if candidate.is_file():
            return candidate
        if directory == PROJECT_ROOT:
            break
    raise FileNotFoundError(f"No run_config.json found above {summary_path}")


def selection_from_checkpoint(checkpoint_path: Path, model_kind: str = "auto") -> ModelSelection:
    checkpoint = checkpoint_path.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    summary_path = checkpoint.parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary.json beside checkpoint: {summary_path}")
    summary = read_json(summary_path)
    run_config_path = find_run_config(summary_path)
    run_config = read_json(run_config_path)
    kind = infer_model_kind(summary, run_config)
    if model_kind != "auto" and kind != model_kind:
        raise ValueError(f"Requested model_kind={model_kind}, but checkpoint run is {kind}.")
    resolved_score_key, score = summary_score(summary)
    return ModelSelection(
        summary_path=summary_path,
        run_config_path=run_config_path,
        checkpoint_path=checkpoint,
        score_key=resolved_score_key,
        score=score,
        model_kind=kind,
    )


def pipeline_config_for_selection(
    selection: ModelSelection,
    run_config: Mapping[str, Any],
    device: str | None,
) -> dict[str, Any]:
    if selection.model_kind == "binary":
        raw_config = run_config.get("pipeline_config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"Binary run config missing pipeline_config: {selection.run_config_path}")
        config = dict(raw_config)
    else:
        config_path_value = run_config.get("config_path")
        if not isinstance(config_path_value, str) or not config_path_value:
            raise ValueError(
                f"Generator multitask run config missing config_path: {selection.run_config_path}"
            )
        config_path = resolve_project_path(config_path_value, base_dir=selection.run_config_path.parent)
        config = load_pipeline_yaml(config_path)
    if device is not None:
        config["device"] = device
    return config


def modalities_for_selection(
    selection: ModelSelection,
    summary: Mapping[str, Any],
    run_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    candidates = (
        summary.get("modalities"),
        run_config.get("modalities"),
        run_config.get("base_modalities"),
        config.get("modalities"),
    )
    for value in candidates:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            modalities = tuple(str(item) for item in value)
            if modalities:
                return modalities
    raise ValueError(f"Cannot resolve modalities for {selection.summary_path}")


def generator_names_from_config(run_config: Mapping[str, Any]) -> tuple[str, ...]:
    target = run_config.get("target")
    if not isinstance(target, Mapping):
        return ()
    value = target.get("generator_names")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value)


def normalize_checkpoint_state(state: object) -> Mapping[str, torch.Tensor]:
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint must be a state_dict mapping or contain `state_dict`.")
    return state


def load_inference_model(selection: ModelSelection, device: str | None = None) -> InferenceModel:
    summary = read_json(selection.summary_path)
    run_config = read_json(selection.run_config_path)
    config = pipeline_config_for_selection(selection, run_config=run_config, device=device)
    modalities = modalities_for_selection(selection, summary, run_config, config)
    build_result = build_fusion_pipeline(config=config, modalities=modalities)

    if selection.model_kind == "generator_multitask":
        generator_names = generator_names_from_config(run_config)
        if not generator_names:
            raise ValueError(f"Generator multitask run missing target names: {selection.run_config_path}")
        model = build_generator_multitask_classifier(
            build_result.pipeline,
            dim=int(config["dim"]),
            num_generators=len(generator_names),
            head_config=config.get("head"),
        )
    else:
        generator_names = ()
        model = build_binary_fusion_classifier(
            build_result.pipeline,
            dim=int(config["dim"]),
            head_config=config.get("head"),
        )

    model = model.to(build_result.device)
    state = torch.load(selection.checkpoint_path, map_location=build_result.device, weights_only=False)
    model.load_state_dict(normalize_checkpoint_state(state))
    model.eval()
    return InferenceModel(
        model=model,
        device=build_result.device,
        config=config,
        modalities=modalities,
        model_kind=selection.model_kind,
        checkpoint_path=selection.checkpoint_path,
        summary_path=selection.summary_path,
        run_config_path=selection.run_config_path,
        generator_names=generator_names,
    )


def video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if frame_count <= 0:
        frame_count = 0
    if not math.isfinite(fps) or fps <= 0.0:
        fps = RPPG_DEFAULT_FPS
    duration = float(frame_count) / fps if frame_count > 0 else 0.0
    return VideoInfo(frame_count=frame_count, fps=fps, duration_seconds=duration)


def video_example(video_path: Path) -> VideoExample:
    return VideoExample(
        path=video_path,
        label=0,
        class_name="real",
        source_id=video_path.stem,
        split="test",
        metadata_filename=video_path.name,
        generator_id="real",
    )


def modality_configs(config: Mapping[str, Any], modalities: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    configs: dict[str, Mapping[str, Any]] = {}
    for modality in modalities:
        value = config.get(modality, {})
        if isinstance(value, Mapping):
            configs[modality] = value
    return configs


def build_video_batch(
    video_path: Path,
    config: Mapping[str, Any],
    modalities: Sequence[str],
    decode_mode: str,
) -> tuple[dict[str, Any], float, int]:
    specs = build_feature_cache_specs(config, modalities)
    frame_counts = {name: specs[name].frame_count for name in modalities}
    image_sizes = {name: specs[name].image_size for name in modalities}
    rppg_config = config.get("rppg", {})
    dataset = LabeledVideoDataset(
        examples=[video_example(video_path)],
        num_frames=frame_counts,
        image_size=int(config.get("image_size", 224)),
        decode_mode=decode_mode,
        dataset_root=None,
        image_size_by_modality=image_sizes,
        rppg_config=rppg_config if isinstance(rppg_config, Mapping) else {},
        modality_configs=modality_configs(config, modalities),
        global_config=config,
    )
    start = time.perf_counter()
    item = dataset[0]
    decode_seconds = time.perf_counter() - start
    return collate_labeled_video_batch([item]), decode_seconds, max(frame_counts.values())


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fake_probability_from_output(output: object) -> float:
    typed_output = cast(Any, output)
    if hasattr(output, "binary_probabilities"):
        probabilities = typed_output.binary_probabilities
    elif hasattr(output, "probabilities"):
        probabilities = typed_output.probabilities
    else:
        raise TypeError("Model output has no binary probability tensor.")
    if not isinstance(probabilities, torch.Tensor):
        raise TypeError("Model probability output must be a tensor.")
    return float(probabilities.detach().cpu().view(-1)[0].item())


def generator_prediction_from_output(
    output: object,
    generator_names: Sequence[str],
) -> tuple[str | None, float | None]:
    if not generator_names or not hasattr(output, "generator_probabilities"):
        return None, None
    probabilities = cast(Any, output).generator_probabilities
    if not isinstance(probabilities, torch.Tensor):
        return None, None
    row = probabilities.detach().cpu()[0]
    index = int(torch.argmax(row).item())
    return generator_names[index], float(row[index].item())


def mean_load_timings(batch: Mapping[str, Any]) -> dict[str, float]:
    timings = batch.get("load_timings_by_modality", {})
    if not isinstance(timings, Mapping):
        return {}
    result: dict[str, float] = {}
    for name, values in timings.items():
        if isinstance(values, Sequence) and values:
            result[str(name)] = float(statistics.mean(float(value) for value in values))
    return result


def predict_video_fps(
    loaded: InferenceModel,
    selection: ModelSelection,
    video_path: Path,
    threshold: float = 0.5,
    repeat: int = 1,
    warmup_runs: int = 0,
    decode_mode: str = "scan",
) -> FpsPrediction:
    if repeat <= 0:
        raise ValueError("`repeat` must be positive.")
    if warmup_runs < 0:
        raise ValueError("`warmup_runs` must be non-negative.")

    info = video_info(video_path)
    batch, decode_seconds, sampled_frame_count = build_video_batch(
        video_path=video_path,
        config=loaded.config,
        modalities=loaded.modalities,
        decode_mode=decode_mode,
    )
    device_batch = move_tensor_batch_to_device(batch, loaded.device)
    forward_seconds: list[float] = []
    output: object | None = None
    with torch.no_grad():
        for run_index in range(warmup_runs + repeat):
            synchronize_if_needed(loaded.device)
            start = time.perf_counter()
            output = loaded.model(device_batch)
            synchronize_if_needed(loaded.device)
            elapsed = time.perf_counter() - start
            if run_index >= warmup_runs:
                forward_seconds.append(elapsed)
    if output is None:
        raise RuntimeError("No inference run completed.")

    probability = fake_probability_from_output(output)
    generator_label, generator_probability = generator_prediction_from_output(
        output,
        loaded.generator_names,
    )
    forward_mean = statistics.mean(forward_seconds)
    forward_median = statistics.median(forward_seconds)
    end_to_end = decode_seconds + forward_median
    label = "fake" if probability >= threshold else "real"
    extractor_timings = getattr(loaded.model.pipeline, "last_feature_timings", {})
    return FpsPrediction(
        video_path=video_path,
        label=label,
        fake_probability=probability,
        threshold=threshold,
        model_kind=loaded.model_kind,
        modalities=loaded.modalities,
        checkpoint_path=loaded.checkpoint_path,
        summary_path=loaded.summary_path,
        score_key=selection.score_key,
        score=selection.score,
        device=str(loaded.device),
        video_frame_count=info.frame_count,
        video_fps=info.fps,
        video_duration_seconds=info.duration_seconds,
        sampled_frame_count=sampled_frame_count,
        decode_seconds=decode_seconds,
        forward_seconds=forward_seconds[-1],
        forward_seconds_mean=forward_mean,
        forward_seconds_median=forward_median,
        end_to_end_seconds=end_to_end,
        end_to_end_video_fps=float(info.frame_count) / end_to_end if end_to_end > 0.0 else 0.0,
        forward_sampled_frame_fps=float(sampled_frame_count) / forward_median
        if forward_median > 0.0
        else 0.0,
        repeats=repeat,
        warmup_runs=warmup_runs,
        extractor_seconds={
            str(name): float(value)
            for name, value in extractor_timings.items()
            if isinstance(value, (int, float))
        },
        load_seconds_by_modality=mean_load_timings(batch),
        generator_label=generator_label,
        generator_probability=generator_probability,
    )
