from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from torch.utils.data import Dataset

from dataset import VideoExample
from frame_config import resolve_modality_frame_count

FEATURE_CACHE_VERSION = 3
SPEC_IGNORED_MODALITY_KEYS = frozenset({"frames", "slot_count"})
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_PATH_CONFIG_KEYS = frozenset({"checkpoint_path", "model_path"})
MODALITY_FEATURE_KEYS: dict[str, tuple[str, ...]] = {
    "rgb": ("rgb_features",),
    "fau": ("fau_features", "fau_au_logits", "fau_au_edge_logits"),
    "rppg": ("rppg_features", "rppg_waveform"),
    "eye_gaze": ("eye_gaze",),
    "face_mesh": ("face_mesh",),
    "depth": ("depth_features",),
    "fft": ("fft_features",),
    "stft": ("stft_features",),
}
FEATURE_CACHE_MANIFEST = "manifest.csv"
FEATURE_CACHE_MANIFEST_COLUMNS = (
    "class_name",
    "filename",
    "generator_id",
    "source_path",
    "cache_path",
    "modality",
    "frame_count",
    "status",
    "error",
)


@dataclass(frozen=True)
class FeatureCacheSpec:
    version: int
    modality: str
    frame_count: int
    image_size: int
    extractor_config: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _canonicalize_local_path(value: Any) -> Any:
    if value is None:
        return None
    path = Path(value) if isinstance(value, Path) else Path(str(value))
    if path.is_absolute():
        return str(path)
    candidate = PROJECT_ROOT / path
    return str(candidate) if candidate.exists() else str(value)


def _jsonable_extractor_value(key: str, value: Any) -> Any:
    if key in LOCAL_PATH_CONFIG_KEYS:
        return _canonicalize_local_path(value)
    return _jsonable(value)


def _modality_extractor_config(config: Mapping[str, Any], modality_name: str) -> dict[str, Any]:
    section = config.get(modality_name, {})
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise ValueError(f"`{modality_name}` must be a YAML mapping when provided.")
    return {
        str(key): _jsonable_extractor_value(str(key), value)
        for key, value in sorted(section.items())
        if key not in SPEC_IGNORED_MODALITY_KEYS
    }


def build_feature_cache_spec(
    config: Mapping[str, Any],
    modality: str,
) -> FeatureCacheSpec:
    if modality not in MODALITY_FEATURE_KEYS:
        raise ValueError(f"Unsupported cache modality: {modality}")
    return FeatureCacheSpec(
        version=FEATURE_CACHE_VERSION,
        modality=modality,
        frame_count=resolve_modality_frame_count(config, modality),
        image_size=int(config.get("image_size", 224)),
        extractor_config=_modality_extractor_config(config, modality),
    )


def build_feature_cache_specs(
    config: Mapping[str, Any],
    modalities: Sequence[str],
) -> dict[str, FeatureCacheSpec]:
    return {modality: build_feature_cache_spec(config, modality) for modality in modalities}


def feature_cache_spec_id(spec: FeatureCacheSpec) -> str:
    return f"{spec.modality}-frames_{spec.frame_count}"


def normalize_cache_dataset_root(dataset_root: str | Path | None) -> Path | None:
    if dataset_root is None:
        return None
    root = Path(dataset_root)
    videos_root = root / "videos"
    if videos_root.is_dir():
        return videos_root
    return root


def _validate_metadata_filename(filename: str) -> str:
    path = PurePosixPath(filename)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Invalid metadata filename for cache: {filename!r}")
    return path.as_posix()


def metadata_filename_for_example(
    example: VideoExample,
    dataset_root: str | Path | None = None,
) -> str:
    if example.metadata_filename:
        return _validate_metadata_filename(example.metadata_filename)
    root = normalize_cache_dataset_root(dataset_root)
    if root is None:
        return _validate_metadata_filename(example.path.name)
    class_root = (root / example.class_name).resolve()
    try:
        return _validate_metadata_filename(
            example.path.resolve().relative_to(class_root).as_posix()
        )
    except ValueError:
        return _validate_metadata_filename(example.path.name)


def feature_cache_spec_dir(cache_dir: str | Path, spec: FeatureCacheSpec) -> Path:
    return Path(cache_dir) / spec.modality / f"frames_{spec.frame_count}"


def feature_cache_manifest_path(cache_dir: str | Path, spec: FeatureCacheSpec) -> Path:
    return feature_cache_spec_dir(cache_dir, spec) / FEATURE_CACHE_MANIFEST


def feature_cache_item_path(
    cache_dir: str | Path,
    example: VideoExample,
    spec: FeatureCacheSpec,
    dataset_root: str | Path | None = None,
) -> Path:
    filename = metadata_filename_for_example(example, dataset_root)
    return feature_cache_spec_dir(cache_dir, spec) / example.class_name / f"{filename}.pt"


def _feature_tensors_for_modality(
    item: Mapping[str, Any],
    modality: str,
) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}
    for key in MODALITY_FEATURE_KEYS[modality]:
        value = item.get(key)
        if isinstance(value, torch.Tensor):
            features[key] = value.detach().cpu()
    required_key = MODALITY_FEATURE_KEYS[modality][0]
    if required_key not in features:
        raise KeyError(f"Missing required cached feature `{required_key}` for {modality}.")
    return features


def feature_cache_payload_header(
    example: VideoExample,
    spec: FeatureCacheSpec,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "version": FEATURE_CACHE_VERSION,
        "class_name": example.class_name,
        "filename": metadata_filename_for_example(example, dataset_root),
        "modality": spec.modality,
        "frame_count": spec.frame_count,
    }


def write_feature_cache_item(
    cache_dir: str | Path,
    example: VideoExample,
    spec: FeatureCacheSpec,
    item: Mapping[str, Any],
    dataset_root: str | Path | None = None,
) -> Path:
    cache_path = feature_cache_item_path(cache_dir, example, spec, dataset_root=dataset_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")
    torch.save(
        {
            **feature_cache_payload_header(example, spec, dataset_root),
            "features": _feature_tensors_for_modality(item, spec.modality),
        },
        tmp_path,
    )
    tmp_path.replace(cache_path)
    return cache_path


def _payload_matches_example(
    payload: Mapping[str, Any],
    example: VideoExample,
    spec: FeatureCacheSpec,
    dataset_root: str | Path | None,
) -> bool:
    expected = feature_cache_payload_header(example, spec, dataset_root)
    return all(payload.get(key) == value for key, value in expected.items())


def load_feature_cache_item(
    cache_dir: str | Path,
    example: VideoExample,
    spec: FeatureCacheSpec,
    dataset_root: str | Path | None = None,
) -> dict[str, torch.Tensor] | None:
    cache_path = feature_cache_item_path(cache_dir, example, spec, dataset_root=dataset_root)
    if not cache_path.exists():
        return None
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not _payload_matches_example(
        payload, example, spec, dataset_root
    ):
        return None
    features = payload.get("features")
    if not isinstance(features, Mapping):
        return None
    result = {str(key): value for key, value in features.items() if isinstance(value, torch.Tensor)}
    required_key = MODALITY_FEATURE_KEYS[spec.modality][0]
    return result if required_key in result else None


def feature_cache_item_exists(
    cache_dir: str | Path,
    example: VideoExample,
    spec: FeatureCacheSpec,
    dataset_root: str | Path | None = None,
) -> bool:
    return feature_cache_item_path(cache_dir, example, spec, dataset_root=dataset_root).exists()


def feature_cache_generator_id(
    example: VideoExample, dataset_root: str | Path | None = None
) -> str:
    if example.class_name == "real":
        return example.generator_id or "real"
    if example.generator_id:
        return example.generator_id
    filename = metadata_filename_for_example(example, dataset_root)
    parts = PurePosixPath(filename).parts
    return parts[0] if len(parts) > 1 else ""


def feature_cache_manifest_row(
    cache_dir: str | Path,
    example: VideoExample,
    spec: FeatureCacheSpec,
    dataset_root: str | Path | None,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    cache_path = feature_cache_item_path(cache_dir, example, spec, dataset_root=dataset_root)
    return {
        "class_name": example.class_name,
        "filename": metadata_filename_for_example(example, dataset_root),
        "generator_id": feature_cache_generator_id(example, dataset_root),
        "source_path": str(example.path),
        "cache_path": str(cache_path),
        "modality": spec.modality,
        "frame_count": spec.frame_count,
        "status": status,
        "error": error,
    }


def write_feature_cache_manifest(
    cache_dir: str | Path,
    examples: Sequence[VideoExample],
    spec: FeatureCacheSpec,
    dataset_root: str | Path | None,
    errors_by_path: Mapping[str, str] | None = None,
) -> Path:
    errors = errors_by_path or {}
    rows: list[dict[str, Any]] = []
    for example in examples:
        error = errors.get(str(example.path), "")
        cache_path = feature_cache_item_path(cache_dir, example, spec, dataset_root=dataset_root)
        if cache_path.exists():
            status = "cached"
        elif error:
            status = "failed"
        else:
            status = "missing"
        rows.append(
            feature_cache_manifest_row(
                cache_dir=cache_dir,
                example=example,
                spec=spec,
                dataset_root=dataset_root,
                status=status,
                error=error,
            )
        )
    path = feature_cache_manifest_path(cache_dir, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_CACHE_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def split_feature_batch(
    feature_batch: Mapping[str, Any],
    raw_batch: Mapping[str, Any],
) -> list[dict[str, Any]]:
    labels = raw_batch["label"]
    batch_size = int(labels.shape[0])
    items: list[dict[str, Any]] = []
    for index in range(batch_size):
        item: dict[str, Any] = {
            "label": labels[index].detach().cpu(),
            "path": raw_batch["path"][index],
            "source_id": raw_batch["source_id"][index],
            "split": raw_batch["split"][index],
            "class_name": raw_batch["class_name"][index],
            "identity_id": raw_batch["identity_id"][index],
            "metadata_filename": raw_batch["metadata_filename"][index],
        }
        for key in ("generator_id", "source_id_kind", "age_bin", "gender", "ethnicity", "emotion"):
            if key in raw_batch:
                item[key] = raw_batch[key][index]
        for key, value in feature_batch.items():
            if isinstance(value, torch.Tensor):
                item[key] = value[index].detach().cpu()
        items.append(item)
    return items


class CachedFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        examples: Sequence[VideoExample],
        cache_dir: str | Path,
        spec_by_modality: Mapping[str, FeatureCacheSpec],
        modalities: Sequence[str],
        strict: bool = True,
        dataset_root: str | Path | None = None,
    ) -> None:
        self.examples = list(examples)
        self.cache_dir = Path(cache_dir)
        self.spec_by_modality = dict(spec_by_modality)
        self.modalities = tuple(modalities)
        self.strict = strict
        self.dataset_root = normalize_cache_dataset_root(dataset_root)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        item: dict[str, Any] = {
            "label": torch.tensor([float(example.label)], dtype=torch.float32),
            "path": str(example.path),
            "source_id": example.source_id,
            "split": example.split,
            "class_name": example.class_name,
            "metadata_filename": metadata_filename_for_example(example, self.dataset_root),
            "identity_id": example.identity_id,
            "generator_id": example.generator_id,
            "source_id_kind": example.source_id_kind,
            "age_bin": example.age_bin,
            "gender": example.gender,
            "ethnicity": example.ethnicity,
            "emotion": example.emotion,
        }
        missing: list[str] = []
        for modality in self.modalities:
            spec = self.spec_by_modality[modality]
            features = load_feature_cache_item(
                self.cache_dir,
                example,
                spec,
                dataset_root=self.dataset_root,
            )
            if features is None:
                missing.append(modality)
                continue
            item.update(features)
        if missing and self.strict:
            raise FileNotFoundError(
                f"Missing cached modalities for {example.path}: {','.join(missing)}"
            )
        if missing:
            item["missing_modalities"] = tuple(missing)
        return item


def collate_cached_feature_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
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
