from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from dataset import VideoExample
from frame_config import resolve_modality_frame_count

FEATURE_CACHE_VERSION = 2
SPEC_IGNORED_MODALITY_KEYS = frozenset({"frames", "slot_count"})
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


def _modality_extractor_config(config: Mapping[str, Any], modality_name: str) -> dict[str, Any]:
    section = config.get(modality_name, {})
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise ValueError(f"`{modality_name}` must be a YAML mapping when provided.")
    return {
        str(key): _jsonable(value)
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
    payload = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _safe_stem(path: Path) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    return safe[:96] or "video"


def _cache_key_path(example: VideoExample, dataset_root: str | Path | None) -> str:
    path = example.path.resolve()
    if dataset_root is None:
        return str(path)
    root = Path(dataset_root).resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def video_cache_record(
    example: VideoExample,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    stat = example.path.stat()
    return {
        "relative_path": _cache_key_path(example, dataset_root),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def feature_cache_spec_dir(cache_dir: str | Path, spec: FeatureCacheSpec) -> Path:
    return (
        Path(cache_dir) / spec.modality / f"frames_{spec.frame_count}" / feature_cache_spec_id(spec)
    )


def feature_cache_item_path(
    cache_dir: str | Path,
    example: VideoExample,
    spec: FeatureCacheSpec,
    dataset_root: str | Path | None = None,
) -> Path:
    cache_key = _cache_key_path(example, dataset_root)
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:20]
    filename = f"{_safe_stem(example.path)}-{digest}.pt"
    return feature_cache_spec_dir(cache_dir, spec) / filename


def feature_cache_metadata_path(cache_path: str | Path) -> Path:
    return Path(cache_path).with_suffix(".json")


def feature_cache_metadata(
    example: VideoExample,
    spec: FeatureCacheSpec,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "version": FEATURE_CACHE_VERSION,
        "spec_id": feature_cache_spec_id(spec),
        "spec": asdict(spec),
        "video": video_cache_record(example, dataset_root=dataset_root),
    }


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


def write_feature_cache_item(
    cache_dir: str | Path,
    example: VideoExample,
    spec: FeatureCacheSpec,
    item: Mapping[str, Any],
    dataset_root: str | Path | None = None,
) -> Path:
    cache_path = feature_cache_item_path(cache_dir, example, spec, dataset_root=dataset_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = feature_cache_metadata(example, spec, dataset_root=dataset_root)
    torch.save(
        {
            **metadata,
            "features": _feature_tensors_for_modality(item, spec.modality),
        },
        cache_path,
    )
    feature_cache_metadata_path(cache_path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return cache_path


def _payload_matches_example(
    payload: Mapping[str, Any],
    example: VideoExample,
    spec: FeatureCacheSpec,
    dataset_root: str | Path | None,
) -> bool:
    if payload.get("version") != FEATURE_CACHE_VERSION:
        return False
    if payload.get("spec_id") != feature_cache_spec_id(spec):
        return False
    stored = payload.get("video")
    if not isinstance(stored, Mapping):
        return False
    current = video_cache_record(example, dataset_root=dataset_root)
    return (
        stored.get("relative_path") == current["relative_path"]
        and stored.get("size") == current["size"]
        and stored.get("mtime_ns") == current["mtime_ns"]
    )


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
    if not isinstance(payload, Mapping):
        return None
    if not _payload_matches_example(payload, example, spec, dataset_root):
        return None
    features = payload.get("features")
    if not isinstance(features, Mapping):
        return None
    result = {str(key): value for key, value in features.items() if isinstance(value, torch.Tensor)}
    required_key = MODALITY_FEATURE_KEYS[spec.modality][0]
    if required_key not in result:
        return None
    return result


def _load_feature_cache_metadata(cache_path: Path) -> Mapping[str, Any] | None:
    metadata_path = feature_cache_metadata_path(cache_path)
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def feature_cache_item_exists(
    cache_dir: str | Path,
    example: VideoExample,
    spec: FeatureCacheSpec,
    dataset_root: str | Path | None = None,
) -> bool:
    cache_path = feature_cache_item_path(cache_dir, example, spec, dataset_root=dataset_root)
    if not cache_path.exists():
        return False
    metadata = _load_feature_cache_metadata(cache_path)
    if metadata is not None:
        return _payload_matches_example(metadata, example, spec, dataset_root)
    return load_feature_cache_item(cache_dir, example, spec, dataset_root=dataset_root) is not None


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
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None

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
