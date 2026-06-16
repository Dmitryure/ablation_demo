from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from dataset import VideoExample
from face_crop_config import modality_cache_variant
from frame_config import resolve_modality_frame_count
from frame_sampling import frame_sampling_cache_variant

FEATURE_CACHE_VERSION = 3
SHARDED_CACHE_SCHEMA_VERSION = 2
SPEC_IGNORED_MODALITY_KEYS = frozenset({"frames", "slot_count"})
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_PATH_CONFIG_KEYS = frozenset({"checkpoint_path", "model_path"})
MODALITY_FEATURE_KEYS: dict[str, tuple[str, ...]] = {
    "rgb": ("rgb_features",),
    "fau": ("fau_features", "fau_au_logits", "fau_au_edge_logits"),
    "rppg": ("rppg_features", "rppg_waveform", "rppg_signal_features"),
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
    cache_variant: str | None = None
    frame_sampling_variant: str | None = None


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
        image_size=_resolve_modality_image_size(config, modality),
        extractor_config=_modality_extractor_config(config, modality),
        cache_variant=modality_cache_variant(config, modality),
        frame_sampling_variant=frame_sampling_cache_variant(
            config,
            config.get(modality, {}) if isinstance(config.get(modality, {}), Mapping) else {},
        ),
    )


def _resolve_modality_image_size(config: Mapping[str, Any], modality: str) -> int:
    section = config.get(modality, {})
    value = None
    if isinstance(section, Mapping):
        value = section.get("image_size")
    if value is None:
        value = config.get("image_size", 224)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{modality}.image_size` must be a positive integer.")
    return value


def build_feature_cache_specs(
    config: Mapping[str, Any],
    modalities: Sequence[str],
) -> dict[str, FeatureCacheSpec]:
    return {modality: build_feature_cache_spec(config, modality) for modality in modalities}


def feature_cache_spec_id(spec: FeatureCacheSpec) -> str:
    parts = [spec.modality, f"frames_{spec.frame_count}"]
    if spec.frame_sampling_variant is not None:
        parts.append(spec.frame_sampling_variant)
    if spec.cache_variant is not None:
        parts.append(f"size_{spec.image_size}")
        parts.append(spec.cache_variant)
    return "-".join(parts)


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
    root = Path(cache_dir) / spec.modality
    if spec.frame_sampling_variant is not None:
        root = root / spec.frame_sampling_variant
    if spec.cache_variant is None:
        return root / f"frames_{spec.frame_count}"
    return root / spec.cache_variant / f"frames_{spec.frame_count}_size_{spec.image_size}"


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
    if modality == "rppg":
        missing = [key for key in MODALITY_FEATURE_KEYS[modality] if key not in features]
        if missing:
            raise KeyError(f"Missing required cached rPPG feature(s): {', '.join(missing)}.")
    return features


def feature_cache_payload_header(
    example: VideoExample,
    spec: FeatureCacheSpec,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    header = {
        "version": FEATURE_CACHE_VERSION,
        "class_name": example.class_name,
        "filename": metadata_filename_for_example(example, dataset_root),
        "modality": spec.modality,
        "frame_count": spec.frame_count,
    }
    if spec.cache_variant is not None:
        header["cache_variant"] = spec.cache_variant
    return header


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
    if required_key not in result:
        return None
    if spec.modality == "rppg" and any(
        key not in result for key in MODALITY_FEATURE_KEYS[spec.modality]
    ):
        return None
    return result


def feature_cache_item_exists(
    cache_dir: str | Path,
    example: VideoExample,
    spec: FeatureCacheSpec,
    dataset_root: str | Path | None = None,
) -> bool:
    return load_feature_cache_item(cache_dir, example, spec, dataset_root=dataset_root) is not None


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
        if feature_cache_item_exists(cache_dir, example, spec, dataset_root):
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


@dataclass(frozen=True)
class ShardedFeatureRef:
    example: VideoExample
    shard_index: int
    row_index: int


def cache_example_key(example: VideoExample, dataset_root: str | Path | None = None) -> str:
    return f"{example.class_name}/{metadata_filename_for_example(example, dataset_root)}"


def cache_row_key(row: Mapping[str, Any]) -> str:
    class_name = str(row["class_name"])
    filename = str(row["metadata_filename"])
    return f"{class_name}/{filename}"


class ShardedCachedFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        sharded_cache_dir: str | Path,
        all_examples: Sequence[VideoExample] | None,
        selected_examples: Sequence[VideoExample],
        dataset_root: str | Path | None = None,
        allow_legacy_shards: bool = False,
    ) -> None:
        self.sharded_cache_dir = Path(sharded_cache_dir)
        self.index_path = self.sharded_cache_dir / "index.json"
        self.index = self._read_index(self.index_path)
        self.dataset_root = normalize_cache_dataset_root(dataset_root)
        self.shard_size = int(self.index["shard_size"])
        self.shards = list(self.index["shards"])
        self.allow_legacy_shards = allow_legacy_shards
        locations = self._example_locations(all_examples or ())
        self.refs = [self._ref_for_example(example, locations) for example in selected_examples]
        self._loaded_shard_index: int | None = None
        self._loaded_shard: Mapping[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[index]
        shard = self._load_shard(ref.shard_index)
        self._assert_ref_matches_shard_row(ref, shard)
        features = shard.get("features")
        if not isinstance(features, Mapping):
            raise ValueError(f"Malformed shard missing features: {ref.shard_index}")
        example = ref.example
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
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                item[str(key)] = value[ref.row_index]
        return item

    @staticmethod
    def _read_index(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing sharded cache index: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or "shards" not in payload:
            raise ValueError(f"Malformed sharded cache index: {path}")
        return payload

    def _example_locations(
        self,
        all_examples: Sequence[VideoExample],
    ) -> dict[str, tuple[int, int]]:
        schema_version = int(self.index.get("schema_version", self.index.get("version", 1)))
        if schema_version != SHARDED_CACHE_SCHEMA_VERSION and not self.allow_legacy_shards:
            raise ValueError(
                "Refusing legacy sharded cache for training. "
                f"Expected schema_version={SHARDED_CACHE_SCHEMA_VERSION}; "
                "rebuild mixed v2 shards or set allow_legacy_shards=True for debugging."
            )
        locations = self.index.get("example_locations")
        if isinstance(locations, dict):
            if schema_version == SHARDED_CACHE_SCHEMA_VERSION:
                return self._v2_example_locations(locations)
            return self._legacy_example_locations(locations)
        if schema_version == SHARDED_CACHE_SCHEMA_VERSION:
            raise ValueError("Schema v2 sharded cache requires `example_locations` in index.json.")
        return self._example_locations_from_shards()

    def _v2_example_locations(
        self,
        locations: Mapping[str, Any],
    ) -> dict[str, tuple[int, int]]:
        parsed: dict[str, tuple[int, int]] = {}
        for key, value in locations.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"Malformed v2 shard location for {key}.")
            class_name = str(value.get("class_name", ""))
            filename = str(value.get("filename", ""))
            if f"{class_name}/{filename}" != str(key):
                raise ValueError(f"Malformed v2 shard location key mismatch for {key}.")
            parsed[str(key)] = (int(value["shard_index"]), int(value["row_index"]))
        return parsed

    def _legacy_example_locations(
        self,
        locations: Mapping[str, Any],
    ) -> dict[str, tuple[int, int]]:
        return {
            str(key): (int(value["shard_index"]), int(value["row_index"]))
            for key, value in locations.items()
            if isinstance(value, Mapping)
        }

    def _example_locations_from_shards(self) -> dict[str, tuple[int, int]]:
        locations: dict[str, tuple[int, int]] = {}
        for shard_index, shard_record in enumerate(self.shards):
            path = self.sharded_cache_dir / str(shard_record["path"])
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, Mapping):
                raise ValueError(f"Malformed shard payload: {path}")
            examples = payload.get("examples")
            if not isinstance(examples, list):
                raise ValueError(f"Malformed shard examples: {path}")
            for row_index, row in enumerate(examples):
                if not isinstance(row, Mapping):
                    raise ValueError(f"Malformed shard example row: {path}:{row_index}")
                locations[cache_row_key(row)] = (shard_index, row_index)
        return locations

    def _ref_for_example(
        self,
        example: VideoExample,
        locations: Mapping[str, tuple[int, int]],
    ) -> ShardedFeatureRef:
        key = cache_example_key(example, self.dataset_root)
        location = locations.get(key)
        if location is None:
            raise KeyError(f"Selected example is absent from sharded cache index: {key}")
        return ShardedFeatureRef(
            example=example,
            shard_index=location[0],
            row_index=location[1],
        )

    def _load_shard(self, shard_index: int) -> Mapping[str, Any]:
        if self._loaded_shard_index == shard_index and self._loaded_shard is not None:
            return self._loaded_shard
        shard_record = self.shards[shard_index]
        path = self.sharded_cache_dir / str(shard_record["path"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Malformed shard payload: {path}")
        self._loaded_shard_index = shard_index
        self._loaded_shard = payload
        return payload

    def _assert_ref_matches_shard_row(
        self,
        ref: ShardedFeatureRef,
        shard: Mapping[str, Any],
    ) -> None:
        rows = shard.get("examples")
        if not isinstance(rows, list) or ref.row_index >= len(rows):
            raise ValueError(f"Malformed shard examples for shard {ref.shard_index}.")
        row = rows[ref.row_index]
        if not isinstance(row, Mapping):
            raise ValueError(f"Malformed shard example row for shard {ref.shard_index}.")
        expected_key = cache_example_key(ref.example, self.dataset_root)
        actual_key = cache_row_key(row)
        if expected_key != actual_key:
            raise ValueError(
                "Sharded cache row mismatch: "
                f"expected {expected_key}, got {actual_key} "
                f"at shard={ref.shard_index} row={ref.row_index}."
            )


class ShardedFeatureBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        refs: Sequence[ShardedFeatureRef],
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        batch_strategy: str = "mixed_shard_local",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")
        if batch_strategy not in {"mixed_shard_local", "global_shuffle", "shard_local"}:
            raise ValueError(f"Unsupported sharded batch strategy: {batch_strategy}")
        self.refs = list(refs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.batch_strategy = batch_strategy
        if self.batch_strategy == "mixed_shard_local" and self.shuffle:
            self._reject_class_blocked_groups()

    def __iter__(self) -> Iterator[list[int]]:
        if self.batch_strategy == "global_shuffle":
            yield from self._global_batches()
            self.epoch += 1
            return
        groups = self._groups_by_shard()
        rng = random.Random(self.seed + self.epoch)
        shard_indices = list(groups)
        if self.shuffle:
            rng.shuffle(shard_indices)
        for shard_index in shard_indices:
            indices = list(groups[shard_index])
            if self.batch_strategy == "mixed_shard_local":
                yield from self._mixed_batches(indices, rng)
            else:
                if self.shuffle:
                    rng.shuffle(indices)
                for start in range(0, len(indices), self.batch_size):
                    yield indices[start : start + self.batch_size]
        self.epoch += 1

    def __len__(self) -> int:
        if self.batch_strategy == "global_shuffle":
            return (len(self.refs) + self.batch_size - 1) // self.batch_size
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self._groups_by_shard().values()
        )

    def _groups_by_shard(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for index, ref in enumerate(self.refs):
            groups.setdefault(ref.shard_index, []).append(index)
        return groups

    def _global_batches(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.refs)))
        if self.shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            yield indices[start : start + self.batch_size]

    def _mixed_batches(self, indices: list[int], rng: random.Random) -> Iterator[list[int]]:
        by_label: dict[int, list[int]] = {}
        for index in indices:
            label = int(self.refs[index].example.label)
            by_label.setdefault(label, []).append(index)
        if self.shuffle:
            for label_indices in by_label.values():
                rng.shuffle(label_indices)
        labels = sorted(by_label)
        if self.shuffle:
            rng.shuffle(labels)
        if len(labels) <= 1:
            ordered = list(indices)
            if self.shuffle:
                rng.shuffle(ordered)
            for start in range(0, len(ordered), self.batch_size):
                yield ordered[start : start + self.batch_size]
            return

        positions = dict.fromkeys(labels, 0)
        remaining = sum(len(label_indices) for label_indices in by_label.values())
        label_cursor = 0
        while remaining > 0:
            batch: list[int] = []
            while len(batch) < self.batch_size and remaining > 0:
                attempts = 0
                while attempts < len(labels):
                    label = labels[label_cursor % len(labels)]
                    label_cursor += 1
                    attempts += 1
                    position = positions[label]
                    if position >= len(by_label[label]):
                        continue
                    batch.append(by_label[label][position])
                    positions[label] += 1
                    remaining -= 1
                    break
                else:
                    break
            if batch:
                yield batch

    def _reject_class_blocked_groups(self) -> None:
        global_labels = {int(ref.example.label) for ref in self.refs}
        if len(global_labels) <= 1:
            return
        blocked: list[str] = []
        for shard_index, indices in self._groups_by_shard().items():
            labels = {int(self.refs[index].example.label) for index in indices}
            if len(labels) == 1 and len(indices) >= self.batch_size * 2:
                blocked.append(
                    f"shard={shard_index} label={next(iter(labels))} count={len(indices)}"
                )
        if blocked:
            sample = "; ".join(blocked[:5])
            raise ValueError(
                "Refusing class-blocked sharded batches. "
                "Rebuild schema v2 stratified shards or use batch_strategy=global_shuffle "
                f"for debugging. Examples: {sample}"
            )
