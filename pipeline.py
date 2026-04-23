from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import yaml

from branches.compression import validate_branch_token_config
from encoders import EncoderFactoryResult, build_local_encoders
from extractors import FeatureExtractor, build_extractors_from_encoders
from fusion import FusionOutput, TokenBankFusion, prepare_token_bank
from registry import FIXED_SLOT_MODALITIES, MODALITY_TO_ID, build_registry, registry_slot_counts


OPTIONAL_FEATURE_KEYS: dict[str, tuple[str, ...]] = {
    "fau": ("fau_au_logits", "fau_au_edge_logits"),
    "rppg": ("rppg_waveform",),
}


@dataclass(frozen=True)
class FusionPipelineBuildResult:
    pipeline: "ClipFusionPipeline"
    device: torch.device
    warnings: tuple[str, ...]


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"`{key}` must be a YAML mapping.")
    return value


def _require_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise ValueError(f"`{key}` must be an integer.")
    return value


def _require_float(config: Mapping[str, Any], key: str) -> float:
    value = config.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"`{key}` must be a number.")
    return float(value)


def _require_str(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{key}` must be a non-empty string.")
    return value.strip()


def _require_modalities(config: Mapping[str, Any]) -> tuple[str, ...]:
    value = config.get("modalities")
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError("`modalities` must be a non-empty YAML list of strings.")
    return tuple(item.strip() for item in value if item.strip())


def _encoder_modules_from_result(encoder_result: EncoderFactoryResult) -> nn.ModuleDict:
    modules = nn.ModuleDict()
    if encoder_result.rgb_encoder is not None:
        modules["rgb"] = encoder_result.rgb_encoder
    if encoder_result.fau_encoder is not None:
        modules["fau"] = encoder_result.fau_encoder
    if encoder_result.rppg_encoder is not None:
        modules["rppg"] = encoder_result.rppg_encoder
    return modules


def _optional_path(config: Mapping[str, Any], key: str) -> Path | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{key}` must be a non-empty string path or null.")
    return Path(value)


def load_pipeline_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {config_path} must be a YAML mapping.")
    return data


def resolve_model_device(config: Mapping[str, Any]) -> torch.device:
    device_spec = _require_str(config, "device").lower()
    if device_spec == "cpu":
        return torch.device("cpu")
    if device_spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested `device: cuda`, but CUDA is not available.")
        return torch.device("cuda")
    raise ValueError("`device` must be either `cpu` or `cuda`.")


def build_fusion_from_config(config: Mapping[str, Any]) -> TokenBankFusion:
    fusion = _require_mapping(config, "fusion")
    dim = _require_int(config, "dim")
    return TokenBankFusion(
        dim=dim,
        num_layers=_require_int(fusion, "num_layers"),
        num_heads=_require_int(fusion, "num_heads"),
        mlp_ratio=_require_float(fusion, "mlp_ratio"),
        dropout=_require_float(fusion, "dropout"),
        max_time_steps=_require_int(fusion, "max_time_steps"),
        num_modalities=len(MODALITY_TO_ID),
    )


def load_fusion_checkpoint(
    fusion_module: TokenBankFusion,
    checkpoint_path: Path | None,
) -> bool:
    if checkpoint_path is None:
        return False
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Fusion checkpoint does not exist: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("Fusion checkpoint must be a state_dict mapping or contain `state_dict`.")
    fusion_module.load_state_dict(state)
    return True


def fuse_selected_modalities(
    registry: nn.ModuleDict,
    batch: Mapping[str, torch.Tensor],
    enabled_modalities: Sequence[str],
    fusion_module: TokenBankFusion,
) -> FusionOutput:
    outputs_by_name = {name: registry[name].encode(batch) for name in enabled_modalities}
    token_bank = prepare_token_bank(
        outputs_by_name=outputs_by_name,
        enabled_modalities=enabled_modalities,
        modality_to_id=MODALITY_TO_ID,
        fixed_slot_modalities=FIXED_SLOT_MODALITIES,
        slot_counts=registry_slot_counts(registry),
    )
    cls_token, fused_tokens = fusion_module(
        tokens=token_bank.tokens,
        token_mask=token_bank.token_mask,
        time_ids=token_bank.time_ids,
        modality_ids=token_bank.modality_ids,
    )
    return FusionOutput(
        fused=cls_token,
        tokens=token_bank.tokens,
        token_mask=token_bank.token_mask,
        time_ids=token_bank.time_ids,
        modality_ids=token_bank.modality_ids,
        modality_names=token_bank.modality_names,
        cls_token=cls_token,
        fused_tokens=fused_tokens,
    )


class ClipFusionPipeline(nn.Module):
    def __init__(
        self,
        registry: nn.ModuleDict,
        fusion_module: TokenBankFusion,
        enabled_modalities: Sequence[str],
        extractors: Mapping[str, FeatureExtractor] | None = None,
        encoder_modules: nn.ModuleDict | None = None,
    ) -> None:
        super().__init__()
        self.registry = registry
        self.fusion_module = fusion_module
        self.enabled_modalities = tuple(enabled_modalities)
        self.extractors = dict(extractors or {})
        self.encoder_modules = encoder_modules if encoder_modules is not None else nn.ModuleDict()
        self.last_feature_timings: dict[str, float] = {}

    def _device(self) -> torch.device:
        parameter = next(self.parameters(), None)
        if parameter is not None:
            return parameter.device
        return torch.device("cpu")

    def _move_feature_batch_to_device(self, feature_batch: Mapping[str, Any]) -> dict[str, Any]:
        target_device = self._device()
        moved: dict[str, Any] = {}
        for key, value in feature_batch.items():
            moved[key] = value.to(target_device) if isinstance(value, torch.Tensor) else value
        return moved

    def _has_precomputed_features(self, batch: Mapping[str, Any], modality_name: str) -> bool:
        return all(key in batch for key in self.registry[modality_name].required_keys())

    def _copy_feature_keys(
        self,
        batch: Mapping[str, Any],
        feature_batch: dict[str, Any],
        modality_name: str,
    ) -> None:
        for key in self.registry[modality_name].required_keys():
            feature_batch[key] = batch[key]
        for key in OPTIONAL_FEATURE_KEYS.get(modality_name, ()):
            if key in batch:
                feature_batch[key] = batch[key]

    def prepare_features(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        feature_batch: dict[str, Any] = {}
        feature_timings: dict[str, float] = {}
        for name in self.enabled_modalities:
            if self._has_precomputed_features(batch, name):
                self._copy_feature_keys(batch, feature_batch, name)
                feature_timings[name] = 0.0
                continue
            if name not in self.extractors:
                raise KeyError(
                    f"Missing extractor for modality `{name}` and precomputed features not provided."
                )
            extract_start = time.perf_counter()
            extracted = self.extractors[name].extract(batch)
            feature_timings[name] = time.perf_counter() - extract_start
            feature_batch.update(extracted)
        self.last_feature_timings = feature_timings
        return feature_batch

    def fuse(self, batch: Mapping[str, Any]) -> FusionOutput:
        feature_batch = self._move_feature_batch_to_device(self.prepare_features(batch))
        return fuse_selected_modalities(
            registry=self.registry,
            batch=dict(feature_batch),
            enabled_modalities=self.enabled_modalities,
            fusion_module=self.fusion_module,
        )

    def forward(self, batch: Mapping[str, Any]) -> FusionOutput:
        return self.fuse(batch)

    def close(self) -> None:
        for extractor in self.extractors.values():
            extractor.close()


def build_fusion_pipeline(
    config: Mapping[str, Any],
    modalities: Sequence[str] | None = None,
) -> FusionPipelineBuildResult:
    enabled_modalities = tuple(modalities or _require_modalities(config))
    device = resolve_model_device(config)
    fusion_config = _require_mapping(config, "fusion")
    validate_branch_token_config(
        config,
        modalities=FIXED_SLOT_MODALITIES,
        fusion_max_time_steps=_require_int(fusion_config, "max_time_steps"),
    )

    encoder_result = build_local_encoders(config, modalities=enabled_modalities)
    extractors_result = build_extractors_from_encoders(
        config=config,
        encoder_result=encoder_result,
        modalities=enabled_modalities,
    )
    fusion_module = build_fusion_from_config(config)
    load_fusion_checkpoint(
        fusion_module=fusion_module,
        checkpoint_path=_optional_path(fusion_config, "checkpoint_path"),
    )
    pipeline = ClipFusionPipeline(
        registry=build_registry(dim=_require_int(config, "dim"), config=config),
        fusion_module=fusion_module,
        enabled_modalities=enabled_modalities,
        extractors=extractors_result.extractors,
        encoder_modules=_encoder_modules_from_result(encoder_result),
    )
    pipeline = pipeline.to(device)
    return FusionPipelineBuildResult(
        pipeline=pipeline,
        device=device,
        warnings=extractors_result.warnings,
    )


def build_fusion_pipeline_from_yaml(
    path: str | Path,
    modalities: Sequence[str] | None = None,
) -> FusionPipelineBuildResult:
    return build_fusion_pipeline(load_pipeline_yaml(path), modalities=modalities)
