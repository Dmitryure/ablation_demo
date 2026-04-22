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
from registry import MODALITY_TO_ID, build_registry


OPTIONAL_FEATURE_KEYS: dict[str, tuple[str, ...]] = {
    "fau": ("fau_au_logits", "fau_au_edge_logits"),
    "rppg": ("rppg_waveform",),
}


@dataclass(frozen=True)
class PredictionOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    fusion_output: FusionOutput


@dataclass(frozen=True)
class ClassifierConfig:
    hidden_dim: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    freeze_encoders: bool
    lr_head: float
    lr_fusion: float
    pos_weight: float


@dataclass(frozen=True)
class PredictionBuildResult:
    model: "ClipRealFakePredictor"
    classifier_config: ClassifierConfig
    training_config: TrainingConfig
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


def _require_bool(config: Mapping[str, Any], key: str) -> bool:
    value = config.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"`{key}` must be a boolean.")
    return value


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


def _require_modality_weights(
    config: Mapping[str, Any],
    enabled_modalities: Sequence[str],
) -> dict[str, float]:
    value = config.get("modality_weights")
    if value is None:
        return {name: 1.0 for name in enabled_modalities}
    if not isinstance(value, Mapping):
        raise ValueError("`modality_weights` must be a YAML mapping when provided.")

    weights: dict[str, float] = {}
    for name in enabled_modalities:
        raw_weight = value.get(name, 1.0)
        if not isinstance(raw_weight, (int, float)):
            raise ValueError(f"`modality_weights.{name}` must be a number.")
        weight = float(raw_weight)
        if weight < 0.0:
            raise ValueError(f"`modality_weights.{name}` must be non-negative.")
        weights[name] = weight
    return weights


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


def load_prediction_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {config_path} must be a YAML mapping.")
    return data


def require_classifier_config(config: Mapping[str, Any]) -> ClassifierConfig:
    classifier = _require_mapping(config, "classifier")
    hidden_dim = _require_int(classifier, "hidden_dim")
    dropout = _require_float(classifier, "dropout")
    if hidden_dim <= 0:
        raise ValueError("`classifier.hidden_dim` must be positive.")
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("`classifier.dropout` must be in [0.0, 1.0).")
    return ClassifierConfig(hidden_dim=hidden_dim, dropout=dropout)


def require_training_config(config: Mapping[str, Any]) -> TrainingConfig:
    training = _require_mapping(config, "training")
    lr_head = _require_float(training, "lr_head")
    lr_fusion = _require_float(training, "lr_fusion")
    pos_weight = _require_float(training, "pos_weight")
    if lr_head <= 0.0:
        raise ValueError("`training.lr_head` must be positive.")
    if lr_fusion <= 0.0:
        raise ValueError("`training.lr_fusion` must be positive.")
    if pos_weight <= 0.0:
        raise ValueError("`training.pos_weight` must be positive.")
    return TrainingConfig(
        freeze_encoders=_require_bool(training, "freeze_encoders"),
        lr_head=lr_head,
        lr_fusion=lr_fusion,
        pos_weight=pos_weight,
    )


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


def build_binary_classification_loss(
    training_config: TrainingConfig,
    device: torch.device | None = None,
) -> nn.BCEWithLogitsLoss:
    pos_weight = torch.tensor([training_config.pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def fuse_selected_modalities_for_prediction(
    registry: nn.ModuleDict,
    batch: Mapping[str, torch.Tensor],
    enabled_modalities: Sequence[str],
    modality_weights: Mapping[str, float],
    fusion_module: TokenBankFusion,
) -> FusionOutput:
    outputs_by_name = {name: registry[name].encode(batch) for name in enabled_modalities}
    token_bank = prepare_token_bank(
        outputs_by_name=outputs_by_name,
        enabled_modalities=enabled_modalities,
        modality_to_id=MODALITY_TO_ID,
        modality_weights=modality_weights,
    )
    cls_token, fused_tokens = fusion_module(
        tokens=token_bank.weighted_tokens,
        time_ids=token_bank.time_ids,
        modality_ids=token_bank.modality_ids,
    )
    return FusionOutput(
        fused=cls_token,
        tokens=token_bank.tokens,
        time_ids=token_bank.time_ids,
        modality_ids=token_bank.modality_ids,
        modality_names=token_bank.modality_names,
        cls_token=cls_token,
        fused_tokens=fused_tokens,
    )


class VideoRealFakeHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        if cls_token.ndim != 2:
            raise ValueError(
                f"`cls_token` must have shape [B, dim] for clip-level classification, got {tuple(cls_token.shape)}"
            )
        return self.layers(cls_token)


class ClipRealFakePredictor(nn.Module):
    def __init__(
        self,
        registry: nn.ModuleDict,
        fusion_module: TokenBankFusion,
        classifier_head: VideoRealFakeHead,
        enabled_modalities: Sequence[str],
        modality_weights: Mapping[str, float],
        extractors: Mapping[str, FeatureExtractor] | None = None,
        encoder_modules: nn.ModuleDict | None = None,
    ) -> None:
        super().__init__()
        self.registry = registry
        self.fusion_module = fusion_module
        self.classifier_head = classifier_head
        self.enabled_modalities = tuple(enabled_modalities)
        self.modality_weights = {name: float(modality_weights[name]) for name in self.enabled_modalities}
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
        return fuse_selected_modalities_for_prediction(
            registry=self.registry,
            batch=dict(feature_batch),
            enabled_modalities=self.enabled_modalities,
            modality_weights=self.modality_weights,
            fusion_module=self.fusion_module,
        )

    def freeze_encoder_parameters(self) -> None:
        for parameter in self.encoder_modules.parameters():
            parameter.requires_grad_(False)

    def unfreeze_encoder_parameters(self) -> None:
        for parameter in self.encoder_modules.parameters():
            parameter.requires_grad_(True)

    def optimizer_parameter_groups(self, training_config: TrainingConfig) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        fusion_params = [param for param in self.registry.parameters() if param.requires_grad]
        fusion_params.extend(param for param in self.fusion_module.parameters() if param.requires_grad)
        head_params = [param for param in self.classifier_head.parameters() if param.requires_grad]
        encoder_params = [param for param in self.encoder_modules.parameters() if param.requires_grad]

        if fusion_params:
            groups.append({"params": fusion_params, "lr": training_config.lr_fusion})
        if head_params:
            groups.append({"params": head_params, "lr": training_config.lr_head})
        if encoder_params:
            groups.append({"params": encoder_params, "lr": training_config.lr_fusion})
        return groups

    def build_optimizer(
        self,
        training_config: TrainingConfig,
        weight_decay: float = 0.01,
    ) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            self.optimizer_parameter_groups(training_config),
            weight_decay=weight_decay,
        )

    def forward(self, batch: Mapping[str, Any]) -> PredictionOutput:
        fusion_output = self.fuse(batch)
        logits = self.classifier_head(fusion_output.cls_token)
        probs = torch.sigmoid(logits)
        return PredictionOutput(
            logits=logits,
            probs=probs,
            fusion_output=fusion_output,
        )

    def close(self) -> None:
        for extractor in self.extractors.values():
            extractor.close()


def build_prediction_model(
    config: Mapping[str, Any],
    modalities: Sequence[str] | None = None,
) -> PredictionBuildResult:
    enabled_modalities = tuple(modalities or _require_modalities(config))
    dim = _require_int(config, "dim")
    device = resolve_model_device(config)
    modality_weights = _require_modality_weights(config, enabled_modalities)
    classifier_config = require_classifier_config(config)
    training_config = require_training_config(config)
    fusion_config = _require_mapping(config, "fusion")
    validate_branch_token_config(
        config,
        modalities=enabled_modalities,
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
    model = ClipRealFakePredictor(
        registry=build_registry(dim=dim, config=config),
        fusion_module=fusion_module,
        classifier_head=VideoRealFakeHead(
            input_dim=dim,
            hidden_dim=classifier_config.hidden_dim,
            dropout=classifier_config.dropout,
        ),
        enabled_modalities=enabled_modalities,
        modality_weights=modality_weights,
        extractors=extractors_result.extractors,
        encoder_modules=_encoder_modules_from_result(encoder_result),
    )
    if training_config.freeze_encoders:
        model.freeze_encoder_parameters()
    model = model.to(device)
    return PredictionBuildResult(
        model=model,
        classifier_config=classifier_config,
        training_config=training_config,
        device=device,
        warnings=extractors_result.warnings,
    )


def build_prediction_model_from_yaml(
    path: str | Path,
    modalities: Sequence[str] | None = None,
) -> PredictionBuildResult:
    return build_prediction_model(load_prediction_yaml(path), modalities=modalities)
