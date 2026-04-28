from __future__ import annotations

import ast
import inspect
import json
import textwrap
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "registry_fusion.yaml"

from branches.compression import LatentQueryPooling, TemporalLatentQueryPooling
from extractors import EYE_GAZE_COLUMNS, FACE_MESH_CONTOUR_INDICES
from frame_config import resolve_modality_frame_count, resolve_modality_frame_counts
from fusion import TokenBankFusion, prepare_token_bank
from registry import FIXED_SLOT_MODALITIES, MODALITY_TO_ID, build_registry

MODALITY_COLORS: dict[str, tuple[str, str]] = {
    "rgb": ("#E4572E", "#F6C4B7"),
    "eye_gaze": ("#4C956C", "#C8E5D5"),
    "face_mesh": ("#F4A261", "#FCE1C8"),
    "fau": ("#2E86AB", "#BFDBE8"),
    "rppg": ("#7B5EA7", "#D9CCE9"),
    "depth": ("#5B8E7D", "#CFE4DC"),
}
DEFAULT_COLOR = ("#6C757D", "#D8DDE3")
DISABLED_COLORS = ("#7B8794", "#F3F4F6")


@dataclass(frozen=True)
class SourceRef:
    path: str
    symbol: str
    line: int | None


@dataclass(frozen=True)
class StageSpec:
    id: str
    title: str
    detail: str
    kind: str
    source: SourceRef | None


@dataclass(frozen=True)
class ComponentSpec:
    id: str
    title: str
    kind: str
    cluster: str
    enabled: bool
    input_summary: str
    output_summary: str
    token_formula: str
    token_count: int | None
    note: str
    stroke_color: str
    fill_color: str
    source: SourceRef | None
    stages: tuple[StageSpec, ...]


@dataclass(frozen=True)
class EdgeSpec:
    source: str
    target: str
    label: str | None = None


@dataclass(frozen=True)
class ArchitectureSpec:
    config_path: str
    frames: Mapping[str, int]
    dim: int
    device: str
    enabled_modalities: tuple[str, ...]
    fixed_slot_modalities: tuple[str, ...]
    total_tokens: int
    enabled_token_count: int
    components: tuple[ComponentSpec, ...]
    edges: tuple[EdgeSpec, ...]
    fusion: Mapping[str, Any]


@dataclass(frozen=True)
class AssignmentEvent:
    target: str
    kind: str
    detail: str
    line: int
    attr: str | None = None
    batch_key: str | None = None


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be YAML mapping: {path}")
    return data


def _relative_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def source_ref(obj: object, symbol: str) -> SourceRef | None:
    try:
        source_file = inspect.getsourcefile(obj)
        _, start_line = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        return None
    if source_file is None:
        return None
    return SourceRef(path=_relative_path(source_file), symbol=symbol, line=start_line)


def architecture_spec_to_dict(spec: ArchitectureSpec) -> dict[str, Any]:
    return asdict(spec)


def architecture_spec_to_json(spec: ArchitectureSpec) -> str:
    return json.dumps(architecture_spec_to_dict(spec), indent=2, sort_keys=True) + "\n"


def describe_module(module: nn.Module) -> str:
    if (
        isinstance(module, nn.Sequential)
        and len(module) == 3
        and isinstance(module[0], nn.Linear)
        and isinstance(module[1], nn.GELU)
        and isinstance(module[2], nn.Linear)
    ):
        return f"MLP({module[0].in_features}->{module[0].out_features}->{module[2].out_features})"
    if isinstance(module, nn.LazyLinear):
        return f"LazyLinear(*->{module.out_features})"
    if isinstance(module, nn.Linear):
        return f"Linear({module.in_features}->{module.out_features})"
    if isinstance(module, TemporalLatentQueryPooling):
        return (
            "TemporalLatentQueryPooling("
            f"output_tokens={module.pool.output_tokens}, positional_bias=sinusoidal)"
        )
    if isinstance(module, LatentQueryPooling):
        return f"LatentQueryPooling(output_tokens={module.output_tokens})"
    if isinstance(module, nn.Embedding):
        return f"Embedding(num_embeddings={module.num_embeddings}, dim={module.embedding_dim})"
    if isinstance(module, nn.LayerNorm):
        normalized_shape = module.normalized_shape
        if isinstance(normalized_shape, tuple):
            normalized_shape_text = "x".join(str(item) for item in normalized_shape)
        else:
            normalized_shape_text = str(normalized_shape)
        return f"LayerNorm({normalized_shape_text})"
    if isinstance(module, nn.TransformerEncoder):
        first_layer = module.layers[0]
        return (
            f"TransformerEncoderLayer x{len(module.layers)}"
            f" (heads={first_layer.self_attn.num_heads}, hidden={first_layer.linear1.out_features})"
        )
    return module.__class__.__name__


def _extract_batch_key(expr: ast.AST) -> str | None:
    if (
        isinstance(expr, ast.Subscript)
        and isinstance(expr.value, ast.Name)
        and expr.value.id == "batch"
    ):
        slice_node = expr.slice
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            return slice_node.value
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and isinstance(expr.func.value, ast.Name)
        and expr.func.value.id == "batch"
        and expr.func.attr == "get"
        and expr.args
        and isinstance(expr.args[0], ast.Constant)
        and isinstance(expr.args[0].value, str)
    ):
        return expr.args[0].value
    return None


def _extract_self_attr(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Call):
        return _extract_self_attr(expr.func)
    if isinstance(expr, ast.Attribute):
        if isinstance(expr.value, ast.Name) and expr.value.id == "self":
            return expr.attr
        return _extract_self_attr(expr.value)
    if isinstance(expr, ast.BinOp):
        return _extract_self_attr(expr.left) or _extract_self_attr(expr.right)
    return None


def _describe_assignment(target: str, expr: ast.AST) -> tuple[str, str, str | None, str | None]:
    batch_key = _extract_batch_key(expr)
    if batch_key is not None:
        return "batch_input", f"batch[{batch_key!r}]", None, batch_key

    if isinstance(expr, ast.Call):
        attr = _extract_self_attr(expr.func)
        if attr is not None:
            return "module_call", f"self.{attr}(...)", attr, None
        if isinstance(expr.func, ast.Attribute) and expr.func.attr == "reshape":
            return "tensor_op", "reshape(...)", None, None
        if (
            isinstance(expr.func, ast.Attribute)
            and isinstance(expr.func.value, ast.Name)
            and expr.func.value.id == "torch"
        ):
            return "tensor_op", f"torch.{expr.func.attr}(...) ", None, None
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        attr = _extract_self_attr(expr)
        if attr is not None:
            return "tensor_op", f"add self.{attr}(...) ", attr, None
        return "tensor_op", "add(...)", None, None
    return "other", ast.unparse(expr), None, None


def extract_assignment_events(func: Callable[..., Any]) -> tuple[AssignmentEvent, ...]:
    source_text = textwrap.dedent(inspect.getsource(func))
    _, start_line = inspect.getsourcelines(func)
    module = ast.parse(source_text)
    func_def = next(
        node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    events: list[AssignmentEvent] = []
    for stmt in func_def.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            continue
        kind, detail, attr, batch_key = _describe_assignment(target.id, stmt.value)
        events.append(
            AssignmentEvent(
                target=target.id,
                kind=kind,
                detail=detail,
                attr=attr,
                batch_key=batch_key,
                line=start_line + stmt.lineno - 1,
            )
        )
    return tuple(events)


def _component_colors(name: str, enabled: bool) -> tuple[str, str]:
    if not enabled:
        return DISABLED_COLORS
    return MODALITY_COLORS.get(name, DEFAULT_COLOR)


def _display_name(name: str) -> str:
    labels = {
        "rgb": "RGB",
        "fau": "FAU",
        "rppg": "rPPG",
        "eye_gaze": "Eye Gaze",
        "face_mesh": "Face Mesh",
        "depth": "Depth",
    }
    return labels.get(name, name.replace("_", " ").title())


def _stage(
    component_id: str,
    suffix: str,
    title: str,
    detail: str,
    kind: str,
    source: SourceRef | None,
) -> StageSpec:
    return StageSpec(
        id=f"{component_id}.{suffix}",
        title=title,
        detail=detail,
        kind=kind,
        source=source,
    )


def _branch_input_summary(name: str, config: Mapping[str, Any]) -> str:
    frames = resolve_modality_frame_count(config, name)
    if name == "rgb":
        return "rgb_features [B, N_rgb, F_rgb]"
    if name == "eye_gaze":
        return f"eye_gaze [B, {frames}, {len(EYE_GAZE_COLUMNS)}]"
    if name == "face_mesh":
        return f"face_mesh [B, {frames}, {len(FACE_MESH_CONTOUR_INDICES)}, 3]"
    if name == "fau":
        num_classes = int(config.get("fau", {}).get("num_classes", 12))
        return f"fau_features [B, {frames}, {num_classes}, F_fau]"
    if name == "rppg":
        return f"rppg_features [B, {frames}, F_rppg] + rppg_waveform [B, {frames}]"
    if name == "depth":
        feature_dim = int(config.get("depth", {}).get("feature_dim", 384))
        return f"depth_features [B, {frames}, {feature_dim}]"
    return "batch features"


def _branch_output_summary(branch: nn.Module, dim: int) -> str:
    slot_count = int(branch.slot_count)
    return f"tokens [B, {slot_count}, {dim}] + time_ids [0..{slot_count - 1}]"


def _branch_token_formula(name: str, branch: nn.Module, config: Mapping[str, Any]) -> str:
    frames = resolve_modality_frame_count(config, name)
    slot_count = int(branch.slot_count)
    if hasattr(branch, "frame_pool") and hasattr(branch.frame_pool, "output_tokens"):
        return (
            f"frames={frames}, frame_query_tokens={branch.frame_pool.output_tokens}, "
            f"final_slots={slot_count}"
        )
    if hasattr(branch, "point_pool") and hasattr(branch.point_pool, "output_tokens"):
        return (
            f"frames={frames}, point_query_tokens={branch.point_pool.output_tokens}, "
            f"final_slots={slot_count}"
        )
    return f"slot_count={slot_count}"


def _branch_note(name: str) -> str:
    if name == "rgb":
        return "Temporal clip features are projected to fusion width, then pooled to fixed slots."
    if name == "eye_gaze":
        return "Per-frame gaze blendshapes stay ordered until temporal latent-query pooling."
    if name == "face_mesh":
        return (
            "Per-point MLP happens before point pooling, then frame tokens compress to clip slots."
        )
    if name == "fau":
        return "FAU features use two-stage pooling: within frame first, then across the clip."
    if name == "rppg":
        return "Waveform is a side output; only temporal features enter projection and pooling."
    if name == "depth":
        return (
            "DepthAnything hidden maps are spatially mean-pooled per frame before temporal pooling."
        )
    return "Custom branch."


def _normalize_branch_stage(
    branch: nn.Module, component_id: str, event: AssignmentEvent
) -> StageSpec | None:
    component_ref = source_ref(branch.encode, f"{branch.__class__.__name__}.encode")
    if event.kind == "batch_input" and event.batch_key in branch.required_keys():
        return _stage(
            component_id,
            event.target,
            "Input",
            f"{event.batch_key} from batch",
            "input",
            SourceRef(component_ref.path, component_ref.symbol, event.line)
            if component_ref
            else None,
        )
    if event.kind == "module_call" and event.attr is not None:
        module = getattr(branch, event.attr, None)
        if not isinstance(module, nn.Module):
            return None
        if event.attr == "proj":
            title = "Project"
        elif event.attr == "frame_pool":
            title = "Frame Pool"
        elif event.attr == "point_pool":
            title = "Point Pool"
        elif event.attr == "clip_pool":
            title = "Clip Pool"
        elif event.attr == "pool":
            title = "Temporal Pool"
        else:
            title = event.attr.replace("_", " ").title()
        source = (
            SourceRef(component_ref.path, component_ref.symbol, event.line)
            if component_ref
            else None
        )
        return _stage(component_id, event.attr, title, describe_module(module), "module", source)
    if event.kind == "tensor_op" and event.target == "clip_tokens" and "reshape" in event.detail:
        source = (
            SourceRef(component_ref.path, component_ref.symbol, event.line)
            if component_ref
            else None
        )
        return _stage(
            component_id,
            event.target,
            "Flatten Frame Tokens",
            "reshape pooled frame tokens into clip sequence",
            "tensor_op",
            source,
        )
    return None


def build_branch_component(
    name: str,
    branch: nn.Module,
    config: Mapping[str, Any],
    enabled: bool,
) -> ComponentSpec:
    stroke_color, fill_color = _component_colors(name, enabled)
    events = extract_assignment_events(branch.encode)
    stages = tuple(
        stage
        for event in events
        if (stage := _normalize_branch_stage(branch, name, event)) is not None
    )
    return ComponentSpec(
        id=name,
        title=f"{_display_name(name)} Branch",
        kind="modality",
        cluster="modalities",
        enabled=enabled,
        input_summary=_branch_input_summary(name, config),
        output_summary=_branch_output_summary(branch, int(config["dim"])),
        token_formula=_branch_token_formula(name, branch, config),
        token_count=int(branch.slot_count),
        note=_branch_note(name)
        if enabled
        else "Disabled modality still reserves slots in the fixed token bank.",
        stroke_color=stroke_color,
        fill_color=fill_color,
        source=source_ref(branch.encode, f"{branch.__class__.__name__}.encode"),
        stages=stages,
    )


def build_input_component(config: Mapping[str, Any]) -> ComponentSpec:
    frame_counts = resolve_modality_frame_counts(config, FIXED_SLOT_MODALITIES)
    frames = ", ".join(f"{name}={count}" for name, count in frame_counts.items())
    return ComponentSpec(
        id="input_clip",
        title="Input Clip",
        kind="input",
        cluster="inputs",
        enabled=True,
        input_summary="video_by_modality + video_rgb_frames_by_modality",
        output_summary=f"sampled frames per modality: {frames}",
        token_formula="",
        token_count=None,
        note="Raw video and RGB frame lists feed extractors with modality-specific frame counts.",
        stroke_color="#7B8794",
        fill_color="#FFFFFF",
        source=None,
        stages=(
            _stage(
                "input_clip",
                "sample",
                "Sample Frames",
                f"frames configured in YAML: {frames}",
                "input",
                None,
            ),
        ),
    )


def build_token_bank_component(
    config: Mapping[str, Any],
    components: tuple[ComponentSpec, ...],
) -> ComponentSpec:
    prepare_ref = source_ref(prepare_token_bank, "prepare_token_bank")
    total_tokens = sum(
        component.token_count or 0 for component in components if component.kind == "modality"
    )
    enabled_tokens = sum(
        component.token_count or 0
        for component in components
        if component.kind == "modality" and component.enabled
    )
    return ComponentSpec(
        id="token_bank",
        title="Token Bank",
        kind="token_bank",
        cluster="shared",
        enabled=True,
        input_summary="modality tokens + token_mask + time_ids + modality_ids",
        output_summary=f"tokens [B, {total_tokens}, {config['dim']}] + metadata vectors",
        token_formula=f"enabled_slots={enabled_tokens}, fixed_slots={total_tokens}",
        token_count=total_tokens,
        note="Disabled modalities produce zero-filled reserved slots and False in token_mask.",
        stroke_color="#B38600",
        fill_color="#FFF4CC",
        source=prepare_ref,
        stages=(
            _stage(
                "token_bank",
                "validate",
                "Validate Outputs",
                "check each enabled modality emits the configured slot count",
                "tensor_op",
                prepare_ref,
            ),
            _stage(
                "token_bank",
                "reserve",
                "Reserve Disabled Slots",
                "zero-fill tokens and masks for missing modalities in fixed order",
                "tensor_op",
                prepare_ref,
            ),
            _stage(
                "token_bank",
                "concat",
                "Concatenate Bank",
                "torch.cat tokens, masks, time_ids, and modality_ids in fixed modality order",
                "tensor_op",
                prepare_ref,
            ),
        ),
    )


def _normalize_fusion_stage(
    fusion: TokenBankFusion,
    component_id: str,
    event: AssignmentEvent,
) -> StageSpec | None:
    forward_ref = source_ref(fusion.forward, "TokenBankFusion.forward")
    source = SourceRef(forward_ref.path, forward_ref.symbol, event.line) if forward_ref else None
    if event.kind == "batch_input" and event.batch_key == "tokens":
        return _stage(
            component_id,
            event.target,
            "Input",
            "token bank from prepare_token_bank()",
            "input",
            source,
        )
    if event.kind == "tensor_op" and event.attr == "time_embedding":
        return _stage(
            component_id,
            event.target,
            "Add Time Embedding",
            describe_module(fusion.time_embedding),
            "tensor_op",
            source,
        )
    if event.kind == "tensor_op" and event.attr == "modality_embedding":
        return _stage(
            component_id,
            event.target,
            "Add Modality Embedding",
            describe_module(fusion.modality_embedding),
            "tensor_op",
            source,
        )
    if event.kind == "module_call" and event.attr == "cls_token" and event.target == "cls_tokens":
        return _stage(
            component_id,
            event.target,
            "Expand CLS",
            f"learned cls_token [1, 1, {fusion.dim}] -> batch-aligned",
            "tensor_op",
            source,
        )
    if event.kind == "tensor_op" and event.target == "fused_tokens" and "torch.cat" in event.detail:
        return _stage(
            component_id,
            event.target,
            "Prepend CLS",
            "torch.cat([cls_tokens, token_states], dim=1)",
            "tensor_op",
            source,
        )
    if event.kind == "tensor_op" and event.target == "src_key_padding_mask":
        return _stage(
            component_id,
            event.target,
            "Build Padding Mask",
            "combine cls padding and inverted token_mask",
            "tensor_op",
            source,
        )
    if event.kind == "module_call" and event.attr == "encoder":
        return _stage(
            component_id,
            event.target,
            "Transformer Encoder",
            describe_module(fusion.encoder),
            "module",
            source,
        )
    if event.kind == "module_call" and event.attr == "output_norm":
        return _stage(
            component_id,
            event.target,
            "LayerNorm",
            describe_module(fusion.output_norm),
            "module",
            source,
        )
    return None


def build_fusion_component(config: Mapping[str, Any], total_tokens: int) -> ComponentSpec:
    fusion_config = config["fusion"]
    fusion = TokenBankFusion(
        dim=int(config["dim"]),
        num_layers=int(fusion_config["num_layers"]),
        num_heads=int(fusion_config["num_heads"]),
        mlp_ratio=float(fusion_config["mlp_ratio"]),
        dropout=float(fusion_config["dropout"]),
        max_time_steps=int(fusion_config["max_time_steps"]),
        num_modalities=len(MODALITY_TO_ID),
    )
    events = extract_assignment_events(fusion.forward)
    normalized_stages = [
        _stage(
            "fusion_core",
            "input",
            "Input",
            "token bank from prepare_token_bank()",
            "input",
            source_ref(fusion.forward, "TokenBankFusion.forward"),
        )
    ]
    normalized_stages.extend(
        stage
        for event in events
        if (stage := _normalize_fusion_stage(fusion, "fusion_core", event)) is not None
    )
    stages = tuple(normalized_stages)
    return ComponentSpec(
        id="fusion_core",
        title="TokenBankFusion",
        kind="fusion",
        cluster="shared",
        enabled=True,
        input_summary=f"token bank [B, {total_tokens}, {config['dim']}]",
        output_summary=f"cls_token [B, {config['dim']}] + fused_tokens [B, {1 + total_tokens}, {config['dim']}]",
        token_formula=(
            f"layers={fusion_config['num_layers']}, heads={fusion_config['num_heads']}, "
            f"mlp_ratio={fusion_config['mlp_ratio']}"
        ),
        token_count=None,
        note="Time and modality embeddings are added before CLS prepend, transformer mixing, and output norm.",
        stroke_color="#295C8A",
        fill_color="#DCEAF7",
        source=source_ref(fusion.forward, "TokenBankFusion.forward"),
        stages=stages,
    )


def build_output_components(
    config: Mapping[str, Any], total_tokens: int
) -> tuple[ComponentSpec, ComponentSpec]:
    dim = int(config["dim"])
    cls_output = ComponentSpec(
        id="cls_output",
        title="CLS Output",
        kind="output",
        cluster="outputs",
        enabled=True,
        input_summary="fusion_core.cls_token",
        output_summary=f"[B, {dim}] fused clip summary",
        token_formula="",
        token_count=None,
        note="Primary clip-level fused representation.",
        stroke_color="#295C8A",
        fill_color="#FFFFFF",
        source=None,
        stages=(),
    )
    fused_tokens = ComponentSpec(
        id="fused_tokens",
        title="Fused Tokens",
        kind="output",
        cluster="outputs",
        enabled=True,
        input_summary="fusion_core.fused_tokens",
        output_summary=f"[B, {1 + total_tokens}, {dim}]",
        token_formula="",
        token_count=None,
        note="Full mixed token sequence with prepended CLS token.",
        stroke_color="#295C8A",
        fill_color="#FFFFFF",
        source=None,
        stages=(),
    )
    return cls_output, fused_tokens


def build_architecture_spec(
    config: Mapping[str, Any],
    config_path: Path = DEFAULT_CONFIG,
) -> ArchitectureSpec:
    dim = int(config["dim"])
    enabled_modalities = tuple(
        str(item).strip() for item in config.get("modalities", []) if str(item).strip()
    )
    registry = build_registry(dim=dim, config=config)

    input_component = build_input_component(config)
    modality_components = tuple(
        build_branch_component(
            name=name,
            branch=registry[name],
            config=config,
            enabled=name in enabled_modalities,
        )
        for name in FIXED_SLOT_MODALITIES
    )
    token_bank_component = build_token_bank_component(config, modality_components)
    fusion_component = build_fusion_component(config, token_bank_component.token_count or 0)
    cls_output, fused_tokens = build_output_components(
        config, token_bank_component.token_count or 0
    )

    edges = [
        EdgeSpec(source="input_clip", target=component.id) for component in modality_components
    ]
    edges.extend(
        EdgeSpec(source=component.id, target="token_bank") for component in modality_components
    )
    edges.append(EdgeSpec(source="token_bank", target="fusion_core"))
    edges.append(EdgeSpec(source="fusion_core", target="cls_output"))
    edges.append(EdgeSpec(source="fusion_core", target="fused_tokens"))

    total_tokens = sum(component.token_count or 0 for component in modality_components)
    enabled_token_count = sum(
        component.token_count or 0 for component in modality_components if component.enabled
    )
    return ArchitectureSpec(
        config_path=_relative_path(config_path),
        frames=resolve_modality_frame_counts(config, FIXED_SLOT_MODALITIES),
        dim=dim,
        device=str(config.get("device", "unknown")),
        enabled_modalities=enabled_modalities,
        fixed_slot_modalities=FIXED_SLOT_MODALITIES,
        total_tokens=total_tokens,
        enabled_token_count=enabled_token_count,
        components=(
            input_component,
            *modality_components,
            token_bank_component,
            fusion_component,
            cls_output,
            fused_tokens,
        ),
        edges=tuple(edges),
        fusion=dict(config["fusion"]),
    )
