from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "registry_fusion.yaml"
DEFAULT_DOCS_DIR = PROJECT_ROOT / "docs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from branches.compression import resolve_output_token_count

MODALITY_COLORS: dict[str, tuple[str, str]] = {
    "rgb": ("#E4572E", "#F6C4B7"),
    "eye_gaze": ("#4C956C", "#C8E5D5"),
    "face_mesh": ("#F4A261", "#FCE1C8"),
    "fau": ("#2E86AB", "#BFDBE8"),
    "rppg": ("#7B5EA7", "#D9CCE9"),
}
DEFAULT_COLOR = ("#6C757D", "#D8DDE3")


@dataclass(frozen=True)
class ModalityDoc:
    name: str
    title: str
    encoder_summary: str
    projector_summary: str
    token_formula: str
    token_count: int | None
    raw_weight: float
    normalized_weight: float
    stroke_color: str
    fill_color: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Graphviz-based model architecture docs from current config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to YAML config. Default: configs/registry_fusion.yaml",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help="Output docs directory. Default: docs/",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be YAML mapping: {path}")
    return data


def normalize_weights(modalities: list[str], raw_weights: dict[str, float]) -> dict[str, float]:
    total = sum(raw_weights[name] for name in modalities)
    if total <= 0.0:
        raise ValueError("At least one modality weight must be greater than zero.")
    return {name: raw_weights[name] / total for name in modalities}


def format_float(value: float) -> str:
    rounded = f"{value:.3f}"
    return rounded.rstrip("0").rstrip(".")


def modality_node_id(doc: ModalityDoc) -> str:
    return doc.name.replace("-", "_")


def rgb_token_count(frames: int) -> int | None:
    if frames != 16:
        return None
    return 8


def build_modality_doc(
    name: str,
    config: dict[str, Any],
    frames: int,
    raw_weight: float,
    normalized_weight: float,
    dim: int,
) -> ModalityDoc:
    stroke_color, fill_color = MODALITY_COLORS.get(name, DEFAULT_COLOR)
    if name == "rgb":
        backbone = "mvit_v2_s"
        token_count = rgb_token_count(frames)
        token_formula = str(token_count) if token_count is not None else "unknown"
        return ModalityDoc(
            name=name,
            title="RGB",
            encoder_summary=f"{backbone} temporal encoder",
            projector_summary=f"spatial-mean temporal tokens -> Linear(*, {dim})",
            token_formula=token_formula,
            token_count=token_count,
            raw_weight=raw_weight,
            normalized_weight=normalized_weight,
            stroke_color=stroke_color,
            fill_color=fill_color,
            note="One token per MViT time step after spatial pooling.",
        )
    if name == "eye_gaze":
        token_count = resolve_output_token_count(config, "eye_gaze")
        return ModalityDoc(
            name=name,
            title="Eye Gaze",
            encoder_summary="MediaPipe face landmarker",
            projector_summary=f"8 gaze blendshapes -> MLP -> temporal position encoding -> latent-query pool -> {dim}",
            token_formula=str(token_count),
            token_count=token_count,
            raw_weight=raw_weight,
            normalized_weight=normalized_weight,
            stroke_color=stroke_color,
            fill_color=fill_color,
            note="Per-frame gaze features compressed to a fixed clip token budget with order-aware pooling.",
        )
    if name == "face_mesh":
        contour_points = 36
        output_tokens_per_frame = resolve_output_token_count(config, "face_mesh")
        return ModalityDoc(
            name=name,
            title="Face Mesh",
            encoder_summary="MediaPipe face landmarker contour points",
            projector_summary=f"36 contour landmarks x (x,y,z) -> point MLP -> latent-query pool -> {dim}",
            token_formula=f"{frames} x {output_tokens_per_frame}",
            token_count=frames * output_tokens_per_frame,
            raw_weight=raw_weight,
            normalized_weight=normalized_weight,
            stroke_color=stroke_color,
            fill_color=fill_color,
            note=f"{contour_points} contour points compressed to {output_tokens_per_frame} tokens per frame.",
        )
    if name == "fau":
        fau_config = config.get("fau", {})
        backbone = str(fau_config.get("backbone", "unknown"))
        num_classes = int(fau_config.get("num_classes", 12))
        output_tokens_per_frame = resolve_output_token_count(config, "fau")
        return ModalityDoc(
            name=name,
            title="FAU",
            encoder_summary=f"{backbone} + ME-GraphAU",
            projector_summary=f"AU graph features -> Linear(*, {dim}) -> latent-query pool -> flatten",
            token_formula=f"{frames} x {output_tokens_per_frame}",
            token_count=frames * output_tokens_per_frame,
            raw_weight=raw_weight,
            normalized_weight=normalized_weight,
            stroke_color=stroke_color,
            fill_color=fill_color,
            note=f"{num_classes} AU features compressed to {output_tokens_per_frame} tokens per frame.",
        )
    if name == "rppg":
        token_count = resolve_output_token_count(config, "rppg")
        return ModalityDoc(
            name=name,
            title="rPPG",
            encoder_summary="PhysNet temporal encoder",
            projector_summary=f"waveform + temporal features -> Linear(*, {dim}) -> temporal position encoding -> latent-query pool",
            token_formula=str(token_count),
            token_count=token_count,
            raw_weight=raw_weight,
            normalized_weight=normalized_weight,
            stroke_color=stroke_color,
            fill_color=fill_color,
            note="Temporal features compressed to a fixed clip token budget with order-aware pooling plus raw waveform side output.",
        )
    return ModalityDoc(
        name=name,
        title=name.replace("_", " ").title(),
        encoder_summary="custom branch",
        projector_summary=f"project features -> {dim}",
        token_formula="custom",
        token_count=None,
        raw_weight=raw_weight,
        normalized_weight=normalized_weight,
        stroke_color=stroke_color,
        fill_color=fill_color,
        note="Unknown to doc generator. Update script if branch changes.",
    )


def build_modality_docs(config: dict[str, Any]) -> tuple[list[ModalityDoc], dict[str, Any]]:
    modalities = config.get("modalities")
    if not isinstance(modalities, list) or not modalities:
        raise ValueError("`modalities` must be non-empty list.")
    enabled = [str(item).strip() for item in modalities if str(item).strip()]
    if not enabled:
        raise ValueError("No enabled modalities in config.")

    frames = int(config["frames"])
    dim = int(config["dim"])
    raw_weights = {
        name: float(config.get("modality_weights", {}).get(name, 1.0))
        for name in enabled
    }
    normalized_weights = normalize_weights(enabled, raw_weights)
    docs = [
        build_modality_doc(
            name=name,
            config=config,
            frames=frames,
            raw_weight=raw_weights[name],
            normalized_weight=normalized_weights[name],
            dim=dim,
        )
        for name in enabled
    ]
    total_tokens = sum(doc.token_count for doc in docs if doc.token_count is not None)
    all_known = all(doc.token_count is not None for doc in docs)
    return docs, {
        "frames": frames,
        "dim": dim,
        "total_tokens": total_tokens if all_known else None,
        "device": str(config.get("device", "unknown")),
        "modalities": enabled,
        "fusion": config["fusion"],
        "classifier": config.get("classifier", {}),
        "training": config.get("training", {}),
        "dot_available": shutil.which("dot") is not None,
    }


def markdown_table_row(columns: list[str]) -> str:
    return "| " + " | ".join(columns) + " |"


def dot_label_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("->", "→")


def render_markdown(
    config_path: Path,
    docs: list[ModalityDoc],
    summary: dict[str, Any],
) -> str:
    fusion = summary["fusion"]
    classifier = summary.get("classifier", {})
    training = summary.get("training", {})
    total_tokens = summary["total_tokens"]
    total_tokens_text = str(total_tokens) if total_tokens is not None else "unknown"
    render_hint = (
        "`dot` detected locally. Render with `venv/bin/python scripts/render_graphviz.py`."
        if summary["dot_available"]
        else "`dot` not installed here. Install Graphviz, then run `venv/bin/python scripts/render_graphviz.py`."
    )
    lines = [
        "# Model Architecture",
        "",
        "Generated file. Do not edit by hand.",
        "",
        "Regenerate source with:",
        "",
        "```bash",
        "venv/bin/python scripts/generate_model_docs.py",
        "```",
        "",
        "Render Graphviz SVG with:",
        "",
        "```bash",
        "venv/bin/python scripts/render_graphviz.py",
        "```",
        "",
        "## Current Build",
        "",
        f"- Config: `{config_path.relative_to(PROJECT_ROOT)}`",
        f"- Modalities: `{', '.join(summary['modalities'])}`",
        f"- Frames: `{summary['frames']}`",
        f"- Token dim: `{summary['dim']}`",
        f"- Token bank size: `{total_tokens_text}`",
        f"- Runtime device: `{summary['device']}`",
        f"- Fusion: `TokenBankFusion`, layers=`{fusion['num_layers']}`, heads=`{fusion['num_heads']}`, mlp_ratio=`{fusion['mlp_ratio']}`, max_time_steps=`{fusion['max_time_steps']}`",
        f"- Fusion internals: `CLS token` + `time embedding` + `modality embedding` + `TransformerEncoderLayer x{fusion['num_layers']}` + `LayerNorm`",
        f"- Classifier: `VideoRealFakeHead`, hidden=`{classifier.get('hidden_dim', 'unknown')}`, dropout=`{classifier.get('dropout', 'unknown')}`",
        f"- Training defaults: freeze_encoders=`{training.get('freeze_encoders', 'unknown')}`, lr_head=`{training.get('lr_head', 'unknown')}`, lr_fusion=`{training.get('lr_fusion', 'unknown')}`, pos_weight=`{training.get('pos_weight', 'unknown')}`",
        f"- Graphviz status: {render_hint}",
        "",
        "## Modality Summary",
        "",
        markdown_table_row(
            [
                "Modality",
                "Encoder",
                "Projection",
                "Token Formula",
                "Raw Weight",
                "Normalized Weight",
                "Note",
            ]
        ),
        markdown_table_row(["---", "---", "---", "---", "---", "---", "---"]),
    ]
    for doc in docs:
        lines.append(
            markdown_table_row(
                [
                    doc.title,
                    doc.encoder_summary,
                    doc.projector_summary,
                    doc.token_formula,
                    format_float(doc.raw_weight),
                    format_float(doc.normalized_weight),
                    doc.note,
                ]
            )
        )

    lines.extend(
        [
            "",
            "## Graphviz Files",
            "",
            "- `docs/model_architecture.dot`: source of truth for diagram layout.",
            "- `docs/model_architecture.svg`: rendered Graphviz output after running render script.",
            "- `docs/model_graphics.md`: repo workflow notes.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def render_dot(docs: list[ModalityDoc], summary: dict[str, Any]) -> str:
    fusion = summary["fusion"]
    classifier = summary.get("classifier", {})
    total_tokens = summary["total_tokens"]
    total_tokens_text = str(total_tokens) if total_tokens is not None else "unknown"
    lines = [
        "digraph model_architecture {",
        "  rankdir=LR;",
        '  graph [pad="0.35", nodesep="0.8", ranksep="1.15", bgcolor="white", splines=ortho];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.2,0.14"];',
        '  edge [color="#52606D", penwidth=1.7, arrowsize=0.8];',
        "",
        f'  video [label="Input Video\\n{summary["frames"]} sampled frames", fillcolor="#FFFFFF", color="#7B8794"];',
    ]

    for doc in docs:
        node_id = modality_node_id(doc)
        lines.append(
            (
                f'  {node_id} [label="{dot_label_text(doc.title)}\\n{dot_label_text(doc.encoder_summary)}'
                f'\\n{dot_label_text(doc.projector_summary)}\\ntokens={dot_label_text(doc.token_formula)}'
                f'\\nw={format_float(doc.raw_weight)}'
                f' (norm {format_float(doc.normalized_weight)})", '
                f'fillcolor="{doc.fill_color}", color="{doc.stroke_color}"];'
            )
        )

    lines.extend(
        [
            f'  bank [label="Weighted Token Bank\\ntotal tokens={total_tokens_text}", fillcolor="#FFF4CC", color="#B38600"];',
            f'  time [label="Time Embedding\\nmax_time_steps={fusion["max_time_steps"]}", fillcolor="#FFF4CC", color="#B38600"];',
            f'  cls_in [label="CLS Token\\n[1 x {summary["dim"]}]", fillcolor="#FFF4CC", color="#B38600"];',
            f'  modality [label="Modality Embedding\\n{len(docs)} active ids", fillcolor="#FFF4CC", color="#B38600"];',
            (
                f'  encoder [label="TransformerEncoder\\nTransformerEncoderLayer x{fusion["num_layers"]}'
                f'\\nheads={fusion["num_heads"]}, mlp_ratio={fusion["mlp_ratio"]}", '
                'fillcolor="#DCEAF7", color="#295C8A"];'
            ),
            (
                f'  layer_hint [label="TransformerEncoderLayer x{fusion["num_layers"]}'
                f'\\nheads={fusion["num_heads"]}\\nmlp_ratio={fusion["mlp_ratio"]}", '
                'fillcolor="#EAF2FA", color="#5B7FA3"];'
            ),
            '  norm [label="LayerNorm\\noutput_norm", fillcolor="#DCEAF7", color="#295C8A"];',
            f'  cls [label="CLS Output\\n[1 x {summary["dim"]}]", fillcolor="#DCEAF7", color="#295C8A"];',
            (
                f'  head [label="VideoRealFakeHead\\nMLP {summary["dim"]}->{classifier.get("hidden_dim", "?")}->1'
                f'\\ndropout={classifier.get("dropout", "?")}", fillcolor="#FFFFFF", color="#295C8A", style="rounded,dashed,filled"];'
            ),
            "",
            "  { rank=same; video; " + "; ".join(modality_node_id(doc) for doc in docs) + "; }",
            "  { rank=same; bank; time; cls_in; modality; }",
            "  { rank=same; encoder; layer_hint; norm; cls; head; }",
            "",
        ]
    )

    for doc in docs:
        node_id = modality_node_id(doc)
        lines.append(f"  video -> {node_id};")
        lines.append(f"  {node_id} -> bank;")

    lines.extend(
        [
            "  bank -> encoder;",
            "  time -> encoder;",
            "  cls_in -> encoder;",
            "  modality -> encoder;",
            '  encoder -> layer_hint [style=dashed, arrowhead=none, color="#7B8794"];',
            "  encoder -> norm;",
            "  norm -> cls;",
            "  cls -> head;",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    docs_dir = args.docs_dir.resolve()

    config = load_yaml(config_path)
    docs, summary = build_modality_docs(config)

    markdown = render_markdown(config_path, docs, summary)
    dot_source = render_dot(docs, summary)

    write_text(docs_dir / "model_architecture.md", markdown)
    write_text(docs_dir / "model_architecture.dot", dot_source)

    print(f"wrote: {docs_dir / 'model_architecture.md'}")
    print(f"wrote: {docs_dir / 'model_architecture.dot'}")


if __name__ == "__main__":
    main()
