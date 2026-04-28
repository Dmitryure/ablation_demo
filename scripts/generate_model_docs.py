from __future__ import annotations

import argparse
import html
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "registry_fusion.yaml"
DEFAULT_DOCS_DIR = PROJECT_ROOT / "docs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.model_architecture_spec import (
    ArchitectureSpec,
    ComponentSpec,
    SourceRef,
    architecture_spec_to_json,
    build_architecture_spec,
    load_yaml,
)


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


def markdown_table_row(columns: list[str]) -> str:
    return "| " + " | ".join(columns) + " |"


def _render_hint() -> str:
    if shutil.which("dot") is not None:
        return "`dot` detected locally. Render with `venv/bin/python scripts/render_graphviz.py`."
    return "`dot` not installed here. Install Graphviz, then run `venv/bin/python scripts/render_graphviz.py`."


def _component_rows(spec: ArchitectureSpec, kind: str) -> list[ComponentSpec]:
    return [component for component in spec.components if component.kind == kind]


def _source_label(source: SourceRef | None) -> str:
    if source is None:
        return "generated from live code"
    if source.line is None:
        return f"`{source.path}`"
    return f"`{source.path}:{source.line}`"


def _frame_count_label(frames: object) -> str:
    if isinstance(frames, dict):
        return ", ".join(f"{name}={count}" for name, count in frames.items())
    return str(frames)


def render_markdown(spec: ArchitectureSpec) -> str:
    fusion = spec.fusion
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
        f"- Config: `{spec.config_path}`",
        f"- Enabled modalities: `{', '.join(spec.enabled_modalities)}`",
        f"- Fixed slot layout: `{', '.join(spec.fixed_slot_modalities)}`",
        f"- Frames: `{_frame_count_label(spec.frames)}`",
        f"- Token dim: `{spec.dim}`",
        f"- Enabled token count: `{spec.enabled_token_count}`",
        f"- Fixed token bank size: `{spec.total_tokens}`",
        f"- Runtime device: `{spec.device}`",
        f"- Fusion: `TokenBankFusion`, layers=`{fusion['num_layers']}`, heads=`{fusion['num_heads']}`, mlp_ratio=`{fusion['mlp_ratio']}`, max_time_steps=`{fusion['max_time_steps']}`",
        f"- Graphviz status: {_render_hint()}",
        "",
        "## Modality Components",
        "",
        markdown_table_row(
            [
                "Component",
                "Status",
                "Input",
                "Stages",
                "Output",
                "Source",
            ]
        ),
        markdown_table_row(["---", "---", "---", "---", "---", "---"]),
    ]

    for component in _component_rows(spec, "modality"):
        stage_summary = " -> ".join(f"{stage.title}: {stage.detail}" for stage in component.stages)
        lines.append(
            markdown_table_row(
                [
                    component.title,
                    "enabled" if component.enabled else "reserved",
                    component.input_summary,
                    stage_summary,
                    f"{component.output_summary}; {component.token_formula}",
                    _source_label(component.source),
                ]
            )
        )

    lines.extend(
        [
            "",
            "## Shared Flow",
            "",
            markdown_table_row(["Component", "Stages", "Output", "Source"]),
            markdown_table_row(["---", "---", "---", "---"]),
        ]
    )
    for component in [c for c in spec.components if c.kind in {"token_bank", "fusion"}]:
        stage_summary = " -> ".join(f"{stage.title}: {stage.detail}" for stage in component.stages)
        lines.append(
            markdown_table_row(
                [
                    component.title,
                    stage_summary,
                    component.output_summary,
                    _source_label(component.source),
                ]
            )
        )

    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `docs/model_architecture.json`: machine-readable architecture spec from live code and config.",
            "- `docs/model_architecture.md`: generated human-readable summary.",
            "- `docs/model_architecture.dot`: Graphviz source derived from the architecture spec.",
            "- `docs/model_architecture.svg`: rendered Graphviz output after running render script.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _escape_text(text: str) -> str:
    return html.escape(text, quote=False)


def _source_href(source: SourceRef | None, docs_dir: Path) -> str | None:
    if source is None:
        return None
    source_path = PROJECT_ROOT / source.path
    return str(Path("..") / source_path.relative_to(PROJECT_ROOT))


def _source_tooltip(source: SourceRef | None) -> str | None:
    if source is None:
        return None
    if source.line is None:
        return f"{source.path} ({source.symbol})"
    return f"{source.path}:{source.line} ({source.symbol})"


def _card_label(component: ComponentSpec) -> str:
    rows = [
        ("Status", "enabled" if component.enabled else "reserved"),
        ("Input", component.input_summary),
    ]
    rows.extend((stage.title, stage.detail) for stage in component.stages)
    if component.output_summary:
        rows.append(("Output", component.output_summary))
    if component.token_formula:
        rows.append(("Formula", component.token_formula))
    if component.note:
        rows.append(("Note", component.note))

    parts = [
        "<",
        (
            f'<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="6" '
            f'COLOR="{component.stroke_color}" BGCOLOR="{component.fill_color}">'
        ),
        (
            f'<TR><TD ALIGN="LEFT" BGCOLOR="{component.stroke_color}">'
            f'<FONT COLOR="white"><B>{_escape_text(component.title)}</B></FONT></TD></TR>'
        ),
    ]
    for label, value in rows:
        parts.append(
            '<TR><TD ALIGN="LEFT" BALIGN="LEFT">'
            f'<FONT POINT-SIZE="10"><B>{_escape_text(label)}:</B> {_escape_text(value)}</FONT>'
            "</TD></TR>"
        )
    parts.append("</TABLE>>")
    return "".join(parts)


def _node_attributes(component: ComponentSpec, docs_dir: Path) -> str:
    attributes = [f"label={_card_label(component)}"]
    tooltip = _source_tooltip(component.source)
    href = _source_href(component.source, docs_dir)
    if tooltip is not None:
        attributes.append(f'tooltip="{_escape_text(tooltip)}"')
    if href is not None:
        attributes.append(f'href="{_escape_text(href)}"')
    return ", ".join(attributes)


def render_dot(spec: ArchitectureSpec, docs_dir: Path) -> str:
    lines = [
        "digraph model_architecture {",
        "  rankdir=TB;",
        '  graph [pad="0.35", nodesep="0.5", ranksep="0.8", bgcolor="white", splines=polyline, newrank=true];',
        '  node [shape=plain, fontname="Helvetica"];',
        '  edge [color="#52606D", penwidth=1.7, arrowsize=0.8, fontname="Helvetica"];',
        "",
    ]

    cluster_order = (
        ("inputs", "Inputs", "#D9E2EC", "#F8FAFC"),
        ("modalities", "Modality Tokenization", "#D9E2EC", "#F8FAFC"),
        ("shared", "Token Bank And Fusion", "#F7D070", "#FFF9E6"),
        ("outputs", "Outputs", "#C3D7F0", "#F4F8FC"),
    )
    for cluster_name, label, color, fillcolor in cluster_order:
        cluster_components = [
            component for component in spec.components if component.cluster == cluster_name
        ]
        if not cluster_components:
            continue
        lines.extend(
            [
                f"  subgraph cluster_{cluster_name} {{",
                f'    label="{label}";',
                f'    color="{color}";',
                '    style="rounded,filled";',
                f'    fillcolor="{fillcolor}";',
                f'    pencolor="{color}";',
                "    margin=18;",
            ]
        )
        for component in cluster_components:
            lines.append(f"    {component.id} [{_node_attributes(component, docs_dir)}];")
        lines.append("  }")
        lines.append("")

    for edge in spec.edges:
        label = f' [label="{_escape_text(edge.label)}"]' if edge.label else ""
        lines.append(f"  {edge.source} -> {edge.target}{label};")

    lines.append("}")
    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    docs_dir = args.docs_dir.resolve()

    config = load_yaml(config_path)
    spec = build_architecture_spec(config, config_path=config_path)

    write_text(docs_dir / "model_architecture.json", architecture_spec_to_json(spec))
    write_text(docs_dir / "model_architecture.md", render_markdown(spec))
    write_text(docs_dir / "model_architecture.dot", render_dot(spec, docs_dir=docs_dir))

    print(f"wrote: {docs_dir / 'model_architecture.json'}")
    print(f"wrote: {docs_dir / 'model_architecture.md'}")
    print(f"wrote: {docs_dir / 'model_architecture.dot'}")


if __name__ == "__main__":
    main()
