from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "runs" / "final_v2_ffpp_celebdf_domain_70_15_15"
PNG_PATH = OUTPUT_DIR / "slide16_cross_dataset_recall.png"
SVG_PATH = OUTPUT_DIR / "slide16_cross_dataset_recall.svg"

# Hardcoded from:
# - runs/raw_predictions/celebdf_predictions.csv
# - runs/raw_predictions/ffpp_c23_predictions.csv
# - runs/final_v2_ffpp_celebdf_domain_70_15_15/generator_metrics.csv
BEFORE_EXTERNAL_RECALL = {
    "Celeb-DF": 0.640,
    "FF++ C23": 0.635,
}

AFTER_SOURCE_RECALL = {
    "Celeb-DF": 0.740,
    "FF++ C23": 0.708,
    "dlc": 0.867,
    "liveavatar": 0.959,
    "ltx2": 0.981,
    "sadtalker": 0.973,
    "unknown_or_other": 0.769,
}

HSE_RED = "#9d1a31"
HSE_BLUE = "#006EB4"
HSE_WHITE = "#ffffff"
HSE_GREY = "#333333"
HSE_PURPLE = "#2e358b"
GRID_GREY = "#D9D9D9"
AXIS_GREY = "#B8B8B8"


def apply_axis_style(ax: plt.Axes) -> None:
    ax.set_facecolor(HSE_WHITE)
    ax.grid(axis="y", color=GRID_GREY, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_GREY)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=HSE_GREY, labelsize=9, width=0.8)


def format_source_label(source: str) -> str:
    return source.replace("_", " ")


def add_vertical_bar_labels(ax: plt.Axes, bars: list[plt.Rectangle]) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.008,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=HSE_GREY,
        )


def add_horizontal_bar_labels(ax: plt.Axes, bars: list[plt.Rectangle]) -> None:
    for bar in bars:
        width = bar.get_width()
        ax.text(
            min(width + 0.006, 0.988),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            ha="left",
            va="center",
            fontsize=8,
            color=HSE_GREY,
        )


def plot_before_after_panel(ax: plt.Axes) -> None:
    labels = list(BEFORE_EXTERNAL_RECALL)
    before = [BEFORE_EXTERNAL_RECALL[label] for label in labels]
    after = [AFTER_SOURCE_RECALL[label] for label in labels]
    x_positions = range(len(labels))
    bar_width = 0.34

    before_bars = ax.bar(
        [x - bar_width / 2 for x in x_positions],
        before,
        width=bar_width,
        label="Before",
        color=HSE_BLUE,
        edgecolor=HSE_BLUE,
        linewidth=0.6,
        alpha=0.82,
    )
    after_bars = ax.bar(
        [x + bar_width / 2 for x in x_positions],
        after,
        width=bar_width,
        label="After",
        color=HSE_RED,
        edgecolor=HSE_RED,
        linewidth=0.6,
        alpha=0.82,
    )

    ax.set_title("External datasets: before vs after mixed training", fontsize=11, pad=12)
    ax.set_xticks(list(x_positions), labels)
    ax.set_ylim(0.55, 1.0)
    ax.set_ylabel("Fake recall", fontsize=9, color=HSE_GREY)
    ax.set_yticks([0.55, 0.65, 0.75, 0.85, 0.95, 1.0])
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    add_vertical_bar_labels(ax, list(before_bars) + list(after_bars))
    apply_axis_style(ax)


def plot_after_source_panel(ax: plt.Axes) -> None:
    rows = sorted(AFTER_SOURCE_RECALL.items(), key=lambda item: item[1], reverse=True)
    labels = [format_source_label(source) for source, _ in rows]
    values = [value for _, value in rows]

    bars = ax.barh(
        labels,
        values,
        color=HSE_PURPLE,
        edgecolor=HSE_PURPLE,
        linewidth=0.6,
        alpha=0.82,
    )

    ax.set_title("Fake recall by source after mixed training", fontsize=11, pad=12)
    ax.set_xlim(0.55, 1.0)
    ax.set_xlabel("Fake recall", fontsize=9, color=HSE_GREY)
    ax.set_xticks([0.55, 0.65, 0.75, 0.85, 0.95, 1.0])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID_GREY, linewidth=0.7)
    add_horizontal_bar_labels(ax, list(bars))
    apply_axis_style(ax)
    ax.grid(axis="x", color=GRID_GREY, linewidth=0.7)


def create_figure() -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 7.2),
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )
    fig.patch.set_facecolor(HSE_WHITE)

    plot_before_after_panel(axes[0])
    plot_after_source_panel(axes[1])

    fig.suptitle(
        "Cross-dataset coverage after mixed-source training",
        fontsize=15,
        fontweight="semibold",
        y=0.965,
        color=HSE_GREY,
    )
    fig.text(
        0.5,
        0.035,
        (
            "Metric: fake recall. Before = no-retraining external stress test; "
            "after = mixed-source test split."
        ),
        ha="center",
        va="center",
        fontsize=8.5,
        color=HSE_GREY,
    )
    fig.tight_layout(rect=[0.035, 0.075, 0.985, 0.925], w_pad=3.0)
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = create_figure()
    fig.savefig(PNG_PATH, dpi=220, facecolor=HSE_WHITE)
    fig.savefig(SVG_PATH, facecolor=HSE_WHITE)
    plt.close(fig)
    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
