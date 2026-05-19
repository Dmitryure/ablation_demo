from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import LabeledVideoDataset, VIDEO_EXTENSIONS, VideoExample
from feature_cache import build_feature_cache_specs, feature_cache_spec_id
from video_model_fps import (
    modality_configs,
    modalities_for_selection,
    pipeline_config_for_selection,
    read_json,
    selection_from_checkpoint,
)

DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs/night_sweep/seed0_lr3e4_gen0p15_warm2_e26/best.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs/frame_previews/ffpp_c23"
CLASS_FILTERS = ("all", "real", "fake")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export sampled/cropped/resized input frames used before feature extraction."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--class-filter", choices=CLASS_FILTERS, default="all")
    parser.add_argument("--generator-filter", default=None)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--decode-mode", choices=("scan", "seek"), default="scan")
    parser.add_argument("--sheet-frames", type=int, default=8)
    return parser.parse_args()


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def discover_ffpp_examples(dataset_root: Path) -> list[VideoExample]:
    original_dir = dataset_root / "original"
    if not original_dir.is_dir():
        raise FileNotFoundError(f"Missing FF++ original folder: {original_dir}")

    examples: list[VideoExample] = []
    for path in sorted(original_dir.iterdir()):
        if is_video_file(path):
            examples.append(
                VideoExample(
                    path=path,
                    label=0,
                    class_name="real",
                    source_id=path.stem,
                    split="test",
                    metadata_filename=str(path.relative_to(dataset_root)),
                    generator_id="real",
                    source_id_kind="ffpp",
                )
            )

    fake_dirs = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name not in {"csv", "original"}
    )
    for fake_dir in fake_dirs:
        for path in sorted(fake_dir.iterdir()):
            if is_video_file(path):
                examples.append(
                    VideoExample(
                        path=path,
                        label=1,
                        class_name="fake",
                        source_id=path.stem,
                        split="test",
                        metadata_filename=str(path.relative_to(dataset_root)),
                        generator_id=fake_dir.name,
                        source_id_kind="ffpp",
                    )
                )
    if not examples:
        raise FileNotFoundError(f"No FF++ videos found under {dataset_root}")
    return examples


def safe_name(value: str, max_length: int = 140) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned[:max_length] or "video"


def load_checkpoint_config(checkpoint: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    selection = selection_from_checkpoint(checkpoint, model_kind="auto")
    summary = read_json(selection.summary_path)
    run_config = read_json(selection.run_config_path)
    config = pipeline_config_for_selection(selection, run_config=run_config, device="cpu")
    modalities = modalities_for_selection(selection, summary, run_config, config)
    return config, modalities


def filtered_examples(
    examples: list[VideoExample],
    class_filter: str,
    generator_filter: str | None,
    start_index: int,
    limit: int,
) -> list[VideoExample]:
    if limit <= 0:
        raise ValueError("`--limit` must be positive.")
    if start_index < 0:
        raise ValueError("`--start-index` must be non-negative.")
    selected = [
        example
        for example in examples
        if class_filter == "all" or example.class_name == class_filter
    ]
    if generator_filter is not None:
        selected = [example for example in selected if example.generator_id == generator_filter]
    return selected[start_index : start_index + limit]


def frame_indices(count: int, sheet_frames: int) -> list[int]:
    if count <= 0:
        return []
    if sheet_frames <= 0:
        raise ValueError("`--sheet-frames` must be positive.")
    if count <= sheet_frames:
        return list(range(count))
    return sorted({int(round(value)) for value in np.linspace(0, count - 1, sheet_frames)})


def resize_tile(frame: np.ndarray, size: int = 192) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
    canvas = np.full((size, size, 3), 24, dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def contact_sheet(frames: list[np.ndarray], sheet_frames: int) -> np.ndarray:
    indices = frame_indices(len(frames), sheet_frames)
    tiles = [resize_tile(frames[index]) for index in indices]
    if not tiles:
        raise ValueError("Cannot build contact sheet from empty frames.")
    columns = min(4, len(tiles))
    rows = []
    for offset in range(0, len(tiles), columns):
        row_tiles = tiles[offset : offset + columns]
        if len(row_tiles) < columns:
            blank = np.full_like(tiles[0], 24)
            row_tiles.extend(blank.copy() for _ in range(columns - len(row_tiles)))
        rows.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(rows, axis=0)


def write_rgb_image(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def example_output_dir(output_dir: Path, index: int, example: VideoExample) -> Path:
    return output_dir / f"{index:03d}_{example.class_name}_{safe_name(example.metadata_filename or example.path.name)}"


def dataset_for_example(
    example: VideoExample,
    config: dict[str, Any],
    modalities: tuple[str, ...],
    decode_mode: str,
) -> LabeledVideoDataset:
    specs = build_feature_cache_specs(config, modalities)
    frame_counts = {name: specs[name].frame_count for name in modalities}
    image_sizes = {name: specs[name].image_size for name in modalities}
    rppg_config = config.get("rppg", {})
    return LabeledVideoDataset(
        examples=[example],
        num_frames=frame_counts,
        image_size=int(config.get("image_size", 224)),
        decode_mode=decode_mode,
        dataset_root=None,
        image_size_by_modality=image_sizes,
        rppg_config=rppg_config if isinstance(rppg_config, dict) else {},
        modality_configs=modality_configs(config, modalities),
        global_config=config,
    )


def export_example_frames(
    output_dir: Path,
    index: int,
    example: VideoExample,
    config: dict[str, Any],
    modalities: tuple[str, ...],
    decode_mode: str,
    sheet_frames: int,
) -> dict[str, Any]:
    specs = build_feature_cache_specs(config, modalities)
    dataset = dataset_for_example(
        example=example,
        config=config,
        modalities=modalities,
        decode_mode=decode_mode,
    )
    item = dataset[0]
    frames_by_modality = item.get("video_rgb_frames_by_modality")
    if not isinstance(frames_by_modality, dict):
        frames_by_modality = {modalities[0]: item["video_rgb_frames"]}

    face_status = item.get("face_crop_status_by_modality") or {}
    video_dir = example_output_dir(output_dir, index, example)
    video_dir.mkdir(parents=True, exist_ok=True)
    modality_rows: list[dict[str, Any]] = []
    for modality in modalities:
        frames = list(frames_by_modality[modality])
        modality_dir = video_dir / modality
        modality_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, frame in enumerate(frames):
            write_rgb_image(modality_dir / f"frame_{frame_index:03d}.jpg", frame)
        sheet_path = video_dir / f"{modality}_sheet.jpg"
        write_rgb_image(sheet_path, contact_sheet(frames, sheet_frames=sheet_frames))
        status_value = face_status.get(modality, "")
        modality_rows.append(
            {
                "modality": modality,
                "spec_id": feature_cache_spec_id(specs[modality]),
                "frames": len(frames),
                "image_size": specs[modality].image_size,
                "face_crop_status": status_value,
                "sheet_path": str(sheet_path),
                "frames_dir": str(modality_dir),
            }
        )

    metadata = {
        "path": str(example.path),
        "metadata_filename": example.metadata_filename,
        "class_name": example.class_name,
        "generator_id": example.generator_id,
        "modalities": modality_rows,
    }
    with (video_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "index": index,
        "path": str(example.path),
        "metadata_filename": example.metadata_filename or "",
        "class_name": example.class_name,
        "generator_id": example.generator_id or "",
        "output_dir": str(video_dir),
    }


def write_index(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ("index", "class_name", "generator_id", "metadata_filename", "path", "output_dir")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config, modalities = load_checkpoint_config(args.checkpoint)
    examples = filtered_examples(
        discover_ffpp_examples(args.dataset_root),
        class_filter=args.class_filter,
        generator_filter=args.generator_filter,
        start_index=args.start_index,
        limit=args.limit,
    )
    if not examples:
        raise ValueError("No examples selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        print(f"export {index}/{len(examples)} {example.metadata_filename}", flush=True)
        rows.append(
            export_example_frames(
                output_dir=args.output_dir,
                index=index,
                example=example,
                config=config,
                modalities=modalities,
                decode_mode=args.decode_mode,
                sheet_frames=args.sheet_frames,
            )
        )
    write_index(args.output_dir, rows)
    print(f"wrote: {args.output_dir}", flush=True)
    print(f"summary: {args.output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
