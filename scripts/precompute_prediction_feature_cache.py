from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import VIDEO_EXTENSIONS, VideoExample
from feature_cache import build_feature_cache_specs, feature_cache_spec_id
from scripts.run_iterative_cached_ablation import (
    build_config,
    ensure_feature_cache,
    resolve_base_modalities,
    write_json,
)

DEFAULT_INPUT_DIR = Path("/home/comp/face_detect_app/test1")
DEFAULT_CACHE_DIR = PROJECT_ROOT / "runs" / "prediction_feature_cache" / "test1_unknown"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "prediction_feature_cache_runs"
CSV_ID_COLUMN = "obj_id"
CSV_LABEL_COLUMN = "label"
CSV_GENERATOR_COLUMN = "generator_attrs.generator.name"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute feature cache tensors for prediction-only video folders."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Optional CSV with obj_id and label columns. Auto-detected when omitted.",
    )
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument("--class-name", choices=("real", "fake"), default="fake")
    parser.add_argument("--generator-id", default="unknown_or_other")
    parser.add_argument(
        "--use-csv-generator",
        action="store_true",
        help="Use CSV generator names for fake examples instead of --generator-id.",
    )
    parser.add_argument("--split", default="prediction")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--extract-batch-size", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--skip-failures", action="store_true")
    parser.add_argument("--assume-missing-cache", action="store_true")
    parser.add_argument(
        "--video-decode-mode",
        choices=("seek", "scan"),
        default="scan",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_device(device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("`--device cuda` requested, but CUDA is not available.")


def discover_videos(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def auto_metadata_csv(input_dir: Path, override: Path | None) -> Path | None:
    if override is not None:
        return override
    csv_paths = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix == ".csv"
    )
    if len(csv_paths) == 1:
        return csv_paths[0]
    return None


def class_name_for_label(value: str) -> str:
    normalized = str(value).strip()
    if normalized == "0":
        return "real"
    if normalized == "1":
        return "fake"
    raise ValueError(f"Unsupported CSV label: {value!r}")


def load_metadata_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = [
            column for column in (CSV_ID_COLUMN, CSV_LABEL_COLUMN) if column not in fieldnames
        ]
        if missing:
            raise ValueError(f"Metadata CSV {path} missing columns: {', '.join(missing)}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            obj_id = str(row.get(CSV_ID_COLUMN, "")).strip()
            if not obj_id:
                raise ValueError(f"Metadata CSV {path} contains empty {CSV_ID_COLUMN}.")
            if obj_id in rows:
                raise ValueError(
                    f"Metadata CSV {path} contains duplicate {CSV_ID_COLUMN}: {obj_id}"
                )
            rows[obj_id] = {str(key): str(value) for key, value in row.items()}
    return rows


def generator_id_for_row(
    row: dict[str, str],
    fallback_generator_id: str,
    use_csv_generator: bool,
) -> str:
    if not use_csv_generator:
        return fallback_generator_id
    generator = row.get(CSV_GENERATOR_COLUMN, "").strip()
    return generator or fallback_generator_id


def prediction_example(
    path: Path,
    class_name: str,
    generator_id: str,
    split: str,
) -> VideoExample:
    label = 1 if class_name == "fake" else 0
    metadata_filename = path.name if class_name == "real" else f"{generator_id}/{path.name}"
    return VideoExample(
        path=path,
        label=label,
        class_name=class_name,
        source_id=path.stem,
        split=split,
        metadata_filename=metadata_filename,
        identity_id=generator_id if class_name == "fake" else None,
        generator_id=generator_id if class_name == "fake" else "real",
        source_id_kind="prediction_input",
    )


def build_prediction_examples(
    input_dir: Path,
    class_name: str,
    generator_id: str,
    split: str,
    metadata_csv: Path | None,
    use_csv_generator: bool,
) -> list[VideoExample]:
    videos = discover_videos(input_dir)
    if not videos:
        raise FileNotFoundError(f"No supported videos found in {input_dir}")
    if metadata_csv is not None:
        rows = load_metadata_rows(metadata_csv)
        missing_rows = [path.stem for path in videos if path.stem not in rows]
        missing_videos = [obj_id for obj_id in rows if not (input_dir / f"{obj_id}.mp4").is_file()]
        if missing_rows or missing_videos:
            details: list[str] = []
            if missing_rows:
                details.append(
                    f"{len(missing_rows)} videos missing CSV rows, first={missing_rows[0]}"
                )
            if missing_videos:
                details.append(
                    f"{len(missing_videos)} CSV rows missing mp4, first={missing_videos[0]}"
                )
            raise FileNotFoundError("; ".join(details))
        return [
            prediction_example(
                path=path,
                class_name=class_name_for_label(rows[path.stem][CSV_LABEL_COLUMN]),
                generator_id=generator_id_for_row(
                    rows[path.stem],
                    fallback_generator_id=generator_id,
                    use_csv_generator=use_csv_generator,
                ),
                split=split,
            )
            for path in videos
        ]
    return [
        prediction_example(
            path=path,
            class_name=class_name,
            generator_id=generator_id,
            split=split,
        )
        for path in videos
    ]


def class_counts(examples: list[VideoExample]) -> dict[str, int]:
    counts = {"real": 0, "fake": 0}
    for example in examples:
        counts[example.class_name] += 1
    return counts


def split_summary(examples: list[VideoExample]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for example in examples:
        if example.split not in summary:
            summary[example.split] = {"real": 0, "fake": 0, "total": 0}
        summary[example.split][example.class_name] += 1
        summary[example.split]["total"] += 1
    return summary


def write_prediction_manifest(path: Path, examples: list[VideoExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "path",
                "label",
                "class_name",
                "source_id",
                "split",
                "metadata_filename",
                "generator_id",
            ),
        )
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "path": str(example.path),
                    "label": example.label,
                    "class_name": example.class_name,
                    "source_id": example.source_id,
                    "split": example.split,
                    "metadata_filename": example.metadata_filename or "",
                    "generator_id": example.generator_id or "",
                }
            )


def run_config_payload(
    args: argparse.Namespace,
    modalities: tuple[str, ...],
    specs: dict[str, Any],
    examples: list[VideoExample],
) -> dict[str, Any]:
    return {
        "input_dir": str(args.input_dir),
        "cache_dir": str(args.cache_dir),
        "config": str(args.config),
        "metadata_csv": None if args.metadata_csv is None else str(args.metadata_csv),
        "modalities": list(modalities),
        "class_name": args.class_name,
        "generator_id": args.generator_id,
        "use_csv_generator": args.use_csv_generator,
        "split": args.split,
        "device": args.device,
        "extract_batch_size": args.extract_batch_size,
        "overwrite_cache": args.overwrite_cache,
        "skip_failures": args.skip_failures,
        "assume_missing_cache": args.assume_missing_cache,
        "video_decode_mode": args.video_decode_mode,
        "spec_ids": {modality: feature_cache_spec_id(spec) for modality, spec in specs.items()},
        "selected_count": len(examples),
        "selected_counts": class_counts(examples),
        "selected_summary": split_summary(examples),
    }


def main() -> None:
    args = parse_args()
    require_device(args.device)
    config = build_config(args.config, args.device)
    modalities = resolve_base_modalities(config, args.modalities)
    specs = build_feature_cache_specs(config, modalities)
    metadata_csv = auto_metadata_csv(args.input_dir, args.metadata_csv)
    args.metadata_csv = metadata_csv
    examples = build_prediction_examples(
        input_dir=args.input_dir,
        class_name=args.class_name,
        generator_id=args.generator_id,
        split=args.split,
        metadata_csv=metadata_csv,
        use_csv_generator=args.use_csv_generator,
    )
    output_dir = args.output_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"

    print(f"output_dir={output_dir}", flush=True)
    print(f"input_dir={args.input_dir}", flush=True)
    print(f"metadata_csv={metadata_csv if metadata_csv is not None else '<none>'}", flush=True)
    print(f"cache_dir={args.cache_dir}", flush=True)
    print(f"device={config['device']}", flush=True)
    print(f"modalities={','.join(modalities)}", flush=True)
    print(f"selected={len(examples)} counts={class_counts(examples)}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run_config.json",
        run_config_payload(args, modalities, specs, examples),
    )
    write_prediction_manifest(output_dir / "selected_videos.csv", examples)
    if args.dry_run:
        print(f"wrote: {output_dir / 'run_config.json'}", flush=True)
        return

    progress = ensure_feature_cache(
        examples=examples,
        cache_dir=args.cache_dir,
        specs=specs,
        modalities=modalities,
        config=config,
        dataset_root=args.input_dir,
        extract_batch_size=args.extract_batch_size,
        overwrite=args.overwrite_cache,
        skip_failures=args.skip_failures,
        progress_every=args.progress_every,
        label="prediction",
        progress_bar=False,
        group_by_modality=False,
        assume_missing_cache=args.assume_missing_cache,
        video_decode_mode=args.video_decode_mode,
    )
    write_json(
        output_dir / "summary.json",
        {**run_config_payload(args, modalities, specs, examples), "progress": progress},
    )
    print(f"wrote: {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
