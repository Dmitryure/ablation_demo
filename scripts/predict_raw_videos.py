from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import VIDEO_EXTENSIONS, VideoExample
from feature_cache import (
    build_feature_cache_specs,
    write_feature_cache_item,
    write_feature_cache_manifest,
)
from video_model_fps import (
    FpsPrediction,
    load_inference_model,
    predict_video_fps,
    selection_from_checkpoint,
)

DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs/night_sweep/seed0_lr3e4_gen0p15_warm2_e26/best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs/raw_predictions/ffpp_c23_predictions.csv"
CELEBDF_REAL_DIRS: tuple[str, ...] = ("Celeb-real", "YouTube-real")
CELEBDF_FAKE_DIR = "Celeb-synthesis"

PREDICTION_COLUMNS: tuple[str, ...] = (
    "path",
    "relative_path",
    "class_name",
    "generator_id",
    "true_label",
    "prediction",
    "binary_prediction",
    "fake_probability",
    "generator_prediction",
    "generator_probability",
    "video_frame_count",
    "video_fps",
    "video_duration_seconds",
    "decode_seconds",
    "forward_seconds",
    "end_to_end_seconds",
    "feature_cache_status",
    "feature_cache_error",
    "status",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a checkpoint on raw videos and write one fake/real row per video."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--decode-mode", choices=("scan", "seek"), default="scan")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=None,
        help="Optional new cache root to write extracted per-modality features while predicting.",
    )
    parser.add_argument(
        "--resume-feature-cache",
        action="store_true",
        help="Allow writing into an existing non-empty --feature-cache-dir.",
    )
    return parser.parse_args()


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def discover_ffpp_examples(dataset_root: Path) -> list[dict[str, Any]]:
    original_dir = dataset_root / "original"
    if not original_dir.is_dir():
        raise FileNotFoundError(f"Missing FF++ original folder: {original_dir}")

    rows: list[dict[str, Any]] = []
    for path in sorted(original_dir.iterdir()):
        if is_video_file(path):
            rows.append(
                {
                    "path": path,
                    "relative_path": path.relative_to(dataset_root),
                    "class_name": "real",
                    "generator_id": "real",
                    "true_label": 0,
                    "source_id": path.stem,
                    "metadata_filename": str(path.relative_to(dataset_root)),
                    "source_id_kind": "ffpp",
                }
            )

    fake_dirs = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name not in {"csv", "original"}
    )
    for fake_dir in fake_dirs:
        for path in sorted(fake_dir.iterdir()):
            if is_video_file(path):
                rows.append(
                    {
                        "path": path,
                        "relative_path": path.relative_to(dataset_root),
                        "class_name": "fake",
                        "generator_id": fake_dir.name,
                        "true_label": 1,
                        "source_id": path.stem,
                        "metadata_filename": str(path.relative_to(dataset_root)),
                        "source_id_kind": "ffpp",
                    }
                )

    if not rows:
        raise FileNotFoundError(f"No videos found under {dataset_root}")
    return rows


def discover_celebdf_examples(dataset_root: Path) -> list[dict[str, Any]]:
    real_dirs = [dataset_root / name for name in CELEBDF_REAL_DIRS]
    fake_dir = dataset_root / CELEBDF_FAKE_DIR
    missing_dirs = [path for path in (*real_dirs, fake_dir) if not path.is_dir()]
    if missing_dirs:
        missing = ", ".join(str(path) for path in missing_dirs)
        raise FileNotFoundError(f"Missing Celeb-DF folders: {missing}")

    rows: list[dict[str, Any]] = []
    for real_dir in real_dirs:
        for path in sorted(real_dir.iterdir()):
            if is_video_file(path):
                rows.append(
                    {
                        "path": path,
                        "relative_path": path.relative_to(dataset_root),
                        "class_name": "real",
                        "generator_id": "real",
                        "true_label": 0,
                        "source_id": path.stem,
                        "metadata_filename": str(path.relative_to(dataset_root)),
                        "source_id_kind": "celebdf",
                    }
                )

    for path in sorted(fake_dir.iterdir()):
        if is_video_file(path):
            rows.append(
                {
                    "path": path,
                    "relative_path": path.relative_to(dataset_root),
                    "class_name": "fake",
                    "generator_id": CELEBDF_FAKE_DIR,
                    "true_label": 1,
                    "source_id": path.stem,
                    "metadata_filename": str(path.relative_to(dataset_root)),
                    "source_id_kind": "celebdf",
                }
            )

    if not rows:
        raise FileNotFoundError(f"No Celeb-DF videos found under {dataset_root}")
    return rows


def is_celebdf_root(dataset_root: Path) -> bool:
    required_dirs = (*CELEBDF_REAL_DIRS, CELEBDF_FAKE_DIR)
    return all((dataset_root / name).is_dir() for name in required_dirs)


def discover_examples(dataset_root: Path) -> list[dict[str, Any]]:
    if (dataset_root / "original").is_dir():
        return discover_ffpp_examples(dataset_root)
    if is_celebdf_root(dataset_root):
        return discover_celebdf_examples(dataset_root)
    raise FileNotFoundError(
        "Unsupported raw prediction dataset layout. Expected FF++ with an "
        "`original` folder or Celeb-DF with Celeb-real, YouTube-real, and "
        f" Celeb-synthesis folders under {dataset_root}."
    )


def to_video_example(example: dict[str, Any]) -> VideoExample:
    return VideoExample(
        path=example["path"],
        label=int(example["true_label"]),
        class_name=str(example["class_name"]),
        source_id=str(example["source_id"]),
        split="test",
        metadata_filename=str(example["metadata_filename"]),
        generator_id=str(example["generator_id"]),
        source_id_kind=str(example["source_id_kind"]),
    )


def prediction_to_row(example: dict[str, Any], result: FpsPrediction) -> dict[str, Any]:
    binary_prediction = 1 if result.label == "fake" else 0
    return {
        "path": str(example["path"]),
        "relative_path": str(example["relative_path"]),
        "class_name": example["class_name"],
        "generator_id": example["generator_id"],
        "true_label": example["true_label"],
        "prediction": result.label,
        "binary_prediction": binary_prediction,
        "fake_probability": f"{result.fake_probability:.8f}",
        "generator_prediction": result.generator_label or "",
        "generator_probability": ""
        if result.generator_probability is None
        else f"{result.generator_probability:.8f}",
        "video_frame_count": result.video_frame_count,
        "video_fps": f"{result.video_fps:.8f}",
        "video_duration_seconds": f"{result.video_duration_seconds:.8f}",
        "decode_seconds": f"{result.decode_seconds:.8f}",
        "forward_seconds": f"{result.forward_seconds:.8f}",
        "end_to_end_seconds": f"{result.end_to_end_seconds:.8f}",
        "feature_cache_status": "",
        "feature_cache_error": "",
        "status": "ok",
        "error": "",
    }


def failure_row(example: dict[str, Any], error: Exception) -> dict[str, Any]:
    return {
        "path": str(example["path"]),
        "relative_path": str(example["relative_path"]),
        "class_name": example["class_name"],
        "generator_id": example["generator_id"],
        "true_label": example["true_label"],
        "prediction": "",
        "binary_prediction": "",
        "fake_probability": "",
        "generator_prediction": "",
        "generator_probability": "",
        "video_frame_count": "",
        "video_fps": "",
        "video_duration_seconds": "",
        "decode_seconds": "",
        "forward_seconds": "",
        "end_to_end_seconds": "",
        "feature_cache_status": "",
        "feature_cache_error": "",
        "status": "error",
        "error": str(error),
    }


def protected_cache_path(path: Path) -> bool:
    resolved = path.resolve()
    protected_roots = (
        Path("/mnt/d/final_cache").resolve(),
        (PROJECT_ROOT / "shards").resolve(),
    )
    return any(resolved == root or root in resolved.parents for root in protected_roots)


def validate_feature_cache_dir(cache_dir: Path | None, resume: bool) -> None:
    if cache_dir is None:
        return
    if protected_cache_path(cache_dir):
        raise ValueError(
            "Refusing to write prediction cache under protected existing cache/shard roots: "
            f"{cache_dir}"
        )
    if cache_dir.exists() and not cache_dir.is_dir():
        raise NotADirectoryError(f"Feature cache path exists but is not a directory: {cache_dir}")
    if cache_dir.exists() and any(cache_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Feature cache dir exists and is not empty: {cache_dir}. "
            "Use --resume-feature-cache only if this is the intended prediction cache."
        )


def read_existing_prediction_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            column for column in PREDICTION_COLUMNS if column not in (reader.fieldnames or ())
        ]
        if missing:
            raise ValueError(f"Existing output is missing columns: {', '.join(missing)}")
        return [dict(row) for row in reader]


def completed_prefix_count(
    examples: list[dict[str, Any]],
    rows: list[dict[str, str]],
) -> int:
    count = 0
    for example, row in zip(examples, rows, strict=False):
        if str(row.get("relative_path", "")) != str(example["relative_path"]):
            break
        if row.get("status") not in {"ok", "error"}:
            break
        count += 1
    return count


def write_prediction_feature_cache(
    cache_dir: Path | None,
    example: VideoExample,
    specs: dict[str, Any],
    feature_batch: dict[str, Any],
    dataset_root: Path,
) -> tuple[str, str]:
    if cache_dir is None:
        return "", ""
    try:
        for modality, spec in specs.items():
            write_feature_cache_item(
                cache_dir=cache_dir,
                example=example,
                spec=spec,
                item=feature_batch,
                dataset_root=dataset_root,
            )
    except Exception as error:
        return "error", str(error)
    return "cached", ""


def write_summary(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    correct = [row for row in ok_rows if str(row["true_label"]) == str(row["binary_prediction"])]
    by_class = Counter(str(row["class_name"]) for row in ok_rows)
    by_prediction = Counter(str(row["prediction"]) for row in ok_rows)
    payload = {
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "feature_cache_dir": None
        if args.feature_cache_dir is None
        else str(args.feature_cache_dir),
        "output": str(args.output),
        "total": len(rows),
        "ok": len(ok_rows),
        "errors": len(rows) - len(ok_rows),
        "feature_cache_cached": sum(
            1 for row in rows if row.get("feature_cache_status") == "cached"
        ),
        "feature_cache_errors": sum(
            1 for row in rows if row.get("feature_cache_status") == "error"
        ),
        "accuracy": (len(correct) / len(ok_rows)) if ok_rows else 0.0,
        "class_counts": dict(sorted(by_class.items())),
        "prediction_counts": dict(sorted(by_prediction.items())),
    }
    summary_path = path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, whole_seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{whole_seconds:02d}s"
    if minutes:
        return f"{minutes}m{whole_seconds:02d}s"
    return f"{whole_seconds}s"


def progress_line(
    index: int,
    total: int,
    rows: list[dict[str, Any]],
    start_time: float,
    completed_at_start: int = 0,
) -> str:
    elapsed = time.perf_counter() - start_time
    ok = sum(1 for item in rows if item["status"] == "ok")
    errors = index - ok
    predictions = Counter(str(item["prediction"]) for item in rows if item["status"] == "ok")
    cached = sum(1 for item in rows if item.get("feature_cache_status") == "cached")
    cache_errors = sum(1 for item in rows if item.get("feature_cache_status") == "error")
    correct = sum(
        1
        for item in rows
        if item["status"] == "ok" and str(item["true_label"]) == str(item["binary_prediction"])
    )
    accuracy = correct / ok if ok else 0.0
    processed_this_run = max(1, index - completed_at_start)
    avg_seconds = elapsed / processed_this_run
    remaining = max(0, total - index)
    eta = avg_seconds * remaining
    return (
        f"predicted {index}/{total} "
        f"ok={ok} errors={errors} "
        f"acc={accuracy:.4f} "
        f"real={predictions.get('real', 0)} fake={predictions.get('fake', 0)} "
        f"cache={cached} cache_errors={cache_errors} "
        f"avg={avg_seconds:.2f}s/video "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
    )


def main() -> None:
    args = parse_args()
    validate_feature_cache_dir(args.feature_cache_dir, args.resume_feature_cache)
    examples = discover_examples(args.dataset_root)
    if args.limit is not None:
        examples = examples[: args.limit]
    existing_rows = read_existing_prediction_rows(args.output) if args.resume_feature_cache else []
    completed_count = completed_prefix_count(examples, existing_rows)
    if args.resume_feature_cache and existing_rows and completed_count == 0:
        raise ValueError(
            f"Existing output does not match selected dataset order, refusing resume: {args.output}"
        )
    if args.resume_feature_cache and completed_count:
        print(
            f"resume: existing_rows={len(existing_rows)} completed_prefix={completed_count} "
            f"remaining={len(examples) - completed_count}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selection = selection_from_checkpoint(args.checkpoint, model_kind="auto")
    loaded = load_inference_model(selection, device=args.device)
    specs = build_feature_cache_specs(loaded.config, loaded.modalities)
    rows: list[dict[str, Any]] = [dict(row) for row in existing_rows[:completed_count]]
    video_examples: list[VideoExample] = [
        to_video_example(example) for example in examples[:completed_count]
    ]
    cache_errors_by_path: dict[str, str] = {}
    start_time = time.perf_counter()
    try:
        output_mode = "a" if completed_count and completed_count == len(existing_rows) else "w"
        with args.output.open(output_mode, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
            if output_mode == "w":
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            for index, example in enumerate(examples[completed_count:], start=completed_count + 1):
                video_example = to_video_example(example)
                video_examples.append(video_example)
                try:
                    result = predict_video_fps(
                        loaded=loaded,
                        selection=selection,
                        video_path=example["path"],
                        threshold=args.threshold,
                        repeat=1,
                        warmup_runs=0,
                        decode_mode=args.decode_mode,
                    )
                    row = prediction_to_row(example, result)
                    cache_status, cache_error = write_prediction_feature_cache(
                        cache_dir=args.feature_cache_dir,
                        example=video_example,
                        specs=specs,
                        feature_batch=loaded.model.pipeline.last_feature_batch,
                        dataset_root=args.dataset_root,
                    )
                    row["feature_cache_status"] = cache_status
                    row["feature_cache_error"] = cache_error
                    if cache_error:
                        cache_errors_by_path[str(video_example.path)] = cache_error
                except Exception as error:
                    row = failure_row(example, error)
                    cache_errors_by_path[str(video_example.path)] = str(error)
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                if index == len(examples) or index % args.progress_every == 0:
                    print(
                        progress_line(
                            index,
                            len(examples),
                            rows,
                            start_time,
                            completed_at_start=completed_count,
                        ),
                        flush=True,
                    )
    finally:
        loaded.model.pipeline.close()

    if args.feature_cache_dir is not None:
        for modality, spec in specs.items():
            manifest_path = write_feature_cache_manifest(
                cache_dir=args.feature_cache_dir,
                examples=video_examples,
                spec=spec,
                dataset_root=args.dataset_root,
                errors_by_path=cache_errors_by_path,
            )
            print(f"feature cache manifest {modality}: {manifest_path}", flush=True)
    write_summary(args.output, rows, args)
    print(f"wrote: {args.output}", flush=True)
    print(f"summary: {args.output.with_suffix('.summary.json')}", flush=True)


if __name__ == "__main__":
    main()
