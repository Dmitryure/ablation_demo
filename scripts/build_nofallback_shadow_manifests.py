from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import load_video_clip
from face_crop_config import (
    FACE_CROP_STATUS_FALLBACK,
    face_crop_cache_variant,
    merged_face_crop_config,
)
from feature_cache import (
    FEATURE_CACHE_MANIFEST_COLUMNS,
    FeatureCacheSpec,
    build_feature_cache_specs,
    feature_cache_manifest_path,
    feature_cache_spec_id,
)
from pipeline import load_pipeline_yaml
from scripts.run_iterative_cached_ablation import (
    DEFAULT_DATASET_ROOT,
    training_run_section,
    write_json,
)
from scripts.run_readonly_cached_smoke import (
    DEFAULT_EXPECTED_MANIFEST_ROWS,
    manifest_key,
    reject_output_inside_inputs,
)

DEFAULT_CONFIG = PROJECT_ROOT / "runs" / "configs" / (
    "train_readonly_cache_v2_balanced_fullreal_no_fau_with_v1_rppg_lr1e4.yaml"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "cache_filters" / "v2_nofallback_manifests"
DEFAULT_PROGRESS_EVERY = 500
NO_FACE_ERROR = "face_crop_not_detected"


@dataclass(frozen=True)
class DetectionGroup:
    group_id: str
    modalities: tuple[str, ...]
    frame_count: int
    image_size: int
    cache_variant: str
    face_crop_config: dict[str, Any]


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy v2 cache manifests and mark face-crop fallback rows as failed."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modalities", nargs="+", required=True)
    parser.add_argument(
        "--apply-to-modalities",
        nargs="*",
        default=None,
        help=(
            "Write shadow manifests for these modalities using fallback keys detected from "
            "--modalities. Useful for RGB-only filtering across an all-modality run."
        ),
    )
    parser.add_argument("--expected-manifest-rows", type=int, default=DEFAULT_EXPECTED_MANIFEST_ROWS)
    parser.add_argument("--video-decode-mode", choices=("seek", "scan"), default="scan")
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument(
        "--skip-failures",
        action="store_true",
        help="Mark videos that cannot be decoded as failed in shadow manifests instead of aborting.",
    )
    return parser.parse_args()


def resolve_cache_dir(config: Mapping[str, Any], cli_cache_dir: Path | None) -> Path:
    if cli_cache_dir is not None:
        return cli_cache_dir
    run_config = training_run_section(config)
    cache_dir = run_config.get("cache_dir")
    if cache_dir is None:
        raise ValueError("Shadow manifest build requires `training.run.cache_dir` or --cache-dir.")
    return Path(cache_dir)


def reject_existing_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Shadow manifest output already exists and is not empty: {output_dir}")


def read_manifest_rows(path: Path, expected_rows: int) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Malformed manifest without header: {path}")
        missing = set(FEATURE_CACHE_MANIFEST_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest missing columns {sorted(missing)}: {path}")
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(rows)}: {path}")
    return rows, fieldnames


def write_manifest_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def row_key(row: Mapping[str, str]) -> str:
    return manifest_key(str(row["class_name"]), str(row["filename"]))


def dataset_video_root(dataset_root: Path) -> Path:
    videos = dataset_root / "videos"
    return videos if videos.is_dir() else dataset_root


def row_video_path(dataset_root: Path, row: Mapping[str, str]) -> Path:
    return dataset_video_root(dataset_root) / str(row["class_name"]) / str(row["filename"])


def modality_face_crop_config(config: Mapping[str, Any], modality: str) -> dict[str, Any]:
    section = config.get(modality, {})
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ValueError(f"`{modality}` config must be a mapping.")
    return merged_face_crop_config(
        global_config=config,
        modality_config=section,
        default_enabled=False,
    )


def detection_groups(
    config: Mapping[str, Any],
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
) -> list[DetectionGroup]:
    grouped: dict[tuple[int, int, str], list[str]] = defaultdict(list)
    configs_by_group: dict[tuple[int, int, str], dict[str, Any]] = {}
    for modality in modalities:
        if modality == "rppg":
            raise ValueError("rPPG no-fallback shadow manifests are out of scope for this script.")
        crop_config = modality_face_crop_config(config, modality)
        if not crop_config["enabled"]:
            raise ValueError(f"Face crop must be enabled for no-fallback audit: {modality}")
        cache_variant = face_crop_cache_variant(
            global_config=config,
            modality_config=config.get(modality, {}),
            default_enabled=False,
        )
        if cache_variant is None:
            raise ValueError(f"Missing face-crop cache variant for {modality}.")
        spec = specs[modality]
        key = (spec.frame_count, spec.image_size, cache_variant)
        grouped[key].append(modality)
        configs_by_group[key] = crop_config

    groups: list[DetectionGroup] = []
    for index, (key, group_modalities) in enumerate(sorted(grouped.items())):
        frame_count, image_size, cache_variant = key
        groups.append(
            DetectionGroup(
                group_id=f"group_{index:02d}_frames_{frame_count}_size_{image_size}",
                modalities=tuple(group_modalities),
                frame_count=frame_count,
                image_size=image_size,
                cache_variant=cache_variant,
                face_crop_config=configs_by_group[key],
            )
        )
    return groups


def cached_rows_for_group(
    rows_by_modality: Mapping[str, Sequence[Mapping[str, str]]],
    modalities: Sequence[str],
) -> dict[str, Mapping[str, str]]:
    rows: dict[str, Mapping[str, str]] = {}
    for modality in modalities:
        for row in rows_by_modality[modality]:
            if str(row.get("status", "")).strip() != "cached":
                continue
            rows.setdefault(row_key(row), row)
    return rows


def detect_fallback_keys(
    rows_by_key: Mapping[str, Mapping[str, str]],
    group: DetectionGroup,
    dataset_root: Path,
    decode_mode: str,
    progress_every: int,
    skip_failures: bool,
) -> tuple[set[str], set[str], dict[str, Any]]:
    fallback_keys: set[str] = set()
    failure_keys: set[str] = set()
    started = time.monotonic()
    total = len(rows_by_key)
    for index, (key, row) in enumerate(sorted(rows_by_key.items()), start=1):
        try:
            clip = load_video_clip(
                row_video_path(dataset_root, row),
                num_frames=group.frame_count,
                image_size=group.image_size,
                decode_mode=decode_mode,
                face_crop_config=group.face_crop_config,
            )
        except Exception as exc:
            if not skip_failures:
                raise
            failure_keys.add(key)
            log(f"detect failure skipped: group={group.group_id} key={key} error={exc}")
            clip = None
        if clip is not None and str(clip.get("face_crop_status")) == FACE_CROP_STATUS_FALLBACK:
            fallback_keys.add(key)
        if progress_every > 0 and (index % progress_every == 0 or index == total):
            elapsed = time.monotonic() - started
            log(
                f"detect {group.group_id}: checked={index}/{total} "
                f"fallback={len(fallback_keys)} failures={len(failure_keys)} elapsed={elapsed:.1f}s"
            )
    return fallback_keys, failure_keys, {
        "checked": total,
        "fallback": len(fallback_keys),
        "decode_failures": len(failure_keys),
        "detected": total - len(fallback_keys) - len(failure_keys),
        "elapsed_seconds": time.monotonic() - started,
    }


def mark_fallback_rows_failed(
    rows: Sequence[Mapping[str, str]],
    fallback_keys: set[str],
    decode_failure_keys: set[str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    output: list[dict[str, str]] = []
    counts = {"cached": 0, "failed": 0, "marked_failed": 0, "decode_failed": 0}
    for row in rows:
        updated = dict(row)
        key = row_key(updated)
        if str(updated.get("status", "")).strip() == "cached":
            if key in fallback_keys:
                updated["status"] = "failed"
                updated["error"] = NO_FACE_ERROR
                counts["marked_failed"] += 1
            elif key in decode_failure_keys:
                updated["status"] = "failed"
                updated["error"] = "video_decode_failed"
                counts["decode_failed"] += 1
        if str(updated.get("status", "")).strip() == "cached":
            counts["cached"] += 1
        else:
            counts["failed"] += 1
        output.append(updated)
    return output, counts


def fallback_key_path(output_dir: Path, group: DetectionGroup) -> Path:
    return output_dir / f"{group.group_id}_fallback_keys.json"


def build_shadow_manifests(
    config: Mapping[str, Any],
    cache_dir: Path,
    output_dir: Path,
    dataset_root: Path,
    modalities: Sequence[str],
    apply_to_modalities: Sequence[str] | None,
    expected_rows: int,
    decode_mode: str,
    progress_every: int,
    skip_failures: bool = False,
) -> dict[str, Any]:
    detect_modalities = tuple(dict.fromkeys(modalities))
    output_modalities = tuple(dict.fromkeys([*(apply_to_modalities or ()), *detect_modalities]))
    specs = build_feature_cache_specs(config, output_modalities)
    rows_by_modality: dict[str, list[dict[str, str]]] = {}
    fieldnames_by_modality: dict[str, list[str]] = {}
    for modality in output_modalities:
        path = feature_cache_manifest_path(cache_dir, specs[modality])
        rows, fieldnames = read_manifest_rows(path, expected_rows)
        rows_by_modality[modality] = rows
        fieldnames_by_modality[modality] = fieldnames
        log(f"manifest read: modality={modality} path={path} rows={len(rows)}")

    fallback_by_modality: dict[str, set[str]] = {modality: set() for modality in output_modalities}
    decode_failures_by_modality: dict[str, set[str]] = {
        modality: set() for modality in output_modalities
    }
    detected_fallback_union: set[str] = set()
    detected_failure_union: set[str] = set()
    groups_summary: dict[str, Any] = {}
    for group in detection_groups(config, specs, detect_modalities):
        rows_by_key = cached_rows_for_group(rows_by_modality, group.modalities)
        log(
            f"detect start: group={group.group_id} modalities={','.join(group.modalities)} "
            f"rows={len(rows_by_key)} frame_count={group.frame_count} image_size={group.image_size}"
        )
        fallback_keys, failure_keys, detection_summary = detect_fallback_keys(
            rows_by_key,
            group,
            dataset_root,
            decode_mode,
            progress_every,
            skip_failures,
        )
        for modality in group.modalities:
            fallback_by_modality[modality].update(fallback_keys)
            decode_failures_by_modality[modality].update(failure_keys)
        detected_fallback_union.update(fallback_keys)
        detected_failure_union.update(failure_keys)
        key_path = fallback_key_path(output_dir, group)
        write_json(
            key_path,
            {
                "group": group.group_id,
                "modalities": list(group.modalities),
                "frame_count": group.frame_count,
                "image_size": group.image_size,
                "cache_variant": group.cache_variant,
                "fallback_keys": sorted(fallback_keys),
                "decode_failure_keys": sorted(failure_keys),
            },
        )
        groups_summary[group.group_id] = {
            "modalities": list(group.modalities),
            "frame_count": group.frame_count,
            "image_size": group.image_size,
            "cache_variant": group.cache_variant,
            "fallback_keys_path": str(key_path),
            **detection_summary,
        }

    if apply_to_modalities is not None:
        for modality in output_modalities:
            if modality in detect_modalities:
                continue
            fallback_by_modality[modality].update(detected_fallback_union)
            decode_failures_by_modality[modality].update(detected_failure_union)

    manifest_summary: dict[str, Any] = {}
    for modality in output_modalities:
        shadow_rows, counts = mark_fallback_rows_failed(
            rows_by_modality[modality],
            fallback_by_modality[modality],
            decode_failures_by_modality[modality],
        )
        output_path = feature_cache_manifest_path(output_dir, specs[modality])
        write_manifest_rows(output_path, fieldnames_by_modality[modality], shadow_rows)
        manifest_summary[modality] = {
            "path": str(output_path),
            "spec_id": feature_cache_spec_id(specs[modality]),
            "fallback_keys": len(fallback_by_modality[modality]),
            "decode_failure_keys": len(decode_failures_by_modality[modality]),
            **counts,
        }
        log(
            f"manifest wrote: modality={modality} cached={counts['cached']} "
            f"failed={counts['failed']} marked_failed={counts['marked_failed']} path={output_path}"
        )

    summary = {
        "mode": "nofallback_shadow_manifests",
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "dataset_root": str(dataset_root),
        "detected_modalities": list(detect_modalities),
        "output_modalities": list(output_modalities),
        "expected_rows": expected_rows,
        "decode_mode": decode_mode,
        "skip_failures": skip_failures,
        "groups": groups_summary,
        "manifests": manifest_summary,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    config = load_pipeline_yaml(args.config)
    cache_dir = resolve_cache_dir(config, args.cache_dir)
    output_dir = args.output_dir
    reject_output_inside_inputs(output_dir, cache_dir, args.dataset_root)
    reject_existing_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"shadow manifest build start: config={args.config} cache_dir={cache_dir} "
        f"output_dir={output_dir} modalities={','.join(args.modalities)}"
    )
    summary = build_shadow_manifests(
        config=config,
        cache_dir=cache_dir,
        output_dir=output_dir,
        dataset_root=args.dataset_root,
        modalities=tuple(args.modalities),
        apply_to_modalities=None if args.apply_to_modalities is None else tuple(args.apply_to_modalities),
        expected_rows=args.expected_manifest_rows,
        decode_mode=args.video_decode_mode,
        progress_every=args.progress_every,
        skip_failures=args.skip_failures,
    )
    log(f"shadow manifest build done: summary={output_dir / 'summary.json'}")
    log(
        "shadow manifest counts: "
        + ", ".join(
            f"{modality}=cached:{item['cached']} failed:{item['failed']}"
            for modality, item in sorted(summary["manifests"].items())
        )
    )


if __name__ == "__main__":
    main()
