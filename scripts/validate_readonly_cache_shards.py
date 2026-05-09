from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import VideoExample
from feature_cache import (
    SHARDED_CACHE_SCHEMA_VERSION,
    cache_example_key,
    cache_row_key,
    normalize_cache_dataset_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate read-only cache shard index/payloads.")
    parser.add_argument("sharded_cache_dir", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--allow-legacy-shards", action="store_true")
    parser.add_argument(
        "--skip-payloads",
        action="store_true",
        help="Only validate index metadata and shard files, not row payload contents.",
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def read_index(sharded_cache_dir: Path) -> dict[str, Any]:
    index_path = sharded_cache_dir / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing sharded cache index: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed sharded cache index: {index_path}")
    return payload


def validate_readonly_cache_shards(
    sharded_cache_dir: Path,
    dataset_root: Path | None = None,
    selected_examples: list[VideoExample] | None = None,
    allow_legacy_shards: bool = False,
    check_payloads: bool = True,
) -> dict[str, Any]:
    index = read_index(sharded_cache_dir)
    schema_version = int(index.get("schema_version", index.get("version", 1)))
    result: dict[str, Any] = {
        "sharded_cache_dir": str(sharded_cache_dir),
        "schema_version": schema_version,
        "valid": True,
        "errors": [],
        "warnings": [],
        "example_count": int(index.get("example_count", 0)),
        "shard_count": len(index.get("shards", [])),
        "duplicate_keys": 0,
        "missing_selected_keys": 0,
        "missing_shard_files": 0,
        "row_mapping_errors": 0,
        "feature_key_errors": 0,
        "single_label_shards": 0,
    }
    if schema_version != SHARDED_CACHE_SCHEMA_VERSION and not allow_legacy_shards:
        result["errors"].append(
            f"expected schema_version={SHARDED_CACHE_SCHEMA_VERSION}, got {schema_version}"
        )
    locations = index.get("example_locations")
    if not isinstance(locations, dict):
        result["errors"].append("missing example_locations mapping")
        locations = {}
    if result["example_count"] and len(locations) != result["example_count"]:
        result["warnings"].append(
            f"example_count={result['example_count']} but locations={len(locations)}"
        )
    result["duplicate_keys"] = max(0, result["example_count"] - len(locations))

    normalized_root = normalize_cache_dataset_root(dataset_root)
    if selected_examples is not None:
        missing = [
            cache_example_key(example, normalized_root)
            for example in selected_examples
            if cache_example_key(example, normalized_root) not in locations
        ]
        result["missing_selected_keys"] = len(missing)
        if missing:
            result["errors"].append(
                "selected examples missing from shard index: " + ", ".join(missing[:5])
            )

    shards = index.get("shards", [])
    if not isinstance(shards, list):
        result["errors"].append("index shards must be a list")
        shards = []
    global_classes = {
        class_name
        for shard in shards
        if isinstance(shard, dict)
        for class_name, count in (shard.get("class_counts") or {}).items()
        if int(count) > 0
    }
    for shard_index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            result["errors"].append(f"malformed shard record at index {shard_index}")
            continue
        shard_path = sharded_cache_dir / str(shard.get("path", ""))
        if not shard_path.is_file():
            result["missing_shard_files"] += 1
            result["errors"].append(f"missing shard file: {shard_path}")
        class_counts = {
            str(class_name): int(count)
            for class_name, count in (shard.get("class_counts") or {}).items()
            if int(count) > 0
        }
        if len(global_classes) > 1 and len(class_counts) == 1 and int(shard.get("example_count", 0)) > 1:
            result["single_label_shards"] += 1

    if check_payloads and not result["missing_shard_files"]:
        validate_shard_payloads(sharded_cache_dir, shards, locations, result)
    if result["row_mapping_errors"]:
        result["errors"].append(f"row_mapping_errors={result['row_mapping_errors']}")
    if result["feature_key_errors"]:
        result["errors"].append(f"feature_key_errors={result['feature_key_errors']}")

    result["valid"] = not result["errors"]
    if result["errors"]:
        raise ValueError("Invalid read-only cache shards: " + "; ".join(result["errors"][:5]))
    return result


def validate_shard_payloads(
    sharded_cache_dir: Path,
    shards: list[Any],
    locations: dict[str, Any],
    result: dict[str, Any],
) -> None:
    for shard_index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            continue
        payload = torch.load(
            sharded_cache_dir / str(shard["path"]),
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, dict):
            result["row_mapping_errors"] += 1
            result["errors"].append(f"malformed payload for shard {shard_index}")
            continue
        rows = payload.get("examples")
        features = payload.get("features")
        if not isinstance(rows, list) or not isinstance(features, dict):
            result["row_mapping_errors"] += 1
            result["errors"].append(f"missing rows/features for shard {shard_index}")
            continue
        if len(rows) != int(shard.get("example_count", len(rows))):
            result["row_mapping_errors"] += 1
            result["errors"].append(f"row count mismatch for shard {shard_index}")
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                result["row_mapping_errors"] += 1
                continue
            key = cache_row_key(row)
            location = locations.get(key)
            if (
                not isinstance(location, dict)
                or int(location.get("shard_index", -1)) != shard_index
                or int(location.get("row_index", -1)) != row_index
            ):
                result["row_mapping_errors"] += 1
        for key, value in features.items():
            if not isinstance(value, torch.Tensor) or value.shape[0] != len(rows):
                result["feature_key_errors"] += 1
                result["errors"].append(f"bad feature tensor {key!r} in shard {shard_index}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    summary = validate_readonly_cache_shards(
        sharded_cache_dir=args.sharded_cache_dir,
        dataset_root=args.dataset_root,
        allow_legacy_shards=args.allow_legacy_shards,
        check_payloads=not args.skip_payloads,
    )
    if args.summary_json is not None:
        write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
