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

from feature_cache import cache_row_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add explicit example_locations to an existing shard index."
    )
    parser.add_argument("sharded_cache_dir", type=Path)
    return parser.parse_args()


def read_index(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("shards"), list):
        raise ValueError(f"Malformed shard index: {path}")
    return payload


def shard_locations(sharded_cache_dir: Path, index: dict[str, Any]) -> dict[str, dict[str, int]]:
    locations: dict[str, dict[str, int]] = {}
    shards = index["shards"]
    for shard_index, shard_record in enumerate(shards):
        path = sharded_cache_dir / str(shard_record["path"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        examples = payload.get("examples") if isinstance(payload, dict) else None
        if not isinstance(examples, list):
            raise ValueError(f"Malformed shard examples: {path}")
        for row_index, row in enumerate(examples):
            if not isinstance(row, dict):
                raise ValueError(f"Malformed shard row: {path}:{row_index}")
            key = cache_row_key(row)
            if key in locations:
                raise ValueError(f"Duplicate shard example key: {key}")
            locations[key] = {
                "shard_index": shard_index,
                "row_index": row_index,
            }
        print(
            f"indexed shard {shard_index + 1}/{len(shards)} "
            f"examples={len(examples)} total={len(locations)}",
            flush=True,
        )
    return locations


def write_index_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    index_path = args.sharded_cache_dir / "index.json"
    index = read_index(index_path)
    locations = shard_locations(args.sharded_cache_dir, index)
    expected = int(index.get("example_count", len(locations)))
    if len(locations) != expected:
        raise ValueError(f"Expected {expected} example locations, got {len(locations)}.")
    index["example_locations"] = locations
    write_index_atomic(index_path, index)
    print(f"wrote: {index_path}", flush=True)


if __name__ == "__main__":
    main()
