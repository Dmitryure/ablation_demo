from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import build_fusion_pipeline, load_pipeline_yaml

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "registry_fusion.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that configured modality models and extractors initialize."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Override config device for this check.",
    )
    return parser.parse_args()


def with_device_override(config: dict[str, Any], device: str | None) -> dict[str, Any]:
    if device is None:
        return config
    return {**config, "device": device}


def summarize_pipeline(modalities: Sequence[str], device: torch.device) -> None:
    print(f"device={device}", flush=True)
    print(f"modalities={','.join(modalities)}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}", flush=True)


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT.resolve()
    config_path = args.config.resolve()

    print(f"project_root={project_root}", flush=True)
    print(f"config={config_path}", flush=True)
    config = with_device_override(load_pipeline_yaml(config_path), args.device)

    pipeline = None
    try:
        result = build_fusion_pipeline(config, modalities=args.modalities)
        pipeline = result.pipeline
        summarize_pipeline(result.pipeline.enabled_modalities, result.device)
        for warning in result.warnings:
            print(f"warning={warning}", flush=True)
        print("model_initialization=ok", flush=True)
    finally:
        if pipeline is not None:
            pipeline.close()


if __name__ == "__main__":
    main()
