from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_model_fps import (
    FpsPrediction,
    load_inference_model,
    predict_video_fps,
    selection_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure end-to-end FPS for a real/fake model checkpoint on one video."
    )
    parser.add_argument("video_path", type=Path)
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument(
        "--model-kind",
        choices=("auto", "binary", "generator_multitask"),
        default="auto",
        help="Expected model family. Leave auto to infer from run_config.json.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--decode-mode", choices=("scan", "seek"), default="scan")
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    return value


def print_text_result(result: FpsPrediction) -> None:
    print(f"prediction={result.label} fake_probability={result.fake_probability:.6f}")
    if result.generator_label is not None and result.generator_probability is not None:
        print(
            "generator="
            f"{result.generator_label} generator_probability={result.generator_probability:.6f}"
        )
    print(
        "fps="
        f"{result.end_to_end_video_fps:.2f} "
        f"end_to_end_seconds={result.end_to_end_seconds:.4f} "
        f"forward_seconds={result.forward_seconds_median:.4f}"
    )
    print(
        "model="
        f"{result.model_kind} device={result.device} "
        f"score={result.score:.6f} score_key={result.score_key}"
    )
    print(f"checkpoint={result.checkpoint_path}")
    print(f"summary={result.summary_path}")


def main() -> None:
    args = parse_args()
    video_path = args.video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    selection = selection_from_checkpoint(args.checkpoint_path, model_kind=args.model_kind)
    loaded = load_inference_model(selection, device=args.device)
    try:
        result = predict_video_fps(
            loaded=loaded,
            selection=selection,
            video_path=video_path,
            threshold=args.threshold,
            repeat=args.repeat,
            warmup_runs=args.warmup_runs,
            decode_mode=args.decode_mode,
        )
    finally:
        loaded.model.pipeline.close()

    if args.json:
        print(json.dumps(json_ready(asdict(result)), indent=2, sort_keys=True))
    else:
        print_text_result(result)


if __name__ == "__main__":
    main()
