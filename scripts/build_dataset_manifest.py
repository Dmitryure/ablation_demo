from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (
    build_real_fake_examples,
    format_split_audit,
    summarize_examples,
    write_dataset_manifest,
)

DEFAULT_REAL_DIR = Path("/home/comp/video_loader/cut_vids1/chunks")
DEFAULT_FAKE_DIR = Path("/home/comp/facefusion/output_f_faceswap")
DEFAULT_OUTPUT = Path("/home/comp/ablation_task/data/real_fake_manifest.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a labeled real/fake video manifest.")
    parser.add_argument("--real-dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--fake-dir", type=Path, default=DEFAULT_FAKE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = build_real_fake_examples(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    manifest_path = write_dataset_manifest(examples, args.output)
    summary = summarize_examples(examples)

    print(f"wrote: {manifest_path}")
    print(f"total_examples: {len(examples)}")
    for split, split_summary in summary.items():
        print(
            f"{split}: total={split_summary['total']} real={split_summary['real']} fake={split_summary['fake']}"
        )
    for line in format_split_audit(examples):
        print(line)


if __name__ == "__main__":
    main()
