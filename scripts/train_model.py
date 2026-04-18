from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import RunConfig, train_from_yaml


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "registry_fusion.yaml"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "real_fake_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"
DEFAULT_WEIGHT_DECAY = 0.01


def _parse_modalities(values: list[str] | None) -> tuple[str, ...] | None:
    if not values:
        return None
    modalities: list[str] = []
    for value in values:
        for item in value.split(","):
            stripped = item.strip()
            if stripped:
                modalities.append(stripped)
    return tuple(modalities) or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train registry fusion real/fake classifier.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--modalities", action="append", default=None)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_config = RunConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        save_every_epoch=args.save_every_epoch,
        log_interval=args.log_interval,
    )
    result = train_from_yaml(
        config_path=args.config,
        manifest_path=args.manifest,
        run_config=run_config,
        modalities=_parse_modalities(args.modalities),
    )

    for item in result.history:
        train_metrics = item["train"]
        val_metrics = item["val"]
        val_roc_auc = "nan" if val_metrics["roc_auc"] is None else f"{val_metrics['roc_auc']:.4f}"
        print(
            "epoch="
            f"{item['epoch']} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_roc_auc={val_roc_auc}"
        )
    print(f"best_epoch={result.best_epoch}")
    print(f"best_checkpoint={result.best_checkpoint_path}")
    print(f"last_checkpoint={result.last_checkpoint_path}")


if __name__ == "__main__":
    main()
