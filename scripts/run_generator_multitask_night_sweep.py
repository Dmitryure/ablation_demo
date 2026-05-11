from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs" / "train_generator_multitask_v2.yaml"
CONFIG_DIR = PROJECT_ROOT / "runs" / "configs" / "night_sweep"
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "night_sweep"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "run_generator_multitask_training.py"


@dataclass(frozen=True)
class Trial:
    name: str
    seed: int
    lr: float
    generator_loss_weight: float
    binary_warmup_epochs: int
    epochs: int = 26


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the grouped-unknown generator multitask overnight hyperparameter sweep."
    )
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--write-configs-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def sweep_trials() -> list[Trial]:
    trials: list[Trial] = []

    for seed in range(5):
        trials.append(
            Trial(
                name=f"seed{seed}_lr3e4_gen025_warm2_e26",
                seed=seed,
                lr=0.0003,
                generator_loss_weight=0.25,
                binary_warmup_epochs=2,
            )
        )

    for generator_loss_weight in (0.10, 0.15):
        for seed in range(3):
            gen_name = str(generator_loss_weight).replace(".", "p")
            trials.append(
                Trial(
                    name=f"seed{seed}_lr3e4_gen{gen_name}_warm2_e26",
                    seed=seed,
                    lr=0.0003,
                    generator_loss_weight=generator_loss_weight,
                    binary_warmup_epochs=2,
                )
            )

    for lr in (0.0002, 0.0001):
        for seed in range(3):
            lr_name = "2e4" if lr == 0.0002 else "1e4"
            trials.append(
                Trial(
                    name=f"seed{seed}_lr{lr_name}_gen025_warm2_e26",
                    seed=seed,
                    lr=lr,
                    generator_loss_weight=0.25,
                    binary_warmup_epochs=2,
                )
            )

    for seed in range(3):
        trials.append(
            Trial(
                name=f"seed{seed}_lr3e4_gen025_warm4_e26",
                seed=seed,
                lr=0.0003,
                generator_loss_weight=0.25,
                binary_warmup_epochs=4,
            )
        )

    return trials


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def repo_relative_string(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def trial_config(base: dict[str, Any], trial: Trial, output_root: Path) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["seed"] = trial.seed
    run = config.setdefault("training", {}).setdefault("run", {})
    run["output_dir"] = repo_relative_string(output_root / trial.name)
    run["epochs"] = trial.epochs
    run["lr"] = trial.lr
    run["generator_loss_weight"] = trial.generator_loss_weight
    run["binary_warmup_epochs"] = trial.binary_warmup_epochs
    run["seed"] = trial.seed
    run["dry_run"] = False
    return config


def write_trial_configs(
    base_config: Path,
    config_dir: Path,
    output_root: Path,
    trials: list[Trial],
) -> list[Path]:
    base = load_yaml(base_config)
    config_paths: list[Path] = []
    for index, trial in enumerate(trials, start=1):
        config = trial_config(base, trial, output_root)
        path = config_dir / f"{index:02d}_{trial.name}.yaml"
        write_yaml(path, config)
        config_paths.append(path)
    return config_paths


def run_trial(config_path: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "sweep_stdout.log"
    command = [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path)]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        return process.wait()


def read_summary(output_dir: Path) -> dict[str, Any] | None:
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        return None
    return summary


def compact_summary(trial: Trial, output_dir: Path) -> dict[str, Any]:
    summary = read_summary(output_dir)
    row: dict[str, Any] = {
        "name": trial.name,
        "output_dir": str(output_dir),
        "seed": trial.seed,
        "lr": trial.lr,
        "generator_loss_weight": trial.generator_loss_weight,
        "binary_warmup_epochs": trial.binary_warmup_epochs,
        "epochs": trial.epochs,
        "complete": summary is not None,
    }
    if summary is None:
        return row

    test_binary = summary["split_summaries"]["test"]["binary_metrics"]
    holdout = summary.get("extra_fake_holdout_summary", {})
    row.update(
        {
            "best_checkpoint_score": summary.get("best_checkpoint_score"),
            "test_balanced_accuracy": test_binary.get("balanced_accuracy"),
            "test_fake_recall": test_binary.get("recall"),
            "test_specificity": test_binary.get("specificity"),
            "test_false_positive": test_binary.get("false_positive"),
            "test_false_negative": test_binary.get("false_negative"),
            "extra_fake_holdout_recall": holdout.get("fake_recall"),
        }
    )
    return row


def write_sweep_summary(trials: list[Trial], output_root: Path) -> None:
    rows = [compact_summary(trial, output_root / trial.name) for trial in trials]
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"runs": rows}, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    trials = sweep_trials()
    if args.limit is not None:
        trials = trials[: args.limit]

    config_paths = write_trial_configs(
        args.base_config,
        args.config_dir,
        args.output_root,
        trials,
    )

    if args.list or args.write_configs_only:
        for trial, config_path in zip(trials, config_paths, strict=True):
            print(f"{trial.name}\t{config_path}")
        write_sweep_summary(trials, args.output_root)
        return

    for trial, config_path in zip(trials, config_paths, strict=True):
        output_dir = args.output_root / trial.name
        if read_summary(output_dir) is not None and not args.rerun:
            print(f"skip complete: {trial.name}", flush=True)
            continue
        print(f"start: {trial.name}", flush=True)
        return_code = run_trial(config_path, output_dir)
        print(f"done: {trial.name} return_code={return_code}", flush=True)
        write_sweep_summary(trials, args.output_root)
        if return_code != 0 and not args.continue_on_error:
            raise SystemExit(return_code)

    write_sweep_summary(trials, args.output_root)


if __name__ == "__main__":
    main()
