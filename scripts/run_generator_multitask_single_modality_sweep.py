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
CONFIG_DIR = PROJECT_ROOT / "runs" / "configs" / "single_modality_best"
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "single_modality_best"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "run_generator_multitask_training.py"


@dataclass(frozen=True)
class Trial:
    modality: str

    @property
    def name(self) -> str:
        return f"{self.modality}_seed0_lr3e4_gen0p15_warm2_e26"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-modality versions of the current best grouped-unknown run."
    )
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--write-configs-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def trials() -> list[Trial]:
    return [Trial(modality) for modality in ("rgb", "eye_gaze", "face_mesh", "depth")]


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
    config["modalities"] = [trial.modality]
    config["seed"] = 0
    run = config.setdefault("training", {}).setdefault("run", {})
    run["output_dir"] = repo_relative_string(output_root / trial.name)
    run["modalities"] = [trial.modality]
    run["epochs"] = 26
    run["lr"] = 0.0003
    run["generator_loss_weight"] = 0.15
    run["binary_warmup_epochs"] = 2
    run["seed"] = 0
    run["dry_run"] = False
    return config


def write_trial_configs(
    base_config: Path,
    config_dir: Path,
    output_root: Path,
    trial_list: list[Trial],
) -> list[Path]:
    base = load_yaml(base_config)
    config_paths: list[Path] = []
    for index, trial in enumerate(trial_list, start=1):
        path = config_dir / f"{index:02d}_{trial.name}.yaml"
        write_yaml(path, trial_config(base, trial, output_root))
        config_paths.append(path)
    return config_paths


def read_summary(output_dir: Path) -> dict[str, Any] | None:
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        return None
    return summary


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


def compact_summary(trial: Trial, output_dir: Path) -> dict[str, Any]:
    summary = read_summary(output_dir)
    row: dict[str, Any] = {
        "name": trial.name,
        "modality": trial.modality,
        "output_dir": str(output_dir),
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


def write_sweep_summary(trial_list: list[Trial], output_root: Path) -> None:
    rows = [compact_summary(trial, output_root / trial.name) for trial in trial_list]
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "sweep_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": rows}, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    trial_list = trials()
    config_paths = write_trial_configs(
        args.base_config,
        args.config_dir,
        args.output_root,
        trial_list,
    )

    if args.list or args.write_configs_only:
        for trial, config_path in zip(trial_list, config_paths, strict=True):
            print(f"{trial.name}\t{config_path}")
        write_sweep_summary(trial_list, args.output_root)
        return

    for trial, config_path in zip(trial_list, config_paths, strict=True):
        output_dir = args.output_root / trial.name
        if read_summary(output_dir) is not None and not args.rerun:
            print(f"skip complete: {trial.name}", flush=True)
            continue
        print(f"start: {trial.name}", flush=True)
        return_code = run_trial(config_path, output_dir)
        print(f"done: {trial.name} return_code={return_code}", flush=True)
        write_sweep_summary(trial_list, args.output_root)
        if return_code != 0 and not args.continue_on_error:
            raise SystemExit(return_code)

    write_sweep_summary(trial_list, args.output_root)


if __name__ == "__main__":
    main()
