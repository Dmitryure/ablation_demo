from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_metrics import BinaryMetrics, safe_divide, write_dict_rows

DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.975,
    0.99,
    0.995,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report binary metrics at multiple thresholds for a predictions CSV."
    )
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--threshold", type=float, action="append", default=None)
    return parser.parse_args()


def load_prediction_scores(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((int(row["binary_label"]), float(row["binary_probability"])))
    if not rows:
        raise ValueError(f"No prediction rows found: {path}")
    return rows


def metrics_at_threshold(rows: list[tuple[int, float]], threshold: float) -> BinaryMetrics:
    tp = tn = fp = fn = 0
    for label, probability in rows:
        prediction = int(probability >= threshold)
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and prediction == 0:
            tn += 1
        elif label == 0 and prediction == 1:
            fp += 1
        elif label == 1 and prediction == 0:
            fn += 1
    precision = safe_divide(float(tp), float(tp + fp))
    recall = safe_divide(float(tp), float(tp + fn))
    specificity = safe_divide(float(tn), float(tn + fp))
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    return BinaryMetrics(
        accuracy=safe_divide(float(tp + tn), float(tp + tn + fp + fn)),
        balanced_accuracy=(recall + specificity) / 2.0,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
    )


def threshold_rows(
    rows: list[tuple[int, float]], thresholds: list[float]
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for threshold in thresholds:
        metrics = metrics_at_threshold(rows, threshold)
        result.append(
            {
                "threshold": f"{threshold:.6f}",
                **{
                    key: f"{value:.8f}" if isinstance(value, float) else value
                    for key, value in asdict(metrics).items()
                },
            }
        )
    return result


def print_rows(rows: list[dict[str, object]]) -> None:
    fields = (
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "specificity",
        "recall",
        "precision",
        "f1",
        "false_positive",
        "false_negative",
    )
    print(",".join(fields))
    for row in rows:
        print(",".join(str(row[field]) for field in fields))


def main() -> None:
    args = parse_args()
    thresholds = sorted(args.threshold or list(DEFAULT_THRESHOLDS))
    rows = threshold_rows(load_prediction_scores(args.predictions_csv), thresholds)
    print_rows(rows)
    if args.output is not None:
        write_dict_rows(args.output, rows)


if __name__ == "__main__":
    main()
