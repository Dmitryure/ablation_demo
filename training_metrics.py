from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


@dataclass(frozen=True)
class PredictionRecord:
    path: str
    split: str
    class_name: str
    generator_id: str
    binary_label: int
    binary_probability: float
    binary_prediction: int
    generator_label: str
    generator_prediction: str
    generator_probability: float
    generator_top2_prediction: str = ""
    generator_top2_probability: float = 0.0
    generator_margin: float = 0.0
    calibrated_output: str = ""
    binary_confidence: float = 0.0
    binary_confidence_label: str = ""
    generator_confidence_label: str = ""


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0.0 else numerator / denominator


def binary_metrics(records: Sequence[PredictionRecord]) -> BinaryMetrics:
    tp = sum(1 for record in records if record.binary_label == 1 and record.binary_prediction == 1)
    tn = sum(1 for record in records if record.binary_label == 0 and record.binary_prediction == 0)
    fp = sum(1 for record in records if record.binary_label == 0 and record.binary_prediction == 1)
    fn = sum(1 for record in records if record.binary_label == 1 and record.binary_prediction == 0)
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


def generator_metrics(records: Sequence[PredictionRecord]) -> list[dict[str, object]]:
    by_generator: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        if record.binary_label == 1:
            by_generator[record.generator_label].append(record)

    rows: list[dict[str, object]] = []
    for generator, group in sorted(by_generator.items()):
        correct = sum(1 for record in group if record.generator_prediction == generator)
        binary_detected = sum(1 for record in group if record.binary_prediction == 1)
        rows.append(
            {
                "generator_id": generator,
                "count": len(group),
                "generator_recall": safe_divide(float(correct), float(len(group))),
                "binary_recall": safe_divide(float(binary_detected), float(len(group))),
            }
        )
    return rows


def prediction_summary(records: Sequence[PredictionRecord]) -> dict[str, object]:
    if not records:
        return {
            "binary_predicted_fake_rate": 0.0,
            "real_binary_probability_mean": 0.0,
            "fake_binary_probability_mean": 0.0,
            "generator_prediction_counts": {},
        }
    real_probs = [record.binary_probability for record in records if record.binary_label == 0]
    fake_probs = [record.binary_probability for record in records if record.binary_label == 1]
    generator_counts: Counter[str] = Counter(
        record.generator_prediction for record in records if record.binary_label == 1
    )
    return {
        "binary_predicted_fake_rate": sum(record.binary_prediction for record in records)
        / len(records),
        "real_binary_probability_mean": sum(real_probs) / len(real_probs) if real_probs else 0.0,
        "fake_binary_probability_mean": sum(fake_probs) / len(fake_probs) if fake_probs else 0.0,
        "generator_prediction_counts": dict(sorted(generator_counts.items())),
    }


def macro_generator_recall(rows: Sequence[Mapping[str, object]]) -> float:
    values = [float(row["generator_recall"]) for row in rows]
    return sum(values) / len(values) if values else 0.0


def worst_generator_recall(rows: Sequence[Mapping[str, object]]) -> float:
    values = [float(row["generator_recall"]) for row in rows]
    return min(values) if values else 0.0


def known_generator_precision_at_coverage(
    records: Sequence[PredictionRecord],
    unknown_group_name: str,
    min_coverage: float = 0.30,
) -> dict[str, float]:
    known = [
        record
        for record in records
        if record.binary_label == 1 and record.generator_label != unknown_group_name
    ]
    named = [
        record
        for record in known
        if record.binary_prediction == 1 and record.generator_prediction != unknown_group_name
    ]
    correct = [record for record in named if record.generator_prediction == record.generator_label]
    coverage = safe_divide(float(len(named)), float(len(known)))
    precision = safe_divide(float(len(correct)), float(len(named)))
    coverage_factor = min(1.0, safe_divide(coverage, min_coverage))
    return {
        "known_generator_count": float(len(known)),
        "known_generator_named_count": float(len(named)),
        "known_generator_correct_named_count": float(len(correct)),
        "known_generator_precision": precision,
        "known_generator_coverage": coverage,
        "known_generator_precision_at_coverage": precision * coverage_factor,
    }


def composite_checkpoint_score(
    binary: BinaryMetrics,
    macro_recall: float,
    known_precision_at_coverage: float,
) -> float:
    binary_robust = min(binary.specificity, binary.recall)
    return (
        0.50 * binary_robust
        + 0.30 * binary.balanced_accuracy
        + 0.15 * known_precision_at_coverage
        + 0.05 * macro_recall
    )


def binary_robust_checkpoint_score(binary: BinaryMetrics) -> float:
    binary_robust = min(binary.specificity, binary.recall)
    return 0.60 * binary_robust + 0.40 * binary.balanced_accuracy


def generator_confusion(records: Sequence[PredictionRecord]) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for record in records:
        if record.binary_label == 1:
            counts[(record.split, record.generator_label, record.generator_prediction)] += 1
    return [
        {
            "split": split,
            "generator_label": label,
            "generator_prediction": prediction,
            "count": count,
        }
        for (split, label, prediction), count in sorted(counts.items())
    ]


def write_predictions(path: Path, records: Sequence[PredictionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(PredictionRecord.__annotations__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_dict_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
