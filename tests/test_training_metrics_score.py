import pytest

from training_metrics import (
    BinaryMetrics,
    binary_robust_checkpoint_score,
    composite_checkpoint_score,
)


def test_composite_checkpoint_score_uses_robust_binary_first() -> None:
    binary = BinaryMetrics(
        accuracy=0.8,
        balanced_accuracy=0.75,
        precision=0.8,
        recall=0.6,
        f1=0.7,
        specificity=0.9,
        true_positive=6,
        true_negative=9,
        false_positive=1,
        false_negative=4,
    )

    score = composite_checkpoint_score(
        binary,
        macro_recall=0.8,
        known_precision_at_coverage=0.9,
    )

    assert score == pytest.approx(0.50 * 0.6 + 0.30 * 0.75 + 0.15 * 0.9 + 0.05 * 0.8)


def test_binary_robust_checkpoint_score_ignores_generator_metrics() -> None:
    binary = BinaryMetrics(
        accuracy=0.8,
        balanced_accuracy=0.75,
        precision=0.8,
        recall=0.6,
        f1=0.7,
        specificity=0.9,
        true_positive=6,
        true_negative=9,
        false_positive=1,
        false_negative=4,
    )

    assert binary_robust_checkpoint_score(binary) == pytest.approx(0.60 * 0.6 + 0.40 * 0.75)
