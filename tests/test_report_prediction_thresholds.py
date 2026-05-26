import pytest

from scripts.report_prediction_thresholds import metrics_at_threshold


def test_metrics_at_threshold() -> None:
    rows = [
        (0, 0.1),
        (0, 0.8),
        (1, 0.4),
        (1, 0.9),
    ]

    metrics = metrics_at_threshold(rows, threshold=0.5)

    assert metrics.true_negative == 1
    assert metrics.false_positive == 1
    assert metrics.true_positive == 1
    assert metrics.false_negative == 1
    assert metrics.specificity == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.balanced_accuracy == pytest.approx(0.5)
