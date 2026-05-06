from __future__ import annotations

import unittest

import torch

from scripts.probe_cached_features import evaluate_predictions


class ProbeCachedFeaturesMetricsTest(unittest.TestCase):
    def test_evaluate_predictions_reports_confusion_and_binary_scores(self):
        labels = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
        probabilities = torch.tensor([0.2, 0.8, 0.7, 0.4])

        metrics = evaluate_predictions(labels, probabilities)

        self.assertEqual(metrics.true_negative, 1)
        self.assertEqual(metrics.false_positive, 1)
        self.assertEqual(metrics.true_positive, 1)
        self.assertEqual(metrics.false_negative, 1)
        self.assertEqual(metrics.accuracy, 0.5)
        self.assertEqual(metrics.balanced_accuracy, 0.5)
        self.assertEqual(metrics.precision, 0.5)
        self.assertEqual(metrics.recall, 0.5)
        self.assertEqual(metrics.f1, 0.5)
        self.assertEqual(metrics.specificity, 0.5)
        self.assertEqual(metrics.negative_predictive_value, 0.5)
        self.assertEqual(metrics.false_positive_rate, 0.5)
        self.assertEqual(metrics.false_negative_rate, 0.5)
        self.assertEqual(metrics.matthews_corrcoef, 0.0)


if __name__ == "__main__":
    unittest.main()
