from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from fusion import FusionOutput
from task_model import BinaryFusionClassifier, BinaryFusionHead


def build_fusion_output(fused: torch.Tensor) -> FusionOutput:
    batch, dim = fused.shape
    tokens = torch.zeros(batch, 1, dim)
    token_ids = torch.zeros(1, dtype=torch.long)
    return FusionOutput(
        fused=fused,
        tokens=tokens,
        token_mask=torch.ones(1, dtype=torch.bool),
        time_ids=token_ids,
        modality_ids=token_ids,
        modality_names=("dummy",),
        cls_token=fused,
        fused_tokens=torch.cat([fused.unsqueeze(1), tokens], dim=1),
    )


class DummyFusionPipeline(nn.Module):
    def forward(self, batch):
        return build_fusion_output(batch["fused"])


class TaskModelTest(unittest.TestCase):
    def test_binary_fusion_head_returns_one_logit_per_example(self):
        head = BinaryFusionHead(dim=4)

        logits = head(torch.randn(3, 4))

        self.assertEqual(tuple(logits.shape), (3, 1))

    def test_binary_fusion_head_rejects_non_batched_fused_tensor(self):
        head = BinaryFusionHead(dim=4)

        with self.assertRaisesRegex(ValueError, "shape"):
            head(torch.randn(4))

    def test_binary_classifier_returns_probabilities_labels_and_paths(self):
        classifier = BinaryFusionClassifier(
            pipeline=DummyFusionPipeline(),
            head=BinaryFusionHead(dim=2),
        )
        batch = {
            "fused": torch.tensor([[0.25, -0.25], [0.5, 0.75]]),
            "label": torch.tensor([[0.0], [1.0]]),
            "path": ["a.mp4", "b.mp4"],
        }

        output = classifier(batch)

        self.assertEqual(tuple(output.logits.shape), (2, 1))
        self.assertTrue(torch.all(output.probabilities >= 0.0))
        self.assertTrue(torch.all(output.probabilities <= 1.0))
        self.assertIs(output.labels, batch["label"])
        self.assertEqual(output.paths, ("a.mp4", "b.mp4"))


if __name__ == "__main__":
    unittest.main()
