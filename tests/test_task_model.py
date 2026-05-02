from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from fusion import FusionOutput
from task_models import (
    AttentionMILBinaryHead,
    BinaryFusionClassifier,
    BinaryFusionHead,
    BinaryHeadResult,
    ModalityGatedMILBinaryHead,
    build_binary_head,
)


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


def build_token_fusion_output() -> FusionOutput:
    fused = torch.tensor([[0.2, -0.3, 0.4, 0.5], [-0.1, 0.7, -0.2, 0.3]])
    tokens = torch.tensor(
        [
            [
                [0.1, 0.2, 0.3, 0.4],
                [9.0, 9.0, 9.0, 9.0],
                [0.5, 0.4, 0.3, 0.2],
                [8.0, 8.0, 8.0, 8.0],
            ],
            [
                [0.4, 0.3, 0.2, 0.1],
                [7.0, 7.0, 7.0, 7.0],
                [0.2, 0.3, 0.4, 0.5],
                [6.0, 6.0, 6.0, 6.0],
            ],
        ]
    )
    token_mask = torch.tensor([True, False, True, False])
    modality_ids = torch.tensor([0, 0, 1, 1])
    return FusionOutput(
        fused=fused,
        tokens=tokens,
        token_mask=token_mask,
        time_ids=torch.arange(4),
        modality_ids=modality_ids,
        modality_names=("rgb", "fau"),
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

    def test_default_head_builder_preserves_linear_head(self):
        head = build_binary_head(dim=4)

        self.assertIsInstance(head, BinaryFusionHead)

    def test_all_configured_heads_return_one_logit_per_example(self):
        fusion = build_token_fusion_output()

        for head_type in ("cls_linear", "cls_mlp", "attention_mil", "modality_gated_mil"):
            with self.subTest(head_type=head_type):
                head = build_binary_head(
                    dim=4,
                    head_config={"type": head_type, "hidden_dim": 8, "dropout": 0.0},
                )
                output = head(fusion.fused) if isinstance(head, BinaryFusionHead) else head(fusion)
                logits = output.logits if isinstance(output, BinaryHeadResult) else output

                self.assertEqual(tuple(logits.shape), (2, 1))

    def test_attention_mil_ignores_masked_tokens(self):
        torch.manual_seed(0)
        head = AttentionMILBinaryHead(dim=4, hidden_dim=8, dropout=0.0)
        fusion = build_token_fusion_output()
        perturbed_tokens = fusion.tokens.clone()
        perturbed_tokens[:, 1, :] = 1000.0
        perturbed_tokens[:, 3, :] = -1000.0
        perturbed = FusionOutput(
            fused=fusion.fused,
            tokens=perturbed_tokens,
            token_mask=fusion.token_mask,
            time_ids=fusion.time_ids,
            modality_ids=fusion.modality_ids,
            modality_names=fusion.modality_names,
            cls_token=fusion.cls_token,
            fused_tokens=torch.cat([fusion.fused.unsqueeze(1), perturbed_tokens], dim=1),
        )

        original = head(fusion)
        changed = head(perturbed)

        self.assertTrue(torch.allclose(original.logits, changed.logits))

    def test_modality_gated_mil_ignores_disabled_modality(self):
        torch.manual_seed(0)
        head = ModalityGatedMILBinaryHead(dim=4, hidden_dim=8, dropout=0.0)
        fusion = build_token_fusion_output()
        disabled_mask = torch.tensor([True, False, False, False])
        disabled_fusion = FusionOutput(
            fused=fusion.fused,
            tokens=fusion.tokens,
            token_mask=disabled_mask,
            time_ids=fusion.time_ids,
            modality_ids=fusion.modality_ids,
            modality_names=fusion.modality_names,
            cls_token=fusion.cls_token,
            fused_tokens=fusion.fused_tokens,
        )

        output = head(disabled_fusion)

        self.assertEqual(tuple(output.logits.shape), (2, 1))
        self.assertTrue(
            torch.all(output.diagnostics["modality_valid_mask"] == torch.tensor([True, False]))
        )
        self.assertTrue(
            torch.allclose(output.diagnostics["modality_gate_weights"][:, 1], torch.zeros(2))
        )

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

    def test_binary_classifier_returns_diagnostics_for_new_heads(self):
        classifier = BinaryFusionClassifier(
            pipeline=DummyFusionPipeline(),
            head=build_binary_head(
                dim=2,
                head_config={"type": "modality_gated_mil", "hidden_dim": 4, "dropout": 0.0},
            ),
        )

        output = classifier({"fused": torch.tensor([[0.25, -0.25], [0.5, 0.75]])})

        self.assertIn("modality_gate_weights", output.diagnostics)
        self.assertIn("token_attention_weights", output.diagnostics)


if __name__ == "__main__":
    unittest.main()
