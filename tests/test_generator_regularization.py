from __future__ import annotations

import random
from types import SimpleNamespace

import torch
import yaml

from scripts.run_generator_multitask_training import (
    ModalityDropoutConfig,
    PseudoUnknownConfig,
    apply_modality_dropout,
    nonnegative_float_value,
    pseudo_unknown_group_for_epoch,
    resolve_checkpoint_score_type,
    resolve_modality_dropout_config,
    resolve_pseudo_unknown_config,
    scheduled_generator_weight,
    train_epoch,
)
from training_losses import (
    auxiliary_modality_binary_loss,
    binary_probability_margin_loss,
    generator_max_probability_loss,
    multitask_loss,
    supervised_binary_contrastive_loss,
)
from training_targets import GeneratorTargetSpec


def test_generator_max_probability_loss_handles_empty_logits() -> None:
    logits = torch.empty(0, 3)

    loss = generator_max_probability_loss(logits)

    assert loss.item() == 0.0


def test_generator_max_probability_loss_penalizes_confident_logits_more() -> None:
    uniform_logits = torch.zeros(2, 3)
    confident_logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])

    assert generator_max_probability_loss(confident_logits) > generator_max_probability_loss(
        uniform_logits
    )


def test_multitask_loss_allows_empty_generator_ce_with_suppression() -> None:
    binary_logits = torch.zeros(3, 1)
    binary_labels = torch.tensor([[0.0], [1.0], [1.0]])
    generator_logits = torch.empty(0, 2)
    generator_labels = torch.empty(0, dtype=torch.long)
    real_generator_logits = torch.tensor([[4.0, 0.0]])
    pseudo_unknown_logits = torch.tensor([[0.0, 4.0], [3.0, 0.0]])

    loss, parts = multitask_loss(
        binary_logits=binary_logits,
        binary_labels=binary_labels,
        generator_logits=generator_logits,
        generator_labels=generator_labels,
        binary_weight=1.0,
        generator_weight=0.15,
        generator_loss=torch.nn.CrossEntropyLoss(),
        real_generator_logits=real_generator_logits,
        real_generator_suppression_weight=0.05,
        pseudo_unknown_generator_logits=pseudo_unknown_logits,
        pseudo_unknown_suppression_weight=0.05,
    )

    assert torch.isfinite(loss)
    assert parts["generator_loss"] == 0.0
    assert parts["real_generator_suppression_loss"] > 0.0
    assert parts["pseudo_unknown_suppression_loss"] > 0.0


def test_binary_probability_margin_loss_penalizes_uncertain_real_and_fake() -> None:
    good_logits = torch.tensor([[-4.0], [4.0]])
    weak_logits = torch.tensor([[0.0], [0.0]])
    labels = torch.tensor([[0.0], [1.0]])

    good = binary_probability_margin_loss(
        good_logits,
        labels,
        real_margin=0.2,
        fake_margin=0.8,
    )
    weak = binary_probability_margin_loss(
        weak_logits,
        labels,
        real_margin=0.2,
        fake_margin=0.8,
    )

    assert weak > good


def test_auxiliary_modality_binary_loss_uses_only_valid_modalities() -> None:
    logits = torch.tensor([[-3.0, 3.0], [3.0, -3.0]])
    labels = torch.tensor([[0.0], [1.0]])
    mask = torch.tensor([True, False])

    loss = auxiliary_modality_binary_loss(logits, labels, mask)

    assert loss < 0.1


def test_multitask_loss_includes_auxiliary_binary_loss() -> None:
    binary_logits = torch.zeros(2, 1)
    binary_labels = torch.tensor([[0.0], [1.0]])
    generator_logits = torch.zeros(1, 2)
    generator_labels = torch.tensor([0])
    auxiliary_logits = torch.zeros(2, 2)

    loss_without_aux, parts_without_aux = multitask_loss(
        binary_logits=binary_logits,
        binary_labels=binary_labels,
        generator_logits=generator_logits,
        generator_labels=generator_labels,
        binary_weight=1.0,
        generator_weight=0.0,
        generator_loss=torch.nn.CrossEntropyLoss(),
    )
    loss_with_aux, parts_with_aux = multitask_loss(
        binary_logits=binary_logits,
        binary_labels=binary_labels,
        generator_logits=generator_logits,
        generator_labels=generator_labels,
        binary_weight=1.0,
        generator_weight=0.0,
        generator_loss=torch.nn.CrossEntropyLoss(),
        auxiliary_binary_logits=auxiliary_logits,
        modality_valid_mask=torch.tensor([True, True]),
        auxiliary_binary_weight=0.2,
    )

    assert parts_without_aux["auxiliary_binary_loss"] == 0.0
    assert parts_with_aux["auxiliary_binary_loss"] > 0.0
    assert loss_with_aux > loss_without_aux


def test_supervised_binary_contrastive_loss_prefers_same_class_clusters() -> None:
    labels = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    clustered = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ]
    )
    mixed = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.9, 0.1],
            [-0.9, -0.1],
        ]
    )

    assert supervised_binary_contrastive_loss(clustered, labels, 0.2) < (
        supervised_binary_contrastive_loss(mixed, labels, 0.2)
    )


def test_multitask_loss_includes_contrastive_loss() -> None:
    binary_logits = torch.zeros(4, 1)
    binary_labels = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    generator_logits = torch.zeros(2, 2)
    generator_labels = torch.tensor([0, 1])
    embeddings = torch.randn(4, 3)

    loss_without_contrastive, parts_without_contrastive = multitask_loss(
        binary_logits=binary_logits,
        binary_labels=binary_labels,
        generator_logits=generator_logits,
        generator_labels=generator_labels,
        binary_weight=1.0,
        generator_weight=0.0,
        generator_loss=torch.nn.CrossEntropyLoss(),
    )
    loss_with_contrastive, parts_with_contrastive = multitask_loss(
        binary_logits=binary_logits,
        binary_labels=binary_labels,
        generator_logits=generator_logits,
        generator_labels=generator_labels,
        binary_weight=1.0,
        generator_weight=0.0,
        generator_loss=torch.nn.CrossEntropyLoss(),
        contrastive_embeddings=embeddings,
        contrastive_weight=0.1,
        contrastive_temperature=0.2,
    )

    assert parts_without_contrastive["contrastive_loss"] == 0.0
    assert parts_with_contrastive["contrastive_loss"] > 0.0
    assert loss_with_contrastive > loss_without_contrastive


def test_apply_modality_dropout_never_drops_all_modalities() -> None:
    config = ModalityDropoutConfig(
        enabled=True,
        probability=1.0,
        max_drop=3,
        modalities=("rgb", "depth", "eye_gaze"),
    )

    batch, drop_count = apply_modality_dropout(
        {"label": torch.zeros(2, 1)},
        enabled_modalities=("rgb", "depth", "eye_gaze"),
        config=config,
        rng=random.Random(0),
    )

    assert drop_count == 2
    assert len(batch["dropped_modalities"]) == 2


def test_resolve_modality_dropout_config_validates_modalities() -> None:
    config = resolve_modality_dropout_config(
        {
            "modality_dropout": {
                "enabled": True,
                "probability": 0.2,
                "max_drop": 1,
                "modalities": ["rgb", "depth"],
            }
        },
        enabled_modalities=("rgb", "depth", "eye_gaze"),
    )

    assert config == ModalityDropoutConfig(
        enabled=True,
        probability=0.2,
        max_drop=1,
        modalities=("rgb", "depth"),
    )


def test_scheduled_generator_weight_warmup_and_ramp() -> None:
    assert scheduled_generator_weight(1, 0.15, warmup_epochs=2, ramp_epochs=3) == 0.0
    assert scheduled_generator_weight(2, 0.15, warmup_epochs=2, ramp_epochs=3) == 0.0
    assert abs(scheduled_generator_weight(3, 0.15, warmup_epochs=2, ramp_epochs=3) - 0.05) < 1e-9
    assert abs(scheduled_generator_weight(4, 0.15, warmup_epochs=2, ramp_epochs=3) - 0.10) < 1e-9
    assert abs(scheduled_generator_weight(5, 0.15, warmup_epochs=2, ramp_epochs=3) - 0.15) < 1e-9


def test_pseudo_unknown_group_rotates_and_excludes_unknown_group() -> None:
    target = GeneratorTargetSpec(
        generator_names=("dlc", "liveavatar", "unknown_or_other"),
    )
    config = PseudoUnknownConfig(enabled=True, exclude_groups=("unknown_or_other",))

    assert pseudo_unknown_group_for_epoch(1, target, config) == "dlc"
    assert pseudo_unknown_group_for_epoch(2, target, config) == "liveavatar"
    assert pseudo_unknown_group_for_epoch(3, target, config) == "dlc"


def test_pseudo_unknown_group_handles_single_eligible_group() -> None:
    target = GeneratorTargetSpec(
        generator_names=("dlc", "unknown_or_other"),
    )
    config = PseudoUnknownConfig(enabled=True, exclude_groups=("unknown_or_other",))

    assert pseudo_unknown_group_for_epoch(1, target, config) == "dlc"
    assert pseudo_unknown_group_for_epoch(5, target, config) == "dlc"


def test_best_run_config_leaves_new_regularizers_disabled_by_default() -> None:
    with open(
        "runs/configs/night_sweep/09_seed0_lr3e4_gen0p15_warm2_e26.yaml",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)
    run = config["training"]["run"]

    pseudo_unknown = resolve_pseudo_unknown_config(run)

    assert pseudo_unknown.enabled is False
    assert nonnegative_float_value(run, "real_generator_suppression_weight", 0.0) == 0.0


def test_new_regularizer_config_fields_parse() -> None:
    run = {
        "pseudo_unknown": {
            "enabled": True,
            "mode": "epoch_rotate",
            "exclude_groups": ["unknown_or_other", "ovi"],
            "suppression_weight": 0.07,
        },
        "real_generator_suppression_weight": 0.03,
    }

    pseudo_unknown = resolve_pseudo_unknown_config(run)

    assert pseudo_unknown == PseudoUnknownConfig(
        enabled=True,
        mode="epoch_rotate",
        exclude_groups=("unknown_or_other", "ovi"),
        suppression_weight=0.07,
    )
    assert nonnegative_float_value(run, "real_generator_suppression_weight", 0.0) == 0.03


def test_checkpoint_score_type_defaults_and_parses() -> None:
    assert resolve_checkpoint_score_type({}) == "generator_aware"
    assert resolve_checkpoint_score_type({"checkpoint_score": {"type": "binary_robust"}}) == (
        "binary_robust"
    )


class DummyGeneratorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, batch: dict[str, object]) -> SimpleNamespace:
        del batch
        binary_logits = torch.tensor([[-1.0], [1.0], [1.0]], requires_grad=True) + self.bias
        generator_logits = (
            torch.tensor(
                [
                    [4.0, 0.0],
                    [4.0, 0.0],
                    [0.0, 4.0],
                ],
                requires_grad=True,
            )
            + self.bias
        )
        diagnostics = {
            "binary_modality_expert_logits": torch.tensor(
                [[-1.0, -0.5], [1.0, 0.5], [0.5, 1.0]],
                requires_grad=True,
            )
            + self.bias,
            "modality_valid_mask": torch.tensor([True, True]),
        }
        fusion = SimpleNamespace(cls_token=torch.randn(3, 4) + self.bias)
        return SimpleNamespace(
            binary_logits=binary_logits,
            generator_logits=generator_logits,
            diagnostics=diagnostics,
            fusion=fusion,
        )


def test_train_epoch_masks_pseudo_unknown_fake_and_suppresses_real() -> None:
    model = DummyGeneratorModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = {
        "label": torch.tensor([[0.0], [1.0], [1.0]]),
        "class_name": ["real", "fake", "fake"],
        "generator_id": ["real", "dlc", "liveavatar"],
    }
    target = GeneratorTargetSpec(generator_names=("dlc", "liveavatar"))

    result = train_epoch(
        model=model,  # type: ignore[arg-type]
        loader=[batch],
        optimizer=optimizer,
        generator_loss=torch.nn.CrossEntropyLoss(),
        target=target,
        binary_weight=1.0,
        generator_weight=0.15,
        auxiliary_binary_weight=0.2,
        binary_margin_weight=0.1,
        real_probability_margin=0.2,
        fake_probability_margin=0.8,
        real_generator_suppression_weight=0.05,
        pseudo_unknown_group="liveavatar",
        pseudo_unknown_suppression_weight=0.05,
        contrastive_weight=0.1,
        contrastive_temperature=0.2,
    )

    assert result["binary_loss"] > 0.0
    assert result["binary_margin_loss"] >= 0.0
    assert result["auxiliary_binary_loss"] > 0.0
    assert result["contrastive_loss"] > 0.0
    assert result["generator_loss"] > 0.0
    assert result["real_generator_suppression_loss"] > 0.0
    assert result["pseudo_unknown_suppression_loss"] > 0.0
