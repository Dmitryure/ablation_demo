from __future__ import annotations

from types import SimpleNamespace

import torch
import yaml

from scripts.run_generator_multitask_training import (
    PseudoUnknownConfig,
    nonnegative_float_value,
    pseudo_unknown_group_for_epoch,
    resolve_checkpoint_score_type,
    resolve_pseudo_unknown_config,
    scheduled_generator_weight,
    train_epoch,
)
from training_losses import (
    binary_probability_margin_loss,
    generator_max_probability_loss,
    multitask_loss,
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
        return SimpleNamespace(
            binary_logits=binary_logits,
            generator_logits=generator_logits,
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
        binary_margin_weight=0.1,
        real_probability_margin=0.2,
        fake_probability_margin=0.8,
        real_generator_suppression_weight=0.05,
        pseudo_unknown_group="liveavatar",
        pseudo_unknown_suppression_weight=0.05,
    )

    assert result["binary_loss"] > 0.0
    assert result["binary_margin_loss"] >= 0.0
    assert result["generator_loss"] > 0.0
    assert result["real_generator_suppression_loss"] > 0.0
    assert result["pseudo_unknown_suppression_loss"] > 0.0
