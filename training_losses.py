from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from training_targets import GeneratorTargetSpec


def effective_number_weights(
    counts: Mapping[str, int],
    label_names: Sequence[str],
    beta: float,
    device: torch.device,
) -> torch.Tensor:
    if beta < 0.0 or beta >= 1.0:
        raise ValueError("`beta` must be in [0.0, 1.0).")
    weights: list[float] = []
    for label in label_names:
        count = max(1, int(counts.get(label, 0)))
        weight = (1.0 - beta) / (1.0 - beta**count) if beta > 0.0 else 1.0
        weights.append(weight)
    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return tensor / tensor.mean().clamp_min(torch.finfo(tensor.dtype).tiny)


class ClassBalancedFocalLoss(torch.nn.Module):
    def __init__(
        self,
        counts: Mapping[str, int],
        target: GeneratorTargetSpec,
        beta: float = 0.999,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if gamma < 0.0:
            raise ValueError("`gamma` must be non-negative.")
        self.counts = dict(counts)
        self.target = target
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.numel() == 0:
            return logits.sum() * 0.0
        weights = effective_number_weights(
            self.counts,
            self.target.generator_names,
            self.beta,
            logits.device,
        )
        ce = F.cross_entropy(logits, labels, weight=weights, reduction="none")
        probabilities = torch.softmax(logits, dim=-1)
        true_probabilities = probabilities.gather(1, labels.view(-1, 1)).squeeze(1)
        focal = (1.0 - true_probabilities).clamp_min(0.0).pow(self.gamma)
        return (focal * ce).mean()


def generator_loss_fn(
    loss_type: str,
    counts: Mapping[str, int],
    target: GeneratorTargetSpec,
    beta: float,
    gamma: float,
) -> torch.nn.Module:
    if loss_type == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    if loss_type == "class_balanced_cross_entropy":

        class WeightedCrossEntropy(torch.nn.Module):
            def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                weights = effective_number_weights(
                    counts,
                    target.generator_names,
                    beta,
                    logits.device,
                )
                return F.cross_entropy(logits, labels, weight=weights)

        return WeightedCrossEntropy()
    if loss_type == "class_balanced_focal":
        return ClassBalancedFocalLoss(counts=counts, target=target, beta=beta, gamma=gamma)
    raise ValueError(f"Unsupported generator loss type: {loss_type}")


def generator_max_probability_loss(logits: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    probabilities = torch.softmax(logits, dim=-1)
    max_probabilities = probabilities.max(dim=-1).values
    return max_probabilities.square().mean()


def safe_generator_loss(
    generator_loss: torch.nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if labels.numel() == 0:
        return logits.sum() * 0.0
    return generator_loss(logits, labels)


def binary_probability_margin_loss(
    binary_logits: torch.Tensor,
    binary_labels: torch.Tensor,
    real_margin: float,
    fake_margin: float,
) -> torch.Tensor:
    if real_margin < 0.0 or real_margin > 1.0:
        raise ValueError("`real_margin` must be in [0.0, 1.0].")
    if fake_margin < 0.0 or fake_margin > 1.0:
        raise ValueError("`fake_margin` must be in [0.0, 1.0].")
    if real_margin >= fake_margin:
        raise ValueError("`real_margin` must be smaller than `fake_margin`.")
    probabilities = torch.sigmoid(binary_logits)
    real_mask = binary_labels <= 0.5
    fake_mask = binary_labels > 0.5
    zero = binary_logits.sum() * 0.0
    real_loss = (
        (probabilities[real_mask] - real_margin).clamp_min(0.0).square().mean()
        if torch.any(real_mask)
        else zero
    )
    fake_loss = (
        (fake_margin - probabilities[fake_mask]).clamp_min(0.0).square().mean()
        if torch.any(fake_mask)
        else zero
    )
    return real_loss + fake_loss


def auxiliary_modality_binary_loss(
    auxiliary_logits: torch.Tensor | None,
    binary_labels: torch.Tensor,
    modality_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if auxiliary_logits is None:
        return binary_labels.sum() * 0.0
    if auxiliary_logits.ndim != 2:
        raise ValueError(
            "`auxiliary_logits` must have shape [batch, modalities], "
            f"got {tuple(auxiliary_logits.shape)}."
        )
    labels = binary_labels.view(-1, 1).to(
        device=auxiliary_logits.device, dtype=auxiliary_logits.dtype
    )
    if labels.shape[0] != auxiliary_logits.shape[0]:
        raise ValueError("`binary_labels` batch size must match `auxiliary_logits`.")
    losses = F.binary_cross_entropy_with_logits(
        auxiliary_logits,
        labels.expand_as(auxiliary_logits),
        reduction="none",
    )
    if modality_valid_mask is None:
        return losses.mean()
    if modality_valid_mask.ndim != 1 or modality_valid_mask.shape[0] != auxiliary_logits.shape[1]:
        raise ValueError("`modality_valid_mask` must have shape [modalities].")
    valid = modality_valid_mask.to(device=auxiliary_logits.device, dtype=torch.bool).view(1, -1)
    if not torch.any(valid):
        return auxiliary_logits.sum() * 0.0
    return losses.masked_select(valid.expand_as(losses)).mean()


def supervised_binary_contrastive_loss(
    embeddings: torch.Tensor | None,
    binary_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if embeddings is None:
        return binary_labels.sum() * 0.0
    if temperature <= 0.0:
        raise ValueError("`temperature` must be positive.")
    if embeddings.ndim != 2:
        raise ValueError(
            f"`embeddings` must have shape [batch, dim], got {tuple(embeddings.shape)}"
        )

    labels = binary_labels.view(-1).to(device=embeddings.device)
    if labels.shape[0] != embeddings.shape[0]:
        raise ValueError("`binary_labels` batch size must match `embeddings`.")
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0

    normalized = F.normalize(embeddings, dim=1)
    logits = torch.matmul(normalized, normalized.transpose(0, 1)) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
    valid_anchor_mask = positive_mask.any(dim=1)
    if not torch.any(valid_anchor_mask):
        return embeddings.sum() * 0.0

    logits = logits.masked_fill(self_mask, torch.finfo(logits.dtype).min)
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_counts = positive_mask.sum(dim=1).clamp_min(1)
    mean_positive_log_probability = (
        log_probabilities.masked_fill(~positive_mask, 0.0).sum(dim=1) / positive_counts
    )
    return -mean_positive_log_probability[valid_anchor_mask].mean()


def multitask_loss(
    binary_logits: torch.Tensor,
    binary_labels: torch.Tensor,
    generator_logits: torch.Tensor,
    generator_labels: torch.Tensor,
    binary_weight: float,
    generator_weight: float,
    generator_loss: torch.nn.Module,
    real_generator_logits: torch.Tensor | None = None,
    real_generator_suppression_weight: float = 0.0,
    pseudo_unknown_generator_logits: torch.Tensor | None = None,
    pseudo_unknown_suppression_weight: float = 0.0,
    binary_margin_weight: float = 0.0,
    real_probability_margin: float = 0.20,
    fake_probability_margin: float = 0.80,
    auxiliary_binary_logits: torch.Tensor | None = None,
    modality_valid_mask: torch.Tensor | None = None,
    auxiliary_binary_weight: float = 0.0,
    contrastive_embeddings: torch.Tensor | None = None,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.20,
) -> tuple[torch.Tensor, dict[str, float]]:
    binary = F.binary_cross_entropy_with_logits(binary_logits, binary_labels)
    binary_margin = (
        binary_probability_margin_loss(
            binary_logits=binary_logits,
            binary_labels=binary_labels,
            real_margin=real_probability_margin,
            fake_margin=fake_probability_margin,
        )
        if binary_margin_weight > 0.0
        else binary_logits.sum() * 0.0
    )
    generator = safe_generator_loss(generator_loss, generator_logits, generator_labels)
    real_generator_suppression = (
        generator_max_probability_loss(real_generator_logits)
        if real_generator_logits is not None and real_generator_suppression_weight > 0.0
        else binary_logits.sum() * 0.0
    )
    pseudo_unknown_suppression = (
        generator_max_probability_loss(pseudo_unknown_generator_logits)
        if pseudo_unknown_generator_logits is not None and pseudo_unknown_suppression_weight > 0.0
        else binary_logits.sum() * 0.0
    )
    auxiliary_binary = (
        auxiliary_modality_binary_loss(
            auxiliary_logits=auxiliary_binary_logits,
            binary_labels=binary_labels,
            modality_valid_mask=modality_valid_mask,
        )
        if auxiliary_binary_weight > 0.0
        else binary_logits.sum() * 0.0
    )
    contrastive = (
        supervised_binary_contrastive_loss(
            embeddings=contrastive_embeddings,
            binary_labels=binary_labels,
            temperature=contrastive_temperature,
        )
        if contrastive_weight > 0.0
        else binary_logits.sum() * 0.0
    )
    total = (
        binary_weight * binary
        + binary_margin_weight * binary_margin
        + generator_weight * generator
        + real_generator_suppression_weight * real_generator_suppression
        + pseudo_unknown_suppression_weight * pseudo_unknown_suppression
        + auxiliary_binary_weight * auxiliary_binary
        + contrastive_weight * contrastive
    )
    return total, {
        "binary_loss": float(binary.detach().item()),
        "binary_margin_loss": float(binary_margin.detach().item()),
        "auxiliary_binary_loss": float(auxiliary_binary.detach().item()),
        "contrastive_loss": float(contrastive.detach().item()),
        "generator_loss": float(generator.detach().item()),
        "real_generator_suppression_loss": float(real_generator_suppression.detach().item()),
        "pseudo_unknown_suppression_loss": float(pseudo_unknown_suppression.detach().item()),
        "loss": float(total.detach().item()),
    }
