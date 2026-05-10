from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import torch

from dataset import VideoExample


@dataclass(frozen=True)
class GeneratorTargetSpec:
    generator_names: tuple[str, ...]
    raw_to_group: tuple[tuple[str, str], ...] = ()
    unknown_group_name: str = "unknown_or_other"

    @property
    def num_generators(self) -> int:
        return len(self.generator_names)

    @property
    def name_to_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.generator_names)}

    @property
    def raw_to_group_map(self) -> dict[str, str]:
        return dict(self.raw_to_group)

    def generator_name_for(self, raw_generator_name: str) -> str:
        return self.raw_to_group_map.get(raw_generator_name, raw_generator_name)

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["raw_to_group"] = dict(self.raw_to_group)
        return payload


def example_generator_name(example: VideoExample) -> str:
    if example.class_name != "fake":
        return "real"
    return str(example.generator_id or "unknown")


def filter_excluded_generators(
    examples: Sequence[VideoExample],
    excluded_generators: Sequence[str],
) -> list[VideoExample]:
    excluded = {str(generator) for generator in excluded_generators}
    return [
        example
        for example in examples
        if example.class_name != "fake" or example_generator_name(example) not in excluded
    ]


def grouped_generator_name(example: VideoExample, target: GeneratorTargetSpec) -> str:
    return target.generator_name_for(example_generator_name(example))


def normalize_generator_groups(
    generator_groups: Mapping[str, Sequence[str]] | None,
) -> tuple[tuple[str, str], ...]:
    if not generator_groups:
        return ()
    raw_to_group: dict[str, str] = {}
    for group_name, raw_names in sorted(generator_groups.items()):
        group = str(group_name)
        for raw_name in raw_names:
            raw = str(raw_name)
            if raw in raw_to_group:
                raise ValueError(
                    f"Raw generator {raw!r} is assigned to both "
                    f"{raw_to_group[raw]!r} and {group!r}."
                )
            raw_to_group[raw] = group
    return tuple(sorted(raw_to_group.items()))


def build_generator_target_spec(
    examples: Sequence[VideoExample],
    generator_groups: Mapping[str, Sequence[str]] | None = None,
    unknown_group_name: str = "unknown_or_other",
) -> GeneratorTargetSpec:
    raw_to_group = normalize_generator_groups(generator_groups)
    raw_mapping = dict(raw_to_group)
    generator_names = sorted(
        {
            raw_mapping.get(example_generator_name(example), example_generator_name(example))
            for example in examples
            if example.class_name == "fake"
        }
    )
    if not generator_names:
        raise ValueError("Generator target requires at least one fake generator.")
    return GeneratorTargetSpec(
        generator_names=tuple(generator_names),
        raw_to_group=raw_to_group,
        unknown_group_name=unknown_group_name,
    )


def binary_labels_for_batch(batch: Mapping[str, object], device: torch.device) -> torch.Tensor:
    labels = batch["label"]
    if not isinstance(labels, torch.Tensor):
        raise TypeError("Batch `label` must be a tensor.")
    return labels.to(device=device, dtype=torch.float32).view(-1, 1)


def generator_labels_for_batch(
    batch: Mapping[str, object],
    target: GeneratorTargetSpec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    class_names = [str(value) for value in batch["class_name"]]  # type: ignore[index]
    generator_ids = [str(value or "") for value in batch.get("generator_id", [])]  # type: ignore[arg-type]
    if len(generator_ids) != len(class_names):
        generator_ids = ["" for _ in class_names]

    mapping = target.name_to_index
    fake_indices: list[int] = []
    labels: list[int] = []
    for index, class_name in enumerate(class_names):
        if class_name != "fake":
            continue
        generator_name = target.generator_name_for(generator_ids[index] or "unknown")
        if generator_name not in mapping:
            raise ValueError(f"Unknown fake generator in batch: {generator_name}")
        fake_indices.append(index)
        labels.append(mapping[generator_name])

    return (
        torch.tensor(fake_indices, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def fake_generator_counts(
    examples: Sequence[VideoExample],
    target: GeneratorTargetSpec | None = None,
) -> dict[str, int]:
    counts: Counter[str] = Counter(
        (
            example_generator_name(example)
            if target is None
            else grouped_generator_name(example, target)
        )
        for example in examples
        if example.class_name == "fake"
    )
    return dict(sorted(counts.items()))


def real_fake_counts(examples: Sequence[VideoExample]) -> dict[str, int]:
    counts: Counter[str] = Counter(example.class_name for example in examples)
    return {"real": int(counts.get("real", 0)), "fake": int(counts.get("fake", 0))}
