from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from math import ceil

from torch.utils.data import Sampler

from dataset import VideoExample
from training_targets import GeneratorTargetSpec, example_generator_name, fake_generator_counts


def sample_with_replacement(
    indices: Sequence[int],
    count: int,
    rng: random.Random,
) -> list[int]:
    if count <= 0:
        return []
    if not indices:
        raise ValueError("Cannot sample from an empty group.")
    result: list[int] = []
    while len(result) < count:
        chunk = list(indices)
        rng.shuffle(chunk)
        result.extend(chunk[: count - len(result)])
    return result


def generator_epoch_quotas(
    counts: Mapping[str, int],
    target_quota: int,
    max_repeat: int,
) -> dict[str, int]:
    if target_quota <= 0:
        raise ValueError("`target_quota` must be positive.")
    if max_repeat <= 0:
        raise ValueError("`max_repeat` must be positive.")
    return {
        generator: min(target_quota, max(1, count) * max_repeat)
        for generator, count in sorted(counts.items())
    }


class MultitaskGeneratorBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        examples: Sequence[VideoExample],
        batch_size: int,
        real_per_batch: int,
        fake_per_batch: int,
        target_quota: int,
        max_repeat: int,
        seed: int,
        target: GeneratorTargetSpec | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")
        if real_per_batch < 0 or fake_per_batch <= 0:
            raise ValueError("`real_per_batch` must be non-negative and `fake_per_batch` positive.")
        if real_per_batch + fake_per_batch > batch_size:
            raise ValueError("real/fake slots cannot exceed `batch_size`.")
        self.examples = list(examples)
        self.batch_size = batch_size
        self.real_per_batch = real_per_batch
        self.fake_per_batch = fake_per_batch
        self.target_quota = target_quota
        self.max_repeat = max_repeat
        self.seed = seed
        self.target = target
        self.epoch = 0

        self.real_indices = [
            index for index, example in enumerate(self.examples) if example.class_name == "real"
        ]
        self.fake_indices_by_generator: dict[str, list[int]] = defaultdict(list)
        for index, example in enumerate(self.examples):
            if example.class_name == "fake":
                generator_name = example_generator_name(example)
                if target is not None:
                    generator_name = target.generator_name_for(generator_name)
                self.fake_indices_by_generator[generator_name].append(index)
        if not self.real_indices:
            raise ValueError("Sampler requires at least one real example.")
        if not self.fake_indices_by_generator:
            raise ValueError("Sampler requires at least one fake generator.")
        self.quotas = generator_epoch_quotas(
            fake_generator_counts(self.examples, target=target),
            target_quota=target_quota,
            max_repeat=max_repeat,
        )

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        fake_indices: list[int] = []
        for generator in sorted(self.fake_indices_by_generator):
            fake_indices.extend(
                sample_with_replacement(
                    self.fake_indices_by_generator[generator],
                    self.quotas[generator],
                    rng,
                )
            )
        rng.shuffle(fake_indices)
        batch_count = ceil(len(fake_indices) / self.fake_per_batch)
        padded_fake_count = batch_count * self.fake_per_batch
        if len(fake_indices) < padded_fake_count:
            all_fake_indices = [
                index
                for generator in sorted(self.fake_indices_by_generator)
                for index in self.fake_indices_by_generator[generator]
            ]
            fake_indices.extend(
                sample_with_replacement(
                    all_fake_indices,
                    padded_fake_count - len(fake_indices),
                    rng,
                )
            )
        real_indices = sample_with_replacement(
            self.real_indices,
            batch_count * self.real_per_batch,
            rng,
        )
        for batch_index in range(batch_count):
            fake_start = batch_index * self.fake_per_batch
            real_start = batch_index * self.real_per_batch
            batch = [
                *real_indices[real_start : real_start + self.real_per_batch],
                *fake_indices[fake_start : fake_start + self.fake_per_batch],
            ]
            rng.shuffle(batch)
            yield batch
        self.epoch += 1

    def __len__(self) -> int:
        return ceil(sum(self.quotas.values()) / self.fake_per_batch)

    def summary(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "real_per_batch": self.real_per_batch,
            "fake_per_batch": self.fake_per_batch,
            "target_quota": self.target_quota,
            "max_repeat": self.max_repeat,
            "epoch_batches": len(self),
            "epoch_fake_samples": len(self) * self.fake_per_batch,
            "epoch_real_samples": len(self) * self.real_per_batch,
            "base_fake_generator_samples": sum(self.quotas.values()),
            "padding_fake_samples": len(self) * self.fake_per_batch - sum(self.quotas.values()),
            "fake_generator_quotas": dict(self.quotas),
        }
