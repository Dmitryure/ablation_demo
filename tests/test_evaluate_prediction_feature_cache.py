from pathlib import Path

import torch

from feature_cache import FeatureCacheSpec, feature_cache_spec_dir
from scripts.evaluate_prediction_feature_cache import (
    cache_key_from_path,
    collate_prediction_feature_batch,
    examples_from_cache_dir,
)


def spec(modality: str, frame_count: int) -> FeatureCacheSpec:
    return FeatureCacheSpec(
        version=3,
        modality=modality,
        frame_count=frame_count,
        image_size=224,
        extractor_config={},
        cache_variant="facecrop_v2_opencv_haar_df16_box1p5_fallback_full_frame",
    )


def write_payload(
    cache_dir: Path, item_spec: FeatureCacheSpec, class_name: str, filename: str
) -> None:
    path = feature_cache_spec_dir(cache_dir, item_spec) / class_name / f"{filename}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_key = {
        "rgb": "rgb_features",
        "depth": "depth_features",
    }[item_spec.modality]
    torch.save(
        {
            "version": 3,
            "class_name": class_name,
            "filename": filename,
            "modality": item_spec.modality,
            "frame_count": item_spec.frame_count,
            "cache_variant": item_spec.cache_variant,
            "features": {feature_key: torch.zeros(1)},
        },
        path,
    )


def test_cache_key_from_nested_prediction_cache_path(tmp_path: Path) -> None:
    item_spec = spec("rgb", 16)
    spec_dir = feature_cache_spec_dir(tmp_path, item_spec)
    path = spec_dir / "real" / "Celeb-real" / "id0_0000.mp4.pt"

    assert cache_key_from_path(path, spec_dir) == ("real", "Celeb-real/id0_0000.mp4")


def test_examples_from_cache_dir_uses_only_complete_modalities(tmp_path: Path) -> None:
    specs = {
        "rgb": spec("rgb", 16),
        "depth": spec("depth", 32),
    }
    write_payload(tmp_path, specs["rgb"], "real", "Celeb-real/id0_0000.mp4")
    write_payload(tmp_path, specs["depth"], "real", "Celeb-real/id0_0000.mp4")
    write_payload(tmp_path, specs["rgb"], "real", "Celeb-real/missing_depth.mp4")
    write_payload(tmp_path, specs["rgb"], "fake", "Deepfakes/001_002.mp4")
    write_payload(tmp_path, specs["depth"], "fake", "Deepfakes/001_002.mp4")

    real_examples = examples_from_cache_dir(
        tmp_path,
        specs=specs,
        modalities=("rgb", "depth"),
        class_filter="real",
    )
    all_examples = examples_from_cache_dir(
        tmp_path,
        specs=specs,
        modalities=("rgb", "depth"),
        class_filter="all",
    )

    assert [example.metadata_filename for example in real_examples] == ["Celeb-real/id0_0000.mp4"]
    assert [(example.class_name, example.generator_id) for example in all_examples] == [
        ("fake", "Deepfakes"),
        ("real", "real"),
    ]


def test_collate_prediction_feature_batch_squeezes_single_video_cache_dim() -> None:
    items = [
        {
            "label": torch.tensor([0.0]),
            "path": "a.mp4",
            "rgb_features": torch.zeros(1, 8, 768),
            "depth_features": torch.zeros(1, 32, 384),
        },
        {
            "label": torch.tensor([0.0]),
            "path": "b.mp4",
            "rgb_features": torch.zeros(1, 8, 768),
            "depth_features": torch.zeros(1, 32, 384),
        },
    ]

    batch = collate_prediction_feature_batch(
        items,
        feature_keys={"rgb_features", "depth_features"},
    )

    assert batch["rgb_features"].shape == (2, 8, 768)
    assert batch["depth_features"].shape == (2, 32, 384)
