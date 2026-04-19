from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker_v2_with_blendshapes.task"
DEFAULT_MODEL_CANDIDATES: tuple[Path, ...] = (
    DEFAULT_MODEL_PATH,
    Path("/models/face_landmarker_v2_with_blendshapes.task"),
)


def optional_model_path(config: dict[str, Any] | Any, key: str) -> Path | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return Path(stripped)
    raise ValueError(f"`{key}` must be a string path or null.")


def import_mediapipe() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe feature extraction requires `mediapipe` in this repo's active environment."
        ) from exc
    return mp


def resolve_face_landmarker_model_path(model_path: Path | None) -> Path:
    if model_path is not None:
        if not model_path.exists():
            raise FileNotFoundError(f"Face landmarker model path does not exist: {model_path}")
        return model_path

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find `face_landmarker_v2_with_blendshapes.task`. "
        "Set the modality-specific `model_path` in the YAML config."
    )


def create_face_landmarker(
    model_path: Path,
    *,
    output_face_blendshapes: bool,
):
    mp = import_mediapipe()
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=output_face_blendshapes,
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.7,
        num_faces=2,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)
