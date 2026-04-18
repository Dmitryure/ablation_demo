from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from extractors.base import FeatureExtractor


EYE_GAZE_COLUMNS: tuple[str, ...] = (
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker_v2_with_blendshapes.task"
DEFAULT_MODEL_CANDIDATES: tuple[Path, ...] = (
    DEFAULT_MODEL_PATH,
    Path("/models/face_landmarker_v2_with_blendshapes.task"),
)


def _optional_path(config: Mapping[str, Any], key: str) -> Path | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return Path(stripped)
    raise ValueError(f"`{key}` must be a string path or null.")


def _import_mediapipe() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "Eye-gaze extraction requires `mediapipe` in this repo's active environment."
        ) from exc
    return mp


def _resolve_model_path(model_path: Path | None) -> Path:
    if model_path is not None:
        if not model_path.exists():
            raise FileNotFoundError(f"Eye-gaze model path does not exist: {model_path}")
        return model_path

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find `face_landmarker_v2_with_blendshapes.task`. "
        "Set `eye_gaze.model_path` in the YAML config."
    )


class EyeGazeExtractor(FeatureExtractor):
    name = "eye_gaze"

    def __init__(
        self,
        model_path: str | Path | None = None,
        detect_features_fn: Callable[[np.ndarray], Mapping[str, float] | None] | None = None,
    ):
        self._landmarker = None
        self.model_path: Path | None = None
        if detect_features_fn is not None:
            self._detect_features = detect_features_fn
            return

        self.model_path = _resolve_model_path(Path(model_path) if isinstance(model_path, str) else model_path)
        mp = _import_mediapipe()
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.7,
            num_faces=2,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

        def detect_features(frame_rgb: np.ndarray) -> Mapping[str, float] | None:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(frame_rgb),
            )
            result = self._landmarker.detect(mp_image)
            if len(result.face_blendshapes) != 1:
                return None

            features = {name: 0.0 for name in EYE_GAZE_COLUMNS}
            for blendshape in result.face_blendshapes[0]:
                if blendshape.category_name in features:
                    features[blendshape.category_name] = float(blendshape.score)
            return features

        self._detect_features = detect_features

    def required_keys(self) -> tuple[str, ...]:
        return ("video_rgb_frames",)

    def extract_tensor(self, frames_rgb: Sequence[np.ndarray]) -> torch.Tensor:
        rows: list[list[float]] = []
        for frame_rgb in frames_rgb:
            if not isinstance(frame_rgb, np.ndarray) or frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
                raise ValueError(
                    "Each eye-gaze frame must be an RGB numpy array with shape [H, W, 3], "
                    f"got {type(frame_rgb)}"
                )
            features = self._detect_features(frame_rgb)
            rows.append([0.0 if features is None else float(features.get(name, 0.0)) for name in EYE_GAZE_COLUMNS])
        return torch.tensor(rows, dtype=torch.float32)

    def extract(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        frames_rgb = batch["video_rgb_frames"]
        if not isinstance(frames_rgb, Sequence) or isinstance(frames_rgb, (str, bytes)):
            raise ValueError("`video_rgb_frames` must be a sequence of RGB frame arrays.")
        if frames_rgb and isinstance(frames_rgb[0], Sequence) and not isinstance(frames_rgb[0], np.ndarray):
            return {
                "eye_gaze": torch.stack(
                    [self.extract_tensor(clip_frames) for clip_frames in frames_rgb],
                    dim=0,
                ),
            }
        return {
            "eye_gaze": self.extract_tensor(frames_rgb).unsqueeze(0),
        }

    def close(self) -> None:
        close = getattr(self._landmarker, "close", None)
        if callable(close):
            close()


def build_eye_gaze_extractor(config: Mapping[str, Any]) -> EyeGazeExtractor:
    eye_gaze_config = config.get("eye_gaze")
    if eye_gaze_config is None:
        eye_gaze_config = {}
    if not isinstance(eye_gaze_config, Mapping):
        raise ValueError("`eye_gaze` must be a YAML mapping when provided.")

    return EyeGazeExtractor(
        model_path=_optional_path(eye_gaze_config, "model_path"),
    )
