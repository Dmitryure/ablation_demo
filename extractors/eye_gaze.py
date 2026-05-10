from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from extractors.base import FeatureExtractor
from extractors.mediapipe_face_landmarker import (
    create_face_landmarker,
    optional_model_path,
    resolve_face_landmarker_model_path,
)

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
EYE_GAZE_RICH_FEATURE_VARIANT = "rich_v1"
EYE_GAZE_LEGACY_FEATURE_VARIANT = "legacy"
EYE_GAZE_FEATURE_VARIANTS = (
    EYE_GAZE_LEGACY_FEATURE_VARIANT,
    EYE_GAZE_RICH_FEATURE_VARIANT,
)
EYE_GAZE_RICH_BLENDSHAPE_COLUMNS: tuple[str, ...] = EYE_GAZE_COLUMNS + (
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
)
LEFT_EYE_GEOMETRY_INDICES: tuple[int, ...] = (33, 133, 159, 145, 160, 144)
RIGHT_EYE_GEOMETRY_INDICES: tuple[int, ...] = (362, 263, 386, 374, 385, 380)
EYE_GEOMETRY_COLUMNS: tuple[str, ...] = (
    "left_eye_center_x",
    "left_eye_center_y",
    "left_eye_width",
    "left_eye_height",
    "left_eye_aspect",
    "left_eye_upper_y",
    "left_eye_lower_y",
    "right_eye_center_x",
    "right_eye_center_y",
    "right_eye_width",
    "right_eye_height",
    "right_eye_aspect",
    "right_eye_upper_y",
    "right_eye_lower_y",
    "eye_center_dx",
    "eye_center_dy",
    "eye_width_diff",
    "eye_aspect_diff",
)
EYE_GAZE_RICH_COLUMNS: tuple[str, ...] = EYE_GAZE_RICH_BLENDSHAPE_COLUMNS + EYE_GEOMETRY_COLUMNS
EYE_GAZE_LEGACY_FEATURE_DIM = len(EYE_GAZE_COLUMNS)
EYE_GAZE_RICH_FEATURE_DIM = len(EYE_GAZE_RICH_COLUMNS)


def _validate_feature_variant(feature_variant: str | None) -> str:
    if feature_variant is None:
        return EYE_GAZE_LEGACY_FEATURE_VARIANT
    resolved = str(feature_variant)
    if resolved not in EYE_GAZE_FEATURE_VARIANTS:
        raise ValueError(
            f"Unsupported eye_gaze.feature_variant: {resolved!r}. "
            f"Expected one of {EYE_GAZE_FEATURE_VARIANTS}."
        )
    return resolved


def _blendshape_scores(result: Any, columns: Sequence[str]) -> dict[str, float] | None:
    blendshapes = getattr(result, "face_blendshapes", None)
    if blendshapes is None or len(blendshapes) != 1:
        return None
    features = dict.fromkeys(columns, 0.0)
    for blendshape in blendshapes[0]:
        category_name = getattr(blendshape, "category_name", "")
        if category_name in features:
            features[category_name] = float(getattr(blendshape, "score", 0.0))
    return features


def _legacy_features_from_result(result: Any) -> Mapping[str, float] | None:
    return _blendshape_scores(result, EYE_GAZE_COLUMNS)


def _landmark_xy(landmarks: Sequence[Any], index: int) -> np.ndarray:
    landmark = landmarks[index]
    return np.array([float(landmark.x), float(landmark.y)], dtype=np.float32)


def _eye_geometry(landmarks: Sequence[Any], indices: tuple[int, ...]) -> tuple[float, ...]:
    outer, inner, upper, lower, upper_outer, lower_outer = (
        _landmark_xy(landmarks, index) for index in indices
    )
    points = np.stack((outer, inner, upper, lower, upper_outer, lower_outer), axis=0)
    center = points.mean(axis=0)
    width = float(np.linalg.norm(outer - inner))
    height = float(np.linalg.norm(upper - lower))
    aspect = height / max(width, 1e-6)
    return (
        float(center[0]),
        float(center[1]),
        width,
        height,
        aspect,
        float(upper[1]),
        float(lower[1]),
    )


def _geometry_features_from_result(result: Any) -> tuple[float, ...] | None:
    face_landmarks = getattr(result, "face_landmarks", None)
    if face_landmarks is None or len(face_landmarks) != 1:
        return None
    landmarks = face_landmarks[0]
    required_index = max(max(LEFT_EYE_GEOMETRY_INDICES), max(RIGHT_EYE_GEOMETRY_INDICES))
    if len(landmarks) <= required_index:
        return None
    left = _eye_geometry(landmarks, LEFT_EYE_GEOMETRY_INDICES)
    right = _eye_geometry(landmarks, RIGHT_EYE_GEOMETRY_INDICES)
    return (
        *left,
        *right,
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
        left[4] - right[4],
    )


def _rich_features_from_result(result: Any) -> list[float]:
    blendshapes = _blendshape_scores(result, EYE_GAZE_RICH_BLENDSHAPE_COLUMNS)
    geometry = _geometry_features_from_result(result)
    if blendshapes is None or geometry is None:
        return [0.0] * EYE_GAZE_RICH_FEATURE_DIM
    return [float(blendshapes.get(name, 0.0)) for name in EYE_GAZE_RICH_BLENDSHAPE_COLUMNS] + [
        float(value) for value in geometry
    ]


class EyeGazeExtractor(FeatureExtractor):
    name = "eye_gaze"

    def __init__(
        self,
        model_path: str | Path | None = None,
        detect_features_fn: Callable[[np.ndarray], Mapping[str, float] | None] | None = None,
        detect_result_fn: Callable[[np.ndarray], Any] | None = None,
        feature_variant: str | None = None,
    ):
        self.feature_variant = _validate_feature_variant(feature_variant)
        self._landmarker = None
        self.model_path: Path | None = None
        if detect_features_fn is not None:
            if self.feature_variant != EYE_GAZE_LEGACY_FEATURE_VARIANT:
                raise ValueError("`detect_features_fn` only supports legacy eye-gaze features.")
            self._detect_features = detect_features_fn
            self._detect_result = None
            return

        if detect_result_fn is not None:
            self._detect_result = detect_result_fn
            self._detect_features = None
            return

        self.model_path = resolve_face_landmarker_model_path(
            Path(model_path) if isinstance(model_path, str) else model_path
        )
        self._mp, self._landmarker = create_face_landmarker(
            self.model_path,
            output_face_blendshapes=True,
        )

        def detect_result(frame_rgb: np.ndarray) -> Any:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(frame_rgb),
            )
            return self._landmarker.detect(mp_image)

        self._detect_result = detect_result
        self._detect_features = None

    def required_keys(self) -> tuple[str, ...]:
        return ("video_rgb_frames",)

    def extract_tensor(self, frames_rgb: Sequence[np.ndarray]) -> torch.Tensor:
        rows: list[list[float]] = []
        for frame_rgb in frames_rgb:
            if (
                not isinstance(frame_rgb, np.ndarray)
                or frame_rgb.ndim != 3
                or frame_rgb.shape[-1] != 3
            ):
                raise ValueError(
                    "Each eye-gaze frame must be an RGB numpy array with shape [H, W, 3], "
                    f"got {type(frame_rgb)}"
                )
            if self.feature_variant == EYE_GAZE_RICH_FEATURE_VARIANT:
                rows.append(_rich_features_from_result(self._detect_result(frame_rgb)))
                continue
            features = (
                self._detect_features(frame_rgb)
                if self._detect_features is not None
                else _legacy_features_from_result(self._detect_result(frame_rgb))
            )
            rows.append(
                [
                    0.0 if features is None else float(features.get(name, 0.0))
                    for name in EYE_GAZE_COLUMNS
                ]
            )
        return torch.tensor(rows, dtype=torch.float32)

    def extract(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        frames_rgb = batch["video_rgb_frames"]
        if not isinstance(frames_rgb, Sequence) or isinstance(frames_rgb, (str, bytes)):
            raise ValueError("`video_rgb_frames` must be a sequence of RGB frame arrays.")
        if (
            frames_rgb
            and isinstance(frames_rgb[0], Sequence)
            and not isinstance(frames_rgb[0], np.ndarray)
        ):
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
        model_path=optional_model_path(eye_gaze_config, "model_path"),
        feature_variant=eye_gaze_config.get("feature_variant"),
    )
