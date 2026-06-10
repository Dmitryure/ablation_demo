from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from extractors.base import FeatureExtractor
from extractors.eye_gaze import (
    EYE_GAZE_COLUMNS,
    EYE_GAZE_RICH_FEATURE_VARIANT,
    _legacy_features_from_result,
    _rich_features_from_result,
    _validate_feature_variant,
)
from extractors.face_mesh import FACE_MESH_CONTOUR_INDICES
from extractors.mediapipe_face_landmarker import (
    create_face_landmarker,
    optional_model_path,
    resolve_face_landmarker_model_path,
)


def _validate_frame(frame_rgb: np.ndarray) -> None:
    if not isinstance(frame_rgb, np.ndarray) or frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
        raise ValueError(
            "Each face-landmarker frame must be an RGB numpy array with shape [H, W, 3], "
            f"got {type(frame_rgb)}"
        )


def _mesh_from_result(result: Any) -> np.ndarray:
    face_landmarks = getattr(result, "face_landmarks", None)
    if face_landmarks is None or len(face_landmarks) != 1:
        return np.full((len(FACE_MESH_CONTOUR_INDICES), 3), -1.0, dtype=np.float32)

    landmarks = face_landmarks[0]
    points = np.array(
        [
            [float(landmarks[index].x), float(landmarks[index].y), float(landmarks[index].z)]
            for index in FACE_MESH_CONTOUR_INDICES
        ],
        dtype=np.float32,
    )
    points[:, :2] = np.clip(points[:, :2], 0.0, 1.0)
    return points


def _eye_gaze_from_result(result: Any, feature_variant: str) -> list[float]:
    if feature_variant == EYE_GAZE_RICH_FEATURE_VARIANT:
        return _rich_features_from_result(result)
    features = _legacy_features_from_result(result)
    return [
        0.0 if features is None else float(features.get(name, 0.0)) for name in EYE_GAZE_COLUMNS
    ]


class FaceLandmarkerCombinedExtractor(FeatureExtractor):
    name = "face_landmarker"

    def __init__(
        self,
        model_path: str | Path | None = None,
        feature_variant: str | None = None,
        detect_result_fn: Callable[[np.ndarray], Any] | None = None,
    ) -> None:
        self.feature_variant = _validate_feature_variant(feature_variant)
        self._detect_result_fn = detect_result_fn
        self._mp = None
        self._landmarker = None
        if detect_result_fn is not None:
            self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
            return
        self.model_path = resolve_face_landmarker_model_path(
            Path(model_path) if isinstance(model_path, str) else model_path
        )
        self._mp, self._landmarker = create_face_landmarker(
            self.model_path,
            output_face_blendshapes=True,
        )

    def required_keys(self) -> tuple[str, ...]:
        return ("video_rgb_frames",)

    def _detect_result(self, frame_rgb: np.ndarray) -> Any:
        if self._detect_result_fn is not None:
            return self._detect_result_fn(frame_rgb)
        if self._mp is None or self._landmarker is None:
            raise RuntimeError("Face landmarker is not initialized.")
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        return self._landmarker.detect(mp_image)

    def extract_tensors(self, frames_rgb: Sequence[np.ndarray]) -> dict[str, torch.Tensor]:
        eye_rows: list[list[float]] = []
        mesh_rows: list[np.ndarray] = []
        for frame_rgb in frames_rgb:
            _validate_frame(frame_rgb)
            result = self._detect_result(frame_rgb)
            eye_rows.append(_eye_gaze_from_result(result, self.feature_variant))
            mesh_rows.append(_mesh_from_result(result))
        return {
            "eye_gaze": torch.tensor(eye_rows, dtype=torch.float32),
            "face_mesh": torch.tensor(np.stack(mesh_rows, axis=0), dtype=torch.float32),
        }

    def extract(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        frames_rgb = batch["video_rgb_frames"]
        if not isinstance(frames_rgb, Sequence) or isinstance(frames_rgb, (str, bytes)):
            raise ValueError("`video_rgb_frames` must be a sequence of RGB frame arrays.")
        if (
            frames_rgb
            and isinstance(frames_rgb[0], Sequence)
            and not isinstance(frames_rgb[0], np.ndarray)
        ):
            items = [self.extract_tensors(clip_frames) for clip_frames in frames_rgb]
            return {
                "eye_gaze": torch.stack([item["eye_gaze"] for item in items], dim=0),
                "face_mesh": torch.stack([item["face_mesh"] for item in items], dim=0),
            }
        item = self.extract_tensors(frames_rgb)
        return {
            "eye_gaze": item["eye_gaze"].unsqueeze(0),
            "face_mesh": item["face_mesh"].unsqueeze(0),
        }

    def close(self) -> None:
        if self._landmarker is None:
            return
        close = getattr(self._landmarker, "close", None)
        if callable(close):
            close()


def build_face_landmarker_combined_extractor(
    config: Mapping[str, Any],
) -> FaceLandmarkerCombinedExtractor:
    eye_gaze_config = config.get("eye_gaze")
    if eye_gaze_config is None:
        eye_gaze_config = {}
    if not isinstance(eye_gaze_config, Mapping):
        raise ValueError("`eye_gaze` must be a YAML mapping when provided.")

    face_mesh_config = config.get("face_mesh")
    if face_mesh_config is None:
        face_mesh_config = {}
    if not isinstance(face_mesh_config, Mapping):
        raise ValueError("`face_mesh` must be a YAML mapping when provided.")

    return FaceLandmarkerCombinedExtractor(
        model_path=optional_model_path(eye_gaze_config, "model_path")
        or optional_model_path(face_mesh_config, "model_path"),
        feature_variant=eye_gaze_config.get("feature_variant"),
    )
