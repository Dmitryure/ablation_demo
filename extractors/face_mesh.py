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

FACE_MESH_CONTOUR_INDICES: tuple[int, ...] = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)


class FaceMeshExtractor(FeatureExtractor):
    name = "face_mesh"

    def __init__(
        self,
        model_path: str | Path | None = None,
        detect_landmarks_fn: Callable[[np.ndarray], np.ndarray | None] | None = None,
    ) -> None:
        self._landmarker = None
        self.model_path: Path | None = None
        if detect_landmarks_fn is not None:
            self._detect_landmarks_fn = detect_landmarks_fn
            return

        self.model_path = resolve_face_landmarker_model_path(
            Path(model_path) if isinstance(model_path, str) else model_path
        )
        self._mp, self._landmarker = create_face_landmarker(
            self.model_path,
            output_face_blendshapes=False,
        )
        self._detect_landmarks_fn = self._detect_landmarks

    def required_keys(self) -> tuple[str, ...]:
        return ("video_rgb_frames",)

    def _detect_landmarks(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        result = self._landmarker.detect(mp_image)
        if len(result.face_landmarks) != 1:
            return None

        landmarks = result.face_landmarks[0]
        points = np.array(
            [
                [float(landmarks[index].x), float(landmarks[index].y), float(landmarks[index].z)]
                for index in FACE_MESH_CONTOUR_INDICES
            ],
            dtype=np.float32,
        )
        points[:, :2] = np.clip(points[:, :2], 0.0, 1.0)
        return points

    def extract_tensor(self, frames_rgb: Sequence[np.ndarray]) -> torch.Tensor:
        rows: list[np.ndarray] = []
        for frame_rgb in frames_rgb:
            if (
                not isinstance(frame_rgb, np.ndarray)
                or frame_rgb.ndim != 3
                or frame_rgb.shape[-1] != 3
            ):
                raise ValueError(
                    "Each face-mesh frame must be an RGB numpy array with shape [H, W, 3], "
                    f"got {type(frame_rgb)}"
                )
            contour = self._detect_landmarks_fn(frame_rgb)
            if contour is None:
                contour = np.full((len(FACE_MESH_CONTOUR_INDICES), 3), -1.0, dtype=np.float32)
            rows.append(contour)
        return torch.tensor(np.stack(rows, axis=0), dtype=torch.float32)

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
                "face_mesh": torch.stack(
                    [self.extract_tensor(clip_frames) for clip_frames in frames_rgb],
                    dim=0,
                ),
            }
        return {
            "face_mesh": self.extract_tensor(frames_rgb).unsqueeze(0),
        }

    def close(self) -> None:
        close = getattr(self._landmarker, "close", None)
        if callable(close):
            close()


def build_face_mesh_extractor(config: Mapping[str, Any]) -> FaceMeshExtractor:
    face_mesh_config = config.get("face_mesh")
    if face_mesh_config is None:
        face_mesh_config = {}
    if not isinstance(face_mesh_config, Mapping):
        raise ValueError("`face_mesh` must be a YAML mapping when provided.")

    return FaceMeshExtractor(
        model_path=optional_model_path(face_mesh_config, "model_path"),
    )
