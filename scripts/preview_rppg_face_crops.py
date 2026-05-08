from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (
    RPPG_DEFAULT_FPS,
    build_real_fake_examples,
    crop_resize_rgb_frames,
    resolve_rppg_face_crop_box,
    sample_contiguous_center_indices,
)

DEFAULT_DATASET_ROOT = Path("/mnt/d/final_dataset")
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "registry_fusion.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "feature_cache_runs" / "rppg_face_crop_preview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview rPPG face crop extraction.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument("--eval-real-count", type=int, default=500)
    parser.add_argument("--eval-fake-count", type=int, default=500)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def resolve_video_root(dataset_root: Path) -> Path:
    videos_root = dataset_root / "videos"
    return videos_root if videos_root.is_dir() else dataset_root


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:180]


def select_examples(examples: list[Any], limit: int, seed: int) -> list[Any]:
    if limit <= 0:
        raise ValueError("`--limit` must be positive.")
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    return shuffled[:limit]


def read_contiguous_rgb_frames(path: Path, frame_count: int) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = RPPG_DEFAULT_FPS
    indices = sample_contiguous_center_indices(total_frames, frame_count)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
    frames: list[np.ndarray] = []
    for frame_index in indices:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to read frame {frame_index} from {path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def draw_box(frame: np.ndarray, box: tuple[int, int, int, int] | None, status: str) -> np.ndarray:
    canvas = frame.copy()
    color = (40, 220, 70) if box is not None else (240, 190, 60)
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 3)
    cv2.putText(
        canvas,
        status,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    return canvas


def resize_for_tile(frame: np.ndarray, size: int = 192) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def make_contact_sheet(
    original_frames: list[np.ndarray],
    cropped_frames: list[np.ndarray],
    box: tuple[int, int, int, int] | None,
    status: str,
) -> np.ndarray:
    indices = [
        0,
        len(original_frames) // 4,
        len(original_frames) // 2,
        3 * len(original_frames) // 4,
    ]
    top = [resize_for_tile(draw_box(original_frames[index], box, status)) for index in indices]
    bottom = [resize_for_tile(cropped_frames[index]) for index in indices]
    return np.concatenate([np.concatenate(top, axis=1), np.concatenate(bottom, axis=1)], axis=0)


def write_rgb_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError("Cannot write empty video.")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def make_side_by_side_frames(
    original_frames: list[np.ndarray],
    cropped_frames: list[np.ndarray],
    box: tuple[int, int, int, int] | None,
    status: str,
    size: int = 256,
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for original, crop in zip(original_frames, cropped_frames, strict=True):
        original_tile = resize_for_tile(draw_box(original, box, status), size=size)
        crop_tile = resize_for_tile(crop, size=size)
        frames.append(np.concatenate([original_tile, crop_tile], axis=1))
    return frames


def write_preview_index(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        cards.append(
            "<section>"
            f"<h3>{row['index']}. {row['class_name']} - {row['status']}</h3>"
            f"<p>{row['metadata_filename']}</p>"
            f'<img src="{Path(row["preview_path"]).name}" />'
            f'<p><a href="{Path(row["crop_video_path"]).name}">crop video</a> '
            f'<a href="{Path(row["side_by_side_video_path"]).name}">side-by-side video</a></p>'
            "</section>"
        )
    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<style>body{font-family:sans-serif;margin:24px;background:#f6f7f9}"
        "section{margin:0 0 28px;padding:16px;background:white;border:1px solid #ddd}"
        "img{max-width:100%;height:auto}p{word-break:break-all;color:#444}</style>"
        "</head><body><h1>rPPG Face Crop Preview</h1>" + "\n".join(cards) + "</body></html>\n"
    )
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    rppg_config = config.get("rppg", {})
    if not isinstance(rppg_config, dict):
        raise ValueError("`rppg` config must be a mapping.")
    face_crop_config = rppg_config.get("face_crop", {})
    if not isinstance(face_crop_config, dict):
        raise ValueError("`rppg.face_crop` must be a mapping.")
    frame_count = int(rppg_config.get("frames", 128))
    image_size = int(rppg_config.get("image_size", 128))
    detection_frequency = int(face_crop_config.get("detection_frequency", 16))
    large_box_coef = float(face_crop_config.get("large_box_coef", 1.5))

    video_root = resolve_video_root(args.dataset_root)
    examples = build_real_fake_examples(
        real_dir=video_root / "real",
        fake_dir=video_root / "fake",
        seed=args.seed,
        eval_real_count=args.eval_real_count,
        eval_fake_count=args.eval_fake_count,
    )
    selected = select_examples(examples, args.limit, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, example in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {example.path}", flush=True)
        frames, fps = read_contiguous_rgb_frames(example.path, frame_count)
        box, status = resolve_rppg_face_crop_box(
            frames,
            detection_frequency=detection_frequency,
            large_box_coef=large_box_coef,
        )
        cropped = crop_resize_rgb_frames(frames, box, image_size)
        sheet = make_contact_sheet(frames, cropped, box, status)
        preview_name = f"{index:03d}_{example.class_name}_{safe_stem(example.metadata_filename or example.path.name)}.jpg"
        preview_path = args.output_dir / preview_name
        cv2.imwrite(str(preview_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        video_stem = preview_path.with_suffix("").name
        crop_video_path = args.output_dir / f"{video_stem}_crop.mp4"
        side_by_side_video_path = args.output_dir / f"{video_stem}_side_by_side.mp4"
        write_rgb_video(crop_video_path, cropped, args.video_fps)
        write_rgb_video(
            side_by_side_video_path,
            make_side_by_side_frames(frames, cropped, box, status),
            args.video_fps,
        )
        rows.append(
            {
                "index": index,
                "class_name": example.class_name,
                "status": status,
                "box": "" if box is None else ",".join(str(value) for value in box),
                "fps": f"{fps:.3f}",
                "metadata_filename": example.metadata_filename or example.path.name,
                "path": str(example.path),
                "preview_path": str(preview_path),
                "crop_video_path": str(crop_video_path),
                "side_by_side_video_path": str(side_by_side_video_path),
            }
        )

    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "index",
                "class_name",
                "status",
                "box",
                "fps",
                "metadata_filename",
                "path",
                "preview_path",
                "crop_video_path",
                "side_by_side_video_path",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    write_preview_index(args.output_dir, rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    print(f"wrote={args.output_dir}", flush=True)
    print(f"status_counts={status_counts}", flush=True)


if __name__ == "__main__":
    main()
