from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MODELS_DIR = PROJECT_ROOT / "models"

SourceKind = Literal["url", "gdrive", "local"]


@dataclass(frozen=True)
class WeightEntry:
    source: str
    source_kind: SourceKind
    output_dir: Path
    filename: str
    extract_zip: bool = False
    local_fallbacks: tuple[str, ...] = ()


BACKBONE_WEIGHTS: dict[str, WeightEntry] = {
    "resnet18": WeightEntry(
        source="https://download.pytorch.org/models/resnet18-5c106cde.pth",
        source_kind="url",
        output_dir=CHECKPOINTS_DIR,
        filename="resnet18-5c106cde.pth",
    ),
    "resnet34": WeightEntry(
        source="https://download.pytorch.org/models/resnet34-333f7ec4.pth",
        source_kind="url",
        output_dir=CHECKPOINTS_DIR,
        filename="resnet34-333f7ec4.pth",
    ),
    "resnet50": WeightEntry(
        source="https://download.pytorch.org/models/resnet50-19c8e357.pth",
        source_kind="url",
        output_dir=CHECKPOINTS_DIR,
        filename="resnet50-19c8e357.pth",
    ),
    "resnet101": WeightEntry(
        source="https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
        source_kind="url",
        output_dir=CHECKPOINTS_DIR,
        filename="resnet101-5d3b4d8f.pth",
    ),
    "resnet152": WeightEntry(
        source="https://download.pytorch.org/models/resnet152-b121ed2d.pth",
        source_kind="url",
        output_dir=CHECKPOINTS_DIR,
        filename="resnet152-b121ed2d.pth",
    ),
    "swin_transformer_tiny": WeightEntry(
        source="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
        source_kind="url",
        output_dir=CHECKPOINTS_DIR,
        filename="swin_tiny_patch4_window7_224.pth",
    ),
    "swin_transformer_small": WeightEntry(
        source="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
        source_kind="url",
        output_dir=CHECKPOINTS_DIR,
        filename="swin_small_patch4_window7_224.pth",
    ),
    "swin_transformer_base": WeightEntry(
        source="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth",
        source_kind="url",
        output_dir=CHECKPOINTS_DIR,
        filename="swin_base_patch4_window7_224.pth",
    ),
}
RGB_WEIGHTS: dict[str, WeightEntry] = {
    model: entry for model, entry in BACKBONE_WEIGHTS.items() if model.startswith("resnet")
}

FAU_BP4D_WEIGHTS: dict[str, WeightEntry] = {
    "resnet50": WeightEntry(
        source="1gYVHRjIj6ounxtTdXMje4WNHFMfCTVF9",
        source_kind="gdrive",
        output_dir=CHECKPOINTS_DIR / "fau" / "bp4d" / "resnet50",
        filename="ME-GraphAU_resnet50_BP4D.zip",
        extract_zip=True,
        local_fallbacks=(
            "/home/comp/detectors/src/backbones/MEGraphAU/checkpoints/MEFARG_resnet50_BP4D_fold1.pth",
            "/home/comp/detectors/src/backbones/MEGraphAU/checkpoints/MEFARG_resnet50_BP4D.pth",
        ),
    ),
    "resnet101": WeightEntry(
        source="1i-ra0dtoEhwIep6goZ55PvEgwE3kecbl",
        source_kind="gdrive",
        output_dir=CHECKPOINTS_DIR / "fau" / "bp4d" / "resnet101",
        filename="ME-GraphAU_resnet101_BP4D.zip",
        extract_zip=True,
        local_fallbacks=(
            "/home/comp/detectors/src/backbones/MEGraphAU/checkpoints/MEFARG_resnet101_BP4D_fold1.pth",
            "/home/comp/detectors/src/backbones/MEGraphAU/checkpoints/MEFARG_resnet101_BP4D.pth",
        ),
    ),
    "swin_transformer_tiny": WeightEntry(
        source="1BT4n7_5Wr6bGxHWVf3WrT7uBT0Zg9B5c",
        source_kind="gdrive",
        output_dir=CHECKPOINTS_DIR / "fau" / "bp4d" / "swin_transformer_tiny",
        filename="ME-GraphAU_swin_tiny_BP4D.zip",
        extract_zip=True,
        local_fallbacks=(
            "/home/comp/detectors/src/backbones/MEGraphAU/checkpoints/MEFARG_swin_tiny_BP4D_fold1.pth",
            "/home/comp/detectors/src/backbones/MEGraphAU/checkpoints/MEFARG_swin_tiny_BP4D.pth",
        ),
    ),
    "swin_transformer_small": WeightEntry(
        source="1EiQd6q7x1bEO6JBLi3s2y5348EuVdP3L",
        source_kind="gdrive",
        output_dir=CHECKPOINTS_DIR / "fau" / "bp4d" / "swin_transformer_small",
        filename="ME-GraphAU_swin_small_BP4D.zip",
        extract_zip=True,
        local_fallbacks=(
            "/home/comp/detectors/src/backbones/MEGraphAU/checkpoints/MEFARG_swin_small_BP4D_fold1.pth",
            "/home/comp/detectors/src/backbones/MEGraphAU/checkpoints/MEFARG_swin_small_BP4D.pth",
        ),
    ),
    "swin_transformer_base": WeightEntry(
        source="1Ti0auMA5o94toJfszuHoMlSlWUumm9L8",
        source_kind="gdrive",
        output_dir=CHECKPOINTS_DIR / "fau" / "bp4d" / "swin_transformer_base",
        filename="ME-GraphAU_swin_base_BP4D.zip",
        extract_zip=True,
        local_fallbacks=(
            "/home/comp/detectors/src/backbones/MEGraphAU/checkpoints/MEFARG_swin_base_BP4D_fold1.pth",
            "/home/comp/detectors/src/backbones/MEGraphAU/checkpoints/MEFARG_swin_base_BP4D.pth",
        ),
    ),
}

FAU_DISFA_WEIGHTS: dict[str, WeightEntry] = {
    "resnet50": WeightEntry(
        source="1V-imbmhg-OgcP2d9SETT5iswNtCA0f8_",
        source_kind="gdrive",
        output_dir=CHECKPOINTS_DIR / "fau" / "disfa" / "resnet50",
        filename="ME-GraphAU_resnet50_DISFA.zip",
        extract_zip=True,
        local_fallbacks=(
            "/home/comp/detectors/src/backbones/MEGraphAU/checkpoints/MEFARG_resnet50_DISFA_fold2.pth",
            "/home/comp/detectors/src/backbones/MEGraphAU/checkpoints/MEFARG_resnet50_DISFA.pth",
        ),
    ),
    "swin_transformer_base": WeightEntry(
        source="1T44KPDaUhi4J_C-fWa6RxXNkY3yoDwIi",
        source_kind="gdrive",
        output_dir=CHECKPOINTS_DIR / "fau" / "disfa" / "swin_transformer_base",
        filename="ME-GraphAU_swin_base_DISFA.zip",
        extract_zip=True,
        local_fallbacks=(
            "/home/comp/detectors/src/backbones/MEGraphAU/checkpoints/MEFARG_swin_base_DISFA_fold2.pth",
            "/home/comp/detectors/src/backbones/MEGraphAU/checkpoints/MEFARG_swin_base_DISFA.pth",
        ),
    ),
}

WEIGHTS_DB: dict[str, dict[str, WeightEntry]] = {
    "backbone": BACKBONE_WEIGHTS,
    "rgb": RGB_WEIGHTS,
    "fau-bp4d": FAU_BP4D_WEIGHTS,
    "fau-disfa": FAU_DISFA_WEIGHTS,
    "rppg": {
        "physnet-pure": WeightEntry(
            source="/home/comp/detectors/src/backbones/rPPGToolbox/final_model_release/PURE_PhysNet_DiffNormalized.pth",
            source_kind="local",
            output_dir=CHECKPOINTS_DIR,
            filename="PURE_PhysNet_DiffNormalized.pth",
        ),
    },
    "mediapipe": {
        "face-landmarker-v2-blendshapes": WeightEntry(
            source="https://huggingface.co/fudan-generative-ai/hallo3/resolve/main/face_analysis/models/face_landmarker_v2_with_blendshapes.task",
            source_kind="url",
            output_dir=MODELS_DIR,
            filename="face_landmarker_v2_with_blendshapes.task",
        ),
    },
}

ALIASES: dict[str, str] = {
    "swin-tiny": "swin_transformer_tiny",
    "swin-small": "swin_transformer_small",
    "swin-base": "swin_transformer_base",
    "physnet": "physnet-pure",
    "pure-physnet": "physnet-pure",
    "eye-gaze": "face-landmarker-v2-blendshapes",
    "face-landmarker": "face-landmarker-v2-blendshapes",
}


def _target_path(entry: WeightEntry) -> Path:
    return entry.output_dir / entry.filename


def _normalize_model_name(model: str) -> str:
    return ALIASES.get(model, model)


def _has_extracted_payload(entry: WeightEntry) -> bool:
    if not entry.extract_zip or not entry.output_dir.exists():
        return False
    return any(path.is_file() for path in entry.output_dir.rglob("*"))


def _download_url(entry: WeightEntry) -> Path:
    path = _target_path(entry)
    if path.exists():
        print(f"exists: {path}")
        return path

    entry.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"download: {entry.source} -> {path}")
    urllib.request.urlretrieve(entry.source, path)
    return path


def _download_gdrive(entry: WeightEntry) -> Path:
    path = _target_path(entry)
    if path.exists():
        print(f"exists: {path}")
        return path

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Google Drive weights need `gdown`. Install with `pip install gdown`.") from exc

    entry.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"gdrive: {entry.source} -> {path}")
    downloaded = gdown.download(id=entry.source, output=str(path), quiet=False, fuzzy=True)
    if downloaded is None:
        raise RuntimeError(f"gdown failed for: {entry.source}")
    return Path(downloaded)


def _copy_local(entry: WeightEntry) -> Path:
    path = _target_path(entry)
    if path.exists():
        print(f"exists: {path}")
        return path

    source_path = Path(entry.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Local source does not exist: {source_path}")

    entry.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"copy: {source_path} -> {path}")
    shutil.copy2(source_path, path)
    return path


def _copy_local_fallback(entry: WeightEntry) -> Path | None:
    for source in entry.local_fallbacks:
        source_path = Path(source)
        if not source_path.exists():
            continue

        entry.output_dir.mkdir(parents=True, exist_ok=True)
        target_path = entry.output_dir / source_path.name
        if target_path.exists():
            print(f"exists: {target_path}")
            return target_path

        print(f"copy-local-fallback: {source_path} -> {target_path}")
        shutil.copy2(source_path, target_path)
        return target_path
    return None


def _extract_zip(path: Path, output_dir: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"Expected zip archive: {path}")

    print(f"extract: {path} -> {output_dir}")
    with zipfile.ZipFile(path) as zip_ref:
        zip_ref.extractall(output_dir)
    path.unlink()


def fetch_entry(entry: WeightEntry) -> Path:
    if _has_extracted_payload(entry):
        print(f"exists: {entry.output_dir}")
        return entry.output_dir

    local_fallback_path = _copy_local_fallback(entry)
    if local_fallback_path is not None:
        return entry.output_dir if entry.extract_zip else local_fallback_path

    if entry.source_kind == "url":
        path = _download_url(entry)
    elif entry.source_kind == "gdrive":
        path = _download_gdrive(entry)
    elif entry.source_kind == "local":
        path = _copy_local(entry)
    else:
        raise ValueError(f"Unsupported source kind: {entry.source_kind}")

    if entry.extract_zip and path.is_file():
        _extract_zip(path, entry.output_dir)
        return entry.output_dir
    return path


def download(category: str, model: str) -> Path:
    normalized_model = _normalize_model_name(model)
    if category not in WEIGHTS_DB:
        raise ValueError(f"Unknown category `{category}`. Options: {', '.join(sorted(WEIGHTS_DB))}")

    entries = WEIGHTS_DB[category]
    if normalized_model not in entries:
        raise ValueError(f"Unknown model `{model}` for `{category}`. Options: {', '.join(sorted(entries))}")

    if category.startswith("fau") and normalized_model in BACKBONE_WEIGHTS:
        fetch_entry(BACKBONE_WEIGHTS[normalized_model])
    return fetch_entry(entries[normalized_model])


def list_weights() -> None:
    for category, entries in WEIGHTS_DB.items():
        print(category)
        for model, entry in entries.items():
            print(f"  {model}: {_target_path(entry)}")


def parse_args() -> argparse.Namespace:
    legacy_args = sys.argv[1:]
    if len(legacy_args) == 2 and legacy_args[0] in WEIGHTS_DB:
        return argparse.Namespace(command="download", category=legacy_args[0], model=legacy_args[1])

    parser = argparse.ArgumentParser(description="Download or copy local model weights for this repo.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Fetch one weight entry.")
    download_parser.add_argument("category", choices=sorted(WEIGHTS_DB))
    download_parser.add_argument("model")

    subparsers.add_parser("list", help="List available categories and models.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "list":
            list_weights()
            return 0
        path = download(args.category, args.model)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"ready: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
