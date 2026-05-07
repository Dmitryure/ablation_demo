from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import VIDEO_EXTENSIONS, infer_real_source_id

DEFAULT_SOURCE_ROOT = Path("/home/comp/final_dataset")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "local_small_dataset"
DEFAULT_MANIFEST_NAME = "subset_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small local real/fake dataset subset using symlinks by default."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--real-count", type=int, default=6)
    parser.add_argument("--fake-count", type=int, default=6)
    parser.add_argument("--fake-generators", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--copy", action="store_true", help="Copy videos instead of symlinking.")
    parser.add_argument("--force", action="store_true", help="Replace an existing subset tree.")
    return parser.parse_args()


def resolve_video_root(root: Path) -> Path:
    videos_root = root / "videos"
    return videos_root if videos_root.is_dir() else root


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def discover_real_videos(video_root: Path) -> list[Path]:
    real_root = video_root / "real"
    if not real_root.is_dir():
        raise FileNotFoundError(f"Missing real video directory: {real_root}")
    return sorted(path for path in real_root.iterdir() if is_video_file(path))


def discover_fake_videos(video_root: Path) -> Mapping[str, list[Path]]:
    fake_root = video_root / "fake"
    if not fake_root.is_dir():
        raise FileNotFoundError(f"Missing fake video directory: {fake_root}")

    by_generator: dict[str, list[Path]] = defaultdict(list)
    for generator_dir in sorted(path for path in fake_root.iterdir() if path.is_dir()):
        videos = sorted(path for path in generator_dir.iterdir() if is_video_file(path))
        if videos:
            by_generator[generator_dir.name].extend(videos)
    return dict(by_generator)


def validate_count(name: str, count: int) -> None:
    if count <= 0:
        raise ValueError(f"`{name}` must be positive.")


def select_paths(paths: Sequence[Path], count: int, seed: int) -> list[Path]:
    if count > len(paths):
        raise ValueError(f"Requested {count} videos but only {len(paths)} are available.")
    selected = list(paths)
    random.Random(seed).shuffle(selected)
    return sorted(selected[:count])


def group_real_videos(paths: Iterable[Path]) -> Mapping[str, list[Path]]:
    by_source: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_source[infer_real_source_id(path)].append(path)
    return dict(by_source)


def select_grouped_paths(
    paths_by_group: Mapping[str, list[Path]],
    count: int,
    seed: int,
) -> list[Path]:
    total = sum(len(paths) for paths in paths_by_group.values())
    if count > total:
        raise ValueError(f"Requested {count} videos but only {total} are available.")

    rng = random.Random(seed)
    shuffled = {name: list(paths) for name, paths in paths_by_group.items()}
    for paths in shuffled.values():
        rng.shuffle(paths)

    group_names = sorted(shuffled)
    rng.shuffle(group_names)
    selected: list[Path] = []
    while len(selected) < count:
        progressed = False
        for group_name in group_names:
            paths = shuffled[group_name]
            if not paths:
                continue
            selected.append(paths.pop())
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            break
    return sorted(selected)


def filter_fake_generators(
    videos_by_generator: Mapping[str, list[Path]],
    generator_names: Sequence[str] | None,
) -> dict[str, list[Path]]:
    if not generator_names:
        return {name: list(paths) for name, paths in videos_by_generator.items()}

    missing = sorted(set(generator_names) - set(videos_by_generator))
    if missing:
        raise ValueError(f"Unknown fake generators: {', '.join(missing)}")
    return {name: list(videos_by_generator[name]) for name in generator_names}


def select_fake_paths(
    videos_by_generator: Mapping[str, list[Path]],
    count: int,
    seed: int,
) -> list[Path]:
    return select_grouped_paths(videos_by_generator, count, seed)


def ensure_empty_subset_root(output_root: Path, force: bool) -> None:
    if not output_root.exists():
        return
    if not force:
        raise FileExistsError(f"Output already exists: {output_root}. Use --force to replace it.")
    shutil.rmtree(output_root)


def materialize_video(source: Path, destination: Path, copy_file: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_file:
        shutil.copy2(source, destination)
        return
    destination.symlink_to(source)


def fake_relative_path(path: Path) -> Path:
    return Path(path.parent.name) / path.name


def write_manifest(
    output_root: Path,
    real_paths: Sequence[Path],
    fake_paths: Sequence[Path],
) -> Path:
    manifest_path = output_root / DEFAULT_MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ("class_name", "relative_path", "source_path")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for path in real_paths:
            writer.writerow(
                {
                    "class_name": "real",
                    "relative_path": str(Path("videos") / "real" / path.name),
                    "source_path": str(path),
                }
            )
        for path in fake_paths:
            writer.writerow(
                {
                    "class_name": "fake",
                    "relative_path": str(Path("videos") / "fake" / fake_relative_path(path)),
                    "source_path": str(path),
                }
            )
    return manifest_path


def create_subset(
    source_root: Path,
    output_root: Path,
    real_count: int,
    fake_count: int,
    fake_generators: Sequence[str] | None,
    seed: int,
    copy_files: bool,
    force: bool,
) -> Path:
    validate_count("real-count", real_count)
    validate_count("fake-count", fake_count)
    video_root = resolve_video_root(source_root)
    real_paths = select_grouped_paths(
        group_real_videos(discover_real_videos(video_root)), real_count, seed
    )
    fake_by_generator = filter_fake_generators(discover_fake_videos(video_root), fake_generators)
    fake_paths = select_fake_paths(fake_by_generator, fake_count, seed + 1)

    ensure_empty_subset_root(output_root, force)
    for path in real_paths:
        materialize_video(path, output_root / "videos" / "real" / path.name, copy_files)
    for path in fake_paths:
        materialize_video(
            path,
            output_root / "videos" / "fake" / fake_relative_path(path),
            copy_files,
        )
    return write_manifest(output_root, real_paths, fake_paths)


def main() -> None:
    args = parse_args()
    manifest_path = create_subset(
        source_root=args.source_root,
        output_root=args.output_root,
        real_count=args.real_count,
        fake_count=args.fake_count,
        fake_generators=args.fake_generators,
        seed=args.seed,
        copy_files=args.copy,
        force=args.force,
    )
    print(f"wrote: {args.output_root}")
    print(f"wrote: {manifest_path}")


if __name__ == "__main__":
    main()
