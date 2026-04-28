from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "docs" / "model_architecture.dot"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "model_architecture.svg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Graphviz DOT source to SVG or another Graphviz format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="DOT input path. Default: docs/model_architecture.dot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Rendered output path. Default: docs/model_architecture.svg",
    )
    parser.add_argument(
        "--format",
        default="svg",
        help="Graphviz output format passed to `dot -T`. Default: svg",
    )
    return parser.parse_args()


def require_dot() -> str:
    dot_path = shutil.which("dot")
    if dot_path is None:
        raise SystemExit("Missing `dot` executable. Install Graphviz, then rerun this script.")
    return dot_path


def main() -> None:
    args = parse_args()
    dot_path = require_dot()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [dot_path, f"-T{args.format}", str(input_path), "-o", str(output_path)],
        check=True,
    )
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
