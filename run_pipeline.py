from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    steps = [
        [sys.executable, str(root / "src" / "data" / "build_manifest.py")],
        [sys.executable, str(root / "src" / "ml" / "train_ml.py")],
        [sys.executable, str(root / "src" / "dl" / "train_mlp.py")],
        [sys.executable, str(root / "src" / "visualization" / "build_assets.py")],
    ]
    for cmd in steps:
        run_step(cmd)
    print("\nPipeline finished. Check outputs/json, outputs/figures, docs/assets.")


if __name__ == "__main__":
    main()

