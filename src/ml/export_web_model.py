from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))
from seed_dataset import load_manifest, load_trial_level_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a browser-usable LR model for uploaded MAT inference.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/json/manifest.jsonl"))
    parser.add_argument("--label-map", type=Path, default=Path("outputs/json/label_map.json"))
    parser.add_argument("--feature-type", type=str, default="de_lds", choices=["de", "de_lds", "psd"])
    parser.add_argument("--output", type=Path, default=Path("docs/assets/web_model.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    if not rows:
        raise RuntimeError("Manifest is empty. Run src/data/build_manifest.py first.")
    X, y, _ = load_trial_level_features(rows)

    with args.label_map.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    labels = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, random_state=args.seed)),
        ]
    )
    clf.fit(X, y)

    scaler: StandardScaler = clf.named_steps["scaler"]
    lr: LogisticRegression = clf.named_steps["lr"]

    payload = {
        "model_type": "logistic_regression",
        "feature_type": args.feature_type,
        "input_dim": int(X.shape[1]),
        "labels": labels,
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "classifier": {
            "coef": lr.coef_.tolist(),
            "intercept": lr.intercept_.tolist(),
            "classes": lr.classes_.tolist(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"Saved web model to {args.output}")


if __name__ == "__main__":
    main()

