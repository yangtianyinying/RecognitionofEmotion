from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_bar_figure(rows: list[dict], metric: str, output_path: Path) -> None:
    protocols = sorted(set(r["protocol"] for r in rows))
    fig, axes = plt.subplots(1, len(protocols), figsize=(7 * len(protocols), 5), squeeze=False)

    for idx, protocol in enumerate(protocols):
        ax = axes[0][idx]
        sub = [r for r in rows if r["protocol"] == protocol]
        sub = sorted(sub, key=lambda x: x[metric], reverse=True)
        models = [r["model"] for r in sub]
        means = [r[metric] for r in sub]
        errs = [r[f"{metric.replace('_mean', '')}_std"] for r in sub]
        ax.bar(models, means, yerr=errs, capsize=4)
        ax.set_ylim(0, 1.0)
        ax.set_title(protocol.replace("_", " ").title())
        ax.set_ylabel(metric.replace("_", " "))
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_confusion_figure(confusion_data: dict, output_path: Path) -> None:
    keys = sorted(confusion_data.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 4 * len(keys)), squeeze=False)
    for i, key in enumerate(keys):
        ax = axes[i][0]
        labels = confusion_data[key]["labels"]
        matrix = np.array(confusion_data[key]["matrix"])
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(key)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build figures and web JSON assets.")
    parser.add_argument("--ml-results", type=Path, default=Path("outputs/json/ml_results.json"))
    parser.add_argument("--dl-results", type=Path, default=Path("outputs/json/dl_results.json"))
    parser.add_argument("--ml-confusion", type=Path, default=Path("outputs/json/ml_confusion.json"))
    parser.add_argument("--dl-confusion", type=Path, default=Path("outputs/json/dl_confusion.json"))
    parser.add_argument("--dataset-summary", type=Path, default=Path("outputs/json/dataset_summary.json"))
    parser.add_argument("--fig-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--web-json", type=Path, default=Path("docs/assets/results.json"))
    args = parser.parse_args()

    ml = _load_json(args.ml_results)
    dl = _load_json(args.dl_results)
    ml_c = _load_json(args.ml_confusion)
    dl_c = _load_json(args.dl_confusion)
    ds = _load_json(args.dataset_summary)

    summary_rows = ml["summary"] + dl["summary"]
    _save_bar_figure(summary_rows, "accuracy_mean", args.fig_dir / "model_accuracy.png")
    _save_bar_figure(summary_rows, "macro_f1_mean", args.fig_dir / "model_macro_f1.png")
    _save_confusion_figure({**ml_c, **dl_c}, args.fig_dir / "confusions.png")

    web_payload = {
        "dataset": ds,
        "summary": summary_rows,
        "ml_confusion": ml_c,
        "dl_confusion": dl_c,
    }
    args.web_json.parent.mkdir(parents=True, exist_ok=True)
    with args.web_json.open("w", encoding="utf-8") as f:
        json.dump(web_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved web payload to {args.web_json}")


if __name__ == "__main__":
    main()

