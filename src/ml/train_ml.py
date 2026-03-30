from __future__ import annotations

from pathlib import Path
import argparse
import json
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))
from seed_dataset import load_manifest, load_trial_level_features  # noqa: E402


@dataclass
class EvalResult:
    protocol: str
    model: str
    fold: int
    accuracy: float
    macro_f1: float


def make_models(seed: int) -> dict[str, Callable[[], Pipeline]]:
    return {
        "logistic_regression": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "svm_rbf": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", random_state=seed)),
            ]
        ),
        "random_forest": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)),
            ]
        ),
    }


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_names: list[str],
    seed: int,
) -> tuple[list[EvalResult], dict]:
    models = make_models(seed)
    all_results: list[EvalResult] = []
    confusion_payload: dict[str, dict] = {}

    # Protocol 1: subject-dependent (within-subject)
    for model_name, model_fn in models.items():
        fold_idx = 0
        per_fold_cms: list[np.ndarray] = []
        for subject in sorted(np.unique(groups)):
            idx = np.where(groups == subject)[0]
            Xs = X[idx]
            ys = y[idx]
            X_train, X_test, y_train, y_test = train_test_split(
                Xs,
                ys,
                test_size=0.2,
                random_state=seed,
                stratify=ys,
            )
            clf = model_fn()
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            all_results.append(
                EvalResult(
                    protocol="subject_dependent",
                    model=model_name,
                    fold=fold_idx,
                    accuracy=accuracy_score(y_test, pred),
                    macro_f1=f1_score(y_test, pred, average="macro"),
                )
            )
            per_fold_cms.append(confusion_matrix(y_test, pred, labels=np.arange(len(label_names))))
            fold_idx += 1
        confusion_payload[f"subject_dependent::{model_name}"] = {
            "labels": label_names,
            "matrix": np.mean(np.stack(per_fold_cms), axis=0).tolist(),
        }

    # Protocol 2: subject-independent (group k-fold)
    gkf = GroupKFold(n_splits=5)
    for model_name, model_fn in models.items():
        per_fold_cms: list[np.ndarray] = []
        for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
            clf = model_fn()
            clf.fit(X[tr_idx], y[tr_idx])
            pred = clf.predict(X[te_idx])
            all_results.append(
                EvalResult(
                    protocol="subject_independent",
                    model=model_name,
                    fold=fold_idx,
                    accuracy=accuracy_score(y[te_idx], pred),
                    macro_f1=f1_score(y[te_idx], pred, average="macro"),
                )
            )
            per_fold_cms.append(confusion_matrix(y[te_idx], pred, labels=np.arange(len(label_names))))
        confusion_payload[f"subject_independent::{model_name}"] = {
            "labels": label_names,
            "matrix": np.mean(np.stack(per_fold_cms), axis=0).tolist(),
        }

    return all_results, confusion_payload


def aggregate(results: list[EvalResult]) -> list[dict]:
    grouped: dict[tuple[str, str], list[EvalResult]] = {}
    for row in results:
        grouped.setdefault((row.protocol, row.model), []).append(row)
    agg = []
    for (protocol, model), rows in sorted(grouped.items()):
        acc = np.asarray([r.accuracy for r in rows], dtype=np.float64)
        f1 = np.asarray([r.macro_f1 for r in rows], dtype=np.float64)
        agg.append(
            {
                "protocol": protocol,
                "model": model,
                "accuracy_mean": float(acc.mean()),
                "accuracy_std": float(acc.std(ddof=0)),
                "macro_f1_mean": float(f1.mean()),
                "macro_f1_std": float(f1.std(ddof=0)),
                "num_folds": len(rows),
            }
        )
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML baselines on SEED features.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/json/manifest.jsonl"))
    parser.add_argument("--label-map", type=Path, default=Path("outputs/json/label_map.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-output", type=Path, default=Path("outputs/json/ml_results.json"))
    parser.add_argument("--confusion-output", type=Path, default=Path("outputs/json/ml_confusion.json"))
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    X, y, groups = load_trial_level_features(rows)
    with args.label_map.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    results, confusion = evaluate(X=X, y=y, groups=groups, label_names=label_names, seed=args.seed)
    payload = {
        "detail": [asdict(r) for r in results],
        "summary": aggregate(results),
    }
    args.results_output.parent.mkdir(parents=True, exist_ok=True)
    with args.results_output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with args.confusion_output.open("w", encoding="utf-8") as f:
        json.dump(confusion, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

