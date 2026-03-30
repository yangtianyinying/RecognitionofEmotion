from __future__ import annotations

from pathlib import Path
import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    num_classes: int,
    seed: int,
    epochs: int = 40,
    batch_size: int = 64,
) -> MLPClassifier:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    ds_train = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    Xv = torch.from_numpy(X_valid).float().to(device)

    best_state = None
    best_f1 = -1.0
    patience = 8
    no_improve = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xv).argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(y_valid, pred, average="macro")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    return model


def predict(model: MLPClassifier, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float())
        return logits.argmax(dim=1).numpy()


def evaluate_protocols(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_names: list[str],
    seed: int,
) -> tuple[list[EvalResult], dict]:
    all_results: list[EvalResult] = []
    confusion_payload: dict[str, dict] = {}
    num_classes = len(label_names)

    # Subject-dependent
    fold_cms = []
    for fold_idx, subject in enumerate(sorted(np.unique(groups))):
        idx = np.where(groups == subject)[0]
        Xs, ys = X[idx], y[idx]
        X_train, X_test, y_train, y_test = train_test_split(
            Xs,
            ys,
            test_size=0.2,
            random_state=seed,
            stratify=ys,
        )
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.25,
            random_state=seed,
            stratify=y_train,
        )
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

        model = fit_mlp(X_tr, y_tr, X_val, y_val, num_classes=num_classes, seed=seed)
        pred = predict(model, X_test)
        all_results.append(
            EvalResult(
                protocol="subject_dependent",
                model="mlp",
                fold=fold_idx,
                accuracy=accuracy_score(y_test, pred),
                macro_f1=f1_score(y_test, pred, average="macro"),
            )
        )
        fold_cms.append(confusion_matrix(y_test, pred, labels=np.arange(num_classes)))
    confusion_payload["subject_dependent::mlp"] = {
        "labels": label_names,
        "matrix": np.mean(np.stack(fold_cms), axis=0).tolist(),
    }

    # Subject-independent
    fold_cms = []
    gkf = GroupKFold(n_splits=5)
    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=seed,
            stratify=y_train,
        )
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

        model = fit_mlp(X_tr, y_tr, X_val, y_val, num_classes=num_classes, seed=seed)
        pred = predict(model, X_test)
        all_results.append(
            EvalResult(
                protocol="subject_independent",
                model="mlp",
                fold=fold_idx,
                accuracy=accuracy_score(y_test, pred),
                macro_f1=f1_score(y_test, pred, average="macro"),
            )
        )
        fold_cms.append(confusion_matrix(y_test, pred, labels=np.arange(num_classes)))
    confusion_payload["subject_independent::mlp"] = {
        "labels": label_names,
        "matrix": np.mean(np.stack(fold_cms), axis=0).tolist(),
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
    parser = argparse.ArgumentParser(description="Train an MLP baseline on SEED features.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/json/manifest.jsonl"))
    parser.add_argument("--label-map", type=Path, default=Path("outputs/json/label_map.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-output", type=Path, default=Path("outputs/json/dl_results.json"))
    parser.add_argument("--confusion-output", type=Path, default=Path("outputs/json/dl_confusion.json"))
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    X, y, groups = load_trial_level_features(rows)
    with args.label_map.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    results, confusion = evaluate_protocols(X, y, groups, label_names, args.seed)
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

