from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))
from seed_dataset import load_manifest, load_trial_level_features_with_meta  # noqa: E402


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

    ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    Xv = torch.from_numpy(X_valid).float().to(device)

    best_state = None
    best_f1 = -1.0
    no_improve = 0
    patience = 8

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
        f1_macro = _macro_f1(y_valid, pred, num_classes=num_classes)
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    model.eval()
    return model


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    eps = 1e-12
    f1s = []
    for c in range(num_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        f1s.append(f1)
    return float(np.mean(f1s))


def _serialize_scaler(scaler: StandardScaler) -> dict:
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}


def _serialize_lr(clf: LogisticRegression) -> dict:
    return {
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "classes": clf.classes_.tolist(),
    }


def _serialize_mlp(model: MLPClassifier) -> dict:
    state = model.state_dict()
    return {
        "layers": [
            {"weight": state["net.0.weight"].tolist(), "bias": state["net.0.bias"].tolist()},
            {"weight": state["net.3.weight"].tolist(), "bias": state["net.3.bias"].tolist()},
            {"weight": state["net.6.weight"].tolist(), "bias": state["net.6.bias"].tolist()},
        ]
    }


def export_subject_dependent(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    trials: np.ndarray,
    labels: list[str],
    seed: int,
) -> tuple[dict, dict]:
    lr_payload = {"protocol": "subject_dependent", "labels": labels, "subjects": {}}
    mlp_payload = {"protocol": "subject_dependent", "labels": labels, "subjects": {}}

    num_classes = len(labels)
    for subject in sorted(np.unique(groups)):
        idx = np.where(groups == subject)[0]
        Xs, ys, ts = X[idx], y[idx], trials[idx]
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
            Xs,
            ys,
            ts,
            test_size=0.2,
            random_state=seed,
            stratify=ys,
        )

        # LR
        scaler_lr = StandardScaler()
        X_train_lr = scaler_lr.fit_transform(X_train)
        lr = LogisticRegression(max_iter=2000, random_state=seed)
        lr.fit(X_train_lr, y_train)
        lr_payload["subjects"][str(int(subject))] = {
            "scaler": _serialize_scaler(scaler_lr),
            "classifier": _serialize_lr(lr),
            "test_trials": [int(v) for v in t_test.tolist()],
        }

        # MLP
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.25,
            random_state=seed,
            stratify=y_train,
        )
        scaler_mlp = StandardScaler()
        X_tr = scaler_mlp.fit_transform(X_tr).astype(np.float32)
        X_val = scaler_mlp.transform(X_val).astype(np.float32)
        model = fit_mlp(X_tr, y_tr, X_val, y_val, num_classes=num_classes, seed=seed)
        mlp_payload["subjects"][str(int(subject))] = {
            "scaler": _serialize_scaler(scaler_mlp),
            "network": _serialize_mlp(model),
            "classes": list(range(num_classes)),
            "test_trials": [int(v) for v in t_test.tolist()],
        }
    return lr_payload, mlp_payload


def export_subject_independent(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    labels: list[str],
    seed: int,
) -> tuple[dict, dict]:
    lr_payload = {"protocol": "subject_independent", "labels": labels, "folds": []}
    mlp_payload = {"protocol": "subject_independent", "labels": labels, "folds": []}
    num_classes = len(labels)
    gkf = GroupKFold(n_splits=5)
    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, y_train = X[tr_idx], y[tr_idx]
        test_subjects = sorted({int(s) for s in groups[te_idx].tolist()})

        # LR
        scaler_lr = StandardScaler()
        X_train_lr = scaler_lr.fit_transform(X_train)
        lr = LogisticRegression(max_iter=2000, random_state=seed)
        lr.fit(X_train_lr, y_train)
        lr_payload["folds"].append(
            {
                "fold": fold_idx,
                "test_subjects": test_subjects,
                "scaler": _serialize_scaler(scaler_lr),
                "classifier": _serialize_lr(lr),
            }
        )

        # MLP
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=seed,
            stratify=y_train,
        )
        scaler_mlp = StandardScaler()
        X_tr = scaler_mlp.fit_transform(X_tr).astype(np.float32)
        X_val = scaler_mlp.transform(X_val).astype(np.float32)
        model = fit_mlp(X_tr, y_tr, X_val, y_val, num_classes=num_classes, seed=seed)
        mlp_payload["folds"].append(
            {
                "fold": fold_idx,
                "test_subjects": test_subjects,
                "scaler": _serialize_scaler(scaler_mlp),
                "network": _serialize_mlp(model),
                "classes": list(range(num_classes)),
            }
        )
    return lr_payload, mlp_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export protocol-aware LR/MLP models for web inference.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/json/manifest.jsonl"))
    parser.add_argument("--label-map", type=Path, default=Path("outputs/json/label_map.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    X, y, groups, trials = load_trial_level_features_with_meta(rows)
    with args.label_map.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    labels = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    lr_sd, mlp_sd = export_subject_dependent(X, y, groups, trials, labels, args.seed)
    lr_si, mlp_si = export_subject_independent(X, y, groups, labels, args.seed)

    args.outdir.mkdir(parents=True, exist_ok=True)
    files = {
        "protocol_models_lr_subject.json": lr_sd,
        "protocol_models_mlp_subject.json": mlp_sd,
        "protocol_models_lr_fold.json": lr_si,
        "protocol_models_mlp_fold.json": mlp_si,
    }
    for name, payload in files.items():
        with (args.outdir / name).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    print("Exported protocol models:")
    for name in files:
        print(f"- {args.outdir / name}")


if __name__ == "__main__":
    main()

