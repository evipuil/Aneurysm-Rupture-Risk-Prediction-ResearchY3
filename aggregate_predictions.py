# Version 10 source snapshot
"""
aggregate_predictions.py

Averages pooled prediction CSVs from multiple v10 seed runs.
Expects each run to write a pooled_predictions.csv with columns:
- filepath
- label
- prob
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def safe_auc(targets, probs):
    try:
        return roc_auc_score(targets, probs)
    except Exception:
        return 0.5


def safe_ap(targets, probs):
    try:
        return average_precision_score(targets, probs) if len(set(targets)) > 1 else 0.0
    except Exception:
        return 0.0


def metrics_dict(labels, probs, threshold=0.5):
    preds = (np.asarray(probs) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auc": safe_auc(labels, probs),
        "pr_auc": safe_ap(labels, probs),
        "acc": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-dirs", nargs="+", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for seed_dir in args.seed_dirs:
        pred_path = Path(seed_dir) / "pooled_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing pooled predictions: {pred_path}")
        df = pd.read_csv(pred_path)
        required = {"filepath", "label", "prob"}
        if not required.issubset(df.columns):
            raise ValueError(f"{pred_path} must contain columns {sorted(required)}")
        df = df[["filepath", "label", "prob"]].copy()
        df.rename(columns={"prob": Path(seed_dir).name}, inplace=True)
        frames.append(df)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["filepath", "label"], how="inner")

    prob_cols = [c for c in merged.columns if c not in {"filepath", "label"}]
    merged["prob_mean"] = merged[prob_cols].mean(axis=1)
    merged.to_csv(output_dir / "seed_ensemble_predictions.csv", index=False)

    labels = merged["label"].values.astype(int)
    probs = merged["prob_mean"].values.astype(float)
    metrics = metrics_dict(labels, probs)
    if len(set(labels)) > 1:
        fpr, tpr, thr = roc_curve(labels, probs)
    else:
        fpr, tpr, thr = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    pd.DataFrame([metrics]).to_csv(output_dir / "seed_ensemble_metrics.csv", index=False)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
        output_dir / "seed_ensemble_roc.csv", index=False
    )
    print(
        f"Seed ensemble AUC={metrics['auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} ACC={metrics['acc']:.4f}"
    )
    print(f"Wrote ensemble results to {output_dir}")


if __name__ == "__main__":
    main()
