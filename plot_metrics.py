# Version 7 source snapshot
"""
Plot per-epoch metrics and pooled ROC curves for every v7 model.

Improvements vs v6:
- Dynamic model discovery: any subdirectory matching `results_*_v7/` is picked
  up, so adding a new model variant does not require editing this script.
- Extra panel: pooled ROC curves overlaid so the best-performing model is
  obvious at a glance.
- Writes a single summary CSV with best per-fold val AUC / Acc / PR-AUC per
  model for quick tabular comparison.
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

METRIC_PANELS = [
    ("val_auc", "Validation ROC-AUC"),
    ("val_pr_auc", "Validation PR-AUC"),
    ("val_acc", "Validation Accuracy"),
    ("val_loss", "Validation Loss"),
]


def collect_fold_metrics(root: Path):
    """Return a long-form DataFrame with columns (model, fold, epoch, <metric>...)."""
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not re.match(r"results_.*_v7$", d.name):
            continue
        model = d.name.replace("results_", "").replace("_v7", "")
        for fold_csv in glob.glob(str(d / "fold_*_metrics.csv")):
            fold_num_match = re.search(r"fold_(\d+)", os.path.basename(fold_csv))
            if not fold_num_match:
                continue
            try:
                df = pd.read_csv(fold_csv)
            except Exception as exc:
                print(f"  [warn] cannot read {fold_csv}: {exc}")
                continue
            df["model"] = model
            df["fold"] = int(fold_num_match.group(1))
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_pooled_rocs(root: Path):
    rocs = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not re.match(r"results_.*_v7$", d.name):
            continue
        roc_path = d / "pooled_roc.csv"
        if roc_path.exists():
            rocs[d.name.replace("results_", "").replace("_v7", "")] = pd.read_csv(roc_path)
    return rocs


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model, per-fold best val_auc plus corresponding acc and pr_auc."""
    out = []
    for (model, fold), chunk in df.groupby(["model", "fold"]):
        if "val_auc" not in chunk.columns or len(chunk) == 0:
            continue
        best_row = chunk.loc[chunk["val_auc"].idxmax()]
        out.append(
            {
                "model": model,
                "fold": fold,
                "best_epoch": int(best_row.get("epoch", -1)),
                "val_auc": float(best_row.get("val_auc", float("nan"))),
                "val_pr_auc": float(best_row.get("val_pr_auc", float("nan"))),
                "val_acc": float(best_row.get("val_acc", float("nan"))),
                "val_loss": float(best_row.get("val_loss", float("nan"))),
            }
        )
    return pd.DataFrame(out)


def plot_all(root: Path, out_path: Path):
    df = collect_fold_metrics(root)
    if df.empty:
        print("No fold_*_metrics.csv files found.")
        return
    rocs = collect_pooled_rocs(root)

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2)

    for idx, (metric, title) in enumerate(METRIC_PANELS):
        if metric not in df.columns:
            continue
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        sns.lineplot(data=df, x="epoch", y=metric, hue="model", ax=ax, errorbar="sd")
        ax.set_title(title, fontsize=14)
        if "auc" in metric or "acc" in metric:
            ax.set_ylim(bottom=0.3)

    ax = fig.add_subplot(gs[2, :])
    for model, roc in rocs.items():
        if {"fpr", "tpr"}.issubset(roc.columns) and len(roc) > 1:
            auc = np.trapz(roc["tpr"], roc["fpr"])
            ax.plot(roc["fpr"], roc["tpr"], label=f"{model} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Pooled ROC Curves")
    if rocs:
        ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    print(f"Saved comparison plot to {out_path}")

    summary = summarize(df)
    summary_path = out_path.parent / "model_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    pivot = (
        summary.groupby("model")[["val_auc", "val_pr_auc", "val_acc"]].agg(["mean", "std"]).round(4)
    )
    print("\nBest per-fold validation metrics (mean +/- std across folds):")
    print(pivot.to_string())
    pivot.to_csv(out_path.parent / "model_comparison_means.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default=".", help="Root containing results_*_v7 directories"
    )
    parser.add_argument("--out", type=str, default="model_comparison_plots.png")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_path = root / args.out
    plot_all(root, out_path)


if __name__ == "__main__":
    main()
