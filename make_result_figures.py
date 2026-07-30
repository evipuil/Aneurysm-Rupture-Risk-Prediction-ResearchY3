# Version 14 source snapshot
from __future__ import annotations

import argparse
import math
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None


ROOT = Path(__file__).resolve().parent

MODEL_LABELS = {
    "cnn_voxel": "Voxel CNN",
    "cnn_voxel_flow": "Voxel CNN + flow",
    "cnn_voxel_geometry": "Voxel CNN geometry",
    "clinical": "Clinical",
    "flow_geometry_pointnext": "Flow + geometry PointNeXt",
    "geometry_clinical_pointnext": "Geometry + clinical PointNeXt",
    "geometry_flow_clinical_pointnext": "Geometry + flow + clinical PointNeXt",
    "geometry_pointnext": "Geometry PointNeXt",
    "flow_geometry": "Flow + geometry",
    "geometry_clinical": "Geometry + clinical",
    "geometry_flow_clinical": "Geometry + flow + clinical",
    "geometry_pointnet2": "Geometry PointNet++",
    "gnn": "GNN",
    "gnn_geometry": "GNN geometry",
}

MODEL_COLORS = {
    "Voxel CNN": "#6A7F21",
    "Voxel CNN + flow": "#6A7F21",
    "Voxel CNN geometry": "#9AA340",
    "Clinical": "#2F6C9F",
    "Flow + geometry PointNeXt": "#E17C05",
    "Geometry + clinical PointNeXt": "#008B8B",
    "Geometry + flow + clinical PointNeXt": "#7A4EA3",
    "Geometry PointNeXt": "#B84A62",
    "Flow + geometry": "#E17C05",
    "Geometry + clinical": "#008B8B",
    "Geometry + flow + clinical": "#7A4EA3",
    "Geometry PointNet++": "#B84A62",
    "GNN": "#4D4D4D",
    "GNN geometry": "#4D4D4D",
}

MODALITY_COLORS = {
    "clinical": "#2F6C9F",
    "hemodynamic": "#E17C05",
    "geometry": "#008B8B",
    "unknown": "#666666",
}

METRIC_LABELS = {
    "auc": "AUROC",
    "pr_auc": "AUPRC",
    "acc": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1": "F1",
}


def setup_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", context="notebook")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.edgecolor": "#B5B5B5",
            "grid.color": "#E6E6E6",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_heatmap(
    data: pd.DataFrame,
    ax: plt.Axes,
    fmt: str = ".2f",
    cmap: str = "YlGnBu",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
) -> None:
    if sns is not None:
        sns.heatmap(
            data,
            ax=ax,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": cbar_label} if cbar_label else None,
        )
        return

    values = data.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(data.columns)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index)
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    midpoint = (
        (vmin if vmin is not None else np.nanmin(values))
        + (vmax if vmax is not None else np.nanmax(values))
    ) / 2
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = values[row, col]
            color = "white" if value > midpoint else "#222222"
            ax.text(col, row, format(value, fmt), ha="center", va="center", color=color, fontsize=9)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)


def model_from_folder(path: Path) -> str:
    name = path.name
    if "_seed_" in name:
        name = name.rsplit("_seed_", 1)[0]
    return name


def pretty_model(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " ").title())


def clean_feature_name(feature: str) -> str:
    replacements = {
        "clinical_location=": "location: ",
        "clinical_side=": "side: ",
        "clinical_": "clinical: ",
        "hemo_": "hemo: ",
        "geometry_": "geom: ",
    }
    label = feature
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label.replace("_", " ")


def wrap_label(text: str, width: int = 24) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def safe_sem(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def mean_sd_text(row: pd.Series, metric: str, digits: int = 3) -> str:
    return f"{row[f'{metric}_mean']:.{digits}f} +/- {row[f'{metric}_sd']:.{digits}f}"


def save_figure(fig: plt.Figure, filename: str, out_dir: Path, pdf: PdfPages) -> Path:
    path = out_dir / "figures" / filename
    fig.savefig(path, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return path


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")


def load_suite_results(suite_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_frames = []
    epoch_frames = []
    for model_dir in sorted(suite_dir.glob("*_seed_*")):
        model = model_from_folder(model_dir)
        summary_path = model_dir / "fold_summary.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df.insert(0, "fold", range(len(df)))
            df.insert(0, "model", model)
            fold_frames.append(df)

        for log_path in sorted(model_dir.glob("fold_*_epoch_log.csv")):
            log = pd.read_csv(log_path)
            if "model" not in log.columns:
                log.insert(0, "model", model)
            epoch_frames.append(log)

    if not fold_frames:
        raise RuntimeError(f"No fold_summary.csv files found under {suite_dir}")
    if not epoch_frames:
        raise RuntimeError(f"No fold_*_epoch_log.csv files found under {suite_dir}")

    folds = pd.concat(fold_frames, ignore_index=True)
    epochs = pd.concat(epoch_frames, ignore_index=True)
    folds["model_label"] = folds["model"].map(pretty_model)
    for col in ["tn", "fp", "fn", "tp"]:
        folds[col] = pd.to_numeric(folds[col], errors="coerce")
    folds["specificity"] = folds["tn"] / (folds["tn"] + folds["fp"])
    folds["f1"] = np.where(
        (folds["precision"] + folds["recall"]) > 0,
        2 * folds["precision"] * folds["recall"] / (folds["precision"] + folds["recall"]),
        0.0,
    )
    folds["n_val"] = folds[["tn", "fp", "fn", "tp"]].sum(axis=1)
    folds["positives"] = folds["fn"] + folds["tp"]
    folds["negatives"] = folds["tn"] + folds["fp"]
    epochs["model_label"] = epochs["model"].map(pretty_model)
    return folds, epochs


def summarize_models(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auc", "pr_auc", "acc", "precision", "recall", "specificity", "f1"]
    if "loss" in folds.columns:
        metrics.append("loss")
    rows = []
    for model, group in folds.groupby("model_label", sort=False):
        row = {"model_label": model, "folds": int(group["fold"].nunique())}
        row["n_val_total"] = int(group["n_val"].sum())
        row["positive_prevalence"] = float(group["positives"].sum() / group["n_val"].sum())
        for metric in metrics:
            if metric in group.columns:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
                row[f"{metric}_sem"] = safe_sem(group[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc_mean", ascending=False).reset_index(drop=True)


def load_feature_run(feature_dir: Path) -> dict[str, pd.DataFrame]:
    required = [
        "feature_importance_summary.csv",
        "modality_importance_summary.csv",
        "importance_cv_summary.csv",
    ]
    result = {}
    for name in required:
        path = feature_dir / name
        require_file(path)
        result[name] = pd.read_csv(path)

    top_files = sorted(feature_dir.glob("condensed_case_features_top*.csv"))
    if top_files:
        result["condensed_top"] = pd.read_csv(top_files[-1])
    else:
        result["condensed_top"] = pd.DataFrame()
    return result


def feature_cv_summary(current: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cv = current["importance_cv_summary.csv"]
    return pd.DataFrame(
        [
            {
                "run": "Feature extraction",
                "folds": int(cv["fold"].nunique()),
                "auc_mean": float(cv["auc"].mean()),
                "auc_sd": float(cv["auc"].std(ddof=1)),
                "pr_auc_mean": float(cv["pr_auc"].mean()),
                "pr_auc_sd": float(cv["pr_auc"].std(ddof=1)),
                "n_val_mean": float(cv["n_val"].mean()),
                "n_val_total": int(cv["n_val"].sum()),
            }
        ]
    )


def plot_key_findings(
    model_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    current: dict[str, pd.DataFrame],
    version_label: str,
    suite_dir: Path,
    out_dir: Path,
    pdf: PdfPages,
) -> Path:
    best_auc = model_summary.iloc[0]
    best_pr = model_summary.sort_values("pr_auc_mean", ascending=False).iloc[0]
    current_cv = feature_summary.iloc[0]
    top_feature = current["feature_importance_summary.csv"].iloc[0]
    top_modality = (
        current["modality_importance_summary.csv"]
        .sort_values("importance_score", ascending=False)
        .iloc[0]
    )

    cards = [
        (
            f"Best {version_label} suite AUROC",
            f"{best_auc['model_label']}\n{best_auc['auc_mean']:.3f} +/- {best_auc['auc_sd']:.3f}",
            "#2F6C9F",
        ),
        (
            f"Best {version_label} suite AUPRC",
            f"{best_pr['model_label']}\n{best_pr['pr_auc_mean']:.3f} +/- {best_pr['pr_auc_sd']:.3f}",
            "#7A4EA3",
        ),
        (
            "Feature extraction CV",
            f"AUROC {current_cv['auc_mean']:.3f} +/- {current_cv['auc_sd']:.3f}\nAUPRC {current_cv['pr_auc_mean']:.3f} +/- {current_cv['pr_auc_sd']:.3f}",
            "#008B8B",
        ),
        (
            "Dominant feature signal",
            f"{clean_feature_name(str(top_feature['feature']))}\nTop modality: {top_modality['modality']}",
            "#E17C05",
        ),
    ]

    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.axis("off")
    fig.suptitle(f"{version_label} Results Summary", fontsize=24, fontweight="bold", y=0.96)
    fig.text(
        0.5,
        0.89,
        "Cross-validation performance, feature-importance stability, and modality-level signal",
        ha="center",
        va="center",
        fontsize=12,
        color="#4A4A4A",
    )

    positions = [(0.06, 0.52), (0.53, 0.52), (0.06, 0.18), (0.53, 0.18)]
    for (title, value, color), (x, y) in zip(cards, positions):
        box = FancyBboxPatch(
            (x, y),
            0.41,
            0.24,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor="#F7F8FA",
            edgecolor="#D8DCE2",
            linewidth=1.0,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(
            x + 0.025,
            y + 0.18,
            title,
            transform=ax.transAxes,
            fontsize=12,
            color=color,
            fontweight="bold",
        )
        ax.text(
            x + 0.025,
            y + 0.065,
            value,
            transform=ax.transAxes,
            fontsize=16,
            color="#222222",
            fontweight="bold",
            linespacing=1.25,
        )

    source_name = (
        suite_dir.relative_to(ROOT).as_posix() if suite_dir.is_relative_to(ROOT) else str(suite_dir)
    )
    foot = (
        f"{version_label} suite source: {source_name}; "
        f"{int(model_summary['folds'].max())}-fold summaries; "
        f"feature extraction mean validation n={current_cv['n_val_mean']:.0f} per fold."
    )
    fig.text(0.5, 0.055, foot, ha="center", va="center", fontsize=10, color="#555555")
    return save_figure(fig, "fig01_key_findings.png", out_dir, pdf)


def plot_model_performance(
    folds: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, pdf: PdfPages
) -> Path:
    order = summary["model_label"].tolist()[::-1]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.6), sharey=True)
    for ax, metric, title in zip(axes, ["auc", "pr_auc"], ["AUROC", "AUPRC"]):
        means = summary.set_index("model_label").loc[order, f"{metric}_mean"]
        sds = summary.set_index("model_label").loc[order, f"{metric}_sd"]
        colors = [MODEL_COLORS.get(label, "#666666") for label in order]
        ax.barh(y, means, xerr=sds, color=colors, alpha=0.88, capsize=4)
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.set_xlim(0.3, 0.9)
        ax.set_xlabel(f"{title} mean +/- fold SD")
        ax.set_title(title)
        for yi, value in zip(y, means):
            ax.text(value + 0.012, yi, f"{value:.3f}", va="center", fontsize=9)
        if metric == "pr_auc":
            prevalence = folds["positives"].sum() / folds["n_val"].sum()
            ax.axvline(prevalence, color="#A23B3B", linestyle="--", linewidth=1.2)
            ax.text(
                prevalence + 0.01,
                len(order) - 0.3,
                f"prevalence baseline {prevalence:.2f}",
                color="#A23B3B",
                fontsize=9,
            )
    fig.suptitle("Model Performance Across Validation Folds", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "fig02_model_performance_auc_pr.png", out_dir, pdf)


def plot_metric_heatmap(summary: pd.DataFrame, out_dir: Path, pdf: PdfPages) -> Path:
    metrics = ["auc", "pr_auc", "acc", "precision", "recall", "specificity", "f1"]
    heat = summary.set_index("model_label")[[f"{m}_mean" for m in metrics]].copy()
    heat.columns = [METRIC_LABELS[m] for m in metrics]
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    plot_heatmap(heat, ax=ax, cmap="YlGnBu", vmin=0.0, vmax=1.0, cbar_label="Mean fold metric")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Mean Classification Metrics by Model", pad=14)
    fig.tight_layout()
    return save_figure(fig, "fig03_metric_heatmap.png", out_dir, pdf)


def plot_fold_stability(
    folds: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, pdf: PdfPages
) -> Path:
    order = summary["model_label"].tolist()
    auc = folds.pivot_table(index="model_label", columns="fold", values="auc").loc[order]
    pr = folds.pivot_table(index="model_label", columns="fold", values="pr_auc").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), sharey=True)
    for ax, table, title, cmap in [
        (axes[0], auc, "AUROC by fold", "YlGnBu"),
        (axes[1], pr, "AUPRC by fold", "YlOrBr"),
    ]:
        plot_heatmap(table, ax=ax, cmap=cmap, vmin=0.3, vmax=0.85, cbar_label=title.split()[0])
        ax.set_xlabel("Fold")
        ax.set_ylabel("")
        ax.set_title(title)
    fig.suptitle("Fold-Level Stability", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "fig04_fold_stability_heatmaps.png", out_dir, pdf)


def plot_training_curves(
    epochs: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, pdf: PdfPages
) -> Path:
    top_order = summary["model_label"].tolist()
    curve = epochs.groupby(["model_label", "epoch"], as_index=False).agg(
        val_auc_mean=("val_auc", "mean"),
        val_auc_sem=("val_auc", safe_sem),
        val_pr_auc_mean=("val_pr_auc", "mean"),
        val_pr_auc_sem=("val_pr_auc", safe_sem),
        folds=("fold", "nunique"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharex=False)
    for ax, metric, ylabel in [
        (axes[0], "val_auc", "Validation AUROC"),
        (axes[1], "val_pr_auc", "Validation AUPRC"),
    ]:
        for label in top_order:
            data = curve[(curve["model_label"] == label) & (curve["folds"] >= 2)].sort_values(
                "epoch"
            )
            if data.empty:
                continue
            color = MODEL_COLORS.get(label, "#666666")
            mean_col = f"{metric}_mean"
            sem_col = f"{metric}_sem"
            ax.plot(data["epoch"], data[mean_col], label=label, color=color, linewidth=2)
            ax.fill_between(
                data["epoch"],
                data[mean_col] - data[sem_col],
                data[mean_col] + data[sem_col],
                color=color,
                alpha=0.10,
                linewidth=0,
            )
        ax.set_title(ylabel)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.25, 0.9)
    axes[1].legend(loc="lower right", fontsize=8, frameon=True)
    fig.suptitle("Validation Learning Curves Averaged Across Folds", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "fig05_training_curves.png", out_dir, pdf)


def plot_modality_importance(
    current: dict[str, pd.DataFrame], out_dir: Path, pdf: PdfPages
) -> Path:
    modalities = current["modality_importance_summary.csv"].copy()
    modalities["importance_per_feature"] = (
        modalities["importance_score"] / modalities["feature_count"]
    )
    order = [m for m in ["hemodynamic", "clinical", "geometry"] if m in set(modalities["modality"])]
    ordered = modalities.set_index("modality").loc[order].reset_index()
    colors = [MODALITY_COLORS.get(m, MODALITY_COLORS["unknown"]) for m in ordered["modality"]]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8))
    for ax, metric, title in [
        (axes[0], "importance_score", "Total composite importance"),
        (axes[1], "importance_per_feature", "Composite importance per feature"),
    ]:
        ax.bar(ordered["modality"], ordered[metric], color=colors, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        for i, value in enumerate(ordered[metric]):
            ax.text(
                i, value + ordered[metric].max() * 0.03, f"{value:.2f}", ha="center", fontsize=9
            )
    fig.suptitle("Current Modality Importance", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "fig06_modality_importance.png", out_dir, pdf)


def plot_top_features(current: dict[str, pd.DataFrame], out_dir: Path, pdf: PdfPages) -> Path:
    features = current["feature_importance_summary.csv"].head(25).copy()
    features["label"] = features["feature"].map(clean_feature_name).map(lambda s: wrap_label(s, 28))
    features = features.iloc[::-1]
    colors = [MODALITY_COLORS.get(m, MODALITY_COLORS["unknown"]) for m in features["modality"]]
    fig, ax = plt.subplots(figsize=(11, 9.8))
    ax.barh(features["label"], features["importance_score"], color=colors, alpha=0.9)
    ax.set_xlabel("Composite importance score")
    ax.set_ylabel("")
    ax.set_title("Top 25 Current Feature-Importance Signals")
    handles = [
        plt.Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=color, markersize=10, label=modality
        )
        for modality, color in MODALITY_COLORS.items()
        if modality != "unknown"
    ]
    ax.legend(handles=handles, title="Modality", loc="lower right")
    for ytick, value in enumerate(features["importance_score"]):
        ax.text(value + 0.01, ytick, f"{value:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, max(1.08, features["importance_score"].max() * 1.15))
    fig.tight_layout()
    return save_figure(fig, "fig07_top25_feature_importance.png", out_dir, pdf)


def plot_feature_cv_folds(current: dict[str, pd.DataFrame], out_dir: Path, pdf: PdfPages) -> Path:
    cv = current["importance_cv_summary.csv"].copy()
    cv["fold_label"] = cv["fold"].astype(str)
    summary = feature_cv_summary(current).iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8))
    x = np.arange(len(cv))
    width = 0.36
    axes[0].bar(x - width / 2, cv["auc"], width=width, label="AUROC", color="#2F6C9F")
    axes[0].bar(x + width / 2, cv["pr_auc"], width=width, label="AUPRC", color="#E17C05")
    axes[0].axhline(summary["auc_mean"], color="#2F6C9F", linestyle="--", linewidth=1)
    axes[0].axhline(summary["pr_auc_mean"], color="#E17C05", linestyle="--", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cv["fold_label"])
    axes[0].set_ylim(0.25, 0.9)
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Metric")
    axes[0].set_title("Feature-extraction CV by fold")
    axes[0].legend()

    axes[1].bar(cv["fold_label"], cv["n_val"], color="#008B8B", alpha=0.9)
    axes[1].set_title("Validation cases by fold")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Validation observations")
    for i, value in enumerate(cv["n_val"]):
        axes[1].text(i, value + max(cv["n_val"]) * 0.02, f"{int(value)}", ha="center", fontsize=9)
    fig.suptitle("Feature-Extraction Cross-Validation Stability", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "fig08_feature_extraction_cv_by_fold.png", out_dir, pdf)


def infer_modality(feature: str, mapping: dict[str, str]) -> str:
    if feature in mapping:
        return mapping[feature]
    if feature.startswith("hemo_"):
        return "hemodynamic"
    if feature.startswith("geometry_"):
        return "geometry"
    if feature.startswith("clinical_"):
        return "clinical"
    return "unknown"


def plot_case_feature_effects(
    current: dict[str, pd.DataFrame], out_dir: Path, pdf: PdfPages
) -> Path:
    case_df = current["condensed_top"].copy()
    if case_df.empty:
        raise RuntimeError("No condensed top feature file found for current run.")

    feature_info = current["feature_importance_summary.csv"].copy()
    modality_map = dict(zip(feature_info["feature"], feature_info["modality"]))
    ignore = {"case_id", "case_name", "dataset", "vesselFileID", "cutToShow", "target", "run"}
    feature_cols = [c for c in case_df.columns if c not in ignore]
    rows = []
    for feature in feature_cols:
        values = pd.to_numeric(case_df[feature], errors="coerce")
        target = pd.to_numeric(case_df["target"], errors="coerce")
        group0 = values[target == 0].dropna()
        group1 = values[target == 1].dropna()
        if len(group0) < 2 or len(group1) < 2:
            continue
        pooled = math.sqrt(
            ((len(group0) - 1) * group0.var(ddof=1) + (len(group1) - 1) * group1.var(ddof=1))
            / (len(group0) + len(group1) - 2)
        )
        if pooled == 0 or np.isnan(pooled):
            continue
        rows.append(
            {
                "feature": feature,
                "effect": float((group1.mean() - group0.mean()) / pooled),
                "abs_effect": abs(float((group1.mean() - group0.mean()) / pooled)),
                "modality": infer_modality(feature, modality_map),
                "mean_target_1": float(group1.mean()),
                "mean_target_0": float(group0.mean()),
            }
        )
    effects = pd.DataFrame(rows).sort_values("abs_effect", ascending=False).head(22)
    effects.to_csv(out_dir / "current_top_feature_target_effects.csv", index=False)
    effects = effects.iloc[::-1]
    labels = [wrap_label(clean_feature_name(f), 27) for f in effects["feature"]]
    colors = [MODALITY_COLORS.get(m, MODALITY_COLORS["unknown"]) for m in effects["modality"]]
    fig, ax = plt.subplots(figsize=(11.2, 9.0))
    ax.barh(labels, effects["effect"], color=colors, alpha=0.9)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Standardized mean difference: target 1 minus target 0")
    ax.set_title("Largest Target-Associated Shifts Among Current Top Features")
    handles = [
        plt.Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=color, markersize=10, label=modality
        )
        for modality, color in MODALITY_COLORS.items()
        if modality != "unknown"
    ]
    ax.legend(handles=handles, title="Modality", loc="lower right")
    fig.tight_layout()
    return save_figure(fig, "fig09_case_feature_target_effects.png", out_dir, pdf)


def write_captions(
    model_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    current: dict[str, pd.DataFrame],
    paths: list[Path],
    suite_dir: Path,
    feature_dir: Path,
    out_dir: Path,
    version_label: str,
) -> None:
    best_auc = model_summary.iloc[0]
    best_pr = model_summary.sort_values("pr_auc_mean", ascending=False).iloc[0]
    clinical = model_summary[model_summary["model_label"] == "Clinical"].iloc[0]
    gfc_rows = model_summary[
        model_summary["model_label"].isin(
            [
                "Geometry + flow + clinical PointNeXt",
                "Geometry + flow + clinical",
            ]
        )
    ]
    gfc = gfc_rows.iloc[0]
    top_modality = (
        current["modality_importance_summary.csv"]
        .sort_values("importance_score", ascending=False)
        .iloc[0]
    )
    top_features = current["feature_importance_summary.csv"].head(5)
    current_cv = feature_summary.iloc[0]

    model_table = model_summary.copy()
    for metric in ["auc", "pr_auc", "acc", "precision", "recall", "specificity", "f1"]:
        model_table[METRIC_LABELS[metric]] = model_table.apply(
            lambda r: mean_sd_text(r, metric), axis=1
        )
    model_table = model_table[
        ["model_label", "folds", "n_val_total", "positive_prevalence"]
        + list(METRIC_LABELS.values())
    ]
    model_table.to_csv(out_dir / "model_performance_summary.csv", index=False)
    feature_summary.to_csv(out_dir / "feature_extraction_cv_summary.csv", index=False)

    top_feature_text = ", ".join(clean_feature_name(x) for x in top_features["feature"].tolist())
    relative_paths = [p.relative_to(out_dir).as_posix() for p in paths]

    captions = [
        (
            "Figure 1. Key findings graphic.",
            f"The {version_label} summary highlights the strongest cross-validated model, the strongest AUPRC model, the current feature-extraction cross-validation result, and the dominant current feature/modality signals. "
            f"The best AUROC was {best_auc['model_label']} ({best_auc['auc_mean']:.3f} +/- {best_auc['auc_sd']:.3f}); the best AUPRC was {best_pr['model_label']} ({best_pr['pr_auc_mean']:.3f} +/- {best_pr['pr_auc_sd']:.3f}).",
        ),
        (
            "Figure 2. Mean AUROC and AUPRC by model.",
            f"Bars show fold means with fold standard deviation. {clinical['model_label']} reached AUROC {clinical['auc_mean']:.3f} +/- {clinical['auc_sd']:.3f} and AUPRC {clinical['pr_auc_mean']:.3f} +/- {clinical['pr_auc_sd']:.3f}; "
            f"{gfc['model_label']} reached AUROC {gfc['auc_mean']:.3f} +/- {gfc['auc_sd']:.3f} and AUPRC {gfc['pr_auc_mean']:.3f} +/- {gfc['pr_auc_sd']:.3f}. The dashed AUPRC line marks the positive-class prevalence baseline.",
        ),
        (
            "Figure 3. Mean classification metric heatmap.",
            "Rows compare model families and columns summarize mean fold metrics. The heatmap shows the tradeoff between discrimination, precision-recall behavior, sensitivity, and specificity.",
        ),
        (
            "Figure 4. Fold-level AUROC and AUPRC stability.",
            "Each cell is one validation fold. Consistently warm rows indicate stable models; isolated low cells show fold-specific performance swings hidden by a single mean.",
        ),
        (
            "Figure 5. Validation learning curves.",
            "Curves show fold-averaged validation AUROC and AUPRC across epochs with a standard-error band, separating clean training signals from volatile validation behavior.",
        ),
        (
            "Figure 6. Current modality importance.",
            f"The current run ranks {top_modality['modality']} highest by total composite importance (score {top_modality['importance_score']:.2f}). The per-feature panel adjusts for different feature counts by modality.",
        ),
        (
            "Figure 7. Top current feature-importance signals.",
            f"The highest-ranked current features are {top_feature_text}. Color encodes modality, showing which feature families dominate the ranked list.",
        ),
        (
            "Figure 8. Feature-extraction CV by fold.",
            f"The current feature-extraction run used about {current_cv['n_val_mean']:.0f} validation cases per fold and achieved AUROC {current_cv['auc_mean']:.3f} +/- {current_cv['auc_sd']:.3f}; AUPRC {current_cv['pr_auc_mean']:.3f} +/- {current_cv['pr_auc_sd']:.3f}.",
        ),
        (
            "Figure 9. Target-associated shifts among current top features.",
            "Bars show standardized mean differences for top-feature columns, computed as target 1 minus target 0. Positive values are higher in target-positive cases; negative values are higher in target-negative cases. This is descriptive association, not a causal effect estimate.",
        ),
    ]

    suite_text = (
        suite_dir.relative_to(ROOT).as_posix() if suite_dir.is_relative_to(ROOT) else str(suite_dir)
    )
    feature_text = (
        feature_dir.relative_to(ROOT).as_posix()
        if feature_dir.is_relative_to(ROOT)
        else str(feature_dir)
    )
    lines = [
        f"# {version_label} Results Figure Pack",
        "",
        "## Source data",
        "",
        f"- Feature extraction: `{feature_text}`",
        f"- Model suite: `{suite_text}`",
        "",
        "## Main quantitative summary",
        "",
        f"- Best mean AUROC: {best_auc['model_label']} ({best_auc['auc_mean']:.3f} +/- {best_auc['auc_sd']:.3f}).",
        f"- Best mean AUPRC: {best_pr['model_label']} ({best_pr['pr_auc_mean']:.3f} +/- {best_pr['pr_auc_sd']:.3f}).",
        f"- Feature-extraction CV: AUROC {current_cv['auc_mean']:.3f} +/- {current_cv['auc_sd']:.3f}; AUPRC {current_cv['pr_auc_mean']:.3f} +/- {current_cv['pr_auc_sd']:.3f}; mean n={current_cv['n_val_mean']:.0f} per fold.",
        "",
        "## Figures and captions",
        "",
    ]

    for (title, caption), rel_path in zip(captions, relative_paths, strict=True):
        lines.extend([f"### {title}", "", f"![{title}]({rel_path})", "", caption, ""])

    lines.extend(
        [
            "## Notes",
            "",
            "- Error bars in model figures are fold standard deviations, not external test-set confidence intervals.",
            f"- This {version_label} report focuses only on the selected current run.",
            "- Target-associated feature shifts are descriptive univariate summaries and should be interpreted alongside cross-validated model performance.",
            "",
        ]
    )
    (out_dir / "figure_captions.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a current-run-only V14 results figure pack."
    )
    parser.add_argument(
        "--suite-dir",
        default=str(ROOT / "results_V14_suite"),
        help="Folder containing *_seed_* model result folders.",
    )
    parser.add_argument(
        "--feature-dir",
        default=str(ROOT / "results_v14_feature_extraction"),
        help="Folder containing feature importance CSV outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results_v14_figure_pack"),
        help="Output folder for figures, CSV summaries, captions, and PDF.",
    )
    parser.add_argument(
        "--version-label", default="V14", help="Label used in figure titles and captions."
    )
    return parser.parse_args()


def validate_version_provenance(version_label: str, *source_dirs: Path) -> None:
    """Reject source folders that identify a different version than the report."""
    label_match = re.fullmatch(r"v(?:ersion)?\s*(\d+)", version_label.strip(), re.IGNORECASE)
    if label_match is None:
        return

    expected = int(label_match.group(1))
    for source_dir in source_dirs:
        versions = {
            int(match)
            for part in source_dir.parts
            for match in re.findall(r"(?:version|v)[_-]?(\d+)", part, re.IGNORECASE)
        }
        if versions and versions != {expected}:
            found = ", ".join(f"V{version}" for version in sorted(versions))
            raise ValueError(
                f"{source_dir} identifies {found}, but --version-label is {version_label}. "
                "Pass a matching --version-label or select inputs from the requested version."
            )


def main() -> None:
    args = parse_args()
    setup_style()

    suite_dir = Path(args.suite_dir).resolve()
    feature_dir = Path(args.feature_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    validate_version_provenance(args.version_label, suite_dir, feature_dir)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    folds, epochs = load_suite_results(suite_dir)
    model_summary = summarize_models(folds)
    current = load_feature_run(feature_dir)
    feat_summary = feature_cv_summary(current)

    folds.to_csv(out_dir / "suite_fold_metrics_long.csv", index=False)
    model_summary.to_csv(out_dir / "suite_model_summary_raw.csv", index=False)

    figure_paths: list[Path] = []
    pdf_path = out_dir / f"{args.version_label.lower()}_results_figure_pack.pdf"
    with PdfPages(pdf_path) as pdf:
        figure_paths.append(
            plot_key_findings(
                model_summary, feat_summary, current, args.version_label, suite_dir, out_dir, pdf
            )
        )
        figure_paths.append(plot_model_performance(folds, model_summary, out_dir, pdf))
        figure_paths.append(plot_metric_heatmap(model_summary, out_dir, pdf))
        figure_paths.append(plot_fold_stability(folds, model_summary, out_dir, pdf))
        figure_paths.append(plot_training_curves(epochs, model_summary, out_dir, pdf))
        figure_paths.append(plot_modality_importance(current, out_dir, pdf))
        figure_paths.append(plot_top_features(current, out_dir, pdf))
        figure_paths.append(plot_feature_cv_folds(current, out_dir, pdf))
        figure_paths.append(plot_case_feature_effects(current, out_dir, pdf))

    write_captions(
        model_summary,
        feat_summary,
        current,
        figure_paths,
        suite_dir,
        feature_dir,
        out_dir,
        args.version_label,
    )
    print(f"Wrote {len(figure_paths)} figures to {fig_dir}")
    print(f"Wrote captions to {out_dir / 'figure_captions.md'}")
    print(f"Wrote PDF to {pdf_path}")


if __name__ == "__main__":
    main()
