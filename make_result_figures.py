# Version 12 source snapshot
from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
SUITE_DIR = ROOT / "results_V12_suite"
CURRENT_FEATURE_DIR = ROOT / "results_v12_feature_extraction"
BAD_FEATURE_DIR = ROOT / "results_v12_feature_extraction_bad"
OUT_DIR = ROOT / "results_v12_figure_pack"
FIG_DIR = OUT_DIR / "figures"


MODEL_LABELS = {
    "clinical": "Clinical",
    "flow_geometry": "Flow + geometry",
    "geometry_clinical": "Geometry + clinical",
    "geometry_flow_clinical": "Geometry + flow + clinical",
    "geometry_pointnet2": "Geometry PointNet++",
    "gnn": "GNN",
}

MODEL_COLORS = {
    "Clinical": "#2F6C9F",
    "Flow + geometry": "#E17C05",
    "Geometry + clinical": "#008B8B",
    "Geometry + flow + clinical": "#7A4EA3",
    "Geometry PointNet++": "#B84A62",
    "GNN": "#4D4D4D",
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
    sns.set_theme(style="whitegrid", context="notebook")
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


def model_from_folder(path: Path) -> str:
    name = path.name
    if name.endswith("_seed_42"):
        name = name[: -len("_seed_42")]
    return name


def pretty_model(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " ").title())


def clean_feature_name(feature: str) -> str:
    replacements = {
        "clinical_location=": "location: ",
        "clinical_source=": "source: ",
        "clinical_side=": "side: ",
        "clinical_": "clinical: ",
        "hemo_": "hemo: ",
        "geometry_": "geom: ",
    }
    label = feature
    for old, new in replacements.items():
        label = label.replace(old, new)
    label = label.replace("_", " ")
    return label


def wrap_label(text: str, width: int = 24) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def mean_sd_text(row: pd.Series, metric: str, digits: int = 3) -> str:
    return f"{row[f'{metric}_mean']:.{digits}f} +/- {row[f'{metric}_sd']:.{digits}f}"


def safe_sem(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def save_figure(fig: plt.Figure, filename: str, pdf: PdfPages) -> Path:
    path = FIG_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return path


def load_suite_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_frames = []
    epoch_frames = []
    for model_dir in sorted(SUITE_DIR.glob("*_seed_42")):
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
    summary = pd.DataFrame(rows)
    return summary.sort_values("auc_mean", ascending=False).reset_index(drop=True)


def load_feature_run(path: Path, run_label: str) -> dict[str, pd.DataFrame]:
    result = {}
    for name in [
        "feature_importance_summary.csv",
        "modality_importance_summary.csv",
        "importance_cv_summary.csv",
    ]:
        result[name] = pd.read_csv(path / name)
        result[name]["run"] = run_label

    top_files = list(path.glob("condensed_case_features_top*.csv"))
    if top_files:
        result["condensed_top"] = pd.read_csv(top_files[0])
        result["condensed_top"]["run"] = run_label
    else:
        result["condensed_top"] = pd.DataFrame()
    return result


def feature_cv_summary(
    current: dict[str, pd.DataFrame], bad: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for label, data in [
        ("Current feature extraction", current),
        ("Historical bad feature extraction", bad),
    ]:
        cv = data["importance_cv_summary.csv"]
        rows.append(
            {
                "run": label,
                "folds": int(cv["fold"].nunique()),
                "auc_mean": float(cv["auc"].mean()),
                "auc_sd": float(cv["auc"].std(ddof=1)),
                "pr_auc_mean": float(cv["pr_auc"].mean()),
                "pr_auc_sd": float(cv["pr_auc"].std(ddof=1)),
                "n_val_mean": float(cv["n_val"].mean()),
                "n_val_total": int(cv["n_val"].sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_key_findings(
    model_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    current: dict[str, pd.DataFrame],
    bad: dict[str, pd.DataFrame],
    pdf: PdfPages,
) -> Path:
    best_auc = model_summary.iloc[0]
    best_pr = model_summary.sort_values("pr_auc_mean", ascending=False).iloc[0]
    current_cv = feature_summary.loc[feature_summary["run"].str.startswith("Current")].iloc[0]
    bad_cv = feature_summary.loc[feature_summary["run"].str.startswith("Historical")].iloc[0]
    top_feature = current["feature_importance_summary.csv"].iloc[0]
    top_modality = (
        current["modality_importance_summary.csv"]
        .sort_values("importance_score", ascending=False)
        .iloc[0]
    )

    cards = [
        (
            "Best v12 suite AUROC",
            f"{best_auc['model_label']}\n{best_auc['auc_mean']:.3f} +/- {best_auc['auc_sd']:.3f}",
            "#2F6C9F",
        ),
        (
            "Best v12 suite AUPRC",
            f"{best_pr['model_label']}\n{best_pr['pr_auc_mean']:.3f} +/- {best_pr['pr_auc_sd']:.3f}",
            "#7A4EA3",
        ),
        (
            "Feature extraction CV",
            f"Current: AUROC {current_cv['auc_mean']:.3f}\nBad run: AUROC {bad_cv['auc_mean']:.3f}",
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
    fig.suptitle("v12 Results Summary", fontsize=24, fontweight="bold", y=0.96)
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

    foot = (
        f"v12 suite: {int(model_summary['folds'].max())}-fold summaries from results_V12_suite; "
        f"current feature extraction uses mean validation n={current_cv['n_val_mean']:.0f} per fold; "
        f"historical bad run uses mean validation n={bad_cv['n_val_mean']:.0f} per fold."
    )
    fig.text(0.5, 0.055, foot, ha="center", va="center", fontsize=10, color="#555555")
    return save_figure(fig, "fig01_key_findings.png", pdf)


def plot_model_performance(folds: pd.DataFrame, summary: pd.DataFrame, pdf: PdfPages) -> Path:
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
    fig.suptitle("Model Performance Across Five Validation Folds", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "fig02_model_performance_auc_pr.png", pdf)


def plot_metric_heatmap(summary: pd.DataFrame, pdf: PdfPages) -> Path:
    metrics = ["auc", "pr_auc", "acc", "precision", "recall", "specificity", "f1"]
    heat = summary.set_index("model_label")[[f"{m}_mean" for m in metrics]].copy()
    heat.columns = [METRIC_LABELS[m] for m in metrics]
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    sns.heatmap(
        heat,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap=sns.color_palette("crest", as_cmap=True),
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean fold metric"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Mean Classification Metrics by Model", pad=14)
    fig.tight_layout()
    return save_figure(fig, "fig03_metric_heatmap.png", pdf)


def plot_fold_stability(folds: pd.DataFrame, summary: pd.DataFrame, pdf: PdfPages) -> Path:
    order = summary["model_label"].tolist()
    auc = folds.pivot_table(index="model_label", columns="fold", values="auc").loc[order]
    pr = folds.pivot_table(index="model_label", columns="fold", values="pr_auc").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), sharey=True)
    for ax, table, title, cmap in [
        (axes[0], auc, "AUROC by fold", "YlGnBu"),
        (axes[1], pr, "AUPRC by fold", "YlOrBr"),
    ]:
        sns.heatmap(
            table,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=0.3,
            vmax=0.85,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": title.split()[0]},
        )
        ax.set_xlabel("Fold")
        ax.set_ylabel("")
        ax.set_title(title)
    fig.suptitle("Fold-Level Stability", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "fig04_fold_stability_heatmaps.png", pdf)


def plot_training_curves(epochs: pd.DataFrame, summary: pd.DataFrame, pdf: PdfPages) -> Path:
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
    return save_figure(fig, "fig05_training_curves.png", pdf)


def plot_modality_importance(
    current: dict[str, pd.DataFrame], bad: dict[str, pd.DataFrame], pdf: PdfPages
) -> Path:
    modalities = pd.concat(
        [
            current["modality_importance_summary.csv"],
            bad["modality_importance_summary.csv"],
        ],
        ignore_index=True,
    )
    modalities["importance_per_feature"] = (
        modalities["importance_score"] / modalities["feature_count"]
    )
    modalities["run_short"] = modalities["run"].map(
        {
            "Current feature extraction": "Current",
            "Historical bad feature extraction": "Bad",
        }
    )
    order = ["hemodynamic", "clinical", "geometry"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=False)
    for ax, metric, title in [
        (axes[0], "importance_score", "Total composite importance"),
        (axes[1], "importance_per_feature", "Composite importance per feature"),
    ]:
        sns.barplot(
            data=modalities,
            x="modality",
            y=metric,
            hue="run_short",
            order=order,
            ax=ax,
            palette={"Current": "#2F6C9F", "Bad": "#C26D3A"},
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend(title="")
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle(
        "Modality Importance: Current vs Historical Bad Feature Extraction",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    return save_figure(fig, "fig06_modality_importance_current_vs_bad.png", pdf)


def plot_top_features(current: dict[str, pd.DataFrame], pdf: PdfPages) -> Path:
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
    return save_figure(fig, "fig07_top25_feature_importance.png", pdf)


def plot_feature_run_cv(feature_summary: pd.DataFrame, pdf: PdfPages) -> Path:
    plot_df = feature_summary.copy()
    plot_df["run_short"] = plot_df["run"].map(
        {
            "Current feature extraction": "Current",
            "Historical bad feature extraction": "Bad",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8))
    x = np.arange(len(plot_df))
    width = 0.34
    axes[0].bar(
        x - width / 2,
        plot_df["auc_mean"],
        yerr=plot_df["auc_sd"],
        width=width,
        label="AUROC",
        color="#2F6C9F",
        capsize=4,
    )
    axes[0].bar(
        x + width / 2,
        plot_df["pr_auc_mean"],
        yerr=plot_df["pr_auc_sd"],
        width=width,
        label="AUPRC",
        color="#E17C05",
        capsize=4,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["run_short"])
    axes[0].set_ylim(0.25, 1.05)
    axes[0].set_ylabel("Mean +/- fold SD")
    axes[0].set_title("Feature-extraction CV performance")
    axes[0].legend()
    for xi, nval in zip(x, plot_df["n_val_mean"]):
        axes[0].text(
            xi, 0.30, f"n/fold={nval:.0f}", ha="center", va="center", fontsize=9, color="#555555"
        )

    axes[1].bar(
        plot_df["run_short"], plot_df["n_val_total"], color=["#2F6C9F", "#C26D3A"], alpha=0.9
    )
    axes[1].set_title("Validation cases represented across folds")
    axes[1].set_ylabel("Total validation observations")
    for i, value in enumerate(plot_df["n_val_total"]):
        axes[1].text(
            i, value + max(plot_df["n_val_total"]) * 0.02, f"{value}", ha="center", fontsize=10
        )
    fig.suptitle(
        "Current Run Is More Stable Than the Historical Bad Feature Run",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    return save_figure(fig, "fig08_feature_extraction_current_vs_bad_cv.png", pdf)


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


def plot_case_feature_effects(current: dict[str, pd.DataFrame], pdf: PdfPages) -> Path:
    case_df = current["condensed_top"].copy()
    if case_df.empty:
        raise RuntimeError("No condensed top feature file found for current run.")

    feature_info = current["feature_importance_summary.csv"].copy()
    modality_map = dict(zip(feature_info["feature"], feature_info["modality"]))
    ignore = {"case_name", "dataset", "vesselFileID", "cutToShow", "target", "run"}
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
        effect = float((group1.mean() - group0.mean()) / pooled)
        rows.append(
            {
                "feature": feature,
                "effect": effect,
                "abs_effect": abs(effect),
                "modality": infer_modality(feature, modality_map),
                "mean_target_1": float(group1.mean()),
                "mean_target_0": float(group0.mean()),
            }
        )
    effects = pd.DataFrame(rows).sort_values("abs_effect", ascending=False).head(22)
    effects.to_csv(OUT_DIR / "current_top_feature_target_effects.csv", index=False)
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
    return save_figure(fig, "fig09_case_feature_target_effects.png", pdf)


def write_captions(
    model_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    current: dict[str, pd.DataFrame],
    bad: dict[str, pd.DataFrame],
    paths: list[Path],
) -> None:
    best_auc = model_summary.iloc[0]
    best_pr = model_summary.sort_values("pr_auc_mean", ascending=False).iloc[0]
    clinical = model_summary[model_summary["model_label"] == "Clinical"].iloc[0]
    gfc = model_summary[model_summary["model_label"] == "Geometry + flow + clinical"].iloc[0]
    top_modality = (
        current["modality_importance_summary.csv"]
        .sort_values("importance_score", ascending=False)
        .iloc[0]
    )
    top_features = current["feature_importance_summary.csv"].head(5)
    current_cv = feature_summary.loc[feature_summary["run"].str.startswith("Current")].iloc[0]
    bad_cv = feature_summary.loc[feature_summary["run"].str.startswith("Historical")].iloc[0]

    model_table = model_summary.copy()
    for metric in ["auc", "pr_auc", "acc", "precision", "recall", "specificity", "f1"]:
        model_table[METRIC_LABELS[metric]] = model_table.apply(
            lambda r: mean_sd_text(r, metric), axis=1
        )
    model_table = model_table[
        ["model_label", "folds", "n_val_total", "positive_prevalence"]
        + list(METRIC_LABELS.values())
    ]
    model_table.to_csv(OUT_DIR / "model_performance_summary.csv", index=False)
    feature_summary.to_csv(OUT_DIR / "feature_extraction_cv_summary.csv", index=False)

    top_feature_text = ", ".join(clean_feature_name(x) for x in top_features["feature"].tolist())
    relative_paths = [p.relative_to(OUT_DIR).as_posix() for p in paths]

    captions = [
        (
            "Figure 1. Key findings graphic.",
            "The v12 suite is summarized with the strongest cross-validated model, the strongest AUPRC model, the current-vs-historical feature-extraction comparison, and the dominant current feature/modality signals. "
            f"The best AUROC was {best_auc['model_label']} ({best_auc['auc_mean']:.3f} +/- {best_auc['auc_sd']:.3f}); the best AUPRC was {best_pr['model_label']} ({best_pr['pr_auc_mean']:.3f} +/- {best_pr['pr_auc_sd']:.3f}).",
        ),
        (
            "Figure 2. Mean AUROC and AUPRC by model.",
            f"Bars show five-fold means with fold standard deviation. {clinical['model_label']} reached AUROC {clinical['auc_mean']:.3f} +/- {clinical['auc_sd']:.3f} and AUPRC {clinical['pr_auc_mean']:.3f} +/- {clinical['pr_auc_sd']:.3f}; "
            f"{gfc['model_label']} reached AUROC {gfc['auc_mean']:.3f} +/- {gfc['auc_sd']:.3f} and AUPRC {gfc['pr_auc_mean']:.3f} +/- {gfc['pr_auc_sd']:.3f}. The dashed AUPRC line marks the positive-class prevalence baseline.",
        ),
        (
            "Figure 3. Mean classification metric heatmap.",
            "Rows compare v12 model families and columns summarize mean fold metrics. The heatmap makes the tradeoff visible: top AUROC models keep broadly similar accuracy, but differ in precision-recall behavior and specificity.",
        ),
        (
            "Figure 4. Fold-level AUROC and AUPRC stability.",
            "Each cell is one validation fold. Stable models show consistently warm cells across folds; unstable entries indicate fold-specific performance swings that can be hidden by a single mean.",
        ),
        (
            "Figure 5. Validation learning curves.",
            "Curves show fold-averaged validation AUROC and AUPRC across epochs with a standard-error band. These traces distinguish models that peak cleanly from models whose validation signal remains weak or volatile during training.",
        ),
        (
            "Figure 6. Modality importance in current and historical feature extraction.",
            f"The current run ranks {top_modality['modality']} highest by total composite importance (score {top_modality['importance_score']:.2f}), while the historical bad run is included as a cautionary comparison. "
            "The per-feature panel adjusts for different feature counts by modality.",
        ),
        (
            "Figure 7. Top current feature-importance signals.",
            f"The highest-ranked current features are {top_feature_text}. Color encodes modality, showing that the strongest ranked features are distributed across clinical and hemodynamic variables, with geometry contributing additional signal.",
        ),
        (
            "Figure 8. Current vs historical feature-extraction CV.",
            f"The current feature-extraction run used about {current_cv['n_val_mean']:.0f} validation cases per fold and achieved AUROC {current_cv['auc_mean']:.3f} +/- {current_cv['auc_sd']:.3f}; "
            f"the historical bad run used about {bad_cv['n_val_mean']:.0f} cases per fold and achieved AUROC {bad_cv['auc_mean']:.3f} +/- {bad_cv['auc_sd']:.3f}. The historical run therefore has visibly higher sampling instability.",
        ),
        (
            "Figure 9. Target-associated shifts among current top features.",
            "Bars show standardized mean differences for top-feature columns, computed as target 1 minus target 0. Positive values are higher in target-positive cases; negative values are higher in target-negative cases. This is descriptive association, not a causal effect estimate.",
        ),
    ]

    lines = [
        "# v12 Results Figure Pack",
        "",
        "## Source data",
        "",
        f"- Current feature extraction: `{CURRENT_FEATURE_DIR.relative_to(ROOT).as_posix()}`",
        f"- Historical bad feature extraction: `{BAD_FEATURE_DIR.relative_to(ROOT).as_posix()}`",
        f"- v12 model suite: `{SUITE_DIR.relative_to(ROOT).as_posix()}`",
        "",
        "## Main quantitative summary",
        "",
        f"- Best mean AUROC: {best_auc['model_label']} ({best_auc['auc_mean']:.3f} +/- {best_auc['auc_sd']:.3f}).",
        f"- Best mean AUPRC: {best_pr['model_label']} ({best_pr['pr_auc_mean']:.3f} +/- {best_pr['pr_auc_sd']:.3f}).",
        f"- Current feature-extraction CV: AUROC {current_cv['auc_mean']:.3f} +/- {current_cv['auc_sd']:.3f}; AUPRC {current_cv['pr_auc_mean']:.3f} +/- {current_cv['pr_auc_sd']:.3f}; mean n={current_cv['n_val_mean']:.0f} per fold.",
        f"- Historical bad feature-extraction CV: AUROC {bad_cv['auc_mean']:.3f} +/- {bad_cv['auc_sd']:.3f}; AUPRC {bad_cv['pr_auc_mean']:.3f} +/- {bad_cv['pr_auc_sd']:.3f}; mean n={bad_cv['n_val_mean']:.0f} per fold.",
        "",
        "## Figures and captions",
        "",
    ]

    for index, ((title, caption), rel_path) in enumerate(zip(captions, relative_paths), start=1):
        lines.extend(
            [
                f"### {title}",
                "",
                f"![{title}]({rel_path})",
                "",
                caption,
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "- Error bars in model figures are fold standard deviations, not external test-set confidence intervals.",
            "- The historical bad feature-extraction folder is intentionally separated from the current root-level feature-extraction outputs.",
            "- Target-associated feature shifts are descriptive univariate summaries and should be interpreted alongside cross-validated model performance.",
            "",
        ]
    )
    (OUT_DIR / "figure_captions.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    folds, epochs = load_suite_results()
    model_summary = summarize_models(folds)
    current = load_feature_run(CURRENT_FEATURE_DIR, "Current feature extraction")
    bad = load_feature_run(BAD_FEATURE_DIR, "Historical bad feature extraction")
    feat_summary = feature_cv_summary(current, bad)

    folds.to_csv(OUT_DIR / "suite_fold_metrics_long.csv", index=False)
    model_summary.to_csv(OUT_DIR / "suite_model_summary_raw.csv", index=False)

    figure_paths: list[Path] = []
    with PdfPages(OUT_DIR / "v12_results_figure_pack.pdf") as pdf:
        figure_paths.append(plot_key_findings(model_summary, feat_summary, current, bad, pdf))
        figure_paths.append(plot_model_performance(folds, model_summary, pdf))
        figure_paths.append(plot_metric_heatmap(model_summary, pdf))
        figure_paths.append(plot_fold_stability(folds, model_summary, pdf))
        figure_paths.append(plot_training_curves(epochs, model_summary, pdf))
        figure_paths.append(plot_modality_importance(current, bad, pdf))
        figure_paths.append(plot_top_features(current, pdf))
        figure_paths.append(plot_feature_run_cv(feat_summary, pdf))
        figure_paths.append(plot_case_feature_effects(current, pdf))

    write_captions(model_summary, feat_summary, current, bad, figure_paths)
    print(f"Wrote {len(figure_paths)} figures to {FIG_DIR}")
    print(f"Wrote captions to {OUT_DIR / 'figure_captions.md'}")
    print(f"Wrote PDF to {OUT_DIR / 'v12_results_figure_pack.pdf'}")


if __name__ == "__main__":
    main()
