# Version 14 source snapshot
"""Leakage-safe grouped and feature-level permutation importance.

This replaces percentile-rank aggregation with directly interpretable held-out
AUROC decreases. For modality importance, every column in one modality is
permuted together using a shared row permutation, preserving correlations
within the permuted block. Existing feature-extraction results remain intact.
The redundant fluid-shear channel and every feature derived from it are
excluded. Raw RRT summaries are also excluded from descriptor-level ranking:
RRT is deterministically derived from TAWSS and OSI, and its inverse
denominator makes the standard deviation unstable near zero shear.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results_v13_feature_extraction_known_status_only" / "condensed_case_features.csv"
OUTPUT = ROOT / "results_V21_grouped_feature_importance_stable"
SEED = 42
N_SPLITS = 5
N_REPEATS = 20


def modality(feature: str) -> str:
    if feature.startswith("geometry_"):
        return "Geometry"
    if feature.startswith("hemo_"):
        return "Hemodynamics"
    if feature.startswith("clinical_"):
        return "Clinical"
    return "Other"


def display_name(feature: str) -> str:
    name = feature.replace("hemo_von_mises", "fluid_shear_surrogate")
    name = name.replace("hemo_", "").replace("geometry_", "").replace("clinical_", "")
    return name.replace("=", ": ").replace("_", " ")


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")),
        ]
    )


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    array = values.to_numpy(dtype=float)
    return (
        float(np.mean(array)),
        float(np.std(array, ddof=1)),
        float(np.quantile(array, 0.025)),
        float(np.quantile(array, 0.975)),
    )


def main() -> None:
    data = pd.read_csv(SOURCE)
    excluded = {
        "case_id",
        "target",
        "geometry_centroid_x",
        "geometry_centroid_y",
        "geometry_centroid_z",
    }
    excluded.update(
        column
        for column in data.columns
        if column.startswith("hemo_von_mises")
        or column.startswith("hemo_shear_ratio")
        or column.startswith("hemo_rrt")
        or column in {"hemo_tawss_von_mises_corr", "hemo_osi_von_mises_corr"}
    )
    features = [column for column in data.columns if column not in excluded]
    X = data[features].copy()
    y = data["target"].astype(int).to_numpy()
    groups = {
        name: [feature for feature in features if modality(feature) == name]
        for name in ["Geometry", "Hemodynamics", "Clinical"]
    }

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    grouped_rows = []
    feature_rows = []
    coefficient_rows = []
    fold_rows = []
    for fold, (train_index, validation_index) in enumerate(splitter.split(X, y), start=1):
        model = build_model()
        model.fit(X.iloc[train_index], y[train_index])
        validation = X.iloc[validation_index].reset_index(drop=True)
        validation_y = y[validation_index]
        validation_probability = model.predict_proba(validation)[:, 1]
        baseline = roc_auc_score(validation_y, validation_probability)
        baseline_auprc = average_precision_score(validation_y, validation_probability)
        fold_rows.append(
            {
                "fold": fold,
                "baseline_auroc": baseline,
                "baseline_auprc": baseline_auprc,
                "n_validation": len(validation),
            }
        )

        coefficients = model.named_steps["clf"].coef_[0]
        for feature, coefficient in zip(features, coefficients):
            coefficient_rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "modality": modality(feature),
                    "coefficient": coefficient,
                }
            )

        for repeat in range(N_REPEATS):
            rng = np.random.default_rng(SEED + fold * 1000 + repeat)
            for group_name, columns in groups.items():
                permuted = validation.copy()
                order = rng.permutation(len(permuted))
                permuted.loc[:, columns] = validation.loc[order, columns].to_numpy()
                score = roc_auc_score(validation_y, model.predict_proba(permuted)[:, 1])
                grouped_rows.append(
                    {
                        "fold": fold,
                        "repeat": repeat + 1,
                        "modality": group_name,
                        "baseline_auroc": baseline,
                        "permuted_auroc": score,
                        "auroc_decrease": baseline - score,
                        "feature_count": len(columns),
                    }
                )

            for feature in features:
                permuted = validation.copy()
                permuted[feature] = validation[feature].to_numpy()[rng.permutation(len(permuted))]
                score = roc_auc_score(validation_y, model.predict_proba(permuted)[:, 1])
                feature_rows.append(
                    {
                        "fold": fold,
                        "repeat": repeat + 1,
                        "feature": feature,
                        "modality": modality(feature),
                        "auroc_decrease": baseline - score,
                    }
                )

    grouped = pd.DataFrame(grouped_rows)
    feature_table = pd.DataFrame(feature_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    fold_table = pd.DataFrame(fold_rows)

    grouped_summary_rows = []
    for name, subset in grouped.groupby("modality"):
        mean, sd, lower, upper = mean_ci(subset["auroc_decrease"])
        grouped_summary_rows.append(
            {
                "modality": name,
                "feature_count": int(subset["feature_count"].iloc[0]),
                "mean_auroc_decrease": mean,
                "sd_auroc_decrease": sd,
                "empirical_95ci_lower": lower,
                "empirical_95ci_upper": upper,
            }
        )
    grouped_summary = pd.DataFrame(grouped_summary_rows).sort_values(
        "mean_auroc_decrease", ascending=False
    )

    feature_summary = (
        feature_table.groupby(["feature", "modality"], as_index=False)["auroc_decrease"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_auroc_decrease", "std": "sd_auroc_decrease"})
    )
    coefficient_summary = coefficients.groupby(["feature", "modality"], as_index=False).agg(
        mean_standardized_coefficient=("coefficient", "mean"),
        sd_standardized_coefficient=("coefficient", "std"),
        positive_fold_fraction=("coefficient", lambda values: float(np.mean(values > 0))),
    )
    feature_summary = feature_summary.merge(
        coefficient_summary, on=["feature", "modality"], how="left"
    )
    feature_summary["display_name"] = feature_summary["feature"].map(display_name)
    feature_summary = feature_summary.sort_values("mean_auroc_decrease", ascending=False)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(OUTPUT / "grouped_permutation_repeats.csv", index=False)
    grouped_summary.to_csv(OUTPUT / "grouped_permutation_summary.csv", index=False)
    feature_summary.to_csv(OUTPUT / "feature_permutation_summary.csv", index=False)
    coefficients.to_csv(OUTPUT / "standardized_coefficients_by_fold.csv", index=False)
    fold_table.to_csv(OUTPUT / "fold_performance.csv", index=False)

    colors = {"Geometry": "#3B7A57", "Hemodynamics": "#B33A3A", "Clinical": "#6C757D"}
    top = feature_summary.head(15).sort_values("mean_auroc_decrease")
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.8, 1]})
    axes[0].barh(
        top["display_name"],
        top["mean_auroc_decrease"],
        color=[colors.get(value, "#777777") for value in top["modality"]],
        alpha=0.9,
    )
    axes[0].axvline(0, color="#444444", linewidth=0.8)
    axes[0].set_xlabel("Held-out AUROC decrease after feature permutation")
    axes[0].set_title("Feature-level permutation importance")

    grouped_plot = grouped_summary.sort_values("mean_auroc_decrease")
    axes[1].barh(
        grouped_plot["modality"],
        grouped_plot["mean_auroc_decrease"],
        color=[colors.get(value, "#777777") for value in grouped_plot["modality"]],
        alpha=0.9,
    )
    axes[1].axvline(0, color="#444444", linewidth=0.8)
    axes[1].set_xlabel("Held-out AUROC decrease after block permutation")
    axes[1].set_title("Grouped modality importance")
    figure.suptitle("Rupture-Risk Descriptor and Modality Importance")
    figure.tight_layout()
    figure.savefig(OUTPUT / "fig_grouped_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(grouped_summary.to_string(index=False))
    print("\nTop feature-level AUROC decreases:")
    print(feature_summary.head(15).to_string(index=False))
    print(f"\nSaved additive outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
