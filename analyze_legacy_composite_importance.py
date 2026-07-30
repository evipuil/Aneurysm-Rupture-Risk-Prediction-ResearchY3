# Version 14 source snapshot
"""Restore the earlier feature-count-neutral composite importance method.

The feature set retains later stability fixes: translation-dependent centroids,
raw RRT summaries, and redundant shear-derived columns remain excluded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results_v13_feature_extraction_known_status_only" / "condensed_case_features.csv"
OUTPUT = ROOT / "results_V23_legacy_composite_importance_stable"
SEED = 42
COLORS = {"geometry": "#3B7A57", "hemodynamic": "#B33A3A", "clinical": "#6C757D"}


def modality(feature: str) -> str:
    if feature.startswith("geometry_"):
        return "geometry"
    if feature.startswith("hemo_"):
        return "hemodynamic"
    if feature.startswith("clinical_"):
        return "clinical"
    return "other"


def display_name(feature: str) -> str:
    return (
        feature.replace("geometry_", "")
        .replace("hemo_", "")
        .replace("clinical_", "")
        .replace("=", ": ")
        .replace("_", " ")
    )


def main() -> None:
    data = pd.read_csv(SOURCE)
    excluded = {
        "case_id",
        "target",
        "geometry_centroid_x",
        "geometry_centroid_y",
        "geometry_centroid_z",
        "hemo_combined_std",
    }
    excluded.update(
        column
        for column in data.columns
        if column.startswith("hemo_von_mises")
        or column.startswith("hemo_shear_ratio")
        or column.startswith("hemo_rrt")
        or column in {"hemo_tawss_von_mises_corr", "hemo_osi_von_mises_corr"}
    )
    features = [
        column for column in data.columns if column not in excluded and modality(column) != "other"
    ]
    X = data[features]
    y = data["target"].astype(int).to_numpy()

    coefficient_rows = []
    permutation_rows = []
    fold_rows = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(splitter.split(X, y), start=1):
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")),
            ]
        )
        model.fit(X.iloc[train], y[train])
        probability = model.predict_proba(X.iloc[test])[:, 1]
        fold_rows.append(
            {
                "fold": fold,
                "auroc": roc_auc_score(y[test], probability),
                "auprc": average_precision_score(y[test], probability),
            }
        )
        for feature, value in zip(features, model.named_steps["clf"].coef_[0]):
            coefficient_rows.append({"fold": fold, "feature": feature, "signed_coefficient": value})
        perm = permutation_importance(
            model,
            X.iloc[test],
            y[test],
            scoring="roc_auc",
            n_repeats=10,
            random_state=SEED + fold,
        )
        for feature, value in zip(features, perm.importances_mean):
            permutation_rows.append(
                {"fold": fold, "feature": feature, "permutation_decrease": value}
            )

    coefficients = pd.DataFrame(coefficient_rows)
    permutations = pd.DataFrame(permutation_rows)
    summary = (
        coefficients.groupby("feature", as_index=False)
        .agg(
            signed_coefficient=("signed_coefficient", "mean"),
            coefficient_importance=(
                "signed_coefficient",
                lambda values: float(np.mean(np.abs(values))),
            ),
        )
        .merge(
            permutations.groupby("feature", as_index=False)["permutation_decrease"].mean(),
            on="feature",
        )
    )
    summary["coefficient_percentile"] = summary["coefficient_importance"].rank(
        pct=True, method="average"
    )
    summary["permutation_percentile"] = summary["permutation_decrease"].rank(
        pct=True, method="average"
    )
    summary["composite_importance"] = (
        summary["coefficient_percentile"] + summary["permutation_percentile"]
    ) / 2.0
    summary["modality"] = summary["feature"].map(modality)
    summary["display_name"] = summary["feature"].map(display_name)
    summary = summary.sort_values("composite_importance", ascending=False)

    modality_summary = (
        summary.groupby("modality", as_index=False)
        .agg(
            feature_count=("feature", "count"),
            mean_composite_importance=("composite_importance", "mean"),
            summed_composite_importance=("composite_importance", "sum"),
        )
        .sort_values("mean_composite_importance", ascending=False)
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "feature_importance_summary.csv", index=False)
    modality_summary.to_csv(OUTPUT / "modality_importance_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT / "fold_performance.csv", index=False)

    top = summary.head(15).sort_values("composite_importance")
    mod = modality_summary.sort_values("mean_composite_importance")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.8, 1]})
    axes[0].barh(
        top["display_name"], top["composite_importance"], color=[COLORS[m] for m in top["modality"]]
    )
    axes[0].set_xlabel("Composite coefficient/permutation percentile score")
    axes[0].set_title("Feature-level composite importance")
    axes[1].barh(
        mod["modality"].str.title(),
        mod["mean_composite_importance"],
        color=[COLORS[m] for m in mod["modality"]],
    )
    axes[1].set_xlabel("Mean composite importance per retained feature")
    axes[1].set_title("Feature-count-neutral modality importance")
    fig.suptitle("Rupture-Risk Descriptor and Modality Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT / "fig_legacy_composite_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(modality_summary.to_string(index=False))
    print("\nTop features:")
    print(
        summary.head(15)[
            [
                "feature",
                "modality",
                "composite_importance",
                "signed_coefficient",
                "permutation_decrease",
            ]
        ].to_string(index=False)
    )
    print("\nFold performance:")
    print(pd.DataFrame(fold_rows).agg(["mean", "std"]).to_string())
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
