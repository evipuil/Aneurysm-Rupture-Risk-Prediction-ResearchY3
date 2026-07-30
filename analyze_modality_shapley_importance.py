# Version 14 source snapshot
"""Exact refit-based modality importance for clinical, geometry, and hemodynamics.

The analysis fits every modality coalition inside repeated outer cross-validation.
Exact Shapley values allocate held-out predictive performance across modalities;
leave-one-group-out (LOGO) and standalone gains are reported as sensitivity checks.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results_v13_feature_extraction_known_status_only" / "condensed_case_features.csv"
OUTPUT = ROOT / "results_V22_modality_shapley_importance"
MODALITIES = ("Clinical", "Hemodynamics", "Geometry")
COLORS = {"Clinical": "#6C757D", "Hemodynamics": "#B33A3A", "Geometry": "#3B7A57"}
SEED = 42
OUTER_SPLITS = 5
OUTER_REPEATS = 5
C_GRID = (0.01, 0.1, 1.0, 10.0)


def modality(feature: str) -> str:
    if feature.startswith("clinical_"):
        return "Clinical"
    if feature.startswith("hemo_"):
        return "Hemodynamics"
    if feature.startswith("geometry_"):
        return "Geometry"
    return "Other"


def stable_features(data: pd.DataFrame) -> list[str]:
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
    return [
        column
        for column in data.columns
        if column not in excluded and modality(column) in MODALITIES
    ]


def pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")),
        ]
    )


def powerset(items: tuple[str, ...]):
    for size in range(len(items) + 1):
        for subset in combinations(items, size):
            yield frozenset(subset)


def coalition_name(coalition: frozenset[str]) -> str:
    return (
        "Empty" if not coalition else " + ".join(name for name in MODALITIES if name in coalition)
    )


def shapley(values: dict[frozenset[str], float], player: str) -> float:
    others = tuple(name for name in MODALITIES if name != player)
    total = 0.0
    n = len(MODALITIES)
    for subset in powerset(others):
        weight = factorial(len(subset)) * factorial(n - len(subset) - 1) / factorial(n)
        total += weight * (values[subset | {player}] - values[subset])
    return float(total)


def main() -> None:
    data = pd.read_csv(SOURCE)
    features = stable_features(data)
    X = data[features].copy()
    y = data["target"].astype(int).to_numpy()
    columns = {
        name: [feature for feature in features if modality(feature) == name] for name in MODALITIES
    }
    coalitions = list(powerset(MODALITIES))

    outer = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=SEED,
    )
    coalition_rows: list[dict] = []
    importance_rows: list[dict] = []
    for outer_index, (train_index, test_index) in enumerate(outer.split(X, y)):
        repeat = outer_index // OUTER_SPLITS + 1
        fold = outer_index % OUTER_SPLITS + 1
        y_train = y[train_index]
        y_test = y[test_index]
        values_auroc: dict[frozenset[str], float] = {}
        values_auprc: dict[frozenset[str], float] = {}

        for coalition in coalitions:
            if not coalition:
                probability = np.full(len(test_index), float(np.mean(y_train)))
                best_c = np.nan
            else:
                selected = [
                    feature for name in MODALITIES if name in coalition for feature in columns[name]
                ]
                inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + outer_index)
                search = GridSearchCV(
                    pipeline(),
                    {"clf__C": C_GRID},
                    scoring="roc_auc",
                    cv=inner,
                    n_jobs=-1,
                    refit=True,
                )
                search.fit(X.iloc[train_index][selected], y_train)
                probability = search.predict_proba(X.iloc[test_index][selected])[:, 1]
                best_c = float(search.best_params_["clf__C"])

            auroc = float(roc_auc_score(y_test, probability))
            auprc = float(average_precision_score(y_test, probability))
            values_auroc[coalition] = auroc
            values_auprc[coalition] = auprc
            coalition_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "coalition": coalition_name(coalition),
                    "coalition_size": len(coalition),
                    "auroc": auroc,
                    "auprc": auprc,
                    "best_c": best_c,
                    "n_test": len(test_index),
                }
            )

        full = frozenset(MODALITIES)
        empty = frozenset()
        for name in MODALITIES:
            without = full - {name}
            alone = frozenset({name})
            importance_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "modality": name,
                    "shapley_auroc": shapley(values_auroc, name),
                    "shapley_auprc": shapley(values_auprc, name),
                    "logo_auroc": values_auroc[full] - values_auroc[without],
                    "logo_auprc": values_auprc[full] - values_auprc[without],
                    "standalone_auroc": values_auroc[alone] - values_auroc[empty],
                    "standalone_auprc": values_auprc[alone] - values_auprc[empty],
                    "full_auroc": values_auroc[full],
                    "full_auprc": values_auprc[full],
                }
            )

    coalition_table = pd.DataFrame(coalition_rows)
    importance_table = pd.DataFrame(importance_rows)
    summary = importance_table.groupby("modality", as_index=False).agg(
        shapley_auroc=("shapley_auroc", "mean"),
        shapley_auroc_sd=("shapley_auroc", "std"),
        shapley_auprc=("shapley_auprc", "mean"),
        shapley_auprc_sd=("shapley_auprc", "std"),
        logo_auroc=("logo_auroc", "mean"),
        logo_auprc=("logo_auprc", "mean"),
        standalone_auroc=("standalone_auroc", "mean"),
        standalone_auprc=("standalone_auprc", "mean"),
    )
    summary["order"] = summary["modality"].map({name: i for i, name in enumerate(MODALITIES)})
    summary = summary.sort_values("order").drop(columns="order")

    coalition_summary = coalition_table.groupby("coalition", as_index=False).agg(
        mean_auroc=("auroc", "mean"),
        sd_auroc=("auroc", "std"),
        mean_auprc=("auprc", "mean"),
        sd_auprc=("auprc", "std"),
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    coalition_table.to_csv(OUTPUT / "coalition_performance_by_fold.csv", index=False)
    coalition_summary.to_csv(OUTPUT / "coalition_performance_summary.csv", index=False)
    importance_table.to_csv(OUTPUT / "modality_importance_by_fold.csv", index=False)
    summary.to_csv(OUTPUT / "modality_importance_summary.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharey=True)
    labels = summary["modality"].tolist()[::-1]
    colors = [COLORS[name] for name in labels]
    for ax, column, title in [
        (axes[0], "shapley_auroc", "Exact Shapley contribution"),
        (axes[1], "logo_auroc", "Leave-one-group-out loss"),
        (axes[2], "standalone_auroc", "Standalone gain over null"),
    ]:
        values = summary.set_index("modality").loc[labels, column]
        ax.barh(labels, values, color=colors, alpha=0.9)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Held-out AUROC difference")
    figure.suptitle("Refit-Based Modality Importance")
    figure.tight_layout()
    figure.savefig(OUTPUT / "fig_refit_modality_importance.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(summary.to_string(index=False))
    print("\nCoalition performance:")
    print(coalition_summary.sort_values("mean_auroc", ascending=False).to_string(index=False))
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
