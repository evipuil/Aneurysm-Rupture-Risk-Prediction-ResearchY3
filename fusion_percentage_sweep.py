# Version 14 source snapshot
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from base_trainer import classification_report_dict
from fusion_strategy_benchmark import BASE_ROOT, clipped_logit, load_predictions, sigmoid

PROJECT_ROOT = Path(__file__).resolve().parent


def blend(anchor: np.ndarray, flow: np.ndarray, flow_weight: float, space: str) -> np.ndarray:
    anchor_weight = 1.0 - flow_weight
    if space == "probability":
        return anchor_weight * anchor + flow_weight * flow
    return sigmoid(anchor_weight * clipped_logit(anchor) + flow_weight * clipped_logit(flow))


def fold_metrics(labels: np.ndarray, folds: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        rows.append(
            {
                "fold": int(fold),
                "auc": roc_auc_score(labels[mask], probabilities[mask]),
                "pr_auc": average_precision_score(labels[mask], probabilities[mask]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_grid(
    data: pd.DataFrame, weights: np.ndarray
) -> tuple[pd.DataFrame, dict[tuple[str, float], np.ndarray]]:
    labels = data["label"].to_numpy(dtype=int)
    folds = data["fold"].to_numpy(dtype=int)
    anchor = data["geometry_clinical"].to_numpy(dtype=float)
    flow = data["flow_geometry"].to_numpy(dtype=float)
    rows = []
    predictions = {}
    for space in ("probability", "logit"):
        for weight in weights:
            weight = float(weight)
            probabilities = blend(anchor, flow, weight, space)
            predictions[(space, weight)] = probabilities
            per_fold = fold_metrics(labels, folds, probabilities)
            metrics = classification_report_dict(labels, probabilities)
            rows.append(
                {
                    "fusion_space": space,
                    "flow_weight": weight,
                    "geometry_clinical_weight": 1.0 - weight,
                    **metrics,
                    "mean_fold_auc": per_fold["auc"].mean(),
                    "sd_fold_auc": per_fold["auc"].std(ddof=1),
                    "min_fold_auc": per_fold["auc"].min(),
                    "mean_fold_pr_auc": per_fold["pr_auc"].mean(),
                    "min_fold_pr_auc": per_fold["pr_auc"].min(),
                }
            )
    return pd.DataFrame(rows), predictions


def choose_weight(
    labels: np.ndarray,
    folds: np.ndarray,
    predictions: dict[tuple[str, float], np.ndarray],
    space: str,
    weights: np.ndarray,
    held_out_fold: int,
) -> tuple[float, float, float]:
    candidates = []
    training_folds = [fold for fold in sorted(np.unique(folds)) if fold != held_out_fold]
    for weight in weights:
        weight = float(weight)
        probabilities = predictions[(space, weight)]
        aucs, pr_aucs = [], []
        for fold in training_folds:
            mask = folds == fold
            aucs.append(roc_auc_score(labels[mask], probabilities[mask]))
            pr_aucs.append(average_precision_score(labels[mask], probabilities[mask]))
        candidates.append(
            (float(np.mean(aucs)), float(np.mean(pr_aucs)), -abs(weight - 0.30), weight)
        )
    mean_auc, mean_pr_auc, _, selected_weight = max(candidates)
    return selected_weight, mean_auc, mean_pr_auc


def nested_selection(
    data: pd.DataFrame,
    weights: np.ndarray,
    predictions: dict[tuple[str, float], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = data["label"].to_numpy(dtype=int)
    folds = data["fold"].to_numpy(dtype=int)
    all_selection_rows = []
    nested_summaries = []
    nested_prediction_columns = data[["key", "label", "fold"]].copy()

    for space in ("probability", "logit"):
        nested_probabilities = np.zeros(len(data), dtype=float)
        for held_out_fold in sorted(np.unique(folds)):
            selected_weight, train_auc, train_pr_auc = choose_weight(
                labels, folds, predictions, space, weights, int(held_out_fold)
            )
            mask = folds == held_out_fold
            selected_probabilities = predictions[(space, selected_weight)]
            nested_probabilities[mask] = selected_probabilities[mask]
            held_auc = roc_auc_score(labels[mask], selected_probabilities[mask])
            held_pr_auc = average_precision_score(labels[mask], selected_probabilities[mask])
            all_selection_rows.append(
                {
                    "fusion_space": space,
                    "held_out_fold": int(held_out_fold),
                    "selected_flow_weight": selected_weight,
                    "selected_geometry_clinical_weight": 1.0 - selected_weight,
                    "selection_mean_fold_auc": train_auc,
                    "selection_mean_fold_pr_auc": train_pr_auc,
                    "held_out_auc": held_auc,
                    "held_out_pr_auc": held_pr_auc,
                }
            )
        metrics = classification_report_dict(labels, nested_probabilities)
        per_fold = fold_metrics(labels, folds, nested_probabilities)
        nested_summaries.append(
            {
                "fusion_space": space,
                **metrics,
                "mean_fold_auc": per_fold["auc"].mean(),
                "sd_fold_auc": per_fold["auc"].std(ddof=1),
                "min_fold_auc": per_fold["auc"].min(),
                "mean_fold_pr_auc": per_fold["pr_auc"].mean(),
                "min_fold_pr_auc": per_fold["pr_auc"].min(),
            }
        )
        nested_prediction_columns[f"{space}_nested_prob"] = nested_probabilities
    return (
        pd.DataFrame(all_selection_rows),
        pd.DataFrame(nested_summaries),
        nested_prediction_columns,
    )


def paired_bootstrap_comparisons(
    data: pd.DataFrame,
    predictions: dict[tuple[str, float], np.ndarray],
    samples: int,
    seed: int,
) -> pd.DataFrame:
    labels = data["label"].to_numpy(dtype=int)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    comparisons = [
        (
            "probability_30_minus_anchor",
            predictions[("probability", 0.30)],
            predictions[("probability", 0.00)],
        ),
        (
            "probability_33_minus_anchor",
            predictions[("probability", 0.33)],
            predictions[("probability", 0.00)],
        ),
        (
            "probability_33_minus_30",
            predictions[("probability", 0.33)],
            predictions[("probability", 0.30)],
        ),
    ]
    rng = np.random.default_rng(seed)
    rows = []
    for name, candidate, reference in comparisons:
        auc_delta = roc_auc_score(labels, candidate) - roc_auc_score(labels, reference)
        pr_delta = average_precision_score(labels, candidate) - average_precision_score(
            labels, reference
        )
        auc_samples, pr_samples = [], []
        for _ in range(samples):
            indices = np.concatenate(
                [
                    rng.choice(positive, len(positive), replace=True),
                    rng.choice(negative, len(negative), replace=True),
                ]
            )
            auc_samples.append(
                roc_auc_score(labels[indices], candidate[indices])
                - roc_auc_score(labels[indices], reference[indices])
            )
            pr_samples.append(
                average_precision_score(labels[indices], candidate[indices])
                - average_precision_score(labels[indices], reference[indices])
            )
        rows.append(
            {
                "comparison": name,
                "auc_delta": auc_delta,
                "auc_ci_low": np.quantile(auc_samples, 0.025),
                "auc_ci_high": np.quantile(auc_samples, 0.975),
                "pr_auc_delta": pr_delta,
                "pr_auc_ci_low": np.quantile(pr_samples, 0.025),
                "pr_auc_ci_high": np.quantile(pr_samples, 0.975),
                "bootstrap_samples": samples,
            }
        )
    return pd.DataFrame(rows)


def plot_sweep(output_dir: Path, sweep: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = {"probability": "#176B87", "logit": "#B5483A"}
    for space, frame in sweep.groupby("fusion_space"):
        frame = frame.sort_values("flow_weight")
        percentage = 100.0 * frame["flow_weight"]
        axes[0].plot(
            percentage, frame["auc"], label=space.title(), color=colors[space], linewidth=2
        )
        axes[1].plot(
            percentage, frame["pr_auc"], label=space.title(), color=colors[space], linewidth=2
        )
    for axis, label in zip(axes, ("AUROC", "AUPRC")):
        axis.axvline(
            30,
            color="#222222",
            linestyle="--",
            linewidth=1,
            label="Selected 30%" if label == "AUROC" else None,
        )
        axis.set_xlabel("Flow + Geometry weight (%)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(output_dir / "percentage_sweep.png", dpi=300)
    plt.close(fig)


def write_report(
    output_dir: Path,
    sweep: pd.DataFrame,
    selections: pd.DataFrame,
    nested: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    top = sweep.sort_values(["mean_fold_auc", "min_fold_auc", "auc"], ascending=False).head(15)
    lines = [
        "# V14 Fusion Percentage Sweep",
        "",
        "Flow + Geometry weights from 0% to 100% were tested in 1% increments against Geometry + Clinical.",
        "Both probability-space and logit-space blending were evaluated on the same 750 OOF predictions.",
        "",
        "## Top Fixed Percentages",
        "",
        "| Fusion | Flow | Geometry + Clinical | AUROC | AUPRC | Mean fold AUROC | Worst fold AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"| {row.fusion_space} | {100 * row.flow_weight:.0f}% | "
            f"{100 * row.geometry_clinical_weight:.0f}% | {row.auc:.4f} | {row.pr_auc:.4f} | "
            f"{row.mean_fold_auc:.4f} | {row.min_fold_auc:.4f} |"
        )
    lines.extend(["", "## Fold-Wise Percentage Selection", ""])
    for row in selections.itertuples(index=False):
        lines.append(
            f"- {row.fusion_space}, held-out fold {row.held_out_fold}: "
            f"{100 * row.selected_flow_weight:.0f}% flow, held-out AUROC {row.held_out_auc:.4f}."
        )
    lines.extend(["", "## Nested Selection Summary", ""])
    for row in nested.itertuples(index=False):
        lines.append(
            f"- {row.fusion_space}: AUROC {row.auc:.4f}, AUPRC {row.pr_auc:.4f}, "
            f"mean fold AUROC {row.mean_fold_auc:.4f}, worst fold AUROC {row.min_fold_auc:.4f}."
        )
    lines.extend(["", "## Paired Bootstrap Comparisons", ""])
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"- {row.comparison}: AUROC delta {row.auc_delta:.4f} "
            f"(95% CI {row.auc_ci_low:.4f} to {row.auc_ci_high:.4f}); "
            f"AUPRC delta {row.pr_auc_delta:.4f} "
            f"(95% CI {row.pr_auc_ci_low:.4f} to {row.pr_auc_ci_high:.4f})."
        )
    lines.extend(
        [
            "",
            "Fixed-grid rankings are exploratory because the same OOF cohort was used to compare percentages.",
            "The fold-wise analysis reduces direct weight-selection bias by choosing each held-out fold's percentage from the other four folds.",
            "The final percentage should be locked before untouched external validation.",
            "",
        ]
    )
    (output_dir / "PERCENTAGE_SWEEP_REPORT.md").write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep Geometry+Clinical and Flow+Geometry fusion percentages."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results_V14_fusion_percentage_sweep",
    )
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-path", type=Path, default=PROJECT_ROOT / "metadata.csv")
    parser.add_argument("--base-root", type=Path, default=BASE_ROOT)
    args = parser.parse_args()
    if not 0.0 < args.step <= 1.0:
        raise ValueError("--step must be in (0, 1]")

    geometry_clinical = load_predictions(
        args.base_root / "geometry_clinical_pointnext_seed_42",
        "geometry_clinical",
        args.folds,
        args.metadata_path,
    )
    flow_geometry = load_predictions(
        args.base_root / "flow_geometry_pointnext_seed_42",
        "flow_geometry",
        args.folds,
        args.metadata_path,
    )
    data = geometry_clinical.merge(
        flow_geometry,
        on=["key", "label", "fold"],
        how="inner",
        validate="one_to_one",
    )
    if len(data) != len(geometry_clinical) or len(data) != len(flow_geometry):
        raise RuntimeError("Base models do not contain identical OOF cases")

    weights = np.round(np.arange(0.0, 1.0 + args.step / 2.0, args.step), 10)
    weights = weights[weights <= 1.0]
    sweep, predictions = evaluate_grid(data, weights)
    selections, nested, nested_predictions = nested_selection(data, weights, predictions)
    bootstrap = paired_bootstrap_comparisons(data, predictions, args.bootstrap_samples, args.seed)
    ranked = sweep.sort_values(
        ["mean_fold_auc", "min_fold_auc", "auc"], ascending=False
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output_dir / "percentage_sweep.csv", index=False)
    selections.to_csv(args.output_dir / "foldwise_selected_percentages.csv", index=False)
    nested.to_csv(args.output_dir / "nested_selection_summary.csv", index=False)
    nested_predictions.to_csv(args.output_dir / "nested_selection_predictions.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap_comparisons.csv", index=False)
    plot_sweep(args.output_dir, sweep)
    write_report(args.output_dir, ranked, selections, nested, bootstrap)

    print(
        ranked.head(20)[
            [
                "rank",
                "fusion_space",
                "flow_weight",
                "geometry_clinical_weight",
                "auc",
                "pr_auc",
                "mean_fold_auc",
                "min_fold_auc",
                "acc",
                "f1",
            ]
        ].to_string(index=False)
    )
    print("\nFold-wise selected percentages:")
    print(selections.to_string(index=False))
    print("\nNested selection summary:")
    print(nested.to_string(index=False))
    print(f"Wrote percentage sweep outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
