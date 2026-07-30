# Version 14 source snapshot
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from base_trainer import classification_report_dict
from rupture_status import filter_predictions_to_known_status

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_ROOT = PROJECT_ROOT / "results_V14_server_hparams_target8192"
IMPROVED_ROOT = PROJECT_ROOT / "results_V14_improved_late_fusion_target8192"


SOURCES = {
    "geometry": BASE_ROOT / "geometry_pointnext_seed_42",
    "flow_geometry": BASE_ROOT / "flow_geometry_pointnext_seed_42",
    "clinical": BASE_ROOT / "clinical_seed_42",
    "geometry_clinical": BASE_ROOT / "geometry_clinical_pointnext_seed_42",
    "original_full": BASE_ROOT / "geometry_flow_clinical_pointnext_seed_42",
    "revised_full": IMPROVED_ROOT / "geometry_flow_clinical_pointnext_seed_42",
}


def normalize_key(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(chr(92), "/", regex=False)


def read_prediction_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"filepath", "label", "prob"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {sorted(missing)}")
    return frame


def load_predictions(
    run_dir: Path, name: str, folds: int, metadata_path: Path | None = None
) -> pd.DataFrame:
    pooled_path = run_dir / "pooled_predictions.csv"
    if pooled_path.is_file():
        result = read_prediction_file(pooled_path)
        if "fold" not in result.columns:
            raise ValueError(f"{pooled_path} is missing the fold column")
        result = result[["filepath", "label", "prob", "fold"]].copy()
    else:
        frames = []
        for fold in range(1, folds + 1):
            path = run_dir / f"fold_{fold}_completed_predictions.csv"
            if not path.is_file():
                raise FileNotFoundError(
                    f"No pooled predictions for {name} at {pooled_path}; missing legacy file {path}"
                )
            frame = read_prediction_file(path)[["filepath", "label", "prob"]].copy()
            frame["fold"] = fold
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True)

    result["fold"] = pd.to_numeric(result["fold"], errors="raise").astype(int)
    expected_folds = set(range(1, folds + 1))
    observed_folds = set(result["fold"].unique())
    if observed_folds != expected_folds:
        raise ValueError(
            f"{run_dir} contains folds {sorted(observed_folds)}; expected {sorted(expected_folds)}"
        )
    if metadata_path is not None:
        result, _ = filter_predictions_to_known_status(result, metadata_path)
    result["key"] = normalize_key(result["filepath"])
    return result[["key", "label", "fold", "prob"]].rename(columns={"prob": name})


def clipped_logit(probabilities, eps: float = 1e-6):
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0 - eps)
    return np.log(probabilities / (1.0 - probabilities))


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def evaluate_strategy(
    name: str, family: str, definition: str, data: pd.DataFrame, probabilities
) -> tuple[dict, list[dict]]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    pooled = classification_report_dict(data["label"], probabilities)
    fold_rows = []
    for fold in sorted(data["fold"].unique()):
        mask = data["fold"].to_numpy() == fold
        metrics = classification_report_dict(data.loc[mask, "label"], probabilities[mask])
        fold_rows.append({"strategy": name, "family": family, "fold": int(fold), **metrics})
    fold_frame = pd.DataFrame(fold_rows)
    summary = {
        "strategy": name,
        "family": family,
        "definition": definition,
        **pooled,
        "mean_fold_auc": float(fold_frame["auc"].mean()),
        "sd_fold_auc": float(fold_frame["auc"].std(ddof=1)),
        "min_fold_auc": float(fold_frame["auc"].min()),
        "mean_fold_pr_auc": float(fold_frame["pr_auc"].mean()),
        "min_fold_pr_auc": float(fold_frame["pr_auc"].min()),
    }
    return summary, fold_rows


def strategy_registry(data: pd.DataFrame):
    probabilities = {name: data[name].to_numpy(dtype=np.float64) for name in SOURCES}
    margins = {name: clipped_logit(values) for name, values in probabilities.items()}

    strategies = []
    for name in SOURCES:
        strategies.append((name, "baseline", name, probabilities[name]))

    for weight in (0.10, 0.20, 0.25, 0.30, 0.35, 0.40):
        anchor = 1.0 - weight
        strategies.append(
            (
                f"logit_gc_fg_w{weight:.2f}",
                "fixed_logit_blend",
                f"sigmoid({anchor:.2f}*logit(geometry_clinical)+{weight:.2f}*logit(flow_geometry))",
                sigmoid(anchor * margins["geometry_clinical"] + weight * margins["flow_geometry"]),
            )
        )
        strategies.append(
            (
                f"prob_gc_fg_w{weight:.2f}",
                "fixed_probability_blend",
                f"{anchor:.2f}*geometry_clinical+{weight:.2f}*flow_geometry",
                anchor * probabilities["geometry_clinical"]
                + weight * probabilities["flow_geometry"],
            )
        )

    for weight in (0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75):
        residual = margins["flow_geometry"] - margins["geometry"]
        strategies.append(
            (
                f"flow_residual_w{weight:.2f}",
                "geometry_subtracted_flow_residual",
                f"sigmoid(logit(geometry_clinical)+{weight:.2f}*(logit(flow_geometry)-logit(geometry)))",
                sigmoid(margins["geometry_clinical"] + weight * residual),
            )
        )

    for base_weight, extra_weight in ((0.10, 0.10), (0.15, 0.10), (0.15, 0.15), (0.20, 0.10)):
        more_confident = (
            np.abs(margins["flow_geometry"]) > np.abs(margins["geometry_clinical"])
        ).astype(float)
        weight = base_weight + extra_weight * more_confident
        strategies.append(
            (
                f"confidence_gate_b{base_weight:.2f}_e{extra_weight:.2f}",
                "label_free_confidence_gate",
                f"flow weight {base_weight:.2f}+{extra_weight:.2f} when abs(flow margin)>abs(anchor margin)",
                sigmoid(
                    (1.0 - weight) * margins["geometry_clinical"]
                    + weight * margins["flow_geometry"]
                ),
            )
        )

    for weight in (0.15, 0.25, 0.35):
        anchor = 1.0 - weight
        strategies.append(
            (
                f"logit_gc_revised_full_w{weight:.2f}",
                "revised_model_blend",
                f"sigmoid({anchor:.2f}*logit(geometry_clinical)+{weight:.2f}*logit(revised_full))",
                sigmoid(anchor * margins["geometry_clinical"] + weight * margins["revised_full"]),
            )
        )
    return strategies


def paired_bootstrap(
    data: pd.DataFrame,
    prediction_map: dict[str, np.ndarray],
    selected: list[str],
    samples: int,
    seed: int,
):
    labels = data["label"].to_numpy(dtype=int)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    anchor = prediction_map["geometry_clinical"]
    rng = np.random.default_rng(seed)
    rows = []
    for name in selected:
        candidate = prediction_map[name]
        auc_deltas, pr_deltas = [], []
        for _ in range(samples):
            indices = np.concatenate(
                [
                    rng.choice(positive, len(positive), replace=True),
                    rng.choice(negative, len(negative), replace=True),
                ]
            )
            metrics_candidate = classification_report_dict(labels[indices], candidate[indices])
            metrics_anchor = classification_report_dict(labels[indices], anchor[indices])
            auc_deltas.append(metrics_candidate["auc"] - metrics_anchor["auc"])
            pr_deltas.append(metrics_candidate["pr_auc"] - metrics_anchor["pr_auc"])
        rows.append(
            {
                "strategy": name,
                "comparison": "minus_geometry_clinical",
                "auc_delta": classification_report_dict(labels, candidate)["auc"]
                - classification_report_dict(labels, anchor)["auc"],
                "auc_ci_low": float(np.quantile(auc_deltas, 0.025)),
                "auc_ci_high": float(np.quantile(auc_deltas, 0.975)),
                "pr_auc_delta": classification_report_dict(labels, candidate)["pr_auc"]
                - classification_report_dict(labels, anchor)["pr_auc"],
                "pr_auc_ci_low": float(np.quantile(pr_deltas, 0.025)),
                "pr_auc_ci_high": float(np.quantile(pr_deltas, 0.975)),
                "bootstrap_samples": int(samples),
            }
        )
    return pd.DataFrame(rows)


def write_report(output_dir: Path, summary: pd.DataFrame, bootstrap: pd.DataFrame):
    top = summary.head(10)
    lines = [
        "# V14 Fusion Strategy Experiment Report",
        "",
        "## Scope",
        "",
        "This benchmark compares predefined, deployable fusion rules using the same 750 out-of-fold cases and five folds.",
        "The results are development-stage model-selection evidence, not independent external validation.",
        "",
        "## Ranking",
        "",
        "| Rank | Strategy | Family | AUROC | AUPRC | Mean fold AUROC | Worst fold AUROC | Accuracy | F1 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {row.strategy} | {row.family} | {row.auc:.4f} | {row.pr_auc:.4f} | "
            f"{row.mean_fold_auc:.4f} | {row.min_fold_auc:.4f} | {row.acc:.4f} | {row.f1:.4f} |"
        )
    lines.extend(["", "## Paired Bootstrap Versus Geometry + Clinical", ""])
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"- `{row.strategy}`: AUROC delta {row.auc_delta:.4f} "
            f"(95% CI {row.auc_ci_low:.4f} to {row.auc_ci_high:.4f}); "
            f"AUPRC delta {row.pr_auc_delta:.4f} "
            f"(95% CI {row.pr_auc_ci_low:.4f} to {row.pr_auc_ci_high:.4f})."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Flow contains complementary ranking information, but unconstrained joint fusion can dilute the stronger geometry-clinical signal.",
            "Fixed or constrained late fusion is therefore the preferred development direction.",
            "The selected rule and its weight must be locked before evaluation on an external or untouched validation cohort.",
            "",
        ]
    )
    (output_dir / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="ascii")


def main():
    global SOURCES
    parser = argparse.ArgumentParser(description="Benchmark V14 multimodal fusion strategies.")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results_V14_fusion_strategy_benchmark"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-path", type=Path, default=PROJECT_ROOT / "metadata.csv")
    parser.add_argument("--base-root", type=Path, default=BASE_ROOT)
    parser.add_argument("--improved-root", type=Path, default=IMPROVED_ROOT)
    args = parser.parse_args()

    SOURCES = {
        "geometry": args.base_root / "geometry_pointnext_seed_42",
        "flow_geometry": args.base_root / "flow_geometry_pointnext_seed_42",
        "clinical": args.base_root / "clinical_seed_42",
        "geometry_clinical": args.base_root / "geometry_clinical_pointnext_seed_42",
        "original_full": args.base_root / "geometry_flow_clinical_pointnext_seed_42",
        "revised_full": args.improved_root / "geometry_flow_clinical_pointnext_seed_42",
    }

    data = None
    for name, path in SOURCES.items():
        frame = load_predictions(path, name, args.folds, args.metadata_path)
        data = (
            frame
            if data is None
            else data.merge(frame, on=["key", "label", "fold"], how="inner", validate="one_to_one")
        )
    if data is None or len(data) != 735:
        raise RuntimeError(
            f"Expected 735 known-status aligned cases, found {0 if data is None else len(data)}"
        )

    summary_rows, fold_rows, prediction_map = [], [], {}
    for name, family, definition, probabilities in strategy_registry(data):
        summary, folds = evaluate_strategy(name, family, definition, data, probabilities)
        summary_rows.append(summary)
        fold_rows.extend(folds)
        prediction_map[name] = np.asarray(probabilities, dtype=np.float64)

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["mean_fold_auc", "min_fold_auc", "auc"], ascending=False)
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    selected = summary.loc[summary["family"] != "baseline", "strategy"].head(5).tolist()
    bootstrap = paired_bootstrap(data, prediction_map, selected, args.bootstrap_samples, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "strategy_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap_top_strategies.csv", index=False)
    predictions = data[["key", "label", "fold"]].copy()
    for name in selected:
        predictions[name] = prediction_map[name]
    predictions.to_csv(args.output_dir / "top_strategy_predictions.csv", index=False)
    write_report(args.output_dir, summary, bootstrap)

    print(
        summary.head(15)[
            [
                "rank",
                "strategy",
                "family",
                "auc",
                "pr_auc",
                "mean_fold_auc",
                "min_fold_auc",
                "acc",
                "f1",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote benchmark outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
