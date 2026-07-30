# Version 14 source snapshot
from __future__ import annotations

from pathlib import Path

import pandas as pd

from base_trainer import classification_report_dict
from rupture_status import (
    build_case_status_index,
    filter_predictions_to_known_status,
    normalize_rupture_status,
)

ROOT = Path(__file__).resolve().parent
METADATA = ROOT / "metadata.csv"
OUTPUT = ROOT / "results_V14_unknown_status_audit"

RUNS = {
    "Geometry": ROOT
    / "results_V14_server_hparams_target8192"
    / "geometry_pointnext_seed_42"
    / "pooled_predictions.csv",
    "Flow + Geometry": ROOT
    / "results_V14_server_hparams_target8192"
    / "flow_geometry_pointnext_seed_42"
    / "pooled_predictions.csv",
    "Clinical": ROOT
    / "results_V14_server_hparams_target8192"
    / "clinical_seed_42"
    / "pooled_predictions.csv",
    "Geometry + Clinical": ROOT
    / "results_V14_server_hparams_target8192"
    / "geometry_clinical_pointnext_seed_42"
    / "pooled_predictions.csv",
    "Original Joint Fusion": ROOT
    / "results_V14_server_hparams_target8192"
    / "geometry_flow_clinical_pointnext_seed_42"
    / "pooled_predictions.csv",
    "Revised Joint Fusion": ROOT
    / "results_V14_improved_late_fusion_target8192"
    / "geometry_flow_clinical_pointnext_seed_42"
    / "pooled_predictions.csv",
    "70/30 Late Fusion": ROOT
    / "results_V14_improved_probability_ensemble_target8192"
    / "pooled_predictions.csv",
}


def metric_row(model: str, scope: str, frame: pd.DataFrame) -> dict:
    metrics = classification_report_dict(frame["label"], frame["prob"])
    return {"model": model, "scope": scope, "n": len(frame), **metrics}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata, _ = build_case_status_index(METADATA)
    normalized = metadata["status"].map(normalize_rupture_status)
    unknown = metadata.loc[normalized.isna()].copy()
    unknown.insert(0, "metadata_row", unknown.index + 2)
    unknown.to_csv(OUTPUT / "unknown_status_cases.csv", index=False)
    metadata.loc[normalized.notna()].to_csv(OUTPUT / "metadata_known_status_only.csv", index=False)

    counts = pd.DataFrame(
        [
            {"status": "ruptured", "count": int((normalized == "ruptured").sum())},
            {"status": "unruptured", "count": int((normalized == "unruptured").sum())},
            {"status": "unknown_or_blank", "count": int(normalized.isna().sum())},
            {"status": "total", "count": int(len(metadata))},
        ]
    )
    counts.to_csv(OUTPUT / "status_counts.csv", index=False)

    metrics_rows = []
    unknown_score_rows = []
    known_prediction_dir = OUTPUT / "known_status_predictions"
    known_prediction_dir.mkdir(parents=True, exist_ok=True)
    for model, path in RUNS.items():
        frame = pd.read_csv(path)
        if "prob" not in frame.columns:
            raise ValueError(f"{path} has no probability column")
        known, excluded = filter_predictions_to_known_status(frame, METADATA)
        slug = model.lower().replace(" + ", "_").replace("/", "_").replace(" ", "_")
        known.to_csv(known_prediction_dir / f"{slug}.csv", index=False)
        metrics_rows.append(metric_row(model, "original_unknown_as_negative", frame))
        metrics_rows.append(metric_row(model, "known_status_only_posthoc", known))
        for row in excluded.itertuples(index=False):
            unknown_score_rows.append(
                {
                    "model": model,
                    "case_name": getattr(row, "case_name"),
                    "filepath": getattr(row, "filepath"),
                    "assigned_label_in_original": int(getattr(row, "label")),
                    "prob": float(getattr(row, "prob")),
                    "pred_at_0_5": int(float(getattr(row, "prob")) >= 0.5),
                }
            )

    metrics = pd.DataFrame(metrics_rows)
    original = metrics[metrics["scope"] == "original_unknown_as_negative"].set_index("model")
    corrected = metrics[metrics["scope"] == "known_status_only_posthoc"].set_index("model")
    for metric in [
        "auc",
        "pr_auc",
        "acc",
        "precision",
        "recall",
        "specificity",
        "balanced_acc",
        "f1",
    ]:
        delta = corrected[metric] - original[metric]
        metrics.loc[metrics["scope"] == "known_status_only_posthoc", f"delta_{metric}"] = (
            metrics.loc[metrics["scope"] == "known_status_only_posthoc", "model"].map(delta)
        )
    metrics.to_csv(OUTPUT / "original_vs_known_status_metrics.csv", index=False)

    scores = pd.DataFrame(unknown_score_rows)
    scores.to_csv(OUTPUT / "unknown_case_prediction_scores.csv", index=False)
    score_summary = scores.groupby("model", as_index=False).agg(
        unknown_n=("case_name", "size"),
        mean_probability=("prob", "mean"),
        predicted_ruptured_at_0_5=("pred_at_0_5", "sum"),
    )
    score_summary.to_csv(OUTPUT / "unknown_prediction_summary.csv", index=False)

    late_original = original.loc["70/30 Late Fusion"]
    late_corrected = corrected.loc["70/30 Late Fusion"]
    lines = [
        "# Unknown rupture-status audit",
        "",
        f"- Metadata rows: {len(metadata)}.",
        f"- Known ruptured: {(normalized == 'ruptured').sum()}.",
        f"- Known unruptured: {(normalized == 'unruptured').sum()}.",
        f"- Unknown/blank: {normalized.isna().sum()}.",
        "- Previous behavior mapped every non-'ruptured' value to target 0, so all unknown cases were treated as unruptured.",
        "- Corrected behavior excludes unknown status before targets, folds, feature extraction, and evaluation are constructed.",
        "",
        "## Post-hoc known-status sensitivity",
        "",
        f"- 70/30 late-fusion AUROC: {late_original['auc']:.6f} -> {late_corrected['auc']:.6f}.",
        f"- 70/30 late-fusion AUPRC: {late_original['pr_auc']:.6f} -> {late_corrected['pr_auc']:.6f}.",
        f"- Evaluation N: {int(late_original['n'])} -> {int(late_corrected['n'])}.",
        "",
        "These corrected metrics exclude unknown cases from evaluation but reuse models trained under the old labeling rule. Full correction therefore requires retraining the base rupture models on the 735 known-status cases.",
    ]
    (OUTPUT / "AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(counts.to_string(index=False))
    print(metrics[["model", "scope", "n", "auc", "pr_auc", "acc", "f1"]].to_string(index=False))
    print(f"Wrote audit to {OUTPUT}")


if __name__ == "__main__":
    main()
