# Version 14 source snapshot
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from base_trainer import classification_report_dict
from rupture_status import filter_predictions_to_known_status

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SUITE = PROJECT_ROOT / "results_V14_suite"


def read_prediction_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"filepath", "label", "prob"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {sorted(missing)}")
    return frame


def load_fold_predictions(
    run_dir: Path, folds: int, metadata_path: Path | None = None
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
                    f"No pooled predictions at {pooled_path} and missing legacy file {path}"
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
    return result


def logit(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0 - eps)
    return np.log(probabilities / (1.0 - probabilities))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(
        values >= 0, 1.0 / (1.0 + np.exp(-values)), np.exp(values) / (1.0 + np.exp(values))
    )


def add_metrics_metadata(metrics: dict, flow_weight: float, fusion_space: str) -> dict:
    return {
        **metrics,
        "model": "geometry_flow_clinical_late_ensemble",
        "backbone": "pointnext",
        "fusion": f"fixed_{fusion_space}_late_fusion",
        "flow_weight": float(flow_weight),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine geometry-clinical and flow-geometry OOF predictions."
    )
    parser.add_argument(
        "--geometry-clinical-dir",
        type=Path,
        default=DEFAULT_SUITE / "geometry_clinical_pointnext_seed_42",
    )
    parser.add_argument(
        "--flow-geometry-dir",
        type=Path,
        default=DEFAULT_SUITE / "flow_geometry_pointnext_seed_42",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results_V14_improved_late_ensemble_target8192",
    )
    parser.add_argument("--flow-weight", type=float, default=0.30)
    parser.add_argument(
        "--fusion-space",
        choices=("probability", "logit"),
        default="probability",
        help="Combine calibrated probabilities directly or combine logits before applying sigmoid.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--metadata-path", type=Path, default=PROJECT_ROOT / "metadata.csv")
    args = parser.parse_args()

    if not 0.0 <= args.flow_weight <= 0.5:
        raise ValueError("--flow-weight must be between 0 and 0.5 to preserve the clinical anchor")

    geometry_clinical = load_fold_predictions(
        args.geometry_clinical_dir, args.folds, args.metadata_path
    ).rename(columns={"prob": "geometry_clinical_prob"})
    flow_geometry = load_fold_predictions(
        args.flow_geometry_dir, args.folds, args.metadata_path
    ).rename(columns={"prob": "flow_geometry_prob"})
    merged = geometry_clinical.merge(
        flow_geometry,
        on=["filepath", "label", "fold"],
        how="inner",
        validate="one_to_one",
    )
    expected = len(geometry_clinical)
    if len(merged) != expected or len(flow_geometry) != expected:
        raise RuntimeError("Base prediction sets do not contain identical cases")

    flow_weight = float(args.flow_weight)
    anchor_weight = 1.0 - flow_weight
    if args.fusion_space == "probability":
        merged["prob"] = (
            anchor_weight * merged["geometry_clinical_prob"]
            + flow_weight * merged["flow_geometry_prob"]
        )
    else:
        fused_margin = anchor_weight * logit(
            merged["geometry_clinical_prob"]
        ) + flow_weight * logit(merged["flow_geometry_prob"])
        merged["prob"] = sigmoid(fused_margin)
    merged["pred"] = (merged["prob"] > 0.5).astype(int)

    fold_rows = []
    for fold, frame in merged.groupby("fold", sort=True):
        fold_rows.append(
            {
                "fold": int(fold),
                **add_metrics_metadata(
                    classification_report_dict(frame["label"], frame["prob"]),
                    flow_weight,
                    args.fusion_space,
                ),
            }
        )
    pooled = add_metrics_metadata(
        classification_report_dict(merged["label"], merged["prob"]),
        flow_weight,
        args.fusion_space,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "pooled_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_summary.csv", index=False)
    pd.DataFrame([pooled]).to_csv(args.output_dir / "pooled_metrics.csv", index=False)
    config = {
        "geometry_clinical_dir": str(args.geometry_clinical_dir.resolve()),
        "flow_geometry_dir": str(args.flow_geometry_dir.resolve()),
        "folds": int(args.folds),
        "flow_weight": flow_weight,
        "anchor_weight": anchor_weight,
        "fusion": f"fixed_{args.fusion_space}_late_fusion",
        "note": "One fixed flow contribution is used for every fold and case; no fold-specific or case-specific tuning.",
    }
    (args.output_dir / "fusion_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="ascii"
    )

    print(pd.DataFrame([pooled]).to_string(index=False))
    print(f"Wrote late-fusion outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
