# Version 12 source snapshot
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None

try:
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover - import error is reported at runtime
    raise RuntimeError("scikit-learn is required for feature extraction") from exc


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_METADATA_PATH = PROJECT_ROOT / "metadata.csv"
DEFAULT_DATA_DIR = PROJECT_ROOT / "predictions" / "pinn_corrected"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results_v12_feature_extraction"


def _safe_float(value, default=0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(value)


def _safe_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if values.size == 0:
        values = np.array([0.0], dtype=np.float32)
    return {
        f"{prefix}_mean": _safe_float(np.mean(values)),
        f"{prefix}_std": _safe_float(np.std(values)),
        f"{prefix}_min": _safe_float(np.min(values)),
        f"{prefix}_p25": _safe_float(np.percentile(values, 25)),
        f"{prefix}_median": _safe_float(np.median(values)),
        f"{prefix}_p75": _safe_float(np.percentile(values, 75)),
        f"{prefix}_max": _safe_float(np.max(values)),
    }


def _match_case_row(folder_name: str, metadata: pd.DataFrame, key_to_idx: Dict[str, int]):
    if folder_name in key_to_idx:
        return metadata.iloc[key_to_idx[folder_name]]

    base = re.sub(r"_cut\d+$", "", folder_name)
    if base in key_to_idx:
        return metadata.iloc[key_to_idx[base]]

    for suffix in ("cut1", "cut2", "cut3"):
        candidate = f"{base}_{suffix}"
        if candidate in key_to_idx:
            return metadata.iloc[key_to_idx[candidate]]
    return None


def build_metadata_index(metadata_path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    metadata = pd.read_csv(metadata_path)
    key_to_idx: Dict[str, int] = {}

    for idx, row in metadata.iterrows():
        dataset = str(row.get("dataset", "")).strip()
        vessel_id = str(row.get("vesselFileID", "")).strip()
        cut_raw = row.get("cutToShow", "cut1")
        cut = str(cut_raw).strip() if pd.notna(cut_raw) else "cut1"

        for key in (dataset, vessel_id):
            if key:
                key_to_idx.setdefault(key, idx)
                key_to_idx.setdefault(f"{key}_{cut}", idx)
                key_to_idx.setdefault(f"{key}_cut1", idx)

    return metadata, key_to_idx


def load_hemodynamics(case_dir: Path) -> pd.DataFrame:
    csv_path = case_dir / "hemodynamics_aggregate.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path)


def geometry_features(coords: np.ndarray) -> Dict[str, float]:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.size == 0:
        coords = np.zeros((1, 3), dtype=np.float64)

    center = coords.mean(axis=0)
    centered = coords - center
    bounds = np.ptp(coords, axis=0)
    bounds = np.maximum(bounds, 1e-8)
    radial = np.linalg.norm(centered, axis=1)

    cov = np.cov(centered.T) if len(coords) > 1 else np.eye(3)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.maximum(eigvals, 1e-12)

    features = {
        "geometry_n_points": float(len(coords)),
        "geometry_centroid_x": _safe_float(center[0]),
        "geometry_centroid_y": _safe_float(center[1]),
        "geometry_centroid_z": _safe_float(center[2]),
        "geometry_bbox_x": _safe_float(bounds[0]),
        "geometry_bbox_y": _safe_float(bounds[1]),
        "geometry_bbox_z": _safe_float(bounds[2]),
        "geometry_bbox_volume": _safe_float(np.prod(bounds)),
        "geometry_bbox_surface_area": _safe_float(
            2.0 * (bounds[0] * bounds[1] + bounds[1] * bounds[2] + bounds[0] * bounds[2])
        ),
        "geometry_bbox_diagonal": _safe_float(np.linalg.norm(bounds)),
        "geometry_radial_mean": _safe_float(np.mean(radial)),
        "geometry_radial_std": _safe_float(np.std(radial)),
        "geometry_radial_p90": _safe_float(np.percentile(radial, 90)),
        "geometry_pca_eig1": _safe_float(eigvals[0]),
        "geometry_pca_eig2": _safe_float(eigvals[1]),
        "geometry_pca_eig3": _safe_float(eigvals[2]),
        "geometry_pca_linearity": _safe_float(eigvals[0] / (eigvals[1] + 1e-12)),
        "geometry_pca_planarity": _safe_float(eigvals[1] / (eigvals[2] + 1e-12)),
        "geometry_pca_sphericity": _safe_float(eigvals[2] / (eigvals[0] + 1e-12)),
    }

    if ConvexHull is not None and len(coords) >= 4:
        try:
            hull = ConvexHull(coords)
            features["geometry_hull_volume"] = _safe_float(hull.volume)
            features["geometry_hull_area"] = _safe_float(hull.area)
        except Exception:
            features["geometry_hull_volume"] = 0.0
            features["geometry_hull_area"] = 0.0
    else:
        features["geometry_hull_volume"] = 0.0
        features["geometry_hull_area"] = 0.0

    return features


def hemodynamic_features(df: pd.DataFrame) -> Dict[str, float]:
    tawss = (
        pd.to_numeric(df.get("tawss", pd.Series(dtype=float)), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    osi = (
        pd.to_numeric(df.get("osi", pd.Series(dtype=float)), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    von_mises = (
        pd.to_numeric(df.get("von_mises", pd.Series(dtype=float)), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    if len(tawss) == 0:
        tawss = np.array([0.0], dtype=np.float64)
        osi = np.array([0.0], dtype=np.float64)
        von_mises = np.array([0.0], dtype=np.float64)

    low_tawss_threshold = np.percentile(tawss, 20)
    low_tawss = tawss <= low_tawss_threshold
    high_osi = osi >= 0.2
    combined = tawss * (1.0 - 2.0 * osi)
    denom = (1.0 - 2.0 * osi) * tawss
    rrt = np.zeros_like(tawss, dtype=np.float64)
    valid = np.abs(denom) > 1e-8
    rrt[valid] = 1.0 / denom[valid]
    rrt[~np.isfinite(rrt)] = 0.0
    shear_ratio = tawss / (von_mises + 1e-6)
    shear_ratio[~np.isfinite(shear_ratio)] = 0.0

    features: Dict[str, float] = {}
    features.update(_safe_stats(tawss, "hemo_tawss"))
    features.update(_safe_stats(osi, "hemo_osi"))
    features.update(_safe_stats(von_mises, "hemo_von_mises"))
    features["hemo_low_tawss_fraction"] = _safe_float(np.mean(low_tawss))
    features["hemo_high_osi_fraction"] = _safe_float(np.mean(high_osi))
    features["hemo_low_tawss_high_osi_fraction"] = _safe_float(np.mean(low_tawss & high_osi))
    features["hemo_combined_mean"] = _safe_float(np.mean(combined))
    features["hemo_combined_std"] = _safe_float(np.std(combined))
    features["hemo_rrt_mean"] = _safe_float(np.mean(rrt))
    features["hemo_rrt_std"] = _safe_float(np.std(rrt))
    features["hemo_shear_ratio_mean"] = _safe_float(np.mean(shear_ratio))
    features["hemo_shear_ratio_std"] = _safe_float(np.std(shear_ratio))

    if len(tawss) > 1:
        features["hemo_tawss_osi_corr"] = _safe_float(np.corrcoef(tawss, osi)[0, 1])
        features["hemo_tawss_von_mises_corr"] = _safe_float(np.corrcoef(tawss, von_mises)[0, 1])
        features["hemo_osi_von_mises_corr"] = _safe_float(np.corrcoef(osi, von_mises)[0, 1])
    else:
        features["hemo_tawss_osi_corr"] = 0.0
        features["hemo_tawss_von_mises_corr"] = 0.0
        features["hemo_osi_von_mises_corr"] = 0.0

    return features


def clinical_features(meta_row: pd.Series) -> Dict[str, float]:
    age = (
        pd.to_numeric(pd.Series([meta_row.get("age", np.nan)]), errors="coerce").fillna(0.0).iloc[0]
    )
    sex_raw = str(meta_row.get("sex", "unknown")).strip().lower()
    sex_male = 1.0 if sex_raw == "male" else 0.0 if sex_raw == "female" else 0.5

    features = {
        "clinical_age": _safe_float(age),
        "clinical_sex_male": _safe_float(sex_male),
    }

    for field in ("location", "hospital", "source", "side"):
        raw_value = meta_row.get(field, "Unknown")
        if pd.isna(raw_value):
            value = "Unknown"
        else:
            value = str(raw_value).strip()
            if not value or value.lower() in {"nan", "none", "null"}:
                value = "Unknown"
        features[f"clinical_{field}={value}"] = 1.0

    return features


def extract_case_row(case_dir: Path, meta_row: pd.Series) -> Dict[str, float]:
    hemo_df = load_hemodynamics(case_dir)
    coords = hemo_df[["x", "y", "z"]].to_numpy(dtype=np.float64)

    row = {
        "case_name": case_dir.name,
        "dataset": str(meta_row.get("dataset", case_dir.name)),
        "vesselFileID": str(meta_row.get("vesselFileID", case_dir.name)),
        "cutToShow": str(meta_row.get("cutToShow", "cut1")),
        "target": 1 if str(meta_row.get("status", "")).strip().lower() == "ruptured" else 0,
    }
    row.update(geometry_features(coords))
    row.update(hemodynamic_features(hemo_df))
    row.update(clinical_features(meta_row))
    return row


def discover_case_rows(
    data_dir: Path, metadata_path: Path, limit: int | None = None
) -> pd.DataFrame:
    metadata, key_to_idx = build_metadata_index(metadata_path)
    rows: List[Dict[str, float]] = []

    case_dirs = [
        p
        for p in sorted(data_dir.iterdir())
        if p.is_dir() and (p / "hemodynamics_aggregate.csv").exists()
    ]
    if limit is not None:
        case_dirs = case_dirs[:limit]

    for case_dir in case_dirs:
        meta_row = _match_case_row(case_dir.name, metadata, key_to_idx)
        if meta_row is None:
            continue
        rows.append(extract_case_row(case_dir, meta_row))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    numeric_cols = [
        c for c in df.columns if c not in {"case_name", "dataset", "vesselFileID", "cutToShow"}
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def _modality_for_feature(feature_name: str) -> str:
    if feature_name.startswith("geometry_"):
        return "geometry"
    if feature_name.startswith("hemo_"):
        return "hemodynamic"
    if feature_name.startswith("clinical_"):
        return "clinical"
    return "other"


def compute_feature_importance(
    features: pd.DataFrame, target_col: str = "target", seed: int = 42, n_splits: int = 5
):
    feature_cols = [
        c
        for c in features.columns
        if c not in {target_col, "case_name", "dataset", "vesselFileID", "cutToShow"}
    ]
    X = features[feature_cols].copy()
    y = features[target_col].astype(int).to_numpy()

    if X.empty:
        raise ValueError("No feature columns were extracted")

    counts = np.bincount(y)
    min_class = int(counts.min()) if len(counts) > 1 else 0
    if min_class < 2:
        raise ValueError("Need at least two samples per class for importance scoring")
    n_splits = max(2, min(int(n_splits), min_class))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    coef_rows = []
    perm_rows = []
    fold_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y[val_idx]

        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")),
            ]
        )
        pipeline.fit(X_train, y_train)

        val_probs = pipeline.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else 0.5
        fold_ap = average_precision_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else 0.0
        fold_rows.append(
            {"fold": fold_idx, "auc": fold_auc, "pr_auc": fold_ap, "n_val": len(val_idx)}
        )

        clf = pipeline.named_steps["clf"]
        abs_coef = np.abs(clf.coef_[0])
        coef_rows.append(
            pd.DataFrame({"feature": feature_cols, "coef_importance": abs_coef, "fold": fold_idx})
        )

        scoring = "roc_auc" if len(np.unique(y_val)) > 1 else "accuracy"
        perm = permutation_importance(
            pipeline, X_val, y_val, n_repeats=10, random_state=seed + fold_idx, scoring=scoring
        )
        perm_rows.append(
            pd.DataFrame(
                {
                    "feature": feature_cols,
                    "perm_importance": perm.importances_mean,
                    "fold": fold_idx,
                }
            )
        )

    coef_df = (
        pd.concat(coef_rows, ignore_index=True)
        .groupby("feature", as_index=False)["coef_importance"]
        .mean()
    )
    perm_df = (
        pd.concat(perm_rows, ignore_index=True)
        .groupby("feature", as_index=False)["perm_importance"]
        .mean()
    )

    summary = coef_df.merge(perm_df, on="feature", how="outer").fillna(0.0)
    summary["modality"] = summary["feature"].map(_modality_for_feature)
    summary["importance_score"] = (
        summary["coef_importance"].rank(pct=True, method="average")
        + summary["perm_importance"].rank(pct=True, method="average")
    ) / 2.0
    summary["importance_score"] = summary["importance_score"].fillna(0.0)
    summary = summary.sort_values(
        ["importance_score", "perm_importance", "coef_importance"], ascending=False
    ).reset_index(drop=True)

    modality_summary = (
        summary.groupby("modality", as_index=False)
        .agg(
            feature_count=("feature", "count"),
            coef_importance=("coef_importance", "sum"),
            perm_importance=("perm_importance", "sum"),
            importance_score=("importance_score", "sum"),
        )
        .sort_values("importance_score", ascending=False)
        .reset_index(drop=True)
    )

    return summary, modality_summary, pd.DataFrame(fold_rows)


def build_condensed_tables(
    metadata_path: Path,
    data_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    top_k: int = 25,
    seed: int = 42,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    features = discover_case_rows(data_dir, metadata_path, limit=limit)
    if features.empty:
        raise RuntimeError(f"No cases with hemodynamics_aggregate.csv found in {data_dir}")

    feature_path = output_dir / "condensed_case_features.csv"
    features.to_csv(feature_path, index=False)

    importance, modality_summary, fold_metrics = compute_feature_importance(features, seed=seed)
    importance_path = output_dir / "feature_importance_summary.csv"
    modality_path = output_dir / "modality_importance_summary.csv"
    fold_path = output_dir / "importance_cv_summary.csv"

    importance.to_csv(importance_path, index=False)
    modality_summary.to_csv(modality_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)

    selected = importance.head(max(1, min(int(top_k), len(importance))))["feature"].tolist()
    condensed_cols = ["case_name", "dataset", "vesselFileID", "cutToShow", "target"] + selected
    condensed_cols = [c for c in condensed_cols if c in features.columns]
    top_path = output_dir / f"condensed_case_features_top{len(selected)}.csv"
    features[condensed_cols].to_csv(top_path, index=False)

    return {
        "feature_table": feature_path,
        "feature_importance": importance_path,
        "modality_importance": modality_path,
        "cross_validation_summary": fold_path,
        "top_feature_table": top_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract condensed geometric, hemodynamic, and clinical features"
    )
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional limit on the number of cases processed"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of top-ranked features to keep in the condensed table",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata_path = Path(args.metadata_path)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    outputs = build_condensed_tables(
        metadata_path=metadata_path,
        data_dir=data_dir,
        output_dir=output_dir,
        limit=args.limit,
        top_k=args.top_k,
        seed=args.seed,
    )

    print("Feature extraction complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
