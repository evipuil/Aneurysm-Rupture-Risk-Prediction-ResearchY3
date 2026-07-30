# Version 14 source snapshot
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from rupture_status import KNOWN_RUPTURE_STATUSES, normalize_rupture_status

try:
    from scipy.spatial import ConvexHull
    from scipy.stats import mannwhitneyu, spearmanr
except Exception:
    ConvexHull = None
    mannwhitneyu = None
    spearmanr = None

try:
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover - import error is reported at runtime
    raise RuntimeError("scikit-learn is required for feature extraction") from exc


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_METADATA_PATH = PROJECT_ROOT / "metadata.csv"
DEFAULT_DATA_DIR = PROJECT_ROOT / "flow_data" / "full_accuracy2_copy"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results_v14_feature_extraction"
DEFAULT_LEGACY_WSS_SCALE = float(os.environ.get("v14_LEGACY_WSS_SCALE", 1000.0))
SAFE_CLINICAL_CATEGORICAL_FIELDS = ("location", "side")
INTERNAL_METADATA_COLUMNS = {"case_name", "patient_group", "vesselFileID", "cutToShow"}
PUBLIC_METADATA_COLUMNS = ("case_id", "target")
NON_FEATURE_COLUMNS = {"case_id", "target", *INTERNAL_METADATA_COLUMNS}
NON_INFORMATIVE_FEATURES = {
    # Defined by each case's own 20th percentile, so this fraction is approximately
    # 0.20 by construction and cannot meaningfully distinguish cases.
    "hemo_low_tawss_fraction",
}
LEAKAGE_PREFIXES = (
    "clinical_source=",
    "clinical_hospital=",
    "clinical_dataset=",
    "clinical_patient",
    "clinical_vessel",
)


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


def hemodynamic_features(df: pd.DataFrame, wss_scale: float = 1.0) -> Dict[str, float]:
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

    # Legacy V14 outputs differentiated velocity with respect to millimetres.
    # Scale them to per-metre gradients before computing Pa-based quantities.
    tawss = np.maximum(tawss * float(wss_scale), 0.0)
    von_mises = np.maximum(von_mises * float(wss_scale), 0.0)
    osi = np.clip(osi, 0.0, 0.499)

    if len(tawss) == 0:
        tawss = np.array([0.0], dtype=np.float64)
        osi = np.array([0.0], dtype=np.float64)
        von_mises = np.array([0.0], dtype=np.float64)

    low_tawss_threshold = np.percentile(tawss, 20)
    low_tawss = tawss <= low_tawss_threshold
    high_osi = osi >= 0.2
    combined = tawss * (1.0 - 2.0 * osi)
    denom = np.maximum((1.0 - 2.0 * osi) * np.maximum(tawss, 1e-6), 1e-6)
    rrt = 1.0 / denom
    rrt[~np.isfinite(rrt)] = 0.0
    if len(rrt) > 1:
        rrt = np.clip(rrt, 0.0, np.percentile(rrt, 99.0))
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

    for field in SAFE_CLINICAL_CATEGORICAL_FIELDS:
        raw_value = meta_row.get(field, "Unknown")
        if pd.isna(raw_value):
            value = "Unknown"
        else:
            value = str(raw_value).strip()
            if not value or value.lower() in {"nan", "none", "null"}:
                value = "Unknown"
        features[f"clinical_{field}={value}"] = 1.0

    return features


def extract_case_row(
    case_dir: Path, meta_row: pd.Series, legacy_wss_scale: float = DEFAULT_LEGACY_WSS_SCALE
) -> Dict[str, float]:
    hemo_df = load_hemodynamics(case_dir)
    coords = hemo_df[["x", "y", "z"]].to_numpy(dtype=np.float64)

    status = normalize_rupture_status(meta_row.get("status", None))
    if status is None:
        raise ValueError(f"Unknown rupture status for {case_dir.name}")
    row = {
        "case_name": case_dir.name,
        "patient_group": _stable_group_id(meta_row, case_dir.name),
        "vesselFileID": str(meta_row.get("vesselFileID", case_dir.name)),
        "cutToShow": str(meta_row.get("cutToShow", "cut1")),
        "target": KNOWN_RUPTURE_STATUSES[status],
    }
    row.update(geometry_features(coords))
    wss_scale = 1.0 if (case_dir / "hemodynamic_units.json").exists() else float(legacy_wss_scale)
    row.update(hemodynamic_features(hemo_df, wss_scale=wss_scale))
    row.update(clinical_features(meta_row))
    return row


def discover_case_rows(
    data_dir: Path,
    metadata_path: Path,
    limit: int | None = None,
    legacy_wss_scale: float = DEFAULT_LEGACY_WSS_SCALE,
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
        if normalize_rupture_status(meta_row.get("status", None)) is None:
            continue
        rows.append(extract_case_row(case_dir, meta_row, legacy_wss_scale=legacy_wss_scale))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.insert(0, "case_id", [f"case_{i + 1:04d}" for i in range(len(df))])
    numeric_cols = [c for c in df.columns if c not in {"case_id", *INTERNAL_METADATA_COLUMNS}]
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


def _stable_group_id(meta_row: pd.Series, fallback: str) -> str:
    for field in ("patientID", "vesselFileID", "dataset"):
        value = meta_row.get(field, "")
        if pd.notna(value):
            text = str(value).strip()
            if text and text.lower() not in {"nan", "none", "null"}:
                return text
    return fallback


def _is_leakage_feature(feature_name: str) -> bool:
    lowered = feature_name.lower()
    return any(lowered.startswith(prefix) for prefix in LEAKAGE_PREFIXES)


def _feature_columns(features: pd.DataFrame, target_col: str) -> List[str]:
    excluded = set(NON_FEATURE_COLUMNS)
    excluded.add(target_col)
    return [
        c
        for c in features.columns
        if c not in excluded and c not in NON_INFORMATIVE_FEATURES and not _is_leakage_feature(c)
    ]


def _public_feature_table(features: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in INTERNAL_METADATA_COLUMNS if c in features.columns]
    return features.drop(columns=drop_cols, errors="ignore")


def _make_cv_splits(features: pd.DataFrame, y: np.ndarray, seed: int, n_splits: int):
    groups = features.get("patient_group")
    if groups is not None and groups.nunique(dropna=True) >= n_splits:
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return list(
                splitter.split(features, y, groups.astype(str))
            ), "StratifiedGroupKFold(patient_group)"
        except ValueError:
            pass

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(features, y)), "StratifiedKFold"


def compute_feature_importance(
    features: pd.DataFrame, target_col: str = "target", seed: int = 42, n_splits: int = 5
):
    feature_cols = _feature_columns(features, target_col)
    X = features[feature_cols].copy()
    y = features[target_col].astype(int).to_numpy()

    if X.empty:
        raise ValueError("No feature columns were extracted")

    counts = np.bincount(y)
    min_class = int(counts.min()) if len(counts) > 1 else 0
    if min_class < 2:
        raise ValueError("Need at least two samples per class for importance scoring")
    n_splits = max(2, min(int(n_splits), min_class))

    cv_splits, split_strategy = _make_cv_splits(features, y, seed, n_splits)
    coef_rows = []
    perm_rows = []
    fold_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits, start=1):
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
            {
                "fold": fold_idx,
                "auc": fold_auc,
                "pr_auc": fold_ap,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "split_strategy": split_strategy,
            }
        )

        clf = pipeline.named_steps["clf"]
        signed_coef = clf.coef_[0]
        coef_rows.append(
            pd.DataFrame(
                {
                    "feature": feature_cols,
                    "signed_coef": signed_coef,
                    "coef_importance": np.abs(signed_coef),
                    "fold": fold_idx,
                }
            )
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

    coef_folds = pd.concat(coef_rows, ignore_index=True)
    coef_df = coef_folds.groupby("feature", as_index=False).agg(
        signed_coef_mean=("signed_coef", "mean"),
        signed_coef_std=("signed_coef", "std"),
        positive_coef_fraction=(
            "signed_coef",
            lambda values: float(np.mean(np.asarray(values) > 0)),
        ),
        coef_importance=("coef_importance", "mean"),
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


def compute_univariate_associations(
    features: pd.DataFrame, target_col: str = "target"
) -> pd.DataFrame:
    """Describe effect direction separately from multivariable feature importance."""
    if mannwhitneyu is None or spearmanr is None:
        return pd.DataFrame()

    target = features[target_col].astype(int)
    rows = []
    for feature in _feature_columns(features, target_col):
        values = pd.to_numeric(features[feature], errors="coerce")
        valid = values.notna() & target.notna()
        observed = values[valid]
        labels = target[valid]
        negative = observed[labels == 0]
        positive = observed[labels == 1]
        if len(negative) < 2 or len(positive) < 2 or observed.nunique() < 2:
            continue

        rho, rho_p = spearmanr(observed, labels)
        u_stat, mw_p = mannwhitneyu(positive, negative, alternative="two-sided")
        rank_biserial = 2.0 * float(u_stat) / (len(positive) * len(negative)) - 1.0
        rows.append(
            {
                "feature": feature,
                "modality": _modality_for_feature(feature),
                "n_unruptured": len(negative),
                "n_ruptured": len(positive),
                "median_unruptured": float(negative.median()),
                "median_ruptured": float(positive.median()),
                "mean_unruptured": float(negative.mean()),
                "mean_ruptured": float(positive.mean()),
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
                "rank_biserial": rank_biserial,
                "mann_whitney_p": float(mw_p),
                "univariate_auroc": float(roc_auc_score(labels, observed)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "spearman_rho", key=lambda column: column.abs(), ascending=False
    )


def build_condensed_tables(
    metadata_path: Path,
    data_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    top_k: int = 25,
    seed: int = 42,
    legacy_wss_scale: float = DEFAULT_LEGACY_WSS_SCALE,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    features = discover_case_rows(
        data_dir, metadata_path, limit=limit, legacy_wss_scale=legacy_wss_scale
    )
    if features.empty:
        raise RuntimeError(f"No cases with hemodynamics_aggregate.csv found in {data_dir}")

    feature_path = output_dir / "condensed_case_features.csv"
    public_features = _public_feature_table(features)
    public_features.to_csv(feature_path, index=False)

    importance, modality_summary, fold_metrics = compute_feature_importance(features, seed=seed)
    importance_path = output_dir / "feature_importance_summary.csv"
    modality_path = output_dir / "modality_importance_summary.csv"
    fold_path = output_dir / "importance_cv_summary.csv"
    association_path = output_dir / "univariate_feature_associations.csv"

    importance.to_csv(importance_path, index=False)
    modality_summary.to_csv(modality_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)
    compute_univariate_associations(features).to_csv(association_path, index=False)

    selected = importance.head(max(1, min(int(top_k), len(importance))))["feature"].tolist()
    condensed_cols = [*PUBLIC_METADATA_COLUMNS, *selected]
    condensed_cols = [c for c in condensed_cols if c in public_features.columns]
    top_path = output_dir / f"condensed_case_features_top{len(selected)}.csv"
    public_features[condensed_cols].to_csv(top_path, index=False)

    return {
        "feature_table": feature_path,
        "feature_importance": importance_path,
        "modality_importance": modality_path,
        "cross_validation_summary": fold_path,
        "univariate_associations": association_path,
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
    parser.add_argument(
        "--legacy-wss-scale",
        type=float,
        default=DEFAULT_LEGACY_WSS_SCALE,
        help="Scale legacy per-mm WSS outputs to Pa; ignored for cases with hemodynamic_units.json.",
    )
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
        legacy_wss_scale=args.legacy_wss_scale,
    )

    print("Feature extraction complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
