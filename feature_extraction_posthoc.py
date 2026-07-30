# Version 8 source snapshot
"""
feature_extraction_posthoc.py (SELF-CONTAINED)

Compute per-feature associations with rupture using global features extracted
from PINN outputs. Ranks features by AUC and Spearman correlation.
All helpers inlined for independence.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# CORE HELPERS
def safe_auc(targets, probs):
    try:
        return roc_auc_score(targets, probs)
    except ValueError:
        return 0.5


def build_metadata_index(metadata_path: str):
    df = pd.read_csv(metadata_path)
    key_to_idx = {}
    for idx, row in df.iterrows():
        ds = str(row.get("dataset", "")).strip()
        vid = str(row.get("vesselFileID", "")).strip()
        raw_cut = row.get("cutToShow", "cut1")
        cut = str(raw_cut).strip() if pd.notna(raw_cut) else "cut1"
        for key in (ds, vid):
            if key:
                key_to_idx[f"{key}_{cut}"] = idx
                key_to_idx[key] = idx
    return df, key_to_idx


def _match_folder(folder: str, key_to_idx):
    if folder in key_to_idx:
        return key_to_idx[folder]
    base = re.sub(r"_cut\d+$", "", folder)
    return key_to_idx.get(base)


def discover_cases(
    data_dir: str, metadata_path: str, require_file: str = "hemodynamics_aggregate.csv"
):
    df, key_to_idx = build_metadata_index(metadata_path)
    valid_indices, filepaths = [], []
    seen = set()
    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        csv_path = os.path.join(folder_path, require_file)
        if not os.path.exists(csv_path):
            continue
        matched = _match_folder(folder, key_to_idx)
        if matched is not None and matched not in seen:
            seen.add(matched)
            valid_indices.append(matched)
            filepaths.append(csv_path)
    df = df.loc[valid_indices].reset_index(drop=True)
    df["filepath"] = filepaths
    df["target"] = (df["status"].astype(str).str.lower() == "ruptured").astype(int)
    return df


def summarize_global_features(
    raw_feats: np.ndarray, pts: np.ndarray, include_rrt: bool = False
) -> np.ndarray:
    tawss = raw_feats[:, 0]
    osi = raw_feats[:, 1]
    von = raw_feats[:, 2]
    feats = [
        np.mean(tawss),
        np.std(tawss),
        np.max(tawss),
        np.min(tawss),
        np.percentile(tawss, 95),
        np.percentile(tawss, 5),
        float(np.mean(tawss > np.percentile(tawss, 90))),
        np.mean(osi),
        np.std(osi),
        np.max(osi),
        np.percentile(osi, 95),
        float(np.mean(osi > 0.2)),
        np.mean(von),
        np.std(von),
        np.max(von),
        np.percentile(von, 95),
        np.percentile(von, 99),
    ]
    if include_rrt:
        denom = (1.0 - 2.0 * osi) * tawss
        rrt = np.zeros_like(tawss, dtype=np.float32)
        mask = np.abs(denom) > 1e-8
        rrt[mask] = 1.0 / denom[mask]
        rrt[~np.isfinite(rrt)] = 0.0
        feats.extend(
            [
                np.mean(rrt),
                np.std(rrt),
                np.max(rrt),
                np.percentile(rrt, 95),
                float(np.mean(rrt > np.percentile(rrt, 90))),
            ]
        )

    centroid = pts.mean(axis=0)
    dist = np.linalg.norm(pts - centroid, axis=1)
    try:
        cov = np.cov(pts.T)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eig = eig / (eig.sum() + 1e-8)
    except Exception:
        eig = np.array([0.5, 0.3, 0.2])
    feats.extend(
        [
            float(np.max(dist)),
            float(np.std(dist)),
            float(np.max(dist) / (np.mean(dist) + 1e-6)),
            float(eig[0]),
            float(eig[1]),
            float(eig[0] / (eig[2] + 1e-6)),
        ]
    )
    arr = np.array(feats, dtype=np.float32)
    arr = np.sign(arr) * np.log1p(np.abs(arr))
    return np.clip(arr, -10.0, 10.0)


def extract_xyz_and_hemo(df: pd.DataFrame):
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    if {"x", "y", "z"}.issubset(cols):
        pts = df[["x", "y", "z"]].values.astype(np.float32)
    else:
        pts = df.iloc[:, :3].values.astype(np.float32)
    raw_cols = []
    for key in ("tawss", "osi", "von"):
        match = next((c for c in cols if key in c), None)
        raw_cols.append(df[match].values if match else np.zeros(len(df), dtype=np.float32))
    raw_feats = np.stack(raw_cols, axis=1).astype(np.float32)
    return pts, raw_feats


# CONFIGURATION
DATA_DIR = os.environ.get("V8_DATA_DIR", "predictions/pinn_corrected")
METADATA_PATH = os.environ.get("V8_METADATA", "metadata.csv")
OUTPUT_DIR = Path(os.environ.get("V8_OUTPUT_DIR", "results_geometry_v8"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def gather_features(data_dir, metadata_path):
    cases = discover_cases(data_dir, metadata_path, require_file="hemodynamics_aggregate.csv")
    feats = []
    labels = []
    for _, row in cases.iterrows():
        path = row["filepath"]
        lab = (
            int(row.get("target", 0))
            if "target" in row.index
            else int(row.get("status", "") == "ruptured")
        )
        try:
            df = pd.read_csv(path)
            pts, raw_feats = extract_xyz_and_hemo(df)
        except Exception:
            arr = np.loadtxt(path).astype(np.float32)
            raw_feats = arr[:, :3]
            pts = arr[:, :3]
        feat = summarize_global_features(raw_feats, pts, include_rrt=False)
        feats.append(feat)
        labels.append(lab)
    X = np.vstack(feats)
    y = np.array(labels)
    return X, y


def main():
    X, y = gather_features(DATA_DIR, METADATA_PATH)
    n_features = X.shape[1]
    if len(np.unique(y)) < 2:
        print(
            "Only one class present in the dataset; ROC-AUC is undefined. Writing correlation-only output."
        )
    rows = []
    for i in range(n_features):
        col = X[:, i]
        auc = safe_auc(y, col) if len(np.unique(y)) > 1 else np.nan
        try:
            rho, p = spearmanr(col, y)
        except Exception:
            rho, p = np.nan, np.nan
        rows.append({"feature": f"feat_{i}", "auc": auc, "spearman_rho": rho, "spearman_p": p})

    df = pd.DataFrame(rows).sort_values(by="auc", ascending=False)
    out_path = OUTPUT_DIR / "feature_ranking.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote feature ranking to {out_path}")


if __name__ == "__main__":
    main()
