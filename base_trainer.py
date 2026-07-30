# Version 14 source snapshot
from __future__ import annotations

import csv
import os
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rupture_status import KNOWN_RUPTURE_STATUSES, normalize_rupture_status

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

    def _missing_one_cycle_lr(*args, **kwargs):
        return None

    torch = None
    nn = SimpleNamespace(Module=object)
    F = SimpleNamespace()
    optim = SimpleNamespace(lr_scheduler=SimpleNamespace(OneCycleLR=_missing_one_cycle_lr))
    DataLoader = TensorDataset = WeightedRandomSampler = object

try:
    import torch_cluster

    HAS_TORCH_CLUSTER = True
except Exception:
    HAS_TORCH_CLUSTER = False

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected

    HAS_PYG = True
except Exception:
    HAS_PYG = False

SAFE_CLINICAL_CATEGORICAL_FIELDS = ("location", "side")
GROUP_COLUMN_CANDIDATES = ("patientID", "vesselFileID", "dataset")

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
except Exception:

    def roc_curve(labels, probs):
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    def roc_auc_score(targets, probs):
        return 0.5

    def average_precision_score(targets, probs):
        return 0.0

    def accuracy_score(labels, preds):
        labels = np.asarray(labels)
        preds = np.asarray(preds)
        return float(np.mean(labels == preds)) if len(labels) else 0.0

    def precision_score(labels, preds, zero_division=0):
        labels = np.asarray(labels)
        preds = np.asarray(preds)
        tp = np.sum((labels == 1) & (preds == 1))
        fp = np.sum((labels == 0) & (preds == 1))
        denom = tp + fp
        return float(tp / denom) if denom > 0 else float(zero_division)

    def recall_score(labels, preds, zero_division=0):
        labels = np.asarray(labels)
        preds = np.asarray(preds)
        tp = np.sum((labels == 1) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))
        denom = tp + fn
        return float(tp / denom) if denom > 0 else float(zero_division)

    def confusion_matrix(y_true, preds, labels=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(preds)
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        return np.array([[tn, fp], [fn, tp]], dtype=int)

    def StratifiedKFold(*args, **kwargs):
        raise RuntimeError("StratifiedKFold is unavailable in the local fallback environment")

    def StratifiedGroupKFold(*args, **kwargs):
        raise RuntimeError("StratifiedGroupKFold is unavailable in the local fallback environment")


# Metrics and utilities


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def safe_auc(targets, probs):
    try:
        return roc_auc_score(targets, probs)
    except Exception:
        return 0.5


def safe_ap(targets, probs):
    try:
        return average_precision_score(targets, probs) if len(set(targets)) > 1 else 0.0
    except Exception:
        return 0.0


def select_accuracy_threshold(labels, probs):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=np.float32)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.5
    thresholds = np.unique(np.clip(probs, 1e-6, 1.0 - 1e-6))
    thresholds = np.concatenate([[0.5], thresholds])
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        preds = (probs > threshold).astype(int)
        score = accuracy_score(labels, preds)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def select_youden_threshold(labels, probs):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=np.float32)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, probs)
    finite = np.isfinite(thresholds)
    if not np.any(finite):
        return 0.5
    scores = tpr[finite] - fpr[finite]
    return float(np.clip(thresholds[finite][int(np.argmax(scores))], 1e-6, 1.0 - 1e-6))


def classification_report_dict(labels, probs, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs)
    if threshold is None:
        threshold = 0.5
    preds = (probs > threshold).astype(int)
    auc = safe_auc(labels, probs)
    pr_auc = safe_ap(labels, probs)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_acc": 0.5 * (recall + specificity),
        "f1": f1,
        "threshold": float(threshold),
        "youden_threshold": select_youden_threshold(labels, probs),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# Data discovery and loading


def _match_folder(folder: str, key_to_idx: Dict[str, int]):
    if folder in key_to_idx:
        return key_to_idx[folder]
    base = re.sub(r"_cut\d+$", "", folder)
    return key_to_idx.get(base)


def build_metadata_index(metadata_path: str):
    df = pd.read_csv(metadata_path)
    key_to_idx: Dict[str, int] = {}
    for idx, row in df.iterrows():
        ds = str(row.get("dataset", "")).strip()
        vid = str(row.get("vesselFileID", "")).strip()
        raw_cut = row.get("cutToShow", "cut1")
        cut = str(raw_cut).strip() if pd.notna(raw_cut) else "cut1"
        for key in (ds, vid):
            if key:
                key_to_idx.setdefault(key, idx)
                key_to_idx.setdefault(f"{key}_{cut}", idx)
                key_to_idx.setdefault(f"{key}_cut1", idx)
    return df, key_to_idx


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
    df["rupture_status"] = df["status"].map(normalize_rupture_status)
    df = df.loc[df["rupture_status"].notna()].reset_index(drop=True)
    df["target"] = df["rupture_status"].map(KNOWN_RUPTURE_STATUSES).astype(int)
    for column in ["age", "sex", *SAFE_CLINICAL_CATEGORICAL_FIELDS, *GROUP_COLUMN_CANDIDATES]:
        if column not in df.columns:
            df[column] = "Unknown"
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
    df["sex"] = df["sex"].astype(str).str.lower()
    df["split_group"] = build_split_groups(df)
    return df


def fallback_categories():
    return {field: ["Unknown"] for field in SAFE_CLINICAL_CATEGORICAL_FIELDS}


def clean_category_value(value) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return "Unknown"
    return text


def build_split_groups(df: pd.DataFrame) -> pd.Series:
    groups = []
    for _, row in df.iterrows():
        group = ""
        for field in GROUP_COLUMN_CANDIDATES:
            value = row.get(field, "")
            if pd.notna(value):
                text = str(value).strip()
                if text and text.lower() not in {"nan", "none", "null", "unknown"}:
                    group = text
                    break
        if not group:
            group = str(row.get("filepath", row.name))
        groups.append(group)
    return pd.Series(groups, index=df.index, dtype="string")


def make_cv_splits(df: pd.DataFrame, n_splits: int, seed: int, target_col: str = "target"):
    y = df[target_col].astype(int).to_numpy()
    counts = np.bincount(y, minlength=2)
    min_class = int(counts.min()) if len(counts) > 1 else 0
    effective_splits = max(2, min(int(n_splits), min_class))

    groups = df.get("split_group")
    if groups is not None and groups.nunique(dropna=True) >= effective_splits:
        try:
            splitter = StratifiedGroupKFold(
                n_splits=effective_splits, shuffle=True, random_state=seed
            )
            splits = list(splitter.split(df, y, groups.astype(str)))
            return splits, "StratifiedGroupKFold(split_group)"
        except Exception:
            pass

    splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    return list(splitter.split(df, y)), "StratifiedKFold"


# Point cloud operations


def normalize_points(pts: np.ndarray) -> np.ndarray:
    center = pts.mean(axis=0)
    scale = np.max(np.linalg.norm(pts - center, axis=1))
    return (pts - center) / (scale + 1e-8)


def normalize_features_zscore(feat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = feat.mean(axis=0, keepdims=True)
    sigma = feat.std(axis=0, keepdims=True)
    return (feat - mu) / (sigma + eps)


def resample_cloud(
    pts: np.ndarray,
    target_n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    n = len(pts)
    if n == target_n:
        return np.arange(n)
    choice = rng.choice if rng is not None else np.random.choice
    if n > target_n:
        return choice(n, target_n, replace=False)
    return np.concatenate([np.arange(n), choice(n, target_n - n)])


def jitter(pts: np.ndarray, sigma: float = 0.001, clip: float = 0.05) -> np.ndarray:
    noise = np.clip(sigma * np.random.randn(*pts.shape), -clip, clip)
    return pts + noise


def so3_rotate(pts: np.ndarray) -> np.ndarray:
    theta = float(np.random.uniform(0, 2 * np.pi))
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return pts @ rot.T


def random_point_dropout(
    pts: np.ndarray, flow: np.ndarray, p: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.random.rand(len(pts)) > p
    return pts[mask], flow[mask]


# Flow feature engineering

FLOW_CHANNELS = 11
FLOW_CLIP = float(os.environ.get("v14_FLOW_CLIP", 5.0))


def augment_flow_tensor(
    flow: torch.Tensor, channel_dropout: float = 0.0, noise_std: float = 0.0
) -> torch.Tensor:
    if channel_dropout > 0:
        keep_prob = max(1e-6, 1.0 - channel_dropout)
        keep = (torch.rand(flow.shape[0], 1, flow.shape[2], device=flow.device) < keep_prob).float()
        flow = flow * keep / keep_prob
    if noise_std > 0:
        flow = flow + noise_std * torch.randn_like(flow)
    return flow


def derive_flow_channels(raw_feats: np.ndarray) -> np.ndarray:
    raw_feats = np.asarray(raw_feats, dtype=np.float32)
    if raw_feats.ndim == 1:
        raw_feats = raw_feats.reshape(-1, 1)
    if raw_feats.shape[1] < 3:
        pad = np.zeros((raw_feats.shape[0], 3 - raw_feats.shape[1]), dtype=np.float32)
        raw_feats = np.concatenate([raw_feats, pad], axis=1)
    tawss = np.nan_to_num(raw_feats[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    osi = np.nan_to_num(raw_feats[:, 1], nan=0.0, posinf=0.0, neginf=0.0)
    von_mises = np.nan_to_num(raw_feats[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
    tawss_pos = np.clip(tawss, 0.0, None)
    osi_phys = np.clip(osi, 0.0, 0.499)
    von_pos = np.clip(von_mises, 0.0, None)

    low_tawss = (tawss_pos < np.percentile(tawss_pos, 20)).astype(np.float32)
    high_osi = (osi_phys > 0.2).astype(np.float32)
    combined = tawss_pos * (1.0 - 2.0 * osi_phys)
    vm_norm = von_pos / (np.max(von_pos) + 1e-8)
    risk = (low_tawss * high_osi).astype(np.float32)
    denom = np.maximum((1.0 - 2.0 * osi_phys) * np.maximum(tawss_pos, 1e-6), 1e-6)
    rrt = 1.0 / denom
    rrt[~np.isfinite(rrt)] = 0.0
    if len(rrt) > 1:
        rrt = np.clip(rrt, 0.0, np.percentile(rrt, 99.0))
    log_tawss = np.log1p(tawss_pos).astype(np.float32)
    log_von = np.log1p(von_pos).astype(np.float32)
    log_rrt = np.sign(rrt) * np.log1p(np.abs(rrt))
    shear_ratio = tawss_pos / (von_pos + 1e-6)
    shear_ratio[~np.isfinite(shear_ratio)] = 0.0
    shear_ratio = np.sign(shear_ratio) * np.log1p(np.abs(shear_ratio))
    channels = [
        log_tawss,
        osi_phys,
        log_von,
        low_tawss,
        high_osi,
        combined,
        vm_norm,
        risk,
        log_rrt,
        shear_ratio,
        tawss_pos,
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def summarize_global_features(
    raw_feats: np.ndarray, coords: np.ndarray, include_rrt: bool = True
) -> np.ndarray:
    """Compute global hemodynamic summary features."""
    tawss = raw_feats[:, 0] if raw_feats.shape[1] > 0 else np.zeros(len(coords))
    osi = raw_feats[:, 1] if raw_feats.shape[1] > 1 else np.zeros(len(coords))
    von_mises = raw_feats[:, 2] if raw_feats.shape[1] > 2 else np.zeros(len(coords))

    features = [
        np.mean(tawss),
        np.std(tawss),
        np.percentile(tawss, 25),
        np.percentile(tawss, 75),
        np.mean(osi),
        np.std(osi),
        np.percentile(osi, 25),
        np.percentile(osi, 75),
        np.mean(von_mises),
        np.std(von_mises),
        np.percentile(von_mises, 25),
        np.percentile(von_mises, 75),
    ]
    if include_rrt:
        denom = (1.0 - 2.0 * osi) * tawss
        rrt = np.zeros_like(tawss, dtype=np.float32)
        mask = np.abs(denom) > 1e-8
        rrt[mask] = 1.0 / denom[mask]
        rrt[~np.isfinite(rrt)] = 0.0
        features.extend([np.mean(rrt), np.std(rrt)])
    return np.array(features, dtype=np.float32)


# Clinical data


def compute_clinical_categories(df: pd.DataFrame):
    categories = {}
    for field in SAFE_CLINICAL_CATEGORICAL_FIELDS:
        if field in df.columns:
            categories[field] = sorted(df[field].map(clean_category_value).unique().tolist())
        else:
            categories[field] = ["Unknown"]
    return categories


def build_clinical_matrix(
    df_slice: pd.DataFrame, categories: Dict[str, List[str]], train_stats=None
):
    ages = (
        pd.to_numeric(df_slice.get("age", pd.Series(np.zeros(len(df_slice)))), errors="coerce")
        .fillna(0.0)
        .values.astype(np.float32)
    )
    mu = train_stats["age_mean"] if train_stats else float(ages.mean())
    sigma = train_stats["age_std"] if train_stats else float(ages.std())
    ages = (ages - mu) / (sigma + 1e-6)

    sex_raw = df_slice.get("sex", pd.Series(["Unknown"] * len(df_slice)))
    sexes = pd.to_numeric(sex_raw, errors="coerce")
    if sexes.isna().all():
        sexes = sex_raw.fillna("Unknown").astype(str).str.lower().map({"female": 0.0, "male": 1.0})
    sexes = sexes.fillna(0.0).values.astype(np.float32)

    parts = [ages.reshape(-1, 1), sexes.reshape(-1, 1)]
    for field in SAFE_CLINICAL_CATEGORICAL_FIELDS:
        values = (
            df_slice.get(field, pd.Series(["Unknown"] * len(df_slice)))
            .map(clean_category_value)
            .values
        )
        cats = categories[field]
        one_hot = np.zeros((len(df_slice), len(cats)), dtype=np.float32)
        idx_map = {cat: i for i, cat in enumerate(cats)}
        for i, val in enumerate(values):
            if val in idx_map:
                one_hot[i, idx_map[val]] = 1.0
        parts.append(one_hot)

    clinical = np.concatenate(parts, axis=1).astype(np.float32)
    stats = {"age_mean": float(mu), "age_std": float(sigma)}
    return torch.tensor(clinical, dtype=torch.float32), stats


# Optimizers and scheduling


def get_optimizer(model, lr: float, weight_decay: float):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer, epochs: int, steps_per_epoch: int):
    total_steps = max(1, epochs * max(1, steps_per_epoch))
    return optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=optimizer.defaults["lr"], total_steps=total_steps, pct_start=0.1
    )


def load_checkpoint_weights(path, device=None):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def write_epoch_log(output_dir: Path, fold_idx: int, rows: List[dict]):
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"fold_{fold_idx}_epoch_log.csv"
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build_epoch_row(
    model_name: str,
    fold_idx: int,
    epoch: int,
    train_loss: float,
    val_metrics: Dict[str, float],
    optimizer,
    hyperparams: Optional[Dict[str, float]] = None,
):
    row = {
        "model": model_name,
        "fold": int(fold_idx),
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_metrics.get("loss", 0.0)),
        "val_auc": float(val_metrics.get("auc", 0.0)),
        "val_pr_auc": float(val_metrics.get("pr_auc", 0.0)),
        "val_acc": float(val_metrics.get("acc", 0.0)),
        "val_precision": float(val_metrics.get("precision", 0.0)),
        "val_recall": float(val_metrics.get("recall", 0.0)),
        "val_specificity": float(val_metrics.get("specificity", 0.0)),
        "val_balanced_acc": float(val_metrics.get("balanced_acc", 0.0)),
        "val_f1": float(val_metrics.get("f1", 0.0)),
        "val_threshold": float(val_metrics.get("threshold", 0.5)),
        "val_youden_threshold": float(val_metrics.get("youden_threshold", 0.5)),
        "val_tn": int(val_metrics.get("tn", 0)),
        "val_fp": int(val_metrics.get("fp", 0)),
        "val_fn": int(val_metrics.get("fn", 0)),
        "val_tp": int(val_metrics.get("tp", 0)),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }
    if hyperparams:
        row.update(hyperparams)
    return row


# Point tensor datasets


def _read_hemodynamics_csv(filepath: str):
    df = pd.read_csv(filepath)
    coords = df[["x", "y", "z"]].values.astype(np.float32)
    feat_cols = [c for c in ["tawss", "osi", "von_mises"] if c in df.columns]
    if len(feat_cols) < 3:
        values = df.iloc[:, 3:6].values.astype(np.float32)
    else:
        values = df[feat_cols].values.astype(np.float32)
    values[~np.isfinite(values)] = 0.0
    units_path = Path(filepath).parent / "hemodynamic_units.json"
    if not units_path.exists():
        legacy_scale = float(os.environ.get("v14_LEGACY_WSS_SCALE", 1000.0))
        values[:, 0] *= legacy_scale
        values[:, 2] *= legacy_scale
    return coords, values


def build_case_cache(df: pd.DataFrame):
    cache = {}
    for _, row in df.iterrows():
        filepath = str(row["filepath"])
        if filepath in cache:
            continue
        coords, raw_flow = _read_hemodynamics_csv(filepath)
        cache[filepath] = {
            "coords": coords,
            "flow": raw_flow,
            "flow_channels": derive_flow_channels(raw_flow),
        }
    return cache


def _fit_flow_stats(rows: pd.DataFrame, cache):
    pieces = []
    for _, row in rows.iterrows():
        pieces.append(cache[str(row["filepath"])]["flow_channels"])
    if not pieces:
        mu = np.zeros((1, FLOW_CHANNELS), dtype=np.float32)
        sigma = np.ones((1, FLOW_CHANNELS), dtype=np.float32)
    else:
        values = np.concatenate(pieces, axis=0)
        mu = values.mean(axis=0, keepdims=True).astype(np.float32)
        sigma = values.std(axis=0, keepdims=True).astype(np.float32)
        sigma[sigma < 1e-6] = 1.0
    return {"flow_mean": mu, "flow_std": sigma}


def _standardize_flow(flow: np.ndarray, stats) -> np.ndarray:
    z = (flow - stats["flow_mean"]) / stats["flow_std"]
    z = np.clip(z, -FLOW_CLIP, FLOW_CLIP)
    return z.astype(np.float32)


def _case_to_point_tensors(row, cache, flow_stats, target_n: int = 4096, augment: bool = False):
    item = cache[str(row["filepath"])]
    coords = item["coords"]
    flow = _standardize_flow(item["flow_channels"], flow_stats)
    idx = resample_cloud(coords, target_n)
    xyz = normalize_points(coords[idx]).astype(np.float32)
    flow = flow[idx].astype(np.float32)
    if augment:
        xyz = so3_rotate(xyz)
        xyz = jitter(xyz, sigma=0.003, clip=0.02)
    return xyz.astype(np.float32), flow.astype(np.float32)


def compute_point_fold_tensors(
    train_df: pd.DataFrame, val_df: pd.DataFrame, cache, categories, target_n: int = 4096
):
    flow_stats = _fit_flow_stats(train_df, cache)
    train_clin, clin_stats = build_clinical_matrix(train_df, categories)
    val_clin, _ = build_clinical_matrix(val_df, categories, clin_stats)

    train_xyz, train_flow = [], []
    for _, row in train_df.iterrows():
        xyz, flow = _case_to_point_tensors(row, cache, flow_stats, target_n=target_n, augment=True)
        train_xyz.append(xyz)
        train_flow.append(flow)

    val_xyz, val_flow = [], []
    for _, row in val_df.iterrows():
        xyz, flow = _case_to_point_tensors(row, cache, flow_stats, target_n=target_n, augment=False)
        val_xyz.append(xyz)
        val_flow.append(flow)

    train_labels = torch.tensor(train_df["target"].values.astype(np.int64), dtype=torch.long)
    val_labels = torch.tensor(val_df["target"].values.astype(np.int64), dtype=torch.long)
    return (
        torch.tensor(np.stack(train_xyz), dtype=torch.float32),
        torch.tensor(np.stack(train_flow), dtype=torch.float32),
        train_clin,
        train_labels,
        torch.tensor(np.stack(val_xyz), dtype=torch.float32),
        torch.tensor(np.stack(val_flow), dtype=torch.float32),
        val_clin,
        val_labels,
    )


def _case_to_voxel_tensor(
    row,
    cache,
    flow_stats,
    grid_size: int = 24,
    augment: bool = False,
    include_flow: bool = True,
) -> np.ndarray:
    item = cache[str(row["filepath"])]
    coords = normalize_points(item["coords"]).astype(np.float32)
    flow = _standardize_flow(item["flow_channels"], flow_stats) if include_flow else None
    if augment:
        coords = so3_rotate(coords)
        coords = jitter(coords, sigma=0.003, clip=0.02)

    grid_size = int(grid_size)
    coords = np.clip(coords, -1.0, 1.0)
    ijk = np.floor((coords + 1.0) * 0.5 * (grid_size - 1)).astype(np.int64)
    ix, iy, iz = ijk[:, 0], ijk[:, 1], ijk[:, 2]

    counts = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    input_channels = FLOW_CHANNELS + 1 if include_flow else 1
    sums = (
        np.zeros((FLOW_CHANNELS, grid_size, grid_size, grid_size), dtype=np.float32)
        if include_flow
        else None
    )
    np.add.at(counts, (ix, iy, iz), 1.0)
    if include_flow:
        for channel_idx in range(FLOW_CHANNELS):
            np.add.at(sums[channel_idx], (ix, iy, iz), flow[:, channel_idx])

    volume = np.zeros((input_channels, grid_size, grid_size, grid_size), dtype=np.float32)
    occupied = counts > 0
    volume[0, occupied] = 1.0
    if include_flow and np.any(occupied):
        volume[1:, occupied] = sums[:, occupied] / counts[occupied]
    return volume


def compute_voxel_fold_tensors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cache,
    grid_size: int = 24,
    include_flow: bool = True,
):
    flow_stats = _fit_flow_stats(train_df, cache) if include_flow else None

    train_volumes = [
        _case_to_voxel_tensor(
            row, cache, flow_stats, grid_size=grid_size, augment=True, include_flow=include_flow
        )
        for _, row in train_df.iterrows()
    ]
    val_volumes = [
        _case_to_voxel_tensor(
            row, cache, flow_stats, grid_size=grid_size, augment=False, include_flow=include_flow
        )
        for _, row in val_df.iterrows()
    ]

    train_labels = torch.tensor(train_df["target"].values.astype(np.int64), dtype=torch.long)
    val_labels = torch.tensor(val_df["target"].values.astype(np.int64), dtype=torch.long)
    train_aux = torch.zeros((len(train_df), 1), dtype=torch.float32)
    val_aux = torch.zeros((len(val_df), 1), dtype=torch.float32)
    return (
        torch.tensor(np.stack(train_volumes), dtype=torch.float32),
        train_aux,
        train_aux,
        train_labels,
        torch.tensor(np.stack(val_volumes), dtype=torch.float32),
        val_aux,
        val_aux,
        val_labels,
    )


def build_point_loaders(train_tensors, val_tensors, batch_size: int = 6):
    train_dataset = TensorDataset(*train_tensors)
    val_dataset = TensorDataset(*val_tensors)
    labels = train_tensors[-1].numpy()
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = 1.0 / np.maximum(counts[labels], 1.0)
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=len(train_dataset) > batch_size,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate_tensor_model(model, loader, criterion, device, use_amp: bool = True):
    metrics, _, _, _ = predict_tensor_model(model, loader, criterion, device, use_amp)
    return metrics


def predict_tensor_model(model, loader, criterion, device, use_amp: bool = True):
    model.eval()
    total_loss = 0.0
    probs, labels = [], []
    with torch.no_grad():
        for xb, fb, cb, yb in loader:
            xb, fb, cb, yb = xb.to(device), fb.to(device), cb.to(device), yb.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits, _ = model(xb, fb, cb)
                loss = criterion(logits, yb)
            total_loss += float(loss.item()) * len(yb)
            probs.extend(F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
            labels.extend(yb.detach().cpu().numpy())
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    metrics = classification_report_dict(labels, probs)
    metrics["loss"] = total_loss / max(1, len(labels))
    preds = (probs > 0.5).astype(np.int64)
    return metrics, labels, probs, preds


def append_prediction_rows(
    rows: List[dict], fold_idx: int, val_df: pd.DataFrame, labels, probs, preds
):
    for row_idx, (_, row) in enumerate(val_df.iterrows()):
        case_id = row.get("case_id", "")
        if not case_id:
            vessel = str(row.get("vesselFileID", row.get("dataset", ""))).strip()
            cut = str(row.get("cutToShow", "")).strip()
            case_id = (
                f"{vessel}_{cut}"
                if vessel and cut
                else Path(str(row.get("filepath", ""))).parent.name
            )
        rows.append(
            {
                "fold": fold_idx + 1,
                "case_id": case_id,
                "vesselFileID": row.get("vesselFileID", ""),
                "cutToShow": row.get("cutToShow", ""),
                "filepath": row.get("filepath", ""),
                "label": int(labels[row_idx]),
                "prob": float(probs[row_idx]),
                "pred": int(preds[row_idx]),
            }
        )


# Graph data


def _chunked_knn_graph(pos: torch.Tensor, k: int, chunk_size: int = 512) -> torch.Tensor:
    """Build neighbor-to-query edges without allocating an N x N distance matrix."""
    n = int(pos.shape[0])
    k = min(int(k), max(0, n - 1))
    if k == 0:
        return torch.empty((2, 0), dtype=torch.long)

    sources = []
    targets = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        distances = torch.cdist(pos[start:stop], pos)
        local_rows = torch.arange(stop - start)
        global_rows = torch.arange(start, stop)
        distances[local_rows, global_rows] = float("inf")
        neighbors = torch.topk(distances, k=k, largest=False).indices
        sources.append(neighbors.reshape(-1))
        targets.append(global_rows.view(-1, 1).expand(-1, k).reshape(-1))
    return torch.stack([torch.cat(sources), torch.cat(targets)], dim=0)


def load_graph_case(
    filepath: str,
    label: int,
    augment: bool = False,
    k: int = 16,
    target_n: int = 4096,
    include_flow: bool = True,
    make_undirected: bool = True,
    store_edge_attr: bool = True,
    knn_chunk_size: int = 512,
    rng: Optional[np.random.Generator] = None,
):
    if not HAS_PYG:
        raise RuntimeError("torch_geometric is required for graph loading")
    coords, raw_flow = _read_hemodynamics_csv(filepath)
    idx = resample_cloud(coords, target_n, rng=rng)
    xyz = normalize_points(coords[idx]).astype(np.float32)
    flow = normalize_features_zscore(derive_flow_channels(raw_flow)[idx]).astype(np.float32)
    if augment:
        xyz = so3_rotate(jitter(xyz, sigma=0.003, clip=0.02))
    features = np.concatenate([xyz, flow], axis=1) if include_flow else xyz
    x = torch.tensor(features, dtype=torch.float32)
    pos = torch.tensor(xyz, dtype=torch.float32)
    if HAS_TORCH_CLUSTER:
        edge_index = torch_cluster.knn_graph(pos, k=min(k, len(xyz) - 1), loop=False)
    else:
        edge_index = _chunked_knn_graph(pos, k=k, chunk_size=knn_chunk_size)
    if make_undirected:
        edge_index = to_undirected(edge_index)
    y = torch.tensor([int(label)], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, pos=pos, y=y)
    if store_edge_attr:
        src, dst = edge_index
        delta = pos[dst] - pos[src]
        data.edge_attr = torch.cat([delta, torch.norm(delta, dim=1, keepdim=True)], dim=1)
    return data
