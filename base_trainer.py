# Version 12 source snapshot
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
    from sklearn.model_selection import StratifiedKFold
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


# Metrics & Utilities


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


def classification_report_dict(labels, probs, threshold=None):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs)
    if threshold is None:
        threshold = select_accuracy_threshold(labels, probs)
    preds = (probs > threshold).astype(int)
    auc = safe_auc(labels, probs)
    pr_auc = safe_ap(labels, probs)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# Data Discovery & Loading


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
    df["target"] = (df["status"].astype(str).str.lower() == "ruptured").astype(int)
    for column in ["age", "sex", "location", "hospital", "source", "side"]:
        if column not in df.columns:
            df[column] = "Unknown"
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
    df["sex"] = df["sex"].astype(str).str.lower()
    return df


def fallback_categories():
    return {
        "location": ["Unknown"],
        "hospital": ["Unknown"],
        "source": ["Unknown"],
        "side": ["Unknown"],
    }


def clean_category_value(value) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return "Unknown"
    return text


# Point Cloud Operations


def normalize_points(pts: np.ndarray) -> np.ndarray:
    center = pts.mean(axis=0)
    scale = np.max(np.linalg.norm(pts - center, axis=1))
    return (pts - center) / (scale + 1e-8)


def normalize_features_zscore(feat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = feat.mean(axis=0, keepdims=True)
    sigma = feat.std(axis=0, keepdims=True)
    return (feat - mu) / (sigma + eps)


def resample_cloud(pts: np.ndarray, target_n: int) -> np.ndarray:
    n = len(pts)
    if n == target_n:
        return np.arange(n)
    if n > target_n:
        return np.random.choice(n, target_n, replace=False)
    return np.concatenate([np.arange(n), np.random.choice(n, target_n - n)])


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


# Flow Feature Engineering

FLOW_CHANNELS = 11


def derive_flow_channels(raw_feats: np.ndarray) -> np.ndarray:
    raw_feats = np.asarray(raw_feats, dtype=np.float32)
    if raw_feats.ndim == 1:
        raw_feats = raw_feats.reshape(-1, 1)
    if raw_feats.shape[1] < 3:
        pad = np.zeros((raw_feats.shape[0], 3 - raw_feats.shape[1]), dtype=np.float32)
        raw_feats = np.concatenate([raw_feats, pad], axis=1)
    tawss = raw_feats[:, 0]
    osi = raw_feats[:, 1]
    von_mises = raw_feats[:, 2]
    low_tawss = (tawss < np.percentile(tawss, 20)).astype(np.float32)
    high_osi = (osi > 0.2).astype(np.float32)
    combined = tawss * (1.0 - 2.0 * osi)
    vm_norm = von_mises / (np.max(von_mises) + 1e-8)
    risk = (low_tawss * high_osi).astype(np.float32)
    denom = (1.0 - 2.0 * osi) * tawss
    rrt = np.zeros_like(tawss, dtype=np.float32)
    mask = np.abs(denom) > 1e-8
    rrt[mask] = 1.0 / denom[mask]
    rrt[~np.isfinite(rrt)] = 0.0
    log_von = np.log1p(np.abs(von_mises)).astype(np.float32)
    shear_ratio = tawss / (von_mises + 1e-6)
    shear_ratio[~np.isfinite(shear_ratio)] = 0.0
    channels = [
        tawss,
        osi,
        von_mises,
        low_tawss,
        high_osi,
        combined,
        vm_norm,
        risk,
        rrt,
        log_von,
        shear_ratio,
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


# Clinical Data


def compute_clinical_categories(df: pd.DataFrame):
    categories = {}
    for field in ["location", "hospital", "source", "side"]:
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
    for field in ["location", "hospital", "source", "side"]:
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


# Optimizers & Scheduling


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
        "val_tn": int(val_metrics.get("tn", 0)),
        "val_fp": int(val_metrics.get("fp", 0)),
        "val_fn": int(val_metrics.get("fn", 0)),
        "val_tp": int(val_metrics.get("tp", 0)),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }
    if hyperparams:
        row.update(hyperparams)
    return row


# Point Tensor Datasets


def _read_hemodynamics_csv(filepath: str):
    df = pd.read_csv(filepath)
    coords = df[["x", "y", "z"]].values.astype(np.float32)
    feat_cols = [c for c in ["tawss", "osi", "von_mises"] if c in df.columns]
    if len(feat_cols) < 3:
        values = df.iloc[:, 3:6].values.astype(np.float32)
    else:
        values = df[feat_cols].values.astype(np.float32)
    values[~np.isfinite(values)] = 0.0
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
    return ((flow - stats["flow_mean"]) / stats["flow_std"]).astype(np.float32)


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
    metrics = classification_report_dict(labels, probs)
    metrics["loss"] = total_loss / max(1, len(labels))
    return metrics


# Graph Data


def load_graph_case(
    filepath: str, label: int, augment: bool = False, k: int = 16, target_n: int = 4096
):
    if not HAS_PYG:
        raise RuntimeError("torch_geometric is required for graph loading")
    coords, raw_flow = _read_hemodynamics_csv(filepath)
    idx = resample_cloud(coords, target_n)
    xyz = normalize_points(coords[idx]).astype(np.float32)
    flow = normalize_features_zscore(derive_flow_channels(raw_flow)[idx]).astype(np.float32)
    if augment:
        xyz = so3_rotate(jitter(xyz, sigma=0.003, clip=0.02))
    x = torch.tensor(np.concatenate([xyz, flow], axis=1), dtype=torch.float32)
    pos = torch.tensor(xyz, dtype=torch.float32)
    if HAS_TORCH_CLUSTER:
        edge_index = torch_cluster.knn_graph(pos, k=min(k, len(xyz) - 1), loop=False)
    else:
        dists = torch.cdist(pos, pos)
        _, nn_idx = torch.topk(dists, k=min(k + 1, len(xyz)), largest=False)
        src = torch.arange(len(xyz)).view(-1, 1).expand_as(nn_idx[:, 1:]).reshape(-1)
        dst = nn_idx[:, 1:].reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)
    edge_index = to_undirected(edge_index)
    y = torch.tensor([int(label)], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, pos=pos, y=y)
