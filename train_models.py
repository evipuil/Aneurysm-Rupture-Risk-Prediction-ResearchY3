# Version 11 source snapshot
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

try:
    import torch_cluster

    HAS_TORCH_CLUSTER = True
except Exception:
    HAS_TORCH_CLUSTER = False

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.nn import (
        BatchNorm,
        GATv2Conv,
        global_add_pool,
        global_max_pool,
        global_mean_pool,
    )
    from torch_geometric.utils import to_undirected

    HAS_PYG = True
except Exception:
    HAS_PYG = False

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


# Local implementations keep Version 11 self-contained.
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
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


def classification_report_dict(labels, probs, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    preds = (np.asarray(probs) > threshold).astype(int)
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
    df["sex"] = df["sex"].fillna("Unknown").astype(str).str.lower()
    df["location"] = df["location"].fillna("Unknown").astype(str)
    df["hospital"] = df["hospital"].fillna("Unknown").astype(str)
    df["source"] = df["source"].fillna("Unknown").astype(str)
    df["side"] = df["side"].fillna("Unknown").astype(str)
    return df


def normalize_points(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts.mean(axis=0)
    scale = np.max(np.linalg.norm(pts, axis=1)) if len(pts) else 1.0
    return (pts / (scale + 1e-8)).astype(np.float32)


def normalize_features_zscore(feats: np.ndarray, clip: float = 3.0) -> np.ndarray:
    mu = feats.mean(axis=0)
    sigma = feats.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return np.clip((feats - mu) / sigma, -clip, clip).astype(np.float32)


def resample_cloud(pts: np.ndarray, target_n: int, rng: Optional[np.random.Generator] = None):
    rng = rng or np.random.default_rng()
    n = len(pts)
    if n == target_n:
        return np.arange(n)
    if n > target_n:
        return rng.choice(n, target_n, replace=False)
    pad = rng.choice(n, target_n - n, replace=True)
    return np.concatenate([np.arange(n), pad])


def so3_rotate(pts: np.ndarray) -> np.ndarray:
    rx, ry, rz = np.random.uniform(0, 2 * np.pi, 3)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rotation_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    rotation_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rotation_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    rotation = rotation_z @ rotation_y @ rotation_x
    return (pts.astype(np.float32) @ rotation.T).astype(np.float32)


def jitter(pts: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
    noise = np.clip(sigma * np.random.randn(*pts.shape), -clip, clip)
    return (pts + noise).astype(np.float32)


def random_point_dropout(pts: np.ndarray, feats: Optional[np.ndarray] = None, p: float = 0.1):
    dropped = np.random.rand(pts.shape[0]) < p
    if not dropped.any():
        return pts, feats

    kept_indices = np.flatnonzero(~dropped)
    if len(kept_indices) == 0:
        return pts, feats

    replacements = np.random.choice(kept_indices, dropped.sum())
    pts = pts.copy()
    pts[dropped] = pts[replacements]
    if feats is not None:
        feats = feats.copy()
        feats[dropped] = feats[replacements]
    return pts, feats


def summarize_global_features(
    raw_feats: np.ndarray, pts: np.ndarray, include_rrt: bool = False
) -> np.ndarray:
    tawss = raw_feats[:, 0]
    osi = raw_feats[:, 1]
    von_mises = raw_feats[:, 2]
    features = [
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
        np.mean(von_mises),
        np.std(von_mises),
        np.max(von_mises),
        np.percentile(von_mises, 95),
        np.percentile(von_mises, 99),
    ]
    if include_rrt:
        denominator = (1.0 - 2.0 * osi) * tawss
        rrt = np.zeros_like(tawss, dtype=np.float32)
        valid = np.abs(denominator) > 1e-8
        rrt[valid] = 1.0 / denominator[valid]
        rrt[~np.isfinite(rrt)] = 0.0
        features.extend(
            [
                np.mean(rrt),
                np.std(rrt),
                np.max(rrt),
                np.percentile(rrt, 95),
                float(np.mean(rrt > np.percentile(rrt, 90))),
            ]
        )

    distances = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    try:
        eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1]
        eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-8)
    except np.linalg.LinAlgError:
        eigenvalues = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    features.extend(
        [
            float(np.max(distances)),
            float(np.std(distances)),
            float(np.max(distances) / (np.mean(distances) + 1e-6)),
            float(eigenvalues[0]),
            float(eigenvalues[1]),
            float(eigenvalues[0] / (eigenvalues[2] + 1e-6)),
        ]
    )
    summary = np.asarray(features, dtype=np.float32)
    summary = np.sign(summary) * np.log1p(np.abs(summary))
    return np.clip(summary, -10.0, 10.0)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[key] = value.detach().clone()

    def apply_to(self, model):
        backup = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return backup

    def restore(self, model, backup):
        model.load_state_dict(backup)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = float(label_smoothing)
        if alpha is None:
            self.alpha = None
        else:
            # register as buffer so it moves with the module
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, targets):
        try:
            ce = F.cross_entropy(
                logits, targets, reduction="none", label_smoothing=self.label_smoothing
            )
        except TypeError:
            class_count = logits.shape[1]
            log_probabilities = F.log_softmax(logits, dim=1)
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.label_smoothing / max(class_count - 1, 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            ce = -(true_dist * log_probabilities).sum(dim=1)

        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if hasattr(self, "alpha") and self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = loss * alpha_t
        return loss.mean()


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False, dropout=0.0):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.dropout = dropout
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch
        self.need_proj = in_channel != mlp[-1]
        if self.need_proj:
            self.proj = nn.Conv2d(in_channel, mlp[-1], 1)
            self.proj_bn = nn.BatchNorm2d(mlp[-1])

    def _index_points(self, points, idx):
        B = points.shape[0]
        view_shape = [1] * idx.ndim
        view_shape[0] = B
        batch_indices = torch.arange(B, device=points.device).view(view_shape).expand_as(idx)
        return points[batch_indices, idx]

    def _square_distance(self, src, dst):
        return torch.cdist(src, dst).pow(2)

    def farthest_point_sample(self, xyz, npoint):
        B, N, _ = xyz.shape
        device = xyz.device
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.full((B, N), 1e10, device=device)
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_ar = torch.arange(B, dtype=torch.long, device=device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_ar, farthest].unsqueeze(1)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)
            mask = dist < distance
            distance = torch.where(mask, dist, distance)
            farthest = torch.max(distance, dim=-1).indices
        return centroids

    def query_ball_point(self, radius, nsample, xyz, new_xyz):
        B, N, _ = xyz.shape
        S = new_xyz.shape[1]
        device = xyz.device
        group_idx = torch.arange(N, device=device).view(1, 1, N).expand(B, S, N).clone()
        sqrdists = self._square_distance(new_xyz, xyz)
        group_idx[sqrdists > radius**2] = N
        group_idx, _ = torch.sort(group_idx, dim=-1)
        group_idx = group_idx[:, :, :nsample]
        first = group_idx[:, :, :1].expand(-1, -1, nsample)
        mask = group_idx == N
        group_idx = torch.where(mask, first, group_idx)
        return group_idx

    def sample_and_group(self, npoint, radius, nsample, xyz, points):
        B, _, C = xyz.shape
        fps_idx = self.farthest_point_sample(xyz, npoint)
        new_xyz = self._index_points(xyz, fps_idx)
        idx = self.query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = self._index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C)
        if points is not None:
            grouped_points = self._index_points(points, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm
        return new_points, new_xyz

    def sample_and_group_all(self, xyz, points):
        B, N, C = xyz.shape
        new_xyz = torch.zeros(B, 1, C, device=xyz.device)
        grouped_xyz = xyz.view(B, 1, N, C)
        if points is not None:
            new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        else:
            new_points = grouped_xyz
        return new_points, new_xyz

    def forward(self, xyz, points=None):
        if self.group_all:
            new_points, new_xyz = self.sample_and_group_all(xyz, points)
        else:
            new_points, new_xyz = self.sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        x = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            x = bn(conv(x))
            x = F.gelu(x)
            if self.dropout > 0 and self.training:
                x = F.dropout2d(x, p=self.dropout, training=True)
        if self.need_proj:
            sc = self.proj_bn(self.proj(new_points.permute(0, 3, 2, 1)))
            x = x + sc
            x = F.gelu(x)
        new_points = torch.max(x, dim=2).values
        return new_points.permute(0, 2, 1), new_xyz


SEED = int(os.environ.get("V11_SEED", 42))
CV_SEED = int(os.environ.get("V11_CV_SEED", 42))
METADATA_PATH = os.environ.get("V11_METADATA", "metadata.csv")
DATA_DIR = os.environ.get("V11_DATA_DIR", "predictions/pinn_corrected")
OUTPUT_ROOT = Path(os.environ.get("V11_OUTPUT_DIR", "results_v11_suite"))
N_FOLDS = int(os.environ.get("V11_FOLDS", 5))
BATCH_SIZE = int(os.environ.get("V11_BATCH", 6))
EPOCHS = int(os.environ.get("V11_EPOCHS", 220))
LR = float(os.environ.get("V11_LR", 3e-4))
WEIGHT_DECAY = float(os.environ.get("V11_WD", 2e-4))
TARGET_N = int(os.environ.get("V11_TARGET_N", 4096))
EARLY_STOP_PATIENCE = int(os.environ.get("V11_PATIENCE", 35))
USE_AMP = os.environ.get("V11_AMP", "1").lower() in {"1", "true", "yes"}
ONLY_FOLD = None
LABEL_SMOOTHING = float(os.environ.get("V11_LABEL_SMOOTH", 0.05))
AUX_LOSS_WEIGHT = float(os.environ.get("V11_AUX_WEIGHT", 0.15))
DROPOUT = float(os.environ.get("V11_DROPOUT", 0.30))
EMBED_DIM = int(os.environ.get("V11_EMBED_DIM", 256))
GNN_HIDDEN = int(os.environ.get("V11_GNN_HIDDEN", 128))
FLOW_CHANNELS = 11
GRAPH_FEATURE_DIM = 28

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_threshold_metrics(labels, probs, threshold: float = 0.5):
    metrics = classification_report_dict(labels, probs, threshold=threshold)
    preds = (np.asarray(probs) > threshold).astype(int)
    if len(set(labels)) > 1:
        fpr, tpr, thr = roc_curve(labels, probs)
    else:
        fpr, tpr, thr = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
    metrics.update({"threshold": threshold, "preds": preds})
    return metrics, fpr, tpr, thr


def compute_clinical_categories(df: pd.DataFrame):
    categories = {}
    for field in ["location", "hospital", "source", "side"]:
        if field in df.columns:
            categories[field] = sorted(df[field].fillna("Unknown").astype(str).unique().tolist())
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
            .fillna("Unknown")
            .astype(str)
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


def derive_flow_channels(raw_feats: np.ndarray) -> np.ndarray:
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


def load_point_case(path: str, label: int, augment: bool):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    try:
        pts = df[["x", "y", "z"]].values.astype(np.float32)
    except KeyError:
        pts = df.iloc[:, :3].values.astype(np.float32)

    raw_cols = []
    for key in ("tawss", "osi", "von"):
        match = next((c for c in df.columns if key in c), None)
        raw_cols.append(df[match].values if match else np.zeros(len(df)))
    raw = np.stack(raw_cols, axis=1).astype(np.float32)

    idx = resample_cloud(pts, TARGET_N)
    pts = pts[idx]
    flow = derive_flow_channels(raw)[idx]

    pts = normalize_points(pts)
    flow = normalize_features_zscore(flow)

    if augment:
        pts = so3_rotate(pts)
        pts = jitter(pts)
        pts = pts * np.random.uniform(0.95, 1.05)
        pts, flow = random_point_dropout(pts, flow, p=0.1)

    return (
        torch.tensor(pts, dtype=torch.float32),
        torch.tensor(flow, dtype=torch.float32),
    )


def load_graph_case(path: str, label: int, augment: bool):
    if not HAS_PYG:
        raise RuntimeError("torch_geometric is required for the gnn model family")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    coords = df[["x", "y", "z"]].values.astype(np.float32)
    feat_cols = []
    for key in ("tawss", "osi", "von"):
        match = next((c for c in df.columns if key in c), None)
        feat_cols.append(df[match].values if match else np.zeros(len(df)))
    raw = np.stack(feat_cols, axis=1).astype(np.float32)

    graph_feats = summarize_global_features(raw, coords, include_rrt=True)
    idx = resample_cloud(coords, TARGET_N)
    coords = coords[idx]
    raw = raw[idx]
    coords = normalize_points(coords)

    node_feats = np.concatenate(
        [coords, np.clip(np.log1p(np.clip(raw, 1e-6, None)), -3.0, 3.0).astype(np.float32)], axis=1
    )

    pos = torch.tensor(coords, dtype=torch.float32)
    x = torch.tensor(node_feats, dtype=torch.float32)

    if augment:
        theta = float(np.random.uniform(0, 2 * np.pi))
        c, s = np.cos(theta), np.sin(theta)
        rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        pos = pos @ rot.T
        pos = pos + torch.clamp(0.01 * torch.randn_like(pos), -0.03, 0.03)
        pos = pos * float(np.random.uniform(0.9, 1.1))
        x = x.clone()
        x[:, :3] = pos
        x[:, 3:] = x[:, 3:] + torch.clamp(0.02 * torch.randn_like(x[:, 3:]), -0.05, 0.05)

    edge_index = knn_graph(pos, k=16)
    edge_index = to_undirected(edge_index)
    if augment and edge_index.size(1) > 0:
        mask = torch.rand(edge_index.size(1), device=edge_index.device) > 0.1
        edge_index = edge_index[:, mask]
        if edge_index.size(1) < 10:
            edge_index = to_undirected(knn_graph(pos, k=16))
    src, dst = edge_index
    edge_attr = torch.norm(pos[src] - pos[dst], dim=-1, keepdim=True)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        y=torch.tensor(label, dtype=torch.long),
    )
    data.graph_features = torch.from_numpy(graph_feats).float().unsqueeze(0)
    return data


def knn_graph(pos: torch.Tensor, k: int) -> torch.Tensor:
    if HAS_TORCH_CLUSTER:
        return torch_cluster.knn_graph(pos, k=k)
    dists = torch.cdist(pos, pos)
    dists.fill_diagonal_(float("inf"))
    _, idx = torch.topk(dists, k=min(k, max(pos.size(0) - 1, 1)), dim=1, largest=False)
    src = torch.arange(pos.size(0), device=pos.device).unsqueeze(1).expand_as(idx).reshape(-1)
    dst = idx.reshape(-1)
    return torch.stack([src, dst], dim=0)


class PointNeXtSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False, dropout=0.0):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.dropout = dropout
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch
        self.need_proj = in_channel != mlp[-1]
        if self.need_proj:
            self.proj = nn.Conv2d(in_channel, mlp[-1], 1)
            self.proj_bn = nn.BatchNorm2d(mlp[-1])

    def forward(self, xyz, points=None):
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_points, new_xyz = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        x = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            x = bn(conv(x))
            x = F.gelu(x)
            if self.dropout > 0 and self.training:
                x = F.dropout2d(x, p=self.dropout, training=True)
        if self.need_proj:
            sc = self.proj_bn(self.proj(new_points.permute(0, 3, 2, 1)))
            x = F.gelu(x + sc)
        new_points = torch.max(x, dim=2).values
        return new_points.permute(0, 2, 1), new_xyz


def sample_and_group(npoint, radius, nsample, xyz, points):
    def _square_distance(src, dst):
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src**2, -1).view(B, N, 1)
        dist += torch.sum(dst**2, -1).view(B, 1, M)
        return dist

    def farthest_point_sample(xyz, npoint):
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
        distance = torch.full((B, N), 1e10, device=xyz.device)
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
        batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids

    def _index_points(points, idx):
        if idx.dim() == 2:
            B = points.shape[0]
            batch_indices = (
                torch.arange(B, dtype=torch.long, device=points.device).view(B, 1).expand_as(idx)
            )
            return points[batch_indices, idx, :]
        if idx.dim() == 3:
            B = points.shape[0]
            batch_indices = (
                torch.arange(B, dtype=torch.long, device=points.device).view(B, 1, 1).expand_as(idx)
            )
            return points[batch_indices, idx, :]
        raise ValueError(f"Unsupported index tensor shape: {tuple(idx.shape)}")

    def query_ball_point(radius, nsample, xyz, new_xyz):
        B, N, _ = xyz.shape
        S = new_xyz.shape[1]
        device = xyz.device
        group_idx = torch.arange(N, device=device).view(1, 1, N).expand(B, S, N).clone()
        sqrdists = _square_distance(new_xyz, xyz)
        group_idx[sqrdists > radius**2] = N
        group_idx, _ = torch.sort(group_idx, dim=-1)
        group_idx = group_idx[:, :, :nsample]
        first = group_idx[:, :, :1].expand(-1, -1, nsample)
        mask = group_idx == N
        group_idx = torch.where(mask, first, group_idx)
        return group_idx

    B, _, C = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = _index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = _index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C)
    if points is not None:
        grouped_points = _index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_points, new_xyz


def sample_and_group_all(xyz, points):
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=xyz.device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        backbone: str = "pointnet2",
        embed_dim: int = EMBED_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        sa_cls = PointNetSetAbstraction if backbone == "pointnet2" else PointNeXtSetAbstraction
        first_in = 3 + feature_dim if feature_dim > 0 else 3
        self.sa1 = sa_cls(512, 0.2, 32, in_channel=first_in, mlp=[64, 64, 128], dropout=0.1)
        self.sa2 = sa_cls(128, 0.4, 64, in_channel=131, mlp=[128, 128, 256], dropout=0.1)
        self.sa3 = sa_cls(
            None, None, None, in_channel=259, mlp=[256, 512, 1024], group_all=True, dropout=0.1
        )
        self.proj = nn.Sequential(
            nn.Linear(1024, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        self.feature_dim = feature_dim

    def forward(self, xyz, features=None):
        if self.feature_dim > 0:
            l1, l1_xyz = self.sa1(xyz, features)
            l2, l2_xyz = self.sa2(l1_xyz, l1)
            l3, _ = self.sa3(l2_xyz, l2)
        else:
            l1, l1_xyz = self.sa1(xyz, None)
            l2, l2_xyz = self.sa2(l1_xyz, l1)
            l3, _ = self.sa3(l2_xyz, l2)
        token = self.proj(l3.reshape(l3.shape[0], -1))
        aux_logits = self.aux_head(token)
        return token, aux_logits


class ClinicalEncoder(nn.Module):
    def __init__(self, clinical_dim: int, embed_dim: int = EMBED_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clinical_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        token = self.net(x)
        return token, self.aux_head(token)


class BranchFusionClassifier(nn.Module):
    def __init__(self, branch_modules: Sequence[nn.Module], embed_dim: int = EMBED_DIM):
        super().__init__()
        # branch_modules is a sequence of dicts: {"name": str, "module": nn.Module}
        names = [b["name"] for b in branch_modules]
        modules = [b["module"] for b in branch_modules]
        self.branch_names = names
        self.branches = nn.ModuleList(modules)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        fused_dim = embed_dim * 4
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, xyz=None, flow=None, clinical=None):
        tokens = []
        aux_logits = []
        for name, module in zip(self.branch_names, self.branches):
            if name == "geometry":
                token, aux = module(xyz, None)
            elif name == "flow":
                token, aux = module(xyz, flow)
            elif name == "clinical":
                token, aux = module(clinical)
            else:
                raise ValueError(f"Unknown branch type: {name}")
            tokens.append(token)
            aux_logits.append(aux)

        token_stack = torch.stack(tokens, dim=1)
        scores = self.gate(token_stack).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        fused = torch.sum(weights.unsqueeze(-1) * token_stack, dim=1)
        summary = torch.cat(
            [
                fused,
                token_stack.mean(dim=1),
                token_stack.std(dim=1, unbiased=False),
                token_stack.max(dim=1).values,
            ],
            dim=-1,
        )
        logits = self.head(summary)
        return logits, aux_logits


class PairFusionHead(nn.Module):
    """Fuse two modality tokens while retaining a dedicated classifier head."""

    def __init__(self, embed_dim: int = EMBED_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 4, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, first, second):
        token_stack = torch.stack([first, second], dim=1)
        weights = torch.softmax(self.gate(token_stack).squeeze(-1), dim=1)
        fused = torch.sum(weights.unsqueeze(-1) * token_stack, dim=1)
        summary = torch.cat(
            [
                fused,
                token_stack.mean(dim=1),
                token_stack.std(dim=1, unbiased=False),
                token_stack.max(dim=1).values,
            ],
            dim=-1,
        )
        return self.head(summary)


class ClinicalAnchoredLateFusionClassifier(nn.Module):
    """Preserve geometry-clinical prediction and add constrained flow signal."""

    def __init__(
        self,
        geometry: nn.Module,
        flow: nn.Module,
        clinical: nn.Module,
        embed_dim: int = EMBED_DIM,
        max_flow_weight: float = 0.5,
        initial_flow_weight: float = 0.25,
    ):
        super().__init__()
        self.geometry = geometry
        self.flow = flow
        self.clinical = clinical
        self.geometry_clinical_head = PairFusionHead(embed_dim=embed_dim)
        self.geometry_flow_head = PairFusionHead(embed_dim=embed_dim)
        self.max_flow_weight = float(max_flow_weight)
        ratio = min(max(initial_flow_weight / max(self.max_flow_weight, 1e-6), 1e-4), 1.0 - 1e-4)
        self.flow_weight_logit = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        )

    def current_flow_weight(self):
        return self.max_flow_weight * torch.sigmoid(self.flow_weight_logit)

    def forward(self, xyz=None, flow=None, clinical=None):
        geometry_token, geometry_aux = self.geometry(xyz, None)
        flow_token, flow_aux = self.flow(xyz, flow)
        clinical_token, clinical_aux = self.clinical(clinical)

        base_logits = self.geometry_clinical_head(geometry_token, clinical_token)
        flow_logits = self.geometry_flow_head(geometry_token, flow_token)
        base_margin = base_logits[:, 1] - base_logits[:, 0]
        flow_margin = flow_logits[:, 1] - flow_logits[:, 0]
        flow_weight = self.current_flow_weight()
        final_margin = (1.0 - flow_weight) * base_margin + flow_weight * flow_margin
        logits = torch.stack([-0.5 * final_margin, 0.5 * final_margin], dim=1)

        auxiliary = {
            "geometry_clinical": (base_logits, 0.35),
            "geometry_flow": (flow_logits, 0.20),
            "geometry": (geometry_aux, 0.03),
            "flow": (flow_aux, 0.03),
            "clinical": (clinical_aux, 0.03),
        }
        return logits, auxiliary


class GraphEncoder(nn.Module):
    def __init__(
        self, hidden_dim: int = GNN_HIDDEN, embed_dim: int = EMBED_DIM, dropout: float = DROPOUT
    ):
        super().__init__()
        if not HAS_PYG:
            raise RuntimeError("torch_geometric is required for GraphEncoder")
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        in_ch = 6
        for _ in range(3):
            self.convs.append(
                GATv2Conv(in_ch, hidden_dim // 4, heads=4, edge_dim=1, dropout=dropout * 0.5)
            )
            self.bns.append(BatchNorm(hidden_dim))
            in_ch = hidden_dim
        self.graph_fc = nn.Sequential(
            nn.Linear(GRAPH_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 3 + hidden_dim // 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        self.dropout = dropout

    def forward(self, batch):
        x, edge_index, edge_attr, b = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_new = conv(x, edge_index, edge_attr=edge_attr)
            x_new = F.gelu(bn(x_new))
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x_new if i == 0 else x + x_new
        pooled = torch.cat(
            [global_mean_pool(x, b), global_max_pool(x, b), global_add_pool(x, b)], dim=1
        )
        gf = self.graph_fc(batch.graph_features)
        token = self.proj(torch.cat([pooled, gf], dim=1))
        return token, self.aux_head(token)


def build_point_model(model_name: str, backbone: str, clinical_dim: int):
    branches = []
    if model_name == "geometry":
        branches.append({"name": "geometry", "module": PointEncoder(0, backbone=backbone)})
    elif model_name == "flow_geometry":
        branches.append({"name": "geometry", "module": PointEncoder(0, backbone=backbone)})
        branches.append({"name": "flow", "module": PointEncoder(FLOW_CHANNELS, backbone=backbone)})
    elif model_name == "geometry_clinical":
        branches.append({"name": "geometry", "module": PointEncoder(0, backbone=backbone)})
        branches.append({"name": "clinical", "module": ClinicalEncoder(clinical_dim)})
    elif model_name == "geometry_flow_clinical":
        return ClinicalAnchoredLateFusionClassifier(
            geometry=PointEncoder(0, backbone=backbone),
            flow=PointEncoder(FLOW_CHANNELS, backbone=backbone),
            clinical=ClinicalEncoder(clinical_dim),
        )
    elif model_name == "clinical":
        branches.append({"name": "clinical", "module": ClinicalEncoder(clinical_dim)})
    else:
        raise ValueError(f"Unsupported point model: {model_name}")
    return BranchFusionClassifier(branches)


def build_graph_model():
    return GraphEncoder()


def balanced_sampler(labels: torch.Tensor):
    labels_np = labels.detach().cpu().numpy().astype(int)
    class_counts = np.bincount(labels_np, minlength=2).astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    weights = 1.0 / class_counts
    sample_weights = torch.tensor(weights[labels_np], dtype=torch.double, device=labels.device)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def fit_stats(items: Sequence[torch.Tensor]):
    stacked = torch.cat([x.reshape(-1, x.shape[-1]) for x in items], dim=0)
    mu = stacked.mean(dim=0)
    sigma = stacked.std(dim=0)
    sigma = torch.where(sigma < 1e-6, torch.ones_like(sigma), sigma)
    return mu, sigma


def apply_stats(
    items: Sequence[torch.Tensor], mu: torch.Tensor, sigma: torch.Tensor, clip: float = 3.0
):
    return [torch.clamp((x - mu) / sigma, -clip, clip) for x in items]


def build_case_cache(df: pd.DataFrame):
    print("Preloading point samples into memory...")
    cache = {}
    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"  Preloaded {i}/{len(df)}")
        xyz, flow = load_point_case(row["filepath"], int(row["target"]), augment=False)
        cache[row["filepath"]] = (xyz, flow)
    print(f"Preloading complete: {len(cache)} samples")
    return cache


def compute_point_fold_tensors(train_df: pd.DataFrame, val_df: pd.DataFrame, cache, categories):
    train_clin, clin_stats = build_clinical_matrix(train_df, categories)
    val_clin, _ = build_clinical_matrix(val_df, categories, train_stats=clin_stats)

    train_xyz, train_flow, train_labels = [], [], []
    for _, row in train_df.iterrows():
        xyz, flow = cache[row["filepath"]]
        train_xyz.append(xyz.clone())
        train_flow.append(flow.clone())
        train_labels.append(torch.tensor(int(row["target"]), dtype=torch.long))

    val_xyz, val_flow, val_labels = [], [], []
    for _, row in val_df.iterrows():
        xyz, flow = cache[row["filepath"]]
        val_xyz.append(xyz.clone())
        val_flow.append(flow.clone())
        val_labels.append(torch.tensor(int(row["target"]), dtype=torch.long))

    flow_mu, flow_sigma = fit_stats(train_flow)
    train_flow = apply_stats(train_flow, flow_mu, flow_sigma)
    val_flow = apply_stats(val_flow, flow_mu, flow_sigma)

    return (
        torch.stack(train_xyz),
        torch.stack(train_flow),
        train_clin,
        torch.stack(train_labels),
        torch.stack(val_xyz),
        torch.stack(val_flow),
        val_clin,
        torch.stack(val_labels),
    )


def build_point_loaders(train_tensors, val_tensors):
    train_xyz, train_flow, train_clin, train_labels = train_tensors
    val_xyz, val_flow, val_clin, val_labels = val_tensors
    sampler = balanced_sampler(train_labels)
    train_ds = TensorDataset(train_xyz, train_flow, train_clin, train_labels)
    val_ds = TensorDataset(val_xyz, val_flow, val_clin, val_labels)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        drop_last=len(train_ds) > BATCH_SIZE,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def build_graph_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame):
    train_graphs = [
        load_graph_case(row["filepath"], int(row["target"]), augment=True)
        for _, row in train_df.iterrows()
    ]
    val_graphs = [
        load_graph_case(row["filepath"], int(row["target"]), augment=False)
        for _, row in val_df.iterrows()
    ]
    train_loader = PyGDataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = PyGDataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def evaluate_tensor_model(model, loader, criterion, device, use_amp: bool):
    model.eval()
    loss_sum, probs, labels = 0.0, [], []
    with torch.no_grad():
        for xb, fb, cb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            fb = fb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits, _ = model(xb, fb, cb)
                loss = criterion(logits, yb)
            loss_sum += loss.item() * len(yb)
            probs.extend(F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
            labels.extend(yb.detach().cpu().numpy())
    metrics = classification_report_dict(labels, probs)
    return loss_sum / max(len(labels), 1), np.asarray(probs), np.asarray(labels), metrics


def evaluate_graph_model(model, loader, criterion, device, use_amp: bool):
    model.eval()
    loss_sum, probs, labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits, _ = model(batch)
                loss = criterion(logits, batch.y)
            loss_sum += loss.item() * batch.num_graphs
            probs.extend(F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
            labels.extend(batch.y.detach().cpu().numpy())
    metrics = classification_report_dict(labels, probs)
    return loss_sum / max(len(labels), 1), np.asarray(probs), np.asarray(labels), metrics


def compute_auxiliary_loss(aux_logits, criterion, targets):
    if isinstance(aux_logits, dict):
        total = torch.zeros((), device=targets.device)
        for value in aux_logits.values():
            logits, weight = value if isinstance(value, tuple) else (value, 1.0)
            if torch.is_tensor(logits):
                total = total + float(weight) * criterion(logits, targets)
        return total

    if torch.is_tensor(aux_logits):
        aux_logits = [aux_logits]
    aux_logits = list(aux_logits or [])
    if not aux_logits:
        return torch.zeros((), device=targets.device)
    total = torch.zeros((), device=targets.device)
    for logits in aux_logits:
        total = total + criterion(logits, targets)
    return total / len(aux_logits)


def run_tensor_fold_training(
    model,
    train_loader,
    val_loader,
    criterion,
    aux_criterion,
    optimizer,
    scheduler,
    device,
    epochs: int,
    early_stop_patience: int,
    fold_dir: Path,
    fold: int,
    use_ema: bool = False,
    grad_clip: float = 1.0,
    amp: bool = False,
    log_every: int = 25,
):
    fold_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = fold_dir / f"fold_{fold + 1}_metrics.csv"
    roc_dir = fold_dir / f"fold_{fold + 1}_roc"
    roc_dir.mkdir(exist_ok=True)

    ema = EMA(model, decay=0.999) if use_ema else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    best_auc = -math.inf
    best_state = None
    best_probs = None
    best_labels = None
    best_metrics = None
    patience = 0

    header = [
        "epoch",
        "train_loss",
        "train_acc",
        "train_auc",
        "train_pr_auc",
        "val_loss",
        "val_acc",
        "val_auc",
        "val_pr_auc",
        "precision",
        "recall",
        "tn",
        "fp",
        "fn",
        "tp",
        "lr",
        "val_threshold_youden",
        "val_acc_youden",
        "val_balanced_acc_youden",
    ]
    with open(metrics_csv, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum, probs, labels = 0.0, [], []
            for xb, fb, cb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                fb = fb.to(device, non_blocking=True)
                cb = cb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                    logits, aux_logits = model(xb, fb, cb)
                    main_loss = criterion(logits, yb)
                    aux_loss = compute_auxiliary_loss(aux_logits, aux_criterion, yb)
                    if isinstance(aux_logits, dict):
                        loss = main_loss + aux_loss
                    else:
                        loss = main_loss + AUX_LOSS_WEIGHT * aux_loss
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    # Under AMP, skip scheduler step when optimizer step is skipped.
                    if (not amp) or (device.type != "cuda") or (scaler.get_scale() >= prev_scale):
                        scheduler.step()
                if ema is not None:
                    ema.update(model)
                loss_sum += loss.item() * len(yb)
                probs.extend(F.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy())
                labels.extend(yb.detach().cpu().numpy())

            train_loss = loss_sum / max(len(labels), 1)
            train_metrics = classification_report_dict(labels, probs)

            backup = ema.apply_to(model) if ema is not None else None
            val_loss, val_probs, val_labels, val_metrics = evaluate_tensor_model(
                model, val_loader, criterion, device, amp
            )
            if ema is not None:
                ema.restore(model, backup)

            threshold = 0.5
            if len(set(val_labels)) > 1:
                fpr, tpr, thr = roc_curve(val_labels, val_probs)
                youden = tpr - fpr
                best_idx = int(np.argmax(youden))
                threshold = float(thr[best_idx])
            else:
                fpr, tpr, thr = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
            val_metrics_youden = classification_report_dict(
                val_labels, val_probs, threshold=threshold
            )
            specificity = val_metrics_youden["tn"] / max(
                val_metrics_youden["tn"] + val_metrics_youden["fp"], 1
            )
            bal_acc_youden = 0.5 * (val_metrics_youden["recall"] + specificity)
            pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
                roc_dir / f"epoch_{epoch}.csv", index=False
            )

            lr = optimizer.param_groups[0]["lr"]
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_metrics['acc']:.4f}",
                    f"{train_metrics['auc']:.4f}",
                    f"{train_metrics['pr_auc']:.4f}",
                    f"{val_loss:.6f}",
                    f"{val_metrics['acc']:.4f}",
                    f"{val_metrics['auc']:.4f}",
                    f"{val_metrics['pr_auc']:.4f}",
                    f"{val_metrics['precision']:.4f}",
                    f"{val_metrics['recall']:.4f}",
                    val_metrics["tn"],
                    val_metrics["fp"],
                    val_metrics["fn"],
                    val_metrics["tp"],
                    f"{lr:.3e}",
                    f"{threshold:.6f}",
                    f"{val_metrics_youden['acc']:.4f}",
                    f"{bal_acc_youden:.4f}",
                ]
            )
            fp.flush()

            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                best_state = deepcopy(model.state_dict() if ema is None else ema.shadow)
                best_probs = list(val_probs)
                best_labels = list(val_labels)
                best_metrics = {
                    "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    "val_threshold_youden": threshold,
                    "val_acc_youden": val_metrics_youden["acc"],
                    "val_balanced_acc_youden": bal_acc_youden,
                    "val_selection_metric": "auc",
                    "val_selection_score": val_metrics["auc"],
                }
                if hasattr(model, "current_flow_weight"):
                    best_metrics["flow_weight"] = float(model.current_flow_weight().detach().cpu())
                patience = 0
            else:
                patience += 1

            if epoch % log_every == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d} | train_auc={train_metrics['auc']:.4f} val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['acc']:.4f} lr={lr:.2e}"
                )

            if patience >= early_stop_patience:
                break

    if best_state is not None:
        torch.save(best_state, fold_dir / f"best_model_fold_{fold + 1}.pt")

    if best_metrics is None:
        best_metrics = {
            "val_auc": 0.5,
            "val_pr_auc": 0.0,
            "val_acc": 0.0,
            "val_precision": 0.0,
            "val_recall": 0.0,
            "val_tn": 0,
            "val_fp": 0,
            "val_fn": 0,
            "val_tp": 0,
        }
        best_probs = []
        best_labels = []

    return best_metrics, best_probs, best_labels


def run_graph_fold_training(
    model,
    train_loader,
    val_loader,
    criterion,
    aux_criterion,
    optimizer,
    scheduler,
    device,
    epochs: int,
    early_stop_patience: int,
    fold_dir: Path,
    fold: int,
    use_ema: bool = False,
    grad_clip: float = 1.0,
    amp: bool = False,
    log_every: int = 25,
):
    fold_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = fold_dir / f"fold_{fold + 1}_metrics.csv"
    roc_dir = fold_dir / f"fold_{fold + 1}_roc"
    roc_dir.mkdir(exist_ok=True)

    ema = EMA(model, decay=0.999) if use_ema else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    best_auc = -math.inf
    best_state = None
    best_probs = None
    best_labels = None
    best_metrics = None
    patience = 0

    header = [
        "epoch",
        "train_loss",
        "train_acc",
        "train_auc",
        "train_pr_auc",
        "val_loss",
        "val_acc",
        "val_auc",
        "val_pr_auc",
        "precision",
        "recall",
        "tn",
        "fp",
        "fn",
        "tp",
        "lr",
        "val_threshold_youden",
        "val_acc_youden",
        "val_balanced_acc_youden",
    ]
    with open(metrics_csv, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum, probs, labels = 0.0, [], []
            for batch in train_loader:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                    logits, aux_logits = model(batch)
                    main_loss = criterion(logits, batch.y)
                    aux_loss = compute_auxiliary_loss(aux_logits, aux_criterion, batch.y)
                    if isinstance(aux_logits, dict):
                        loss = main_loss + aux_loss
                    else:
                        loss = main_loss + AUX_LOSS_WEIGHT * aux_loss
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    # Under AMP, skip scheduler step when optimizer step is skipped.
                    if (not amp) or (device.type != "cuda") or (scaler.get_scale() >= prev_scale):
                        scheduler.step()
                if ema is not None:
                    ema.update(model)
                loss_sum += loss.item() * batch.num_graphs
                probs.extend(F.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy())
                labels.extend(batch.y.detach().cpu().numpy())

            train_loss = loss_sum / max(len(labels), 1)
            train_metrics = classification_report_dict(labels, probs)

            backup = ema.apply_to(model) if ema is not None else None
            val_loss, val_probs, val_labels, val_metrics = evaluate_graph_model(
                model, val_loader, criterion, device, amp
            )
            if ema is not None:
                ema.restore(model, backup)

            threshold = 0.5
            if len(set(val_labels)) > 1:
                fpr, tpr, thr = roc_curve(val_labels, val_probs)
                youden = tpr - fpr
                best_idx = int(np.argmax(youden))
                threshold = float(thr[best_idx])
            else:
                fpr, tpr, thr = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
            val_metrics_youden = classification_report_dict(
                val_labels, val_probs, threshold=threshold
            )
            specificity = val_metrics_youden["tn"] / max(
                val_metrics_youden["tn"] + val_metrics_youden["fp"], 1
            )
            bal_acc_youden = 0.5 * (val_metrics_youden["recall"] + specificity)
            pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
                roc_dir / f"epoch_{epoch}.csv", index=False
            )

            lr = optimizer.param_groups[0]["lr"]
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_metrics['acc']:.4f}",
                    f"{train_metrics['auc']:.4f}",
                    f"{train_metrics['pr_auc']:.4f}",
                    f"{val_loss:.6f}",
                    f"{val_metrics['acc']:.4f}",
                    f"{val_metrics['auc']:.4f}",
                    f"{val_metrics['pr_auc']:.4f}",
                    f"{val_metrics['precision']:.4f}",
                    f"{val_metrics['recall']:.4f}",
                    val_metrics["tn"],
                    val_metrics["fp"],
                    val_metrics["fn"],
                    val_metrics["tp"],
                    f"{lr:.3e}",
                    f"{threshold:.6f}",
                    f"{val_metrics_youden['acc']:.4f}",
                    f"{bal_acc_youden:.4f}",
                ]
            )
            fp.flush()

            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                best_state = deepcopy(model.state_dict() if ema is None else ema.shadow)
                best_probs = list(val_probs)
                best_labels = list(val_labels)
                best_metrics = {
                    "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    "val_threshold_youden": threshold,
                    "val_acc_youden": val_metrics_youden["acc"],
                    "val_balanced_acc_youden": bal_acc_youden,
                    "val_selection_metric": "auc",
                    "val_selection_score": val_metrics["auc"],
                }
                patience = 0
            else:
                patience += 1

            if epoch % log_every == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d} | train_auc={train_metrics['auc']:.4f} val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['acc']:.4f} lr={lr:.2e}"
                )

            if patience >= early_stop_patience:
                break

    if best_state is not None:
        torch.save(best_state, fold_dir / f"best_model_fold_{fold + 1}.pt")

    if best_metrics is None:
        best_metrics = {
            "val_auc": 0.5,
            "val_pr_auc": 0.0,
            "val_acc": 0.0,
            "val_precision": 0.0,
            "val_recall": 0.0,
            "val_tn": 0,
            "val_fp": 0,
            "val_fn": 0,
            "val_tp": 0,
        }
        best_probs = []
        best_labels = []

    return best_metrics, best_probs, best_labels


def write_fold_summary(
    output_dir: Path,
    model_name: str,
    backbone: str,
    fold_summaries,
    pooled_probs,
    pooled_labels,
    pooled_paths,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(fold_summaries)
    df.insert(0, "backbone", backbone)
    df.insert(0, "model", model_name)
    df.to_csv(output_dir / "fold_summary.csv", index=False)

    pooled_metrics = classification_report_dict(pooled_labels, pooled_probs, threshold=0.5)
    pooled_metrics["model"] = model_name
    pooled_metrics["backbone"] = backbone
    pd.DataFrame([pooled_metrics]).to_csv(output_dir / "pooled_metrics.csv", index=False)
    pd.DataFrame({"filepath": pooled_paths, "label": pooled_labels, "prob": pooled_probs}).to_csv(
        output_dir / "pooled_predictions.csv", index=False
    )


def _fold_artifact_paths(output_dir: Path, fold: int):
    prefix = f"fold_{fold + 1}_completed"
    return {
        "metrics": output_dir / f"{prefix}_metrics.csv",
        "predictions": output_dir / f"{prefix}_predictions.csv",
        "rng": output_dir / f"{prefix}_rng.pt",
        "marker": output_dir / f"{prefix}.flag",
    }


def _save_rng_state(path: Path):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _restore_rng_state(path: Path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:
        print(f"  Warning: could not restore fold RNG state from {path}: {exc}")


def save_completed_tensor_fold(
    output_dir: Path, fold: int, best_metrics, best_probs, best_labels, val_paths
):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _fold_artifact_paths(output_dir, fold)
    metrics_tmp = paths["metrics"].with_suffix(".csv.tmp")
    predictions_tmp = paths["predictions"].with_suffix(".csv.tmp")
    pd.DataFrame([{**best_metrics, "fold": fold + 1}]).to_csv(metrics_tmp, index=False)
    pd.DataFrame(
        {
            "filepath": list(val_paths),
            "label": list(best_labels),
            "prob": list(best_probs),
        }
    ).to_csv(predictions_tmp, index=False)
    os.replace(metrics_tmp, paths["metrics"])
    os.replace(predictions_tmp, paths["predictions"])
    _save_rng_state(paths["rng"])
    paths["marker"].write_text("complete\n", encoding="ascii")


def load_completed_tensor_fold(output_dir: Path, fold: int, expected_paths):
    paths = _fold_artifact_paths(output_dir, fold)
    if not all(paths[key].exists() for key in ("metrics", "predictions", "rng", "marker")):
        return None
    metrics_df = pd.read_csv(paths["metrics"])
    predictions_df = pd.read_csv(paths["predictions"])
    stored_paths = predictions_df["filepath"].astype(str).tolist()
    expected_paths = [str(path) for path in expected_paths]
    if stored_paths != expected_paths:
        print(
            f"  Ignoring stale completed-fold artifacts for fold {fold + 1}: validation paths differ"
        )
        return None
    metrics = metrics_df.iloc[0].to_dict()
    metrics.pop("fold", None)
    _restore_rng_state(paths["rng"])
    return metrics, predictions_df["prob"].tolist(), predictions_df["label"].astype(int).tolist()


def run_tensor_experiment(
    model_name: str, backbone: str, df: pd.DataFrame, categories, output_dir: Path
):
    cache = build_case_cache(df)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    pooled_probs, pooled_labels, pooled_paths = [], [], []
    fold_summaries = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(df)), df["target"].values)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)
        val_paths = val_df["filepath"].astype(str).tolist()

        completed = load_completed_tensor_fold(output_dir, fold, val_paths)
        if completed is not None:
            best_metrics, best_probs, best_labels = completed
            fold_summaries.append({"fold": fold + 1, **best_metrics})
            pooled_probs.extend(list(best_probs))
            pooled_labels.extend(list(best_labels))
            pooled_paths.extend(val_paths)
            print(
                f"  Loaded completed fold; best val AUC: {float(best_metrics.get('val_auc', 0.0)):.4f}"
            )
            continue

        if ONLY_FOLD is not None and fold + 1 != ONLY_FOLD:
            print(f"  Skipping unfinished fold because --only-fold={ONLY_FOLD}")
            continue

        train_tensors = compute_point_fold_tensors(train_df, val_df, cache, categories)
        train_loader, val_loader = build_point_loaders(train_tensors[:4], train_tensors[4:])

        train_xyz, train_flow, train_clin, train_labels, val_xyz, val_flow, val_clin, val_labels = (
            train_tensors
        )
        clinical_dim = train_clin.shape[1]
        model = build_point_model(model_name, backbone, clinical_dim=clinical_dim).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        n_pos = int(train_labels.sum().item())
        n_neg = int(len(train_labels) - n_pos)
        alpha = torch.tensor([1.0 / max(n_neg, 1), 1.0 / max(n_pos, 1)], dtype=torch.float32)
        alpha = alpha / alpha.sum() * 2.0
        criterion = FocalLoss(alpha=alpha.to(DEVICE), gamma=2.0, label_smoothing=LABEL_SMOOTHING)
        aux_criterion = FocalLoss(
            alpha=alpha.to(DEVICE), gamma=2.0, label_smoothing=LABEL_SMOOTHING
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LR,
            total_steps=max(EPOCHS * max(len(train_loader), 1), 1),
            pct_start=0.12,
            div_factor=25.0,
            final_div_factor=100.0,
            anneal_strategy="cos",
        )

        best_metrics, best_probs, best_labels = run_tensor_fold_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            aux_criterion=aux_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            epochs=EPOCHS,
            early_stop_patience=EARLY_STOP_PATIENCE,
            fold_dir=output_dir,
            fold=fold,
            use_ema=True,
            grad_clip=1.0,
            amp=USE_AMP,
        )

        save_completed_tensor_fold(
            output_dir, fold, best_metrics, best_probs, best_labels, val_paths
        )
        fold_summaries.append({"fold": fold + 1, **best_metrics})
        pooled_probs.extend(list(best_probs))
        pooled_labels.extend(list(best_labels))
        pooled_paths.extend(list(val_paths))
        print(f"  Best val AUC: {best_metrics.get('val_auc', 0.0):.4f}")

    completed_count = sum(
        _fold_artifact_paths(output_dir, fold)["marker"].exists() for fold in range(N_FOLDS)
    )
    if completed_count == N_FOLDS:
        write_fold_summary(
            output_dir,
            model_name,
            backbone,
            fold_summaries,
            pooled_probs,
            pooled_labels,
            pooled_paths,
        )
        print(f"Training complete. Results written to {output_dir}")
    else:
        print(
            f"Fold run complete ({completed_count}/{N_FOLDS} folds persisted). Final pooled outputs will be written after all folds complete."
        )


def run_graph_experiment(df: pd.DataFrame, output_dir: Path):
    if not HAS_PYG:
        raise RuntimeError("torch_geometric is not installed; cannot run gnn model")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    pooled_probs, pooled_labels, pooled_paths = [], [], []
    fold_summaries = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(df)), df["target"].values)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)
        train_loader, val_loader = build_graph_loaders(train_df, val_df)

        model = build_graph_model().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        labels_tensor = torch.tensor(train_df["target"].values.astype(int), dtype=torch.long)
        n_pos = int(labels_tensor.sum().item())
        n_neg = int(len(labels_tensor) - n_pos)
        alpha = torch.tensor([1.0 / max(n_neg, 1), 1.0 / max(n_pos, 1)], dtype=torch.float32)
        alpha = alpha / alpha.sum() * 2.0
        criterion = FocalLoss(alpha=alpha.to(DEVICE), gamma=2.0, label_smoothing=LABEL_SMOOTHING)
        aux_criterion = FocalLoss(
            alpha=alpha.to(DEVICE), gamma=2.0, label_smoothing=LABEL_SMOOTHING
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LR,
            total_steps=max(EPOCHS * max(len(train_loader), 1), 1),
            pct_start=0.12,
            div_factor=25.0,
            final_div_factor=100.0,
            anneal_strategy="cos",
        )

        best_metrics, best_probs, best_labels = run_graph_fold_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            aux_criterion=aux_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            epochs=EPOCHS,
            early_stop_patience=EARLY_STOP_PATIENCE,
            fold_dir=output_dir,
            fold=fold,
            use_ema=True,
            grad_clip=1.0,
            amp=USE_AMP,
        )

        val_paths = val_df["filepath"].values.tolist()
        fold_summaries.append({"fold": fold + 1, **best_metrics})
        pooled_probs.extend(list(best_probs))
        pooled_labels.extend(list(best_labels))
        pooled_paths.extend(list(val_paths))
        print(f"  Best val AUC: {best_metrics.get('val_auc', 0.0):.4f}")

    write_fold_summary(
        output_dir, "gnn", "gnn", fold_summaries, pooled_probs, pooled_labels, pooled_paths
    )
    print(f"Training complete. Results written to {output_dir}")


def dry_run(model_name: str, backbone: str, categories: Dict[str, List[str]]):
    clinical_dim = 2 + sum(len(v) for v in categories.values())
    if model_name == "gnn":
        if not HAS_PYG:
            raise RuntimeError("torch_geometric is required for GNN dry-run")
        model = build_graph_model().to("cpu")
        batch = type("Batch", (), {})()
        batch.x = torch.randn(2, 64, 6)
        batch.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        batch.edge_attr = torch.randn(2, 1)
        batch.batch = torch.tensor([0, 0], dtype=torch.long)
        batch.num_graphs = 1
        batch.graph_features = torch.randn(1, GRAPH_FEATURE_DIM)
        batch.y = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            logits, aux = model(batch)
        print(
            "Dry-run forward pass successful; output shape:", logits.shape, "aux_heads:", len(aux)
        )
        return

    model = build_point_model(model_name, backbone, clinical_dim=clinical_dim).to("cpu")
    x = torch.randn(2, TARGET_N, 3)
    f = torch.randn(2, TARGET_N, FLOW_CHANNELS)
    c = torch.randn(2, clinical_dim)
    with torch.no_grad():
        logits, aux = model(x, f, c)
    print("Dry-run forward pass successful; output shape:", logits.shape, "aux_heads:", len(aux))


def fallback_categories():
    return {
        "location": ["Unknown"],
        "hospital": ["Unknown"],
        "source": ["Unknown"],
        "side": ["Unknown"],
    }


def main():
    global \
        SEED, \
        CV_SEED, \
        METADATA_PATH, \
        DATA_DIR, \
        OUTPUT_ROOT, \
        N_FOLDS, \
        BATCH_SIZE, \
        EPOCHS, \
        LR, \
        WEIGHT_DECAY, \
        TARGET_N, \
        EARLY_STOP_PATIENCE, \
        USE_AMP, \
        ONLY_FOLD

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[
            "geometry",
            "flow_geometry",
            "gnn",
            "clinical",
            "geometry_clinical",
            "geometry_flow_clinical",
        ],
        default="geometry",
    )
    parser.add_argument("--backbone", choices=["pointnet2", "pointnext"], default="pointnet2")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cv-seed", type=int, default=CV_SEED)
    parser.add_argument("--metadata-path", default=METADATA_PATH)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--target-n", type=int, default=TARGET_N)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--amp", action="store_true", default=USE_AMP)
    parser.add_argument("--only-fold", type=int, choices=range(1, N_FOLDS + 1), default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()
    SEED = args.seed
    CV_SEED = args.cv_seed
    METADATA_PATH = args.metadata_path
    DATA_DIR = args.data_dir
    N_FOLDS = args.folds
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    TARGET_N = args.target_n
    EARLY_STOP_PATIENCE = args.patience
    USE_AMP = args.amp
    ONLY_FOLD = args.only_fold

    set_seed(SEED)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else OUTPUT_ROOT / f"{args.model}_{args.backbone}_seed_{SEED}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        try:
            df = discover_cases(DATA_DIR, METADATA_PATH)
            categories = compute_clinical_categories(df) if len(df) > 0 else fallback_categories()
        except Exception:
            categories = fallback_categories()
        dry_run(args.model, args.backbone, categories)
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for v11 training but torch.cuda.is_available() is false."
        )

    print(
        f"Building v11 model={args.model} backbone={args.backbone} on cuda (seed={SEED}, cv_seed={CV_SEED})"
    )
    print(
        f"Training config: folds={N_FOLDS} epochs={EPOCHS} batch_size={BATCH_SIZE} lr={LR} wd={WEIGHT_DECAY} patience={EARLY_STOP_PATIENCE} target_n={TARGET_N}"
    )

    df = discover_cases(DATA_DIR, METADATA_PATH)
    if len(df) == 0:
        raise RuntimeError(f"No samples found under {DATA_DIR} using metadata {METADATA_PATH}")
    df = df.copy()
    df["target"] = (df["status"].astype(str).str.lower() == "ruptured").astype(int)
    categories = compute_clinical_categories(df)
    print(
        f"Total samples: {len(df)}   class_counts={{0: {int((df['target'] == 0).sum())}, 1: {int((df['target'] == 1).sum())}}}"
    )

    if args.model == "gnn":
        run_graph_experiment(df, output_dir)
    else:
        run_tensor_experiment(args.model, args.backbone, df, categories, output_dir)


if __name__ == "__main__":
    main()
