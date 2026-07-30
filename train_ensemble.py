# Version 10 source snapshot
"""
train_ensemble.py

V10 multibranch classifier for aneurysm rupture prediction.
Combines:
- geometry branch (xyz)
- flow/hemodynamic branch (xyz + derived hemodynamic channels)
- clinical branch (age, sex, location, hospital, source, side)
- global summary branch (hemodynamic + geometric summaries)

Design goals:
- self-contained, no dependency on version8/v9 helpers
- CUDA-only training path for real runs
- stronger fusion than v9: gated multi-branch fusion plus auxiliary heads
- fold-level normalization and balanced sampling
- fold outputs include pooled predictions for later seed ensembling
"""

from __future__ import annotations

import csv
import math
import os
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

SEED = int(os.environ.get("V10_SEED", 42))
CV_SEED = int(os.environ.get("V10_CV_SEED", 42))
METADATA_PATH = os.environ.get("V10_METADATA", "metadata.csv")
DATA_DIR = os.environ.get("V10_DATA_DIR", "predictions/pinn_corrected")
OUTPUT_ROOT = Path(os.environ.get("V10_OUTPUT_DIR", "results_v10_multibranch"))
N_FOLDS = int(os.environ.get("V10_FOLDS", 5))
BATCH_SIZE = int(os.environ.get("V10_BATCH", 6))
EPOCHS = int(os.environ.get("V10_EPOCHS", 220))
LR = float(os.environ.get("V10_LR", 3e-4))
WEIGHT_DECAY = float(os.environ.get("V10_WD", 2e-4))
TARGET_N = int(os.environ.get("V10_TARGET_N", 4096))
EARLY_STOP_PATIENCE = int(os.environ.get("V10_PATIENCE", 35))
USE_AMP = os.environ.get("V10_AMP", "1").lower() in {"1", "true", "yes"}
LABEL_SMOOTHING = float(os.environ.get("V10_LABEL_SMOOTH", 0.05))
AUX_LOSS_WEIGHT = float(os.environ.get("V10_AUX_WEIGHT", 0.15))
DROPOUT = float(os.environ.get("V10_DROPOUT", 0.30))
FLOW_CHANNELS = 11
GLOBAL_FEATURE_DIM = 33
EMBED_DIM = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


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
    preds = (np.asarray(probs) > threshold).astype(int)
    auc = safe_auc(labels, probs)
    pr_auc = safe_ap(labels, probs)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "acc": acc,
        "precision": prec,
        "recall": rec,
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
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (pts.astype(np.float32) @ (Rz @ Ry @ Rx).T).astype(np.float32)


def jitter(pts: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
    return (pts + np.clip(sigma * np.random.randn(*pts.shape), -clip, clip)).astype(np.float32)


def random_point_dropout(pts: np.ndarray, feats: Optional[np.ndarray] = None, p: float = 0.1):
    n = pts.shape[0]
    drop = np.random.rand(n) < p
    if drop.any():
        keep_idx = np.where(~drop)[0]
        if len(keep_idx) == 0:
            return pts, feats
        replace = np.random.choice(keep_idx, drop.sum())
        pts = pts.copy()
        pts[drop] = pts[replace]
        if feats is not None:
            feats = feats.copy()
            feats[drop] = feats[replace]
    return pts, feats


def derive_hemo_channels(raw_feats: np.ndarray) -> np.ndarray:
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


def summarize_global_features(raw_feats: np.ndarray, pts: np.ndarray) -> np.ndarray:
    tawss = raw_feats[:, 0]
    osi = raw_feats[:, 1]
    von = raw_feats[:, 2]
    combined = tawss * (1.0 - 2.0 * osi)
    denom = (1.0 - 2.0 * osi) * tawss
    rrt = np.zeros_like(tawss, dtype=np.float32)
    mask = np.abs(denom) > 1e-8
    rrt[mask] = 1.0 / denom[mask]
    rrt[~np.isfinite(rrt)] = 0.0
    geom_centroid = pts.mean(axis=0)
    dist = np.linalg.norm(pts - geom_centroid, axis=1)
    try:
        cov = np.cov(pts.T)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eig = eig / (eig.sum() + 1e-8)
    except Exception:
        eig = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    risk = (tawss < np.percentile(tawss, 20)).astype(np.float32) * (osi > 0.2).astype(np.float32)
    vm_norm = von / (np.max(von) + 1e-8)

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
        np.mean(combined),
        np.std(combined),
        np.mean(risk),
        float(np.mean(risk > 0.0)),
        np.mean(rrt),
        np.std(rrt),
        np.max(rrt),
        np.percentile(rrt, 95),
        float(np.mean(rrt > np.percentile(rrt, 90))),
        float(np.max(dist)),
        float(np.std(dist)),
        float(np.max(dist) / (np.mean(dist) + 1e-6)),
        float(eig[0]),
        float(eig[1]),
        float(eig[0] / (eig[2] + 1e-6)),
        float(np.mean(vm_norm)),
    ]
    arr = np.array(feats, dtype=np.float32)
    arr = np.sign(arr) * np.log1p(np.abs(arr))
    return np.clip(arr, -10.0, 10.0)


def normalize_features_zscore(feats: np.ndarray, clip: float = 3.0) -> np.ndarray:
    mu = feats.mean(axis=0)
    sigma = feats.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return np.clip((feats - mu) / sigma, -clip, clip).astype(np.float32)


def build_clinical_matrix(
    df_slice: pd.DataFrame, categories: Dict[str, List[str]], train_stats=None
):
    ages = df_slice["age"].values.astype(np.float32)
    mu = train_stats["age_mean"] if train_stats else float(ages.mean())
    sigma = train_stats["age_std"] if train_stats else float(ages.std())
    ages = (ages - mu) / (sigma + 1e-6)

    sexes = (
        df_slice["sex"]
        .fillna("Unknown")
        .astype(str)
        .str.lower()
        .map({"female": 0.0, "male": 1.0})
        .fillna(0.0)
        .values.astype(np.float32)
    )

    encoded_parts = [ages.reshape(-1, 1), sexes.reshape(-1, 1)]
    for field in ["location", "hospital", "source", "side"]:
        values = df_slice[field].fillna("Unknown").astype(str).values
        cats = categories[field]
        one_hot = np.zeros((len(df_slice), len(cats)), dtype=np.float32)
        idx_map = {cat: i for i, cat in enumerate(cats)}
        for i, val in enumerate(values):
            if val in idx_map:
                one_hot[i, idx_map[val]] = 1.0
        encoded_parts.append(one_hot)
    X = np.concatenate(encoded_parts, axis=1).astype(np.float32)
    stats = {"age_mean": float(mu), "age_std": float(sigma)}
    return torch.tensor(X, dtype=torch.float32), stats


def load_flow_sample(path: str, label: int, augment: bool):
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

    global_feats = summarize_global_features(raw, pts)
    per_point = derive_hemo_channels(raw)

    idx = resample_cloud(pts, TARGET_N)
    pts = pts[idx]
    per_point = per_point[idx]

    pts = pts - pts.mean(axis=0)
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts = pts / max_dist
    per_point = normalize_features_zscore(per_point)

    if augment:
        pts = so3_rotate(pts)
        pts = jitter(pts)
        pts = pts * np.random.uniform(0.95, 1.05)
        pts, per_point = random_point_dropout(pts, per_point, p=0.1)

    return (
        torch.tensor(pts, dtype=torch.float32),
        torch.tensor(per_point, dtype=torch.float32),
        torch.tensor(global_feats, dtype=torch.float32),
        torch.tensor(label, dtype=torch.long),
    )


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if hasattr(self, "alpha") and self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = loss * alpha_t
        return loss.mean()


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if not torch.is_floating_point(param):
                    continue
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return backup

    def restore(self, model, backup):
        model.load_state_dict(backup, strict=True)


class PointSetAbstraction(nn.Module):
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


class BranchEncoder(nn.Module):
    def __init__(self, in_channel: int, embed_dim: int = EMBED_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.sa1 = PointSetAbstraction(
            512, 0.2, 32, in_channel=in_channel, mlp=[64, 64, 128], dropout=0.1
        )
        self.sa2 = PointSetAbstraction(
            128, 0.4, 64, in_channel=128 + 3, mlp=[128, 128, 256], dropout=0.1
        )
        self.sa3 = PointSetAbstraction(
            None, None, None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True, dropout=0.1
        )
        self.proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, xyz, feats=None):
        l1, l1_xyz = self.sa1(xyz, feats)
        l2, l2_xyz = self.sa2(l1_xyz, l1)
        l3, _ = self.sa3(l2_xyz, l2)
        token = self.proj(l3.reshape(l3.shape[0], -1))
        aux_logits = self.aux_head(token)
        return token, aux_logits


class TabularEncoder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = EMBED_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
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
        aux_logits = self.aux_head(token)
        return token, aux_logits


class MultiBranchClassifier(nn.Module):
    def __init__(
        self,
        clinical_dim: int,
        global_dim: int,
        flow_channels: int = FLOW_CHANNELS,
        embed_dim: int = EMBED_DIM,
    ):
        super().__init__()
        self.geom = BranchEncoder(3, embed_dim=embed_dim)
        self.flow = BranchEncoder(3 + flow_channels, embed_dim=embed_dim)
        self.clinical = TabularEncoder(clinical_dim, embed_dim=embed_dim)
        self.global_proj = TabularEncoder(global_dim, embed_dim=embed_dim)
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

    def forward(self, xyz, flow_feats, clinical_feats, global_feats):
        geom_token, geom_aux = self.geom(xyz, None)
        flow_token, flow_aux = self.flow(xyz, flow_feats)
        clin_token, clin_aux = self.clinical(clinical_feats)
        glob_token, glob_aux = self.global_proj(global_feats)

        tokens = torch.stack([geom_token, flow_token, clin_token, glob_token], dim=1)
        scores = self.gate(tokens).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        fused = torch.sum(weights.unsqueeze(-1) * tokens, dim=1)
        summary = torch.cat(
            [
                fused,
                tokens.mean(dim=1),
                tokens.std(dim=1),
                tokens.max(dim=1).values,
            ],
            dim=-1,
        )
        logits = self.head(summary)
        aux_logits = [geom_aux, flow_aux, clin_aux, glob_aux]
        return logits, aux_logits


def balanced_sampler(labels: torch.Tensor):
    labels_np = labels.detach().cpu().numpy().astype(int)
    class_counts = np.bincount(labels_np, minlength=2).astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    weights = 1.0 / class_counts
    sample_weights = torch.tensor(weights[labels_np], dtype=torch.double, device=labels.device)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def fit_global_stats(gs: torch.Tensor):
    mu = gs.mean(dim=0)
    sigma = gs.std(dim=0)
    sigma = torch.where(sigma < 1e-6, torch.ones_like(sigma), sigma)
    return mu, sigma


def apply_global_stats(gs: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, clip: float = 3.0):
    return torch.clamp((gs - mu) / sigma, -clip, clip)


def build_loader_from_tensors(xs, fs, clins, gs, ys, shuffle: bool, batch_size: int):
    sampler = balanced_sampler(ys) if shuffle else None
    return DataLoader(
        TensorDataset(xs, fs, clins, gs, ys),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        drop_last=(shuffle and len(ys) > batch_size),
    )


def preprocess_fold_tensors(df: pd.DataFrame, paths, labels, clinical_tensor, augment: bool):
    xs, fs, gs, ys = [], [], [], []
    for path, label in zip(paths, labels):
        try:
            pts, feats, gf, lab = load_flow_sample(path, int(label), augment=augment)
        except Exception as exc:
            print(f"Error loading {path}: {exc}")
            continue
        xs.append(pts)
        fs.append(feats)
        gs.append(gf)
        ys.append(lab)
    if not ys:
        raise RuntimeError("No valid samples loaded for fold.")
    return torch.stack(xs), torch.stack(fs), clinical_tensor, torch.stack(gs), torch.stack(ys)


def compute_clinical_categories(df: pd.DataFrame):
    categories = {}
    for field in ["location", "hospital", "source", "side"]:
        categories[field] = sorted(df[field].fillna("Unknown").astype(str).unique().tolist())
    return categories


def build_sample_cache(df: pd.DataFrame):
    print("Preloading samples into memory...")
    cache = {}
    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"  Preloaded {i}/{len(df)}")
        pts, feats, gf, _ = load_flow_sample(row["filepath"], int(row["target"]), augment=False)
        cache[row["filepath"]] = (pts, feats, gf)
    print(f"Preloading complete: {len(cache)} samples")
    return cache


def encode_clinical(df_slice: pd.DataFrame, categories: Dict[str, List[str]], train_stats=None):
    ages = df_slice["age"].values.astype(np.float32)
    mu = train_stats["age_mean"] if train_stats else float(ages.mean())
    sigma = train_stats["age_std"] if train_stats else float(ages.std())
    ages = (ages - mu) / (sigma + 1e-6)

    sex_raw = df_slice["sex"]
    sexes = pd.to_numeric(sex_raw, errors="coerce")
    if sexes.isna().all():
        sexes = sex_raw.fillna("Unknown").astype(str).str.lower().map({"female": 0.0, "male": 1.0})
    sexes = sexes.fillna(0.0).values.astype(np.float32)
    parts = [ages.reshape(-1, 1), sexes.reshape(-1, 1)]
    for field in ["location", "hospital", "source", "side"]:
        values = df_slice[field].fillna("Unknown").astype(str).values
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


def compute_fold_normalization(train_pts, train_flow, train_global):
    pts_all = torch.cat([x.reshape(-1, x.shape[-1]) for x in train_pts], dim=0)
    flow_all = torch.cat([x.reshape(-1, x.shape[-1]) for x in train_flow], dim=0)
    global_all = torch.stack(train_global, dim=0)
    pts_mu, pts_sigma = fit_global_stats(pts_all)
    flow_mu, flow_sigma = fit_global_stats(flow_all)
    glob_mu, glob_sigma = fit_global_stats(global_all)
    return (pts_mu, pts_sigma), (flow_mu, flow_sigma), (glob_mu, glob_sigma)


def apply_fold_normalization(items, mu, sigma, clip=3.0):
    out = []
    for tensor in items:
        out.append(torch.clamp((tensor - mu) / sigma, -clip, clip))
    return out


def prepare_fold_data(df: pd.DataFrame, cache, train_idx, val_idx, categories):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_clin, clin_stats = encode_clinical(train_df, categories)
    val_clin, _ = encode_clinical(val_df, categories, train_stats=clin_stats)

    train_pts, train_flow, train_global, train_labels = [], [], [], []
    for _, row in train_df.iterrows():
        pts, feats, gf = cache[row["filepath"]]
        train_pts.append(pts.clone())
        train_flow.append(feats.clone())
        train_global.append(gf.clone())
        train_labels.append(torch.tensor(int(row["target"]), dtype=torch.long))

    val_pts, val_flow, val_global, val_labels = [], [], [], []
    for _, row in val_df.iterrows():
        pts, feats, gf = cache[row["filepath"]]
        val_pts.append(pts.clone())
        val_flow.append(feats.clone())
        val_global.append(gf.clone())
        val_labels.append(torch.tensor(int(row["target"]), dtype=torch.long))

    (pts_mu, pts_sigma), (flow_mu, flow_sigma), (glob_mu, glob_sigma) = compute_fold_normalization(
        train_pts, train_flow, train_global
    )

    train_pts = apply_fold_normalization(train_pts, pts_mu, pts_sigma)
    val_pts = apply_fold_normalization(val_pts, pts_mu, pts_sigma)
    train_flow = apply_fold_normalization(train_flow, flow_mu, flow_sigma)
    val_flow = apply_fold_normalization(val_flow, flow_mu, flow_sigma)
    train_global = apply_fold_normalization(train_global, glob_mu, glob_sigma)
    val_global = apply_fold_normalization(val_global, glob_mu, glob_sigma)

    train_pts = torch.stack(train_pts)
    train_flow = torch.stack(train_flow)
    train_global = torch.stack(train_global)
    train_labels = torch.stack(train_labels)

    val_pts = torch.stack(val_pts)
    val_flow = torch.stack(val_flow)
    val_global = torch.stack(val_global)
    val_labels = torch.stack(val_labels)

    return (
        train_df,
        val_df,
        train_clin,
        val_clin,
        train_pts,
        train_flow,
        train_global,
        train_labels,
        val_pts,
        val_flow,
        val_global,
        val_labels,
    )


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = -math.inf
        self.count = 0
        self.best_epoch = -1

    def step(self, metric: float, epoch: int):
        if metric > self.best:
            self.best = metric
            self.best_epoch = epoch
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience


def evaluate_model(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    probs, labels = [], []
    with torch.no_grad():
        for xb, fb, cb, gb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            fb = fb.to(DEVICE, non_blocking=True)
            cb = cb.to(DEVICE, non_blocking=True)
            gb = gb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP and DEVICE.type == "cuda"):
                logits, _ = model(xb, fb, cb, gb)
                loss = criterion(logits, yb)
            loss_sum += loss.item() * len(yb)
            probs.extend(F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
            labels.extend(yb.detach().cpu().numpy())
    metrics = classification_report_dict(labels, probs)
    return loss_sum / max(len(labels), 1), np.asarray(probs), np.asarray(labels), metrics


def compute_youden_metrics(labels, probs):
    if len(set(labels)) < 2:
        threshold = 0.5
        (np.asarray(probs) > threshold).astype(int)
        metrics = classification_report_dict(labels, probs, threshold=threshold)
        bal_acc = 0.5 * (
            metrics["recall"] + (metrics["tn"] / max(metrics["tn"] + metrics["fp"], 1))
        )
        return threshold, metrics, bal_acc
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    threshold = float(thresholds[best_idx])
    metrics = classification_report_dict(labels, probs, threshold=threshold)
    specificity = metrics["tn"] / max(metrics["tn"] + metrics["fp"], 1)
    bal_acc = 0.5 * (metrics["recall"] + specificity)
    return threshold, metrics, bal_acc


def write_fold_summary(output_dir: Path, fold_summaries, pooled_probs, pooled_labels, pooled_paths):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(fold_summaries)
    df.to_csv(output_dir / "fold_summary.csv", index=False)

    pooled_threshold = 0.5
    pooled_metrics = classification_report_dict(
        pooled_labels, pooled_probs, threshold=pooled_threshold
    )
    pooled_metrics_df = pd.DataFrame([pooled_metrics])
    pooled_metrics_df.to_csv(output_dir / "pooled_metrics.csv", index=False)

    pooled_pred_df = pd.DataFrame(
        {"filepath": pooled_paths, "label": pooled_labels, "prob": pooled_probs}
    )
    pooled_pred_df.to_csv(output_dir / "pooled_predictions.csv", index=False)


def run_fold_training(
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
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    best_auc = -math.inf
    best_state = None
    best_probs = None
    best_labels = None
    best_metrics = None
    patience = 0
    patience_exceeded_at = None
    stopper = EarlyStopper(early_stop_patience)

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
    with open(metrics_csv, "w", newline="") as metrics_fp:
        writer = csv.writer(metrics_fp)
        writer.writerow(header)

        for epoch in range(1, epochs + 1):
            model.train()
            t_loss_sum, t_probs, t_labels = 0.0, [], []
            for xb, fb, cb, gb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                fb = fb.to(device, non_blocking=True)
                cb = cb.to(device, non_blocking=True)
                gb = gb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                    logits, aux_logits = model(xb, fb, cb, gb)
                    main_loss = criterion(logits, yb)
                    aux_loss = 0.0
                    for aux in aux_logits:
                        aux_loss = aux_loss + aux_criterion(aux, yb)
                    aux_loss = aux_loss / max(len(aux_logits), 1)
                    loss = main_loss + AUX_LOSS_WEIGHT * aux_loss
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)
                t_loss_sum += loss.item() * len(yb)
                t_probs.extend(F.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy())
                t_labels.extend(yb.detach().cpu().numpy())

            t_loss = t_loss_sum / max(len(t_labels), 1)
            train_metrics = classification_report_dict(t_labels, t_probs)

            v_backup = ema.apply_to(model) if ema is not None else None
            v_loss, v_probs, v_labels, val_metrics = evaluate_model(model, val_loader, criterion)
            if ema is not None:
                ema.restore(model, v_backup)

            threshold, val_metrics_youden, bal_acc_youden = compute_youden_metrics(
                v_labels, v_probs
            )

            if len(set(v_labels)) > 1:
                fpr, tpr, thr = roc_curve(v_labels, v_probs)
            else:
                fpr, tpr, thr = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
            pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
                roc_dir / f"epoch_{epoch}.csv", index=False
            )

            lr = optimizer.param_groups[0]["lr"]
            writer.writerow(
                [
                    epoch,
                    f"{t_loss:.6f}",
                    f"{train_metrics['acc']:.4f}",
                    f"{train_metrics['auc']:.4f}",
                    f"{train_metrics['pr_auc']:.4f}",
                    f"{v_loss:.6f}",
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
            metrics_fp.flush()

            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                best_state = deepcopy(model.state_dict() if ema is None else ema.shadow)
                best_probs = list(v_probs)
                best_labels = list(v_labels)
                best_metrics = {
                    "val_loss": v_loss,
                    **{f"val_{k}": val for k, val in val_metrics.items()},
                    "val_threshold_youden": threshold,
                    "val_acc_youden": val_metrics_youden["acc"],
                    "val_balanced_acc_youden": bal_acc_youden,
                    "val_precision_youden": val_metrics_youden["precision"],
                    "val_recall_youden": val_metrics_youden["recall"],
                    "val_tn_youden": val_metrics_youden["tn"],
                    "val_fp_youden": val_metrics_youden["fp"],
                    "val_fn_youden": val_metrics_youden["fn"],
                    "val_tp_youden": val_metrics_youden["tp"],
                    "val_selection_metric": "auc",
                    "val_selection_score": val_metrics["auc"],
                }
                patience = 0
            else:
                patience += 1
                if patience_exceeded_at is None and patience >= early_stop_patience:
                    patience_exceeded_at = epoch

            if epoch % log_every == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d} | train_auc={train_metrics['auc']:.4f} "
                    f"val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['acc']:.4f} lr={lr:.2e}"
                )

            if stopper.step(val_metrics["auc"], epoch):
                if patience_exceeded_at is None:
                    patience_exceeded_at = epoch
                break

    if best_state is not None:
        torch.save(best_state, fold_dir / f"best_model_fold_{fold + 1}.pt")

    if best_metrics is None:
        best_metrics = {
            "val_loss": v_loss,
            **{f"val_{k}": val for k, val in val_metrics.items()},
            "val_threshold_youden": threshold,
            "val_acc_youden": val_metrics_youden["acc"],
            "val_balanced_acc_youden": bal_acc_youden,
            "val_precision_youden": val_metrics_youden["precision"],
            "val_recall_youden": val_metrics_youden["recall"],
            "val_tn_youden": val_metrics_youden["tn"],
            "val_fp_youden": val_metrics_youden["fp"],
            "val_fn_youden": val_metrics_youden["fn"],
            "val_tp_youden": val_metrics_youden["tp"],
            "val_selection_metric": "auc",
            "val_selection_score": val_metrics["auc"],
        }
        best_probs = list(v_probs)
        best_labels = list(v_labels)
    best_metrics["patience_exceeded_at"] = patience_exceeded_at or -1
    return best_metrics, best_probs, best_labels


def build_model(clinical_dim: int, global_dim: int):
    return MultiBranchClassifier(
        clinical_dim=clinical_dim,
        global_dim=global_dim,
        flow_channels=FLOW_CHANNELS,
        embed_dim=EMBED_DIM,
    )


def forward_path(model, batch, device, train: bool):
    xb, fb, cb, gb, yb, path = batch
    xb = xb.to(device, non_blocking=True)
    fb = fb.to(device, non_blocking=True)
    cb = cb.to(device, non_blocking=True)
    gb = gb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    logits, _ = model(xb, fb, cb, gb)
    return logits, yb, path


def main():
    import argparse

    global \
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
        SEED, \
        CV_SEED

    parser = argparse.ArgumentParser()
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

    set_seed(SEED)

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"seed_{SEED}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for v10 training but torch.cuda.is_available() is false."
        )

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building v10 multibranch model on {device_name} (seed={SEED}, cv_seed={CV_SEED})")
    print(
        f"Training config: folds={N_FOLDS} epochs={EPOCHS} batch_size={BATCH_SIZE} lr={LR} wd={WEIGHT_DECAY} patience={EARLY_STOP_PATIENCE} target_n={TARGET_N}"
    )

    if args.dry_run:
        categories = {
            "location": ["A", "B"],
            "hospital": ["H1"],
            "source": ["S1"],
            "side": ["L", "R"],
        }
        clinical_dim = 2 + sum(len(v) for v in categories.values())
        model = build_model(clinical_dim=clinical_dim, global_dim=GLOBAL_FEATURE_DIM).to("cpu")
        x = torch.randn(2, TARGET_N, 3)
        f = torch.randn(2, TARGET_N, FLOW_CHANNELS)
        c = torch.randn(2, clinical_dim)
        g = torch.randn(2, GLOBAL_FEATURE_DIM)
        with torch.no_grad():
            logits, aux = model(x, f, c, g)
        print(
            "Dry-run forward pass successful; output shape:", logits.shape, "aux_heads:", len(aux)
        )
        return

    df = discover_cases(DATA_DIR, METADATA_PATH)
    if len(df) == 0:
        raise RuntimeError(f"No samples found under {DATA_DIR} using metadata {METADATA_PATH}")

    categories = compute_clinical_categories(df)
    clinical_dim = 2 + sum(len(v) for v in categories.values())
    print(f"Total samples: {len(df)}   clinical_dim={clinical_dim}")
    print(
        f"Class counts: unruptured={int((df['target'] == 0).sum())}, ruptured={int((df['target'] == 1).sum())}"
    )

    cache = build_sample_cache(df)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    pooled_probs, pooled_labels, pooled_paths = [], [], []
    fold_summaries = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(df)), df["target"].values)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)

        train_clin, clin_stats = encode_clinical(train_df, categories)
        val_clin, _ = encode_clinical(val_df, categories, train_stats=clin_stats)

        train_pts, train_flow, train_global, train_labels = [], [], [], []
        for _, row in train_df.iterrows():
            pts, feats, gf = cache[row["filepath"]]
            train_pts.append(pts.clone())
            train_flow.append(feats.clone())
            train_global.append(gf.clone())
            train_labels.append(torch.tensor(int(row["target"]), dtype=torch.long))

        val_pts, val_flow, val_global, val_labels = [], [], [], []
        for _, row in val_df.iterrows():
            pts, feats, gf = cache[row["filepath"]]
            val_pts.append(pts.clone())
            val_flow.append(feats.clone())
            val_global.append(gf.clone())
            val_labels.append(torch.tensor(int(row["target"]), dtype=torch.long))

        (pts_mu, pts_sigma), (flow_mu, flow_sigma), (glob_mu, glob_sigma) = (
            compute_fold_normalization(train_pts, train_flow, train_global)
        )
        train_pts = apply_fold_normalization(train_pts, pts_mu, pts_sigma)
        val_pts = apply_fold_normalization(val_pts, pts_mu, pts_sigma)
        train_flow = apply_fold_normalization(train_flow, flow_mu, flow_sigma)
        val_flow = apply_fold_normalization(val_flow, flow_mu, flow_sigma)
        train_global = apply_fold_normalization(train_global, glob_mu, glob_sigma)
        val_global = apply_fold_normalization(val_global, glob_mu, glob_sigma)

        train_pts = torch.stack(train_pts)
        train_flow = torch.stack(train_flow)
        train_global = torch.stack(train_global)
        train_labels = torch.stack(train_labels)
        val_pts = torch.stack(val_pts)
        val_flow = torch.stack(val_flow)
        val_global = torch.stack(val_global)
        val_labels = torch.stack(val_labels)

        train_clin = train_clin
        val_clin = val_clin

        train_dataset = TensorDataset(train_pts, train_flow, train_clin, train_global, train_labels)
        val_paths = val_df["filepath"].values.tolist()
        val_dataset = TensorDataset(val_pts, val_flow, val_clin, val_global, val_labels)

        sampler = balanced_sampler(train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            shuffle=False,
            drop_last=len(train_dataset) > BATCH_SIZE,
        )
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        n_pos = int(train_labels.sum().item())
        n_neg = int(len(train_labels) - n_pos)
        alpha = torch.tensor([1.0 / max(n_neg, 1), 1.0 / max(n_pos, 1)], dtype=torch.float32)
        alpha = alpha / alpha.sum() * 2.0

        model = build_model(clinical_dim=clinical_dim, global_dim=GLOBAL_FEATURE_DIM).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = FocalLoss(alpha=alpha.to(DEVICE), gamma=2.0)
        aux_criterion = FocalLoss(alpha=alpha.to(DEVICE), gamma=2.0)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=LR,
            total_steps=max(EPOCHS * max(len(train_loader), 1), 1),
            pct_start=0.12,
            div_factor=25.0,
            final_div_factor=100.0,
            anneal_strategy="cos",
        )

        best_metrics, best_probs, best_labels = run_fold_training(
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

        best_paths = val_paths
        fold_summaries.append({"fold": fold + 1, **best_metrics})
        pooled_probs.extend(list(best_probs))
        pooled_labels.extend(list(best_labels))
        pooled_paths.extend(list(best_paths))
        print(f"  Best val AUC: {best_metrics.get('val_auc', 0.0):.4f}")

    write_fold_summary(output_dir, fold_summaries, pooled_probs, pooled_labels, pooled_paths)
    print(f"Training complete. Results written to {output_dir}")


if __name__ == "__main__":
    main()
