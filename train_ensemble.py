# Version 8 source snapshot
"""
train_ensemble.py (SELF-CONTAINED)

Ensemble-style training that reuses the PointNeXt flow-geometry model
as a base estimator and trains a simple stacking classifier on global features
and model logits. All helpers inlined for independence.
"""

import csv
import math
import os
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional

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
from torch.utils.data import DataLoader, TensorDataset

# CONSTANTS & CONFIG
SEED = 42
METADATA_PATH = os.environ.get("V8_METADATA", "metadata.csv")
DATA_DIR = os.environ.get("V8_DATA_DIR", "predictions/pinn_corrected")
OUTPUT_DIR = Path(os.environ.get("V8_OUTPUT_DIR", "results_ensemble_v8"))
N_FOLDS = int(os.environ.get("V8_FOLDS", 5))
BATCH_SIZE = int(os.environ.get("V8_BATCH", 8))
EPOCHS = int(os.environ.get("V8_EPOCHS", 200))
EARLY_STOP_PATIENCE = int(os.environ.get("V8_PATIENCE", 40))
LR = float(os.environ.get("V8_LR", 1e-4))
WEIGHT_DECAY = float(os.environ.get("V8_WD", 1e-4))
TARGET_N = int(os.environ.get("V8_TARGET_N", 8192))
USE_AMP = os.environ.get("V8_AMP", "1").lower() in {"1", "true", "yes"}
MODEL_TYPE = os.environ.get("V8_MODEL", "pointnext").lower()
FLOW_CHANNELS = 8
GLOBAL_FEATURE_DIM = 23


# CORE HELPERS
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_auc(targets, probs):
    try:
        return roc_auc_score(targets, probs)
    except ValueError:
        return 0.5


def safe_ap(targets, probs):
    try:
        return average_precision_score(targets, probs) if len(set(targets)) > 1 else 0.0
    except ValueError:
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


# POINTNET++ SAMPLING HELPERS
def _index_points(points, idx):
    B = points.shape[0]
    view_shape = [1] * idx.ndim
    view_shape[0] = B
    batch_indices = torch.arange(B, device=points.device).view(view_shape).expand_as(idx)
    return points[batch_indices, idx]


def _square_distance(src, dst):
    return torch.cdist(src, dst).pow(2)


def farthest_point_sample(xyz, npoint):
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


def sample_and_group(npoint, radius, nsample, xyz, points):
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


# HEMODYNAMIC FEATURES
def derive_hemo_channels(raw_feats: np.ndarray, include_rrt: bool = False) -> np.ndarray:
    tawss = raw_feats[:, 0]
    osi = raw_feats[:, 1]
    von_mises = raw_feats[:, 2]
    low_tawss = (tawss < np.percentile(tawss, 20)).astype(np.float32)
    high_osi = (osi > 0.2).astype(np.float32)
    combined = tawss * (1.0 - 2.0 * osi)
    vm_norm = von_mises / (np.max(von_mises) + 1e-8)
    risk = (low_tawss * high_osi).astype(np.float32)

    channels = [tawss, osi, von_mises]
    if include_rrt:
        denom = (1.0 - 2.0 * osi) * tawss
        rrt = np.zeros_like(tawss, dtype=np.float32)
        mask = np.abs(denom) > 1e-8
        rrt[mask] = 1.0 / denom[mask]
        rrt[~np.isfinite(rrt)] = 0.0
        channels.append(rrt)
    channels.extend([low_tawss, high_osi, combined, vm_norm, risk])
    return np.stack(channels, axis=1).astype(np.float32)


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


def normalize_features_zscore(feats: np.ndarray, clip: float = 3.0) -> np.ndarray:
    mu = feats.mean(axis=0)
    sigma = feats.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return np.clip((feats - mu) / sigma, -clip, clip).astype(np.float32)


# TRAINING & METRICS
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def apply_to(self, model: nn.Module):
        state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return state

    def restore(self, model: nn.Module, backup_state):
        model.load_state_dict(backup_state)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        p = F.softmax(logits, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        return ((1.0 - p_t) ** self.gamma * ce).mean()


def balanced_sampler(labels: torch.Tensor):
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    w = torch.where(labels == 1, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1)).double()
    w = w / w.sum()
    return torch.utils.data.WeightedRandomSampler(
        weights=w, num_samples=len(labels), replacement=True
    )


def class_weights(labels: torch.Tensor, device):
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    w = torch.tensor([1.0 / max(n_neg, 1), 1.0 / max(n_pos, 1)])
    return (w / w.sum() * 2.0).to(device)


def stratified_kfold_indices(targets, n_folds: int, seed: int = SEED):
    counts = np.bincount(np.asarray(targets, dtype=int), minlength=2)
    max_splits = int(counts.min())
    if max_splits < 2:
        raise ValueError(
            f"Need at least 2 samples in each class for CV, got counts={counts.tolist()}"
        )
    n_splits = min(n_folds, max_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy_x = np.zeros(len(targets))
    return list(skf.split(dummy_x, targets))


def run_fold_training(
    forward_fn,
    model: nn.Module,
    train_loader,
    val_loader,
    criterion,
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
    patience_exceeded_at = None

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
    ]
    with open(metrics_csv, "w", newline="") as metrics_fp:
        writer = csv.writer(metrics_fp)
        writer.writerow(header)

        for epoch in range(1, epochs + 1):
            model.train()
            t_loss_sum, t_probs, t_labels = 0.0, [], []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                    logits, yb, n = forward_fn(model, batch, device, train=True)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)

                t_loss_sum += loss.item() * n
                t_probs.extend(F.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy())
                t_labels.extend(yb.detach().cpu().numpy())

            scheduler.step() if scheduler is not None else None
            t_loss = t_loss_sum / max(len(t_labels), 1)
            train_metrics = classification_report_dict(t_labels, t_probs)

            v_backup = ema.apply_to(model) if ema is not None else None
            model.eval()
            v_loss_sum, v_probs, v_labels = 0.0, [], []
            with torch.no_grad():
                for batch in val_loader:
                    with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                        logits, yb, n = forward_fn(model, batch, device, train=False)
                        loss = criterion(logits, yb)
                    v_loss_sum += loss.item() * n
                    v_probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
                    v_labels.extend(yb.cpu().numpy())
            if ema is not None:
                ema.restore(model, v_backup)

            v_loss = v_loss_sum / max(len(v_labels), 1)
            val_metrics = classification_report_dict(v_labels, v_probs)

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

    if best_state is not None:
        torch.save(best_state, fold_dir / f"best_model_fold_{fold + 1}.pt")

    if best_metrics is None:
        best_metrics = {"val_loss": v_loss, **{f"val_{k}": val for k, val in val_metrics.items()}}
        best_probs = list(v_probs)
        best_labels = list(v_labels)

    best_metrics["patience_exceeded_at"] = patience_exceeded_at or -1
    return best_metrics, best_probs, best_labels


def write_fold_summary(output_dir: Path, fold_summaries, pooled_probs, pooled_labels):
    sdf = pd.DataFrame(fold_summaries)
    numeric = sdf.drop(
        columns=[c for c in sdf.columns if sdf[c].dtype == object and c != "fold"], errors="ignore"
    )
    avg = numeric.drop(columns=["fold"], errors="ignore").mean(numeric_only=True)
    std = numeric.drop(columns=["fold"], errors="ignore").std(numeric_only=True)
    sdf = pd.concat(
        [
            sdf,
            pd.DataFrame([{"fold": "mean", **avg.to_dict()}]),
            pd.DataFrame([{"fold": "std", **std.to_dict()}]),
        ],
        ignore_index=True,
    )
    sdf.to_csv(output_dir / "fold_averages.csv", index=False)

    pooled_labels = np.array(pooled_labels)
    pooled_probs = np.array(pooled_probs)
    if len(set(pooled_labels)) > 1:
        fpr, tpr, thr = roc_curve(pooled_labels, pooled_probs)
    else:
        fpr, tpr, thr = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
        output_dir / "pooled_roc.csv", index=False
    )

    preds = pooled_probs > 0.5
    tn, fp, fn, tp = confusion_matrix(pooled_labels, preds, labels=[0, 1]).ravel()
    pd.DataFrame({"tn": [tn], "fp": [fp], "fn": [fn], "tp": [tp]}).to_csv(
        output_dir / "pooled_cm.csv", index=False
    )

    print(
        f"\nPooled AUC: {safe_auc(pooled_labels, pooled_probs):.4f}  "
        f"Pooled PR-AUC: {safe_ap(pooled_labels, pooled_probs):.4f}  "
        f"Pooled Acc: {accuracy_score(pooled_labels, preds):.4f}"
    )


# INITIALIZATION
set_seed(SEED)
DEVICE = get_device()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# PointNeXt SA + optional attention pooling
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
            x = nn.GELU()(x)
            if self.dropout > 0 and self.training:
                x = F.dropout2d(x, p=self.dropout, training=True)
        if self.need_proj:
            sc = self.proj_bn(self.proj(new_points.permute(0, 3, 2, 1)))
            x = x + sc
            x = nn.GELU()(x)
        new_points = torch.max(x, dim=2).values
        return new_points.permute(0, 2, 1), new_xyz


class AttentionPool(nn.Module):
    """Simple multihead attention pooling across points for the 1024-d embedding."""

    def __init__(self, dim=1024, n_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, dim) or (B, npoint, dim) - if single vector, return as-is
        if x.dim() == 2:
            return x
        # perform self-attention pooling: use points as sequence, mean query
        q = x.mean(dim=1, keepdim=True)  # (B,1,dim)
        attn_out, _ = self.mha(q, x, x)
        return self.proj(attn_out.squeeze(1))


class AttentionFuse(nn.Module):
    def __init__(self, point_dim: int, global_dim: int):
        super().__init__()
        self.global_up = nn.Linear(global_dim, point_dim)
        self.gate = nn.Sequential(nn.Linear(point_dim + global_dim, 2), nn.Softmax(dim=-1))

    def forward(self, p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        gu = F.gelu(self.global_up(g))
        w = self.gate(torch.cat([p, g], dim=-1))
        return torch.cat([w[:, 0:1] * p, w[:, 1:2] * gu], dim=-1)


# Model builder with option to use attention-pooled fusion
def build_model(model_type="pointnext") -> nn.ModuleDict:
    sa1 = PointNeXtSetAbstraction(
        512, 0.2, 32, in_channel=3 + FLOW_CHANNELS, mlp=[64, 64, 128], dropout=0.1
    )
    sa2 = PointNeXtSetAbstraction(128, 0.4, 64, in_channel=131, mlp=[128, 128, 256], dropout=0.1)
    sa3 = PointNeXtSetAbstraction(
        None, None, None, in_channel=259, mlp=[256, 512, 1024], group_all=True, dropout=0.1
    )
    global_fc = nn.Sequential(
        nn.Linear(GLOBAL_FEATURE_DIM, 128),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.GELU(),
    )
    fuse = AttentionFuse(point_dim=1024, global_dim=64)
    if model_type == "attention":
        pool = AttentionPool(dim=1024, n_heads=8)
        head_in = 1024 + 1024  # point + attended global-up
    else:
        pool = None
        head_in = 1024 * 2

    head = nn.Sequential(
        nn.Linear(head_in, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2),
    )
    return nn.ModuleDict(
        {
            "sa1": sa1,
            "sa2": sa2,
            "sa3": sa3,
            "global_fc": global_fc,
            "fuse": fuse,
            "pool": pool,
            "head": head,
        }
    )


def forward_model(model, xyz, feats, global_feats, model_type="pointnext"):
    B = xyz.shape[0]
    l1, l1_xyz = model["sa1"](xyz, feats)
    l2, l2_xyz = model["sa2"](l1_xyz, l1)
    l3, _ = model["sa3"](l2_xyz, l2)
    # l3: (B, npoint, 1024)
    if model_type == "attention" and model["pool"] is not None:
        point_seq = l3  # (B, npoint, dim)
        p_emb = model["pool"](point_seq)  # (B, dim)
        g = model["global_fc"](global_feats)
        fused = model["fuse"](p_emb, g)
        # fuse already concatenates point and global, so don't concatenate again
    else:
        p_emb = l3.view(B, 1024)
        g = model["global_fc"](global_feats)
        fused = model["fuse"](p_emb, g)
        # fuse already concatenates point and global, so don't concatenate again
    return model["head"](fused)


# Data loading (reuse v7 functions with minor changes)
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

    global_feats = summarize_global_features(raw, pts, include_rrt=False)
    per_point = derive_hemo_channels(raw, include_rrt=False)

    idx = resample_cloud(pts, TARGET_N)
    pts = pts[idx]
    per_point = per_point[idx]

    # version6 normalization for geometry, z-score for features
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


def build_loader(paths, labels, augment: bool, shuffle: bool):
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
    xs = torch.stack(xs)
    fs = torch.stack(fs)
    gs = torch.stack(gs)
    ys = torch.stack(ys)
    sampler = balanced_sampler(ys) if shuffle else None
    return DataLoader(
        TensorDataset(xs, fs, gs, ys),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        drop_last=(shuffle and len(ys) > BATCH_SIZE),
    )


def collect_samples(paths, labels, augment: bool):
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
    return torch.stack(xs), torch.stack(fs), torch.stack(gs), torch.stack(ys)


def fit_global_stats(gs: torch.Tensor):
    mu = gs.mean(dim=0, keepdim=True)
    sigma = gs.std(dim=0, keepdim=True)
    sigma = torch.where(sigma < 1e-6, torch.ones_like(sigma), sigma)
    return mu, sigma


def apply_global_stats(gs: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, clip: float = 3.0):
    return torch.clamp((gs - mu) / sigma, -clip, clip)


def build_loader_from_tensors(xs, fs, gs, ys, shuffle: bool):
    sampler = balanced_sampler(ys) if shuffle else None
    return DataLoader(
        TensorDataset(xs, fs, gs, ys),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        drop_last=(shuffle and len(ys) > BATCH_SIZE),
    )


class StackingHead(nn.Module):
    def __init__(self, logit_dim=2, global_dim=23):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(logit_dim + global_dim, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 2)
        )

    def forward(self, logits, g):
        x = torch.cat([logits, g], dim=-1)
        return self.fc(x)


def discover_labeled_cases():
    cases = discover_cases(DATA_DIR, METADATA_PATH, require_file="hemodynamics_aggregate.csv")
    return cases["filepath"].values, cases["target"].values


def forward_fn_stack(model, stack_head, batch, device, train: bool):
    xb, fb, gb, yb = batch
    xb = xb.to(device)
    fb = fb.to(device)
    gb = gb.to(device)
    yb = yb.to(device)
    logits = forward_model(model, xb, fb, gb, model_type=MODEL_TYPE)
    out = stack_head(logits, gb)
    return out, yb, xb.size(0)


def main():
    paths, labels = discover_labeled_cases()
    print(f"Total samples: {len(paths)}   Device: {DEVICE}")
    class_counts = np.bincount(labels.astype(int), minlength=2)
    print(f"Class counts: unruptured={int(class_counts[0])}, ruptured={int(class_counts[1])}")

    fold_indices = stratified_kfold_indices(labels, N_FOLDS)
    pooled_probs, pooled_labels, fold_summaries = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(fold_indices):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
        tr_x, tr_f, tr_g, tr_y = collect_samples(paths[tr_idx], labels[tr_idx], augment=True)
        va_x, va_f, va_g, va_y = collect_samples(paths[va_idx], labels[va_idx], augment=False)
        g_mu, g_sigma = fit_global_stats(tr_g)
        tr_g = apply_global_stats(tr_g, g_mu, g_sigma)
        va_g = apply_global_stats(va_g, g_mu, g_sigma)
        train_loader = build_loader_from_tensors(tr_x, tr_f, tr_g, tr_y, shuffle=True)
        val_loader = build_loader_from_tensors(va_x, va_f, va_g, va_y, shuffle=False)

        base_model = build_model().to(DEVICE)
        stack_head = StackingHead().to(DEVICE)

        optimizer = optim.AdamW(
            list(base_model.parameters()) + list(stack_head.parameters()),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        best_metrics, best_probs, best_labels = run_fold_training(
            forward_fn=lambda m, b, d, train: forward_fn_stack(base_model, stack_head, b, d, train),
            model=stack_head,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            epochs=EPOCHS,
            early_stop_patience=EARLY_STOP_PATIENCE,
            fold_dir=OUTPUT_DIR,
            fold=fold,
            use_ema=False,
            grad_clip=1.0,
            amp=USE_AMP,
        )
        fold_summaries.append({"fold": fold + 1, **best_metrics})
        pooled_probs.extend(best_probs)
        pooled_labels.extend(best_labels)
        print(f"  Best val AUC: {best_metrics.get('val_auc', 0.0):.4f}")

    write_fold_summary(OUTPUT_DIR, fold_summaries, pooled_probs, pooled_labels)


if __name__ == "__main__":
    main()
