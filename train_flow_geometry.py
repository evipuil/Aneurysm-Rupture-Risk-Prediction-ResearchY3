# Version 9 source snapshot
"""
train_flow_geometry.py

Flexible v9 training script to test multiple fusion strategies:
- FUSION_MODE: 'late' (default), 'attention', 'early'
- EARLY fusion projects global features and tiles them into per-point features

This script includes a `--dry-run` mode (or env DRY_RUN=1) which builds
models for the requested fusion mode and runs a single forward pass on
synthetic data to validate tensor shapes. Use this from the `run_all.sh`
script to validate configurations quickly.

Designed to be self-contained (no v7_common import) and to maximize
flexibility when exploring strategies for best performance.
"""

import csv
import math
import os
import random
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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ------------------------ Configuration -------------------------------------
SEED = int(os.environ.get("V9_SEED", 42))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FUSION_MODE = os.environ.get("FUSION_MODE", "late").lower()  # late | attention | early
METADATA_PATH = os.environ.get("V9_METADATA", "metadata.csv")
DATA_DIR = os.environ.get("V9_DATA_DIR", "predictions/pinn_corrected")
OUTPUT_ROOT = Path(os.environ.get("V9_OUTPUT_DIR", "results_v9_flow_geometry"))
N_FOLDS = int(os.environ.get("V9_FOLDS", 5))
BATCH_SIZE = int(os.environ.get("V9_BATCH", 8))
EPOCHS = int(os.environ.get("V9_EPOCHS", 200))
LR = float(os.environ.get("V9_LR", 1e-4))
WEIGHT_DECAY = float(os.environ.get("V9_WD", 1e-4))
EARLY_STOP_PATIENCE = int(os.environ.get("V9_PATIENCE", 40))
USE_AMP = os.environ.get("V9_AMP", "1").lower() in {"1", "true", "yes"}
TARGET_N = int(os.environ.get("V9_TARGET_N", 8192))
GLOBAL_EMBED_DIM = int(os.environ.get("V9_GLOBAL_EMBED_DIM", 16))  # for early fusion projection
DRY_RUN = os.environ.get("DRY_RUN", "0") in {"1", "true", "yes"}

# Flow channels in source data (tawss, osi, von, etc.)
FLOW_CHANNELS = int(os.environ.get("FLOW_CHANNELS", 8))
GLOBAL_FEATURE_DIM = int(os.environ.get("GLOBAL_FEATURE_DIM", 23))
# Lower smoothing => sharper probabilities; 0.05 often hurts acc@0.5 on imbalanced val.
LABEL_SMOOTHING = float(os.environ.get("V9_LABEL_SMOOTHING", "0.02"))
# Model selection: "balanced_youden" (default; better acc when scores are shifted) or "auc" (pure ranking).
BEST_METRIC = os.environ.get("V9_BEST_METRIC", "balanced_youden").lower()

# ------------------------ Reproducibility ----------------------------------


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


def balanced_accuracy(tn: int, fp: int, fn: int, tp: int) -> float:
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (sens + spec)


def youden_threshold(labels, probs):
    """Threshold maximizing (TPR - FPR) on the given set; stabilizes acc vs raw 0.5 when scores are shifted."""
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels, probs)
    if thr.size == 0:
        return 0.5
    j = tpr - fpr
    idx = int(np.argmax(j))
    t = float(thr[idx])
    if not np.isfinite(t):
        t = 0.5
    return t


def metrics_at_threshold(labels, probs, threshold: float):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)
    acc = float(accuracy_score(labels, preds))
    prec = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    ba = balanced_accuracy(tn, fp, fn, tp)
    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "balanced_acc": ba,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def append_youden_val_metrics(summary: dict, labels, probs):
    """Add val_*_youden fields (same val set, decision threshold tuned by Youden's J)."""
    thr = youden_threshold(labels, probs)
    m = metrics_at_threshold(labels, probs, thr)
    summary["val_threshold_youden"] = thr
    summary["val_acc_youden"] = m["acc"]
    summary["val_balanced_acc_youden"] = m["balanced_acc"]
    summary["val_precision_youden"] = m["precision"]
    summary["val_recall_youden"] = m["recall"]
    summary["val_tn_youden"] = m["tn"]
    summary["val_fp_youden"] = m["fp"]
    summary["val_fn_youden"] = m["fn"]
    summary["val_tp_youden"] = m["tp"]
    return summary


def stratified_fold_splits(labels: np.ndarray, n_folds: int, seed: int = SEED):
    """StratifiedKFold with n_splits capped by minority class count."""
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=2)
    if int(counts.min()) < 2:
        raise ValueError(
            f"Need at least 2 samples per class for stratified CV; counts={counts.tolist()}"
        )
    n_splits = min(int(n_folds), int(counts.min()))
    if n_splits < int(n_folds):
        print(
            f"  Warning: reducing CV folds from {n_folds} to {n_splits} "
            f"(minority class count={int(counts.min())})."
        )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(labels)), labels)), n_splits


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
                key_to_idx.setdefault(key, idx)
                key_to_idx.setdefault(f"{key}_{cut}", idx)
                key_to_idx.setdefault(f"{key}_cut1", idx)
    return df, key_to_idx


def _match_folder(folder: str, key_to_idx):
    if folder in key_to_idx:
        return key_to_idx[folder]
    base = folder
    if folder.endswith("_cut1"):
        base = folder[:-5]
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


def balanced_sampler(labels: torch.Tensor):
    labels_np = labels.detach().cpu().numpy().astype(int)
    class_counts = np.bincount(labels_np, minlength=2).astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    weights = 1.0 / class_counts
    sample_weights = torch.tensor(weights[labels_np], dtype=torch.double, device=labels.device)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


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


def discover_labeled_cases():
    cases = discover_cases(DATA_DIR, METADATA_PATH, require_file="hemodynamics_aggregate.csv")
    return cases["filepath"].values, cases["target"].values


def forward_fn(model, batch, device, fusion_mode: str, train: bool):
    xb, fb, gb, yb = batch
    xb = xb.to(device, non_blocking=True)
    fb = fb.to(device, non_blocking=True)
    gb = gb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    logits = forward_model(model, xb, fb, gb, fusion_mode=fusion_mode)
    return logits, yb, xb.size(0)


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
    selection_metric: Optional[str] = None,
):
    sel = (selection_metric or BEST_METRIC).lower()
    if sel not in ("auc", "balanced_youden"):
        print(f"  Warning: unknown selection_metric={sel!r}, using auc")
        sel = "auc"

    fold_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = fold_dir / f"fold_{fold + 1}_metrics.csv"
    roc_dir = fold_dir / f"fold_{fold + 1}_roc"
    roc_dir.mkdir(exist_ok=True)

    ema = EMA(model, decay=0.999) if use_ema else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    best_selection_score = -math.inf
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

            if scheduler is not None:
                scheduler.step()
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
            thr_y = youden_threshold(v_labels, v_probs)
            val_youden = metrics_at_threshold(v_labels, v_probs, thr_y)
            if sel == "balanced_youden":
                selection_score = val_youden["balanced_acc"]
            else:
                selection_score = val_metrics["auc"]

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

            if selection_score > best_selection_score:
                best_selection_score = selection_score
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
                    f"val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['acc']:.4f} "
                    f"val_bal_acc_youden={val_youden['balanced_acc']:.4f} lr={lr:.2e}"
                )

    if best_state is not None:
        torch.save(best_state, fold_dir / f"best_model_fold_{fold + 1}.pt")

    if best_metrics is None:
        best_metrics = {"val_loss": v_loss, **{f"val_{k}": val for k, val in val_metrics.items()}}
        best_probs = list(v_probs)
        best_labels = list(v_labels)

    append_youden_val_metrics(best_metrics, best_labels, best_probs)
    best_metrics["val_selection_metric"] = sel
    best_metrics["val_selection_score"] = float(best_selection_score)
    best_metrics["patience_exceeded_at"] = patience_exceeded_at or -1
    return best_metrics, best_probs, best_labels


def write_fold_summary(output_dir: Path, fold_summaries, pooled_probs, pooled_labels):
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_summaries).to_csv(output_dir / "fold_summary.csv", index=False)
    pooled_metrics = classification_report_dict(pooled_labels, pooled_probs)
    pthr = youden_threshold(pooled_labels, pooled_probs)
    ptuned = metrics_at_threshold(pooled_labels, pooled_probs, pthr)
    pooled_row = {
        **pooled_metrics,
        "pooled_threshold_youden": pthr,
        "pooled_acc_youden": ptuned["acc"],
        "pooled_balanced_acc_youden": ptuned["balanced_acc"],
        "pooled_precision_youden": ptuned["precision"],
        "pooled_recall_youden": ptuned["recall"],
    }
    pd.DataFrame([pooled_row]).to_csv(output_dir / "pooled_metrics.csv", index=False)


# ------------------------ Small helpers ------------------------------------


def _index_points(points, idx):
    B = points.shape[0]
    view_shape = [1] * idx.ndim
    view_shape[0] = B
    batch_indices = torch.arange(B, device=points.device).view(view_shape).expand_as(idx)
    return points[batch_indices, idx]


def _square_distance(src, dst):
    return torch.cdist(src, dst).pow(2)


def farthest_point_sample(xyz, npoint):
    if npoint is None:
        return None
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
    if radius is None or nsample is None:
        return None
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
    if npoint is None:
        return sample_and_group_all(xyz, points)
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = _index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = _index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(xyz.shape[0], npoint, 1, xyz.shape[-1])
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
    return new_points, new_xyz


# (Torch-available path continues below)

# ------------------------ Model components ---------------------------------


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
            new_points, new_xyz = sample_and_group_all(xyz, points)
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
    def __init__(self, dim=1024, n_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        if x.dim() == 2:
            return x
        q = x.mean(dim=1, keepdim=True)
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
        # output concatenation: point-part and global-up part -> total 2*point_dim
        return torch.cat([w[:, 0:1] * p, w[:, 1:2] * gu], dim=-1)


# ------------------------ Model builder ------------------------------------


def build_model(fusion_mode: str = "late") -> nn.ModuleDict:
    fusion_mode = fusion_mode.lower()
    # choose sa1 in_channel based on early fusion
    if fusion_mode == "early":
        # project global features to a small vector and tile per point
        per_point_extra = GLOBAL_EMBED_DIM
    else:
        per_point_extra = 0

    sa1_in = 3 + FLOW_CHANNELS + per_point_extra
    sa1 = PointNeXtSetAbstraction(512, 0.2, 32, in_channel=sa1_in, mlp=[64, 64, 128], dropout=0.1)
    sa2 = PointNeXtSetAbstraction(
        128, 0.4, 64, in_channel=128 + 3, mlp=[128, 128, 256], dropout=0.1
    )
    sa3 = PointNeXtSetAbstraction(
        None, None, None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True, dropout=0.1
    )

    global_proj = nn.Sequential(
        nn.Linear(GLOBAL_FEATURE_DIM, 128),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.GELU(),
    )
    fuse = AttentionFuse(point_dim=1024, global_dim=64)
    pool = AttentionPool(dim=1024, n_heads=8) if fusion_mode == "attention" else None

    head_in = 2048  # by design: fuse returns 2*1024
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

    # add optional early projection
    early_global_proj = (
        nn.Linear(GLOBAL_FEATURE_DIM, per_point_extra) if per_point_extra > 0 else None
    )

    return nn.ModuleDict(
        {
            "sa1": sa1,
            "sa2": sa2,
            "sa3": sa3,
            "global_proj": global_proj,
            "fuse": fuse,
            "pool": pool,
            "head": head,
            "early_global_proj": early_global_proj,
        }
    )


# ------------------------ Forward pass -------------------------------------


def forward_model(model, xyz, per_point_feats, global_feats, fusion_mode: str = "late"):
    B = xyz.shape[0]
    fusion_mode = fusion_mode.lower()

    # Early fusion: project global features and tile to per-point channels
    if (
        fusion_mode == "early"
        and "early_global_proj" in model
        and model["early_global_proj"] is not None
    ):
        gproj = model["early_global_proj"](global_feats)  # (B, GEMB)
        # tile to (B, N, GEMB)
        N = xyz.shape[1]
        gtile = gproj.unsqueeze(1).expand(-1, N, -1)
        per_point = torch.cat([per_point_feats, gtile], dim=-1)
    else:
        per_point = per_point_feats

    l1, l1_xyz = model["sa1"](xyz, per_point)
    l2, l2_xyz = model["sa2"](l1_xyz, l1)
    l3, _ = model["sa3"](l2_xyz, l2)
    # l3: (B, npoint, 1024)

    if fusion_mode == "attention" and model["pool"] is not None:
        p_emb = model["pool"](l3)  # (B, 1024)
        g = model["global_proj"](global_feats)  # (B, 64)
        fused = model["fuse"](p_emb, g)
    else:
        # late fusion (default): reduce point cloud via max and fuse with global
        p_emb = l3.view(B, -1) if l3.dim() == 2 else l3.view(B, 1024)
        g = model["global_proj"](global_feats)
        fused = model["fuse"](p_emb, g)

    out = model["head"](fused)
    return out


# ------------------------ Shape check utility -------------------------------


def synthetic_case(batch_size=4, n_points=TARGET_N, flow_channels=FLOW_CHANNELS):
    # small synthetic cloud for dry-run: use smaller N to be quick
    N = min(n_points, 1024)
    xyz = np.random.randn(batch_size, N, 3).astype(np.float32)
    per_point = np.random.randn(batch_size, N, flow_channels).astype(np.float32)
    global_feats = np.random.randn(batch_size, GLOBAL_FEATURE_DIM).astype(np.float32)
    return torch.tensor(xyz), torch.tensor(per_point), torch.tensor(global_feats)


# ------------------------ CLI / Main ---------------------------------------


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
        EARLY_STOP_PATIENCE, \
        USE_AMP

    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", default=FUSION_MODE, choices=["late", "attention", "early"])
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN)
    parser.add_argument("--metadata-path", default=METADATA_PATH)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--amp", action="store_true", default=USE_AMP)
    args = parser.parse_args()
    METADATA_PATH = args.metadata_path
    DATA_DIR = args.data_dir
    N_FOLDS = args.folds
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    EARLY_STOP_PATIENCE = args.patience
    USE_AMP = args.amp

    fusion = args.fusion
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"{fusion}"
    print(
        f"Building v9 model with fusion mode={fusion} (GLOBAL_EMBED_DIM={GLOBAL_EMBED_DIM}) on {DEVICE}"
    )
    print(
        f"Training config: folds={N_FOLDS} epochs={EPOCHS} batch_size={BATCH_SIZE} lr={LR} "
        f"wd={WEIGHT_DECAY} patience={EARLY_STOP_PATIENCE} label_smoothing={LABEL_SMOOTHING} "
        f"best_metric={BEST_METRIC}"
    )

    if args.dry_run:
        model = build_model(fusion).to(DEVICE)
        print("Model built successfully")
        xyz, per_point, global_feats = synthetic_case(batch_size=2)
        xyz = xyz.to(DEVICE)
        per_point = per_point.to(DEVICE)
        global_feats = global_feats.to(DEVICE)
        try:
            with torch.no_grad():
                out = forward_model(model, xyz, per_point, global_feats, fusion_mode=fusion)
            print("Dry-run forward pass successful; output shape:", out.shape)
        except Exception:
            print("Dry-run forward pass failed with error:")
            raise
        return

    paths, labels = discover_labeled_cases()
    labels = np.asarray(labels).astype(int)
    if len(paths) == 0:
        raise RuntimeError(
            f"No labeled cases found under {DATA_DIR} using metadata {METADATA_PATH}"
        )

    print(f"Total samples: {len(paths)}   Device: {DEVICE}")
    class_counts = np.bincount(labels, minlength=2)
    print(f"Class counts: unruptured={int(class_counts[0])}, ruptured={int(class_counts[1])}")

    fold_indices, n_splits_eff = stratified_fold_splits(labels, N_FOLDS, SEED)
    pooled_probs, pooled_labels, fold_summaries = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(fold_indices):
        print(f"\n--- Fold {fold + 1}/{n_splits_eff} ---")
        print("  Loading flow-geometry samples...")
        tr_x, tr_f, tr_g, tr_y = collect_samples(paths[tr_idx], labels[tr_idx], augment=True)
        va_x, va_f, va_g, va_y = collect_samples(paths[va_idx], labels[va_idx], augment=False)
        g_mu, g_sigma = fit_global_stats(tr_g)
        tr_g = apply_global_stats(tr_g, g_mu, g_sigma)
        va_g = apply_global_stats(va_g, g_mu, g_sigma)
        train_loader = build_loader_from_tensors(tr_x, tr_f, tr_g, tr_y, shuffle=True)
        val_loader = build_loader_from_tensors(va_x, va_f, va_g, va_y, shuffle=False)

        model = build_model(fusion).to(DEVICE)
        print("  Model built successfully")
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

        def fold_forward(model, batch, device, train):
            return forward_fn(model, batch, device, fusion_mode=fusion, train=train)

        best_metrics, best_probs, best_labels = run_fold_training(
            forward_fn=fold_forward,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
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
            selection_metric=BEST_METRIC,
        )
        fold_summaries.append({"fold": fold + 1, **best_metrics})
        pooled_probs.extend(best_probs)
        pooled_labels.extend(best_labels)
        print(f"  Best val AUC: {best_metrics.get('val_auc', 0.0):.4f}")

    write_fold_summary(output_dir, fold_summaries, pooled_probs, pooled_labels)
    print(f"Training complete. Results written to {output_dir}")


if __name__ == "__main__":
    main()
