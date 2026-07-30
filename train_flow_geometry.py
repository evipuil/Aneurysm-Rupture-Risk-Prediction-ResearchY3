# Version 7 source snapshot
"""
PointNet++ classifier using geometry + hemodynamic features + global summary.

Improvements vs v6:
- Extra per-point flow channels (low-TAWSS flag, high-OSI flag, combined
  stress, normalized von Mises, risk product) so SA layers see richer local
  signals beyond raw TAWSS/OSI/VM.
- Attention-based fusion of the 1024-d PointNet embedding with the global
  summary features (instead of a plain concat), which scales each modality
  per-case.
- Shared common helpers for metrics, training loop, EMA, augmentation.
- Optional AMP autocast and gradient clipping for faster, more stable training.
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

_pointnet_root = None
_env_root = os.environ.get("POINTNET_ROOT")
if _env_root:
    _pointnet_root = Path(_env_root).resolve()
else:
    for _p in (Path(__file__).resolve().parent, *Path(__file__).resolve().parent.parents):
        if _p.name == "pointnet_pytorch":
            _pointnet_root = _p
            break

if _pointnet_root is None:
    _pointnet_root = Path(__file__).resolve().parent.parent

_proj_parent = (
    str(_pointnet_root.parent) if _pointnet_root.name == "pointnet_pytorch" else str(_pointnet_root)
)
if _proj_parent not in sys.path:
    sys.path.insert(0, _proj_parent)

try:
    import importlib.util

    _local_v7 = Path(_pointnet_root) / "common.py"
    if _local_v7.exists():
        _spec = importlib.util.spec_from_file_location("common_local", str(_local_v7))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        vc = _mod
    else:
        try:
            import pointnet_pytorch.common as vc
        except Exception:
            import common as vc
except Exception:
    try:
        import pointnet_pytorch.common as vc
    except Exception:
        import common as vc

METADATA_PATH = os.environ.get("V7_METADATA", "metadata.csv")
DATA_DIR = os.environ.get("V7_DATA_DIR", "predictions/pinn_corrected")
OUTPUT_DIR = Path(os.environ.get("V7_OUTPUT_DIR", "results_flow_geometry_v7"))
N_FOLDS = int(os.environ.get("V7_FOLDS", 5))
BATCH_SIZE = int(os.environ.get("V7_BATCH", 8))
EPOCHS = int(os.environ.get("V7_EPOCHS", 200))
LR = float(os.environ.get("V7_LR", 1e-4))
WEIGHT_DECAY = float(os.environ.get("V7_WD", 1e-4))
TARGET_N = int(os.environ.get("V7_TARGET_N", 8192))
EARLY_STOP_PATIENCE = int(os.environ.get("V7_PATIENCE", 40))
USE_AMP = os.environ.get("V7_AMP", "1").lower() in {"1", "true", "yes"}

# Flow channels: 8 (tawss, osi, vm, low_tawss, high_osi, combined, vm_norm, risk)
FLOW_CHANNELS = 8
GLOBAL_FEATURE_DIM = 23  # matches summarize_global_features with include_rrt=False

vc.set_seed(vc.SEED)
DEVICE = vc.get_device()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Model
class AttentionFuse(nn.Module):
    """Gate-weighted fusion: each modality is scaled by a learned softmax gate."""

    def __init__(self, point_dim: int, global_dim: int):
        super().__init__()
        self.global_up = nn.Linear(global_dim, point_dim)
        self.gate = nn.Sequential(nn.Linear(point_dim + global_dim, 2), nn.Softmax(dim=-1))

    def forward(self, p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        gu = F.gelu(self.global_up(g))
        w = self.gate(torch.cat([p, g], dim=-1))
        return torch.cat([w[:, 0:1] * p, w[:, 1:2] * gu], dim=-1)


def build_model() -> nn.ModuleDict:
    sa1 = vc.PointNetSetAbstraction(
        512, 0.2, 32, in_channel=3 + FLOW_CHANNELS, mlp=[64, 64, 128], dropout=0.1
    )
    sa2 = vc.PointNetSetAbstraction(128, 0.4, 64, in_channel=131, mlp=[128, 128, 256], dropout=0.1)
    sa3 = vc.PointNetSetAbstraction(
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
    head = nn.Sequential(
        nn.Linear(1024 * 2, 512),
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
        {"sa1": sa1, "sa2": sa2, "sa3": sa3, "global_fc": global_fc, "fuse": fuse, "head": head}
    )


def forward_model(model, xyz, feats, global_feats):
    B = xyz.shape[0]
    l1, l1_xyz = model["sa1"](xyz, feats)
    l2, l2_xyz = model["sa2"](l1_xyz, l1)
    l3, _ = model["sa3"](l2_xyz, l2)
    x = l3.view(B, 1024)
    g = model["global_fc"](global_feats)
    fused = model["fuse"](x, g)
    return model["head"](fused)


# Data loading
def load_flow_sample(path: str, label: int, augment: bool):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    try:
        pts = df[["x", "y", "z"]].values.astype(np.float32)
    except KeyError:
        pts = df.iloc[:, :3].values.astype(np.float32)

    # Pull raw tawss / osi / von_mises columns
    raw_cols = []
    for key in ("tawss", "osi", "von"):
        match = next((c for c in df.columns if key in c), None)
        raw_cols.append(df[match].values if match else np.zeros(len(df)))
    raw = np.stack(raw_cols, axis=1).astype(np.float32)

    # Compute summary stats on the FULL data before subsampling
    global_feats = vc.summarize_global_features(raw, pts, include_rrt=False)
    per_point = vc.derive_hemo_channels(raw, include_rrt=False)

    idx = vc.resample_cloud(pts, TARGET_N)
    pts = pts[idx]
    per_point = per_point[idx]

    pts = vc.normalize_points(pts)
    per_point = vc.normalize_features_zscore(per_point)

    if augment:
        pts = vc.so3_rotate(pts)
        pts = vc.jitter(pts)
        pts = pts * np.random.uniform(0.9, 1.1)
        pts, per_point = vc.random_point_dropout(pts, per_point, p=0.1)

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
    sampler = vc.balanced_sampler(ys) if shuffle else None
    return DataLoader(
        TensorDataset(xs, fs, gs, ys),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        drop_last=(shuffle and len(ys) > BATCH_SIZE),
    )


def discover_labeled_cases():
    meta_df = pd.read_csv(METADATA_PATH)
    label_map = {}
    for _, row in meta_df.iterrows():
        ds = str(row.get("dataset", "")).strip()
        vid = str(row.get("vesselFileID", "")).strip()
        cut = (
            str(row.get("cutToShow", "cut1")).strip() if pd.notna(row.get("cutToShow")) else "cut1"
        )
        status = str(row.get("status", "")).strip().lower()
        if status not in {"ruptured", "unruptured"}:
            continue
        label = 1 if status == "ruptured" else 0
        for key in (ds, vid):
            if key:
                label_map[f"{key}_{cut}"] = label
                label_map[key] = label

    paths, labels = [], []
    for folder in sorted(os.listdir(DATA_DIR)):
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        csv_path = os.path.join(folder_path, "hemodynamics_aggregate.csv")
        if not os.path.exists(csv_path):
            continue
        label = label_map.get(folder)
        if label is None:
            base = re.sub(r"_cut\d+$", "", folder)
            label = label_map.get(base)
        if label is not None:
            paths.append(csv_path)
            labels.append(label)
    return np.array(paths), np.array(labels)


def forward_fn(model, batch, device, train: bool):
    xb, fb, gb, yb = batch
    xb = xb.to(device, non_blocking=True)
    fb = fb.to(device, non_blocking=True)
    gb = gb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    logits = forward_model(model, xb, fb, gb)
    return logits, yb, xb.size(0)


def main():
    paths, labels = discover_labeled_cases()
    print(f"Total samples: {len(paths)}   Device: {DEVICE}")

    fold_indices = vc.stratified_kfold_indices(labels, N_FOLDS)
    pooled_probs, pooled_labels, fold_summaries = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(fold_indices):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
        print("  Loading flow-geometry samples...")
        train_loader = build_loader(paths[tr_idx], labels[tr_idx], augment=True, shuffle=True)
        val_loader = build_loader(paths[va_idx], labels[va_idx], augment=False, shuffle=False)

        model = build_model().to(DEVICE)
        y_tr = torch.tensor(labels[tr_idx], dtype=torch.long)
        weights = vc.class_weights(y_tr, DEVICE)
        criterion = vc.FocalLoss(alpha=weights, gamma=2.0, label_smoothing=0.05)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

        best_metrics, best_probs, best_labels = vc.run_fold_training(
            forward_fn=forward_fn,
            model=model,
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
            use_ema=True,
            grad_clip=1.0,
            amp=USE_AMP,
        )
        fold_summaries.append({"fold": fold + 1, **best_metrics})
        pooled_probs.extend(best_probs)
        pooled_labels.extend(best_labels)
        print(f"  Best val AUC: {best_metrics.get('val_auc', 0.0):.4f}")

    vc.write_fold_summary(OUTPUT_DIR, fold_summaries, pooled_probs, pooled_labels)


if __name__ == "__main__":
    main()
