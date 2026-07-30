# Version 7 source snapshot
"""
Geometry-only PointNet++ classifier (point coordinates only).

Improvements vs v6:
- Fewer but wider SA layers with 2D dropout inside the set-abstraction MLPs
  (reduces co-adaptation on small datasets).
- Augmentation adds random point dropout + mixed scale/jitter.
- EMA weights during validation for more stable AUC; best-AUC model tracked
  (rather than "first early-stop trigger").
- AMP autocast on GPU for ~2x faster forward passes; loss is still computed in
  fp32 for numerical stability.
- Uses shared helpers from common so training logic is not duplicated.
"""

import os
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
OUTPUT_DIR = Path(os.environ.get("V7_OUTPUT_DIR", "results_geometry_v7"))
N_FOLDS = int(os.environ.get("V7_FOLDS", 5))
BATCH_SIZE = int(os.environ.get("V7_BATCH", 8))
EPOCHS = int(os.environ.get("V7_EPOCHS", 200))
LR = float(os.environ.get("V7_LR", 1e-4))
WEIGHT_DECAY = float(os.environ.get("V7_WD", 1e-4))
TARGET_N = int(os.environ.get("V7_TARGET_N", 8192))
EARLY_STOP_PATIENCE = int(os.environ.get("V7_PATIENCE", 40))
USE_AMP = os.environ.get("V7_AMP", "1").lower() in {"1", "true", "yes"}

vc.set_seed(vc.SEED)
DEVICE = vc.get_device()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Model
class PointNeXtSetAbstraction(nn.Module):
    """A lightweight PointNeXt-style set abstraction layer.

    This keeps the same functional interface as the existing
    `PointNetSetAbstraction` so it can be swapped in place. It uses the
    same sampling/grouping helpers from `common` (imported as `vc`).
    """

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
            # pointwise conv operating on grouped points (Conv2d: in_ch, out_ch, 1)
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        # lightweight residual projection from input -> final channel (for skip)
        self.need_proj = in_channel != mlp[-1]
        if self.need_proj:
            self.proj = nn.Conv2d(in_channel, mlp[-1], 1)
            self.proj_bn = nn.BatchNorm2d(mlp[-1])

    def forward(self, xyz, points=None):
        # Follow the same unpacking convention as common helpers
        if self.group_all:
            new_xyz, new_points = vc.sample_and_group_all(xyz, points)
            # sample_and_group_all returns (new_xyz, new_points)
        else:
            new_points, new_xyz = vc.sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )

        # new_points: (B, npoint, nsample, C) or (B, 1, N, C) for group_all
        x = new_points.permute(0, 3, 2, 1)  # -> (B, C, nsample, npoint)

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            x = bn(conv(x))
            x = nn.GELU()(x)
            if self.dropout > 0 and self.training:
                x = F.dropout2d(x, p=self.dropout, training=True)

        if self.need_proj:
            sc = self.proj_bn(self.proj(new_points.permute(0, 3, 2, 1)))
            x = x + sc
            x = nn.GELU()(x)

        # channel-wise max pool over local neighborhood (nsample dim)
        new_points = torch.max(x, dim=2).values  # -> (B, out_ch, npoint)
        return new_points.permute(0, 2, 1), new_xyz


def build_model() -> nn.ModuleDict:
    # SA1: 512 centroids, r=0.2, in_channel = 3 (xyz relative)
    sa1 = PointNeXtSetAbstraction(512, 0.2, 32, in_channel=3, mlp=[64, 64, 128], dropout=0.1)
    sa2 = PointNeXtSetAbstraction(128, 0.4, 64, in_channel=131, mlp=[128, 128, 256], dropout=0.1)
    sa3 = PointNeXtSetAbstraction(
        None, None, None, in_channel=259, mlp=[256, 512, 1024], group_all=True, dropout=0.1
    )
    head = nn.Sequential(
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2),
    )
    return nn.ModuleDict({"sa1": sa1, "sa2": sa2, "sa3": sa3, "head": head})


def forward_model(model: nn.ModuleDict, xyz: torch.Tensor) -> torch.Tensor:
    B = xyz.shape[0]
    l1_pts, l1_xyz = model["sa1"](xyz, None)
    l2_pts, l2_xyz = model["sa2"](l1_xyz, l1_pts)
    l3_pts, _ = model["sa3"](l2_xyz, l2_pts)
    return model["head"](l3_pts.view(B, 1024))


# Data loading
def load_geometry_sample(path: str, label: int, augment: bool):
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        try:
            pts = df[["x", "y", "z"]].values.astype(np.float32)
        except KeyError:
            pts = df.iloc[:, :3].values.astype(np.float32)

        idx = vc.resample_cloud(pts, TARGET_N)
        pts = pts[idx]
        pts = vc.normalize_points(pts)

        if augment:
            pts = vc.so3_rotate(pts)
            pts = vc.jitter(pts)
            pts = pts * np.random.uniform(0.9, 1.1)
            pts, _ = vc.random_point_dropout(pts, None, p=0.1)

        return torch.tensor(pts, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    except Exception as exc:
        print(f"Error loading {path}: {exc}")
        return torch.zeros(TARGET_N, 3), torch.tensor(label, dtype=torch.long)


def build_loader(paths, labels, augment: bool, shuffle: bool):
    xs, ys = [], []
    for path, label in zip(paths, labels):
        pts, lab = load_geometry_sample(path, int(label), augment=augment)
        xs.append(pts)
        ys.append(lab)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    sampler = vc.balanced_sampler(ys) if shuffle else None
    return DataLoader(
        TensorDataset(xs, ys),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        drop_last=(shuffle and len(ys) > BATCH_SIZE),
    )


# Label discovery from metadata
def discover_labeled_cases():
    meta_df = pd.read_csv(METADATA_PATH)
    import re

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


# Training
def forward_fn(model, batch, device, train: bool):
    xb, yb = batch
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    logits = forward_model(model, xb)
    return logits, yb, xb.size(0)


def main():
    paths, labels = discover_labeled_cases()
    print(f"Total samples: {len(paths)}   Device: {DEVICE}")

    fold_indices = vc.stratified_kfold_indices(labels, N_FOLDS)
    pooled_probs, pooled_labels, fold_summaries = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(fold_indices):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
        print("  Loading point clouds...")
        train_loader = build_loader(paths[tr_idx], labels[tr_idx], augment=True, shuffle=True)
        val_loader = build_loader(paths[va_idx], labels[va_idx], augment=False, shuffle=False)

        model = build_model().to(DEVICE)
        y_tr = torch.tensor(labels[tr_idx], dtype=torch.long)
        weights = vc.class_weights(y_tr, DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
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
