# Version 5 source snapshot
import csv
import os
import random
import re

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
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# Configuration
METADATA_PATH = "metadata.csv"
DATA_DIR = "predictions/pinn_corrected"
OUTPUT_DIR = "results_geometry"
N_FOLDS = 5
BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
TARGET_N = 8192
EARLY_STOP_PATIENCE = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# PointNet++ utility functions


def index_points(points, idx):
    # Gather points by batch indices
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long, device=points.device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    # Iteratively pick the farthest point from those already chosen
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_ar = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_ar, farthest].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def square_distance(src, dst):
    # Pairwise squared Euclidean distances
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    # Find all points within a ball of given radius around each query point
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    # FPS + ball query + relative coordinates
    B, N, C = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_points, new_xyz


def sample_and_group_all(xyz, points):
    # Global grouping (treat entire cloud as one neighborhood)
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=xyz.device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# Set Abstraction layer


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

    def forward(self, xyz, points=None):
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_points, new_xyz = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C, nsample, npoint]
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]  # max-pool over neighbors
        new_points = new_points.permute(0, 2, 1)  # [B, npoint, D']
        return new_points, new_xyz


# Build PointNet++ geometry model (functional wrapper)


def build_pointnet_geometry():
    # SA1: 512 centroids, radius 0.2, 32 neighbors, xyz-only input (3ch)
    sa1 = PointNetSetAbstraction(512, 0.2, 32, in_channel=3, mlp=[64, 64, 128])
    # SA2: 128 centroids, radius 0.4, 64 neighbors, in = 3 (xyz) + 128 (features)
    sa2 = PointNetSetAbstraction(128, 0.4, 64, in_channel=131, mlp=[128, 128, 256])
    # SA3: global pooling, in = 3 (xyz) + 256 (features)
    sa3 = PointNetSetAbstraction(
        None, None, None, in_channel=259, mlp=[256, 512, 1024], group_all=True
    )

    head = nn.Sequential(
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2),
    )
    return nn.ModuleDict({"sa1": sa1, "sa2": sa2, "sa3": sa3, "head": head})


def forward_pointnet(model, xyz):
    # Hierarchical feature extraction then classification
    B = xyz.shape[0]
    l1_pts, l1_xyz = model["sa1"](xyz, None)
    l2_pts, l2_xyz = model["sa2"](l1_xyz, l1_pts)
    l3_pts, _ = model["sa3"](l2_xyz, l2_pts)
    x = l3_pts.view(B, 1024)
    return model["head"](x)


# Data loading helpers


def normalize_points(pts):
    # Center and scale to unit sphere
    pts = pts - pts.mean(axis=0)
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    return pts / (max_dist + 1e-8)


def so3_rotate(pts):
    # Random rotation in SO(3)
    rx, ry, rz = np.random.uniform(0, 2 * np.pi, 3)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return (pts @ (Rz @ Ry @ Rx).T).astype(np.float32)


def jitter(pts, sigma=0.01, clip=0.05):
    # Small random noise
    return (pts + np.clip(sigma * np.random.randn(*pts.shape), -clip, clip)).astype(np.float32)


def load_geometry_sample(path, label, target_n=TARGET_N, augment=False):
    # Load CSV, extract xyz, resample, normalize, optionally augment
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        try:
            pts = df[["x", "y", "z"]].values.astype(np.float32)
        except KeyError:
            pts = df.iloc[:, :3].values.astype(np.float32)

        n = len(pts)
        if n > target_n:
            idx = np.random.choice(n, target_n, replace=False)
        elif n < target_n:
            idx = np.concatenate([np.arange(n), np.random.choice(n, target_n - n, replace=True)])
        else:
            idx = np.arange(n)
        pts = pts[idx]

        pts = normalize_points(pts)

        if augment:
            pts = so3_rotate(pts)
            pts = jitter(pts)
            pts = pts * np.random.uniform(0.95, 1.05)

        return torch.tensor(pts, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    except Exception as e:
        print(f"Error loading {path}: {e}")
        return torch.zeros(target_n, 3), torch.tensor(label, dtype=torch.long)


def safe_auc(targets, probs):
    try:
        return roc_auc_score(targets, probs)
    except ValueError:
        return 0.5


# Simple list-based dataset (no class inheritance needed)


def build_tensor_batches(file_paths, file_labels, augment, batch_size, shuffle):
    # Load all samples into memory and return a DataLoader
    all_pts, all_labels = [], []
    for path, label in zip(file_paths, file_labels):
        pts, lab = load_geometry_sample(path, label, augment=augment)
        all_pts.append(pts)
        all_labels.append(lab)
    pts_tensor = torch.stack(all_pts)
    lab_tensor = torch.stack(all_labels)
    dataset = torch.utils.data.TensorDataset(pts_tensor, lab_tensor)

    # Balanced sampling for training
    sampler = None
    if shuffle:
        n_pos = int(lab_tensor.sum().item())
        n_neg = len(lab_tensor) - n_pos
        sample_weights = torch.where(lab_tensor == 1, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
        sample_weights = sample_weights / sample_weights.sum()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights.double(),
            num_samples=len(lab_tensor),
            replacement=True,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        drop_last=(shuffle and len(dataset) > batch_size),
    )


# Build metadata label map and walk DATA_DIR for hemodynamics_aggregate.csv
meta_df = pd.read_csv(METADATA_PATH)
label_map = {}
for _, row in meta_df.iterrows():
    ds = str(row.get("dataset", "")).strip()
    vid = str(row.get("vesselFileID", "")).strip()
    cut = str(row.get("cutToShow", "cut1")).strip() if pd.notna(row.get("cutToShow")) else "cut1"
    status = str(row.get("status", "")).strip().lower()
    if status not in ["ruptured", "unruptured"]:
        continue
    label = 1 if status == "ruptured" else 0
    for key in [ds, vid]:
        if key:
            label_map[f"{key}_{cut}"] = label
            label_map[key] = label

file_paths, labels = [], []
for folder in sorted(os.listdir(DATA_DIR)):
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    csv_path = os.path.join(folder_path, "hemodynamics_aggregate.csv")
    if not os.path.exists(csv_path):
        continue
    label = None
    if folder in label_map:
        label = label_map[folder]
    else:
        base = re.sub(r"_cut\d+$", "", folder)
        if base in label_map:
            label = label_map[base]
    if label is not None:
        file_paths.append(csv_path)
        labels.append(label)

file_paths = np.array(file_paths)
labels = np.array(labels)
print(f"Total samples: {len(file_paths)}   Device: {DEVICE}")

# K-fold training
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
pooled_probs, pooled_targets = [], []
fold_summaries = []

for fold, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

    # Re-load with augmentation for training, without for validation
    print("  Loading point clouds ...")
    train_loader = build_tensor_batches(
        file_paths[train_idx], labels[train_idx], augment=True, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = build_tensor_batches(
        file_paths[val_idx], labels[val_idx], augment=False, batch_size=BATCH_SIZE, shuffle=False
    )

    # Class weights
    n_pos = int(labels[train_idx].sum())
    n_neg = len(train_idx) - n_pos
    w = torch.tensor([1.0 / max(n_neg, 1), 1.0 / max(n_pos, 1)])
    w = (w / w.sum() * 2.0).to(DEVICE)

    model = build_pointnet_geometry().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=w)

    # Early-stopping trigger tracking (save model at trigger epoch, not best spike)
    best_val_auc = -1.0
    patience_counter = 0
    early_stop_v_probs = None
    early_stop_v_labels = None
    early_stop_metrics = None
    early_stop_epoch = None

    fold_csv = os.path.join(OUTPUT_DIR, f"fold_{fold + 1}_metrics.csv")
    fold_roc_dir = os.path.join(OUTPUT_DIR, f"fold_{fold + 1}_roc")
    os.makedirs(fold_roc_dir, exist_ok=True)

    with open(fold_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "train_auc",
                "train_pr_auc",
                "val_loss",
                "val_acc",
                "val_auc",
                "val_pr_auc",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        )
        early_stop_announced = False

        for epoch in range(1, EPOCHS + 1):
            # Train
            model.train()
            t_loss, t_probs, t_labels = 0.0, [], []
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                logits = forward_pointnet(model, xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                t_loss += loss.item() * xb.size(0)
                t_probs.extend(F.softmax(logits, 1)[:, 1].detach().cpu().numpy())
                t_labels.extend(yb.cpu().numpy())
            scheduler.step()

            t_loss /= len(train_loader.dataset)
            t_acc = accuracy_score(t_labels, np.array(t_probs) > 0.5)
            t_auc = safe_auc(t_labels, t_probs)
            t_pr_auc = average_precision_score(t_labels, t_probs) if len(set(t_labels)) > 1 else 0.0

            # Validate
            model.eval()
            v_loss, v_probs, v_labels = 0.0, [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = forward_pointnet(model, xb)
                    v_loss += criterion(logits, yb).item() * xb.size(0)
                    v_probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
                    v_labels.extend(yb.cpu().numpy())

            v_loss /= len(val_loader.dataset)
            v_acc = accuracy_score(v_labels, np.array(v_probs) > 0.5)
            v_auc = safe_auc(v_labels, v_probs)
            v_pr_auc = average_precision_score(v_labels, v_probs) if len(set(v_labels)) > 1 else 0.0
            tn, fp, fn, tp = confusion_matrix(
                v_labels, np.array(v_probs) > 0.5, labels=[0, 1]
            ).ravel()

            wr.writerow(
                [
                    epoch,
                    f"{t_loss:.6f}",
                    f"{t_acc:.4f}",
                    f"{t_auc:.4f}",
                    f"{t_pr_auc:.4f}",
                    f"{v_loss:.6f}",
                    f"{v_acc:.4f}",
                    f"{v_auc:.4f}",
                    f"{v_pr_auc:.4f}",
                    tn,
                    fp,
                    fn,
                    tp,
                ]
            )

            # Per-epoch ROC data
            fpr, tpr, thr = roc_curve(v_labels, v_probs)
            pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
                os.path.join(fold_roc_dir, f"epoch_{epoch}.csv"), index=False
            )

            if epoch % 25 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}  train_auc={t_auc:.4f}  val_auc={v_auc:.4f}  val_acc={v_acc:.4f}"
                )

            # Track when early stopping WOULD trigger (run all epochs for full logs)
            if v_auc > best_val_auc:
                best_val_auc = v_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE and not early_stop_announced:
                    print(
                        f"    Early-stop patience reached at epoch {epoch}; continuing to log all remaining epochs."
                    )
                    early_stop_announced = True
                if patience_counter >= EARLY_STOP_PATIENCE and early_stop_v_probs is None:
                    early_stop_v_probs = list(v_probs)
                    early_stop_v_labels = list(v_labels)
                    early_stop_metrics = {
                        "val_loss": v_loss,
                        "val_acc": v_acc,
                        "val_auc": v_auc,
                        "tn": tn,
                        "fp": fp,
                        "fn": fn,
                        "tp": tp,
                    }
                    early_stop_epoch = epoch
                    # Save the model snapshot at the early-stop trigger epoch.
                    state = model.state_dict()
                    torch.save(
                        state, os.path.join(OUTPUT_DIR, f"early_stop_model_fold_{fold + 1}.pt")
                    )
                    # Keep legacy filename for downstream compatibility, but now points to early-stop snapshot.
                    torch.save(state, os.path.join(OUTPUT_DIR, f"best_model_fold_{fold + 1}.pt"))

    # Use early-stop-trigger epoch predictions for pooling; fallback to final epoch if never triggered.
    if early_stop_v_probs is not None:
        pooled_probs.extend(early_stop_v_probs)
        pooled_targets.extend(early_stop_v_labels)
        fold_summaries.append({"fold": fold + 1, **early_stop_metrics})
        print(
            f"  Using early-stop epoch {early_stop_epoch} metrics  val_auc={early_stop_metrics['val_auc']:.4f}"
        )
    else:
        torch.save(
            model.state_dict(), os.path.join(OUTPUT_DIR, f"early_stop_model_fold_{fold + 1}.pt")
        )
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"best_model_fold_{fold + 1}.pt"))
        pooled_probs.extend(v_probs)
        pooled_targets.extend(v_labels)
        fold_summaries.append(
            {
                "fold": fold + 1,
                "val_loss": v_loss,
                "val_acc": v_acc,
                "val_auc": v_auc,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
# Fold averages
sdf = pd.DataFrame(fold_summaries)
avg = sdf.drop(columns=["fold"]).mean()
std = sdf.drop(columns=["fold"]).std()
sdf = pd.concat(
    [
        sdf,
        pd.DataFrame([{"fold": "mean", **avg.to_dict()}]),
        pd.DataFrame([{"fold": "std", **std.to_dict()}]),
    ],
    ignore_index=True,
)
sdf.to_csv(os.path.join(OUTPUT_DIR, "fold_averages.csv"), index=False)

# Pooled ROC & confusion matrix
pooled_targets = np.array(pooled_targets)
pooled_probs = np.array(pooled_probs)

fpr, tpr, thr = roc_curve(pooled_targets, pooled_probs)
pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
    os.path.join(OUTPUT_DIR, "pooled_roc.csv"), index=False
)

cm = confusion_matrix(pooled_targets, pooled_probs > 0.5)
tn, fp, fn, tp = cm.ravel()
pd.DataFrame({"tn": [tn], "fp": [fp], "fn": [fn], "tp": [tp]}).to_csv(
    os.path.join(OUTPUT_DIR, "pooled_cm.csv"), index=False
)

print(
    f"\nPooled AUC: {safe_auc(pooled_targets, pooled_probs):.4f}  "
    f"Pooled Acc: {accuracy_score(pooled_targets, pooled_probs > 0.5):.4f}"
)
