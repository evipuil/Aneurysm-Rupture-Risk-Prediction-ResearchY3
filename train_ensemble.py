# Version 4 source snapshot
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
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# Configuration
METADATA_PATH = "metadata.csv"
DATA_DIR = "predictions/pinn_corrected"
OUTPUT_DIR = "results_ensemble"
N_FOLDS = 5
BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3
SEED = 42
TARGET_N = 4096
LABEL_SMOOTHING = 0.1
GLOBAL_FEATURE_DIM = 23
HEMO_CHANNELS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# PointNet++ utility functions
def index_points(points, idx):
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
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
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
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=xyz.device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# Set Abstraction layer (nn.Module for learnable params)
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
        new_points = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0, 2, 1)
        return new_points, new_xyz


# Build the ensemble: geometry branch + flow branch + clinical MLP + global features
def build_ensemble(n_locations, hemo_channels=HEMO_CHANNELS, global_feature_dim=GLOBAL_FEATURE_DIM):
    # Geometry branch: xyz-only input (3ch)
    geo_sa1 = PointNetSetAbstraction(512, 0.2, 32, in_channel=3, mlp=[64, 64, 128])
    geo_sa2 = PointNetSetAbstraction(128, 0.4, 64, in_channel=131, mlp=[128, 128, 256])
    geo_sa3 = PointNetSetAbstraction(
        None, None, None, in_channel=259, mlp=[256, 512, 1024], group_all=True
    )

    # Flow branch: xyz + derived hemodynamics (3 + hemo_channels = 11 ch)
    flow_sa1 = PointNetSetAbstraction(512, 0.2, 32, in_channel=3 + hemo_channels, mlp=[64, 64, 128])
    flow_sa2 = PointNetSetAbstraction(128, 0.4, 64, in_channel=131, mlp=[128, 128, 256])
    flow_sa3 = PointNetSetAbstraction(
        None, None, None, in_channel=259, mlp=[256, 512], group_all=True
    )

    # Clinical encoder: age (1) + sex (1) + location one-hot (n_locations) -> 64
    clinical_dim = 2 + n_locations
    clinical_enc = nn.Sequential(
        nn.Linear(clinical_dim, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
    )

    # Global feature projection
    global_fc = nn.Sequential(nn.Linear(global_feature_dim, 64), nn.ReLU(), nn.Dropout(0.3))

    # Fusion classifier: geo(1024) + flow(512) + clinical(64) + global(64) = 1664
    head = nn.Sequential(
        nn.Linear(1024 + 512 + 64 + 64, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2),
    )

    return nn.ModuleDict(
        {
            "geo_sa1": geo_sa1,
            "geo_sa2": geo_sa2,
            "geo_sa3": geo_sa3,
            "flow_sa1": flow_sa1,
            "flow_sa2": flow_sa2,
            "flow_sa3": flow_sa3,
            "clinical_enc": clinical_enc,
            "global_fc": global_fc,
            "head": head,
        }
    )


def forward_ensemble(model, xyz, features, clinical, global_feats=None):
    B = xyz.shape[0]

    # Geometry branch
    g1, g1_xyz = model["geo_sa1"](xyz, None)
    g2, g2_xyz = model["geo_sa2"](g1_xyz, g1)
    g3, _ = model["geo_sa3"](g2_xyz, g2)
    geo_feat = g3.view(B, 1024)

    # Flow branch
    f1, f1_xyz = model["flow_sa1"](xyz, features)
    f2, f2_xyz = model["flow_sa2"](f1_xyz, f1)
    f3, _ = model["flow_sa3"](f2_xyz, f2)
    flow_feat = f3.view(B, 512)

    # Clinical branch
    clin_feat = model["clinical_enc"](clinical)

    # Global feature branch
    if global_feats is not None:
        gf = model["global_fc"](global_feats)
    else:
        gf = torch.zeros(B, 64, device=xyz.device)

    # Late fusion
    fused = torch.cat([geo_feat, flow_feat, clin_feat, gf], dim=1)
    return model["head"](fused)


# Data loading helper functions
def normalize_points(pts):
    pts = pts - pts.mean(axis=0)
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    return pts / (max_dist + 1e-8)


def normalize_features(feats):
    mu = feats.mean(axis=0)
    sigma = feats.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return np.clip((feats - mu) / sigma, -3.0, 3.0).astype(np.float32)


def compute_global_features(raw_feats, pts):
    """Compute 23 global summary statistics (hemodynamic + geometric)."""
    tawss = raw_feats[:, 0]
    osi = raw_feats[:, 1]
    von_mises = raw_feats[:, 2]

    features = []

    # TAWSS statistics (7)
    features.extend(
        [
            np.mean(tawss),
            np.std(tawss),
            np.max(tawss),
            np.min(tawss),
            np.percentile(tawss, 95),
            np.percentile(tawss, 5),
            np.sum(tawss > np.percentile(tawss, 90)) / len(tawss),
        ]
    )

    # OSI statistics (5)
    features.extend(
        [
            np.mean(osi),
            np.std(osi),
            np.max(osi),
            np.percentile(osi, 95),
            np.sum(osi > 0.2) / len(osi),
        ]
    )

    # Von Mises statistics (5)
    features.extend(
        [
            np.mean(von_mises),
            np.std(von_mises),
            np.max(von_mises),
            np.percentile(von_mises, 95),
            np.percentile(von_mises, 99),
        ]
    )

    # Geometric features (6) — PCA eigenvalues are key for rupture
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    distances = np.linalg.norm(centered, axis=1)
    try:
        cov = np.cov(pts.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-8)
    except Exception:
        eigenvalues = np.array([0.5, 0.3, 0.2])

    features.extend(
        [
            np.max(distances),
            np.std(distances),
            np.max(distances) / (np.mean(distances) + 1e-6),
            eigenvalues[0],
            eigenvalues[1],
            eigenvalues[0] / (eigenvalues[2] + 1e-6),
        ]
    )

    features = np.array(features, dtype=np.float32)
    features = np.sign(features) * np.log1p(np.abs(features))
    features = np.clip(features, -10, 10)
    return features


def so3_rotate(pts):
    rx, ry, rz = np.random.uniform(0, 2 * np.pi, 3)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return (pts @ (Rz @ Ry @ Rx).T).astype(np.float32)


def jitter(pts, sigma=0.01, clip=0.05):
    return (pts + np.clip(sigma * np.random.randn(*pts.shape), -clip, clip)).astype(np.float32)


def derive_hemo_features(raw_feats):
    tawss = raw_feats[:, 0]
    osi = raw_feats[:, 1]
    von_mises = raw_feats[:, 2]
    low_tawss = (tawss < np.percentile(tawss, 20)).astype(np.float32)
    high_osi = (osi > 0.2).astype(np.float32)
    combined_stress = tawss * (1 - 2 * osi)
    vm_normalized = von_mises / (np.max(von_mises) + 1e-8)
    risk_score = (high_osi * low_tawss).astype(np.float32)
    return np.stack(
        [tawss, osi, von_mises, low_tawss, high_osi, combined_stress, vm_normalized, risk_score],
        axis=1,
    ).astype(np.float32)


def load_sample(path, label, target_n=TARGET_N, augment=False):
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]

        try:
            pts = df[["x", "y", "z"]].values.astype(np.float32)
        except KeyError:
            pts = df.iloc[:, :3].values.astype(np.float32)

        # Extract raw hemodynamic features
        feat_cols = []
        for key in ["tawss", "osi", "von"]:
            match = next((c for c in df.columns if key in c), None)
            feat_cols.append(df[match].values if match else np.zeros(len(df)))
        raw_feats = np.stack(feat_cols, axis=1).astype(np.float32)

        # Global features from FULL data (before subsampling)
        gf = compute_global_features(raw_feats, pts)

        # Derive 8 hemodynamic channels
        derived = derive_hemo_features(raw_feats)

        # Resample to target_n
        n = len(pts)
        if n > target_n:
            idx = np.random.choice(n, target_n, replace=False)
        elif n < target_n:
            idx = np.concatenate([np.arange(n), np.random.choice(n, target_n - n, replace=True)])
        else:
            idx = np.arange(n)
        pts = pts[idx]
        derived = derived[idx]

        pts = normalize_points(pts)
        derived = normalize_features(derived)

        if augment:
            pts = so3_rotate(pts)
            pts = jitter(pts)
            pts = pts * np.random.uniform(0.95, 1.05)

        return (
            torch.tensor(pts, dtype=torch.float32),
            torch.tensor(derived, dtype=torch.float32),
            torch.tensor(gf, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return (
            torch.zeros(target_n, 3),
            torch.zeros(target_n, HEMO_CHANNELS),
            torch.zeros(GLOBAL_FEATURE_DIM, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


def safe_auc(targets, probs):
    try:
        return roc_auc_score(targets, probs)
    except ValueError:
        return 0.5


# Build clinical feature vector for a batch
def encode_clinical(df_slice, all_locations, loc_to_idx, train_stats=None):
    ages = df_slice["age"].values.astype(np.float32)
    mu = train_stats["age_mean"] if train_stats else ages.mean()
    sigma = train_stats["age_std"] if train_stats else ages.std()
    ages = (ages - mu) / (sigma + 1e-6)

    sexes = df_slice["sex_enc"].values.astype(np.float32)

    loc_ohe = np.zeros((len(df_slice), len(all_locations)), dtype=np.float32)
    for i, loc in enumerate(df_slice["location"].values):
        loc_ohe[i, loc_to_idx[loc]] = 1.0

    X = np.column_stack([ages, sexes, loc_ohe])
    return torch.tensor(X, dtype=torch.float32), {"age_mean": float(mu), "age_std": float(sigma)}


# Load metadata and build lookup
df = pd.read_csv(METADATA_PATH)
meta_keys = {}
for idx, row in df.iterrows():
    ds = str(row.get("dataset", "")).strip()
    vid = str(row.get("vesselFileID", "")).strip()
    cut = str(row.get("cutToShow", "cut1")).strip() if pd.notna(row.get("cutToShow")) else "cut1"
    for key in [ds, vid]:
        if key:
            meta_keys[f"{key}_{cut}"] = idx
            meta_keys[key] = idx

# Walk DATA_DIR subdirectories for hemodynamics_aggregate.csv
valid_indices = []
filepaths = []
seen = set()
for folder in sorted(os.listdir(DATA_DIR)):
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    csv_path = os.path.join(folder_path, "hemodynamics_aggregate.csv")
    if not os.path.exists(csv_path):
        continue
    matched_idx = None
    if folder in meta_keys:
        matched_idx = meta_keys[folder]
    else:
        base = re.sub(r"_cut\d+$", "", folder)
        if base in meta_keys:
            matched_idx = meta_keys[base]
    if matched_idx is not None and matched_idx not in seen:
        seen.add(matched_idx)
        valid_indices.append(matched_idx)
        filepaths.append(csv_path)

df = df.loc[valid_indices].reset_index(drop=True)
df["filepath"] = filepaths
print(f"Valid samples: {len(df)}")

df["target"] = (df["status"] == "ruptured").astype(int)
df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
df["sex_enc"] = df["sex"].map({"female": 0, "male": 1}).fillna(0).astype(float)
df["location"] = df["location"].fillna("Unknown")

all_locations = sorted(df["location"].unique())
loc_to_idx = {loc: i for i, loc in enumerate(all_locations)}
print(f"Locations ({len(all_locations)}): {all_locations}")
targets = df["target"].values

# K-fold training
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
pooled_probs, pooled_targets = [], []
fold_summaries = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df, targets)):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Clinical features (fit stats on train, apply to val)
    clin_train, stats = encode_clinical(train_df, all_locations, loc_to_idx)
    clin_val, _ = encode_clinical(val_df, all_locations, loc_to_idx, train_stats=stats)

    # Load point cloud + derived flow data + global features
    print("  Loading point clouds ...")
    train_pts, train_feats, train_gf, train_labels = [], [], [], []
    for _, row in train_df.iterrows():
        pts, feats, gf, lab = load_sample(row["filepath"], row["target"], augment=True)
        train_pts.append(pts)
        train_feats.append(feats)
        train_gf.append(gf)
        train_labels.append(lab)

    val_pts, val_feats, val_gf, val_labels = [], [], [], []
    for _, row in val_df.iterrows():
        pts, feats, gf, lab = load_sample(row["filepath"], row["target"], augment=False)
        val_pts.append(pts)
        val_feats.append(feats)
        val_gf.append(gf)
        val_labels.append(lab)

    train_pts = torch.stack(train_pts)
    train_feats = torch.stack(train_feats)
    train_gf = torch.stack(train_gf)
    train_labels = torch.stack(train_labels)
    val_pts = torch.stack(val_pts)
    val_feats = torch.stack(val_feats)
    val_gf = torch.stack(val_gf)
    val_labels = torch.stack(val_labels)

    train_dataset = torch.utils.data.TensorDataset(
        train_pts, train_feats, train_gf, clin_train, train_labels
    )
    val_dataset = torch.utils.data.TensorDataset(val_pts, val_feats, val_gf, clin_val, val_labels)

    # Balanced sampling: oversample minority class
    n_pos = int(train_labels.sum())
    n_neg = len(train_labels) - n_pos
    sample_weights = torch.where(train_labels == 1, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
    sample_weights = sample_weights / sample_weights.sum()
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(train_labels),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        drop_last=len(train_dataset) > BATCH_SIZE,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Class weights
    w = torch.tensor([1.0 / max(n_neg, 1), 1.0 / max(n_pos, 1)])
    w = (w / w.sum() * 2.0).to(DEVICE)

    model = build_ensemble(len(all_locations)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTHING)

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
                "val_loss",
                "val_acc",
                "val_auc",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        )

        for epoch in range(1, EPOCHS + 1):
            # Train
            model.train()
            t_loss, t_probs, t_labels_all = 0.0, [], []
            for xb, fb, gfb, cb, yb in train_loader:
                xb = xb.to(DEVICE)
                fb = fb.to(DEVICE)
                gfb = gfb.to(DEVICE)
                cb = cb.to(DEVICE)
                yb = yb.to(DEVICE)
                optimizer.zero_grad()
                logits = forward_ensemble(model, xb, fb, cb, gfb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                t_loss += loss.item() * xb.size(0)
                t_probs.extend(F.softmax(logits, 1)[:, 1].detach().cpu().numpy())
                t_labels_all.extend(yb.cpu().numpy())
            scheduler.step()

            t_loss /= len(train_dataset)
            t_acc = accuracy_score(t_labels_all, np.array(t_probs) > 0.5)
            t_auc = safe_auc(t_labels_all, t_probs)

            # Validate
            model.eval()
            v_loss, v_probs, v_labels_all = 0.0, [], []
            with torch.no_grad():
                for xb, fb, gfb, cb, yb in val_loader:
                    xb = xb.to(DEVICE)
                    fb = fb.to(DEVICE)
                    gfb = gfb.to(DEVICE)
                    cb = cb.to(DEVICE)
                    yb = yb.to(DEVICE)
                    logits = forward_ensemble(model, xb, fb, cb, gfb)
                    v_loss += criterion(logits, yb).item() * xb.size(0)
                    v_probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
                    v_labels_all.extend(yb.cpu().numpy())

            v_loss /= len(val_dataset)
            v_acc = accuracy_score(v_labels_all, np.array(v_probs) > 0.5)
            v_auc = safe_auc(v_labels_all, v_probs)
            tn, fp, fn, tp = confusion_matrix(v_labels_all, np.array(v_probs) > 0.5).ravel()

            wr.writerow(
                [
                    epoch,
                    f"{t_loss:.6f}",
                    f"{t_acc:.4f}",
                    f"{t_auc:.4f}",
                    f"{v_loss:.6f}",
                    f"{v_acc:.4f}",
                    f"{v_auc:.4f}",
                    tn,
                    fp,
                    fn,
                    tp,
                ]
            )

            fpr, tpr, thr = roc_curve(v_labels_all, v_probs)
            pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
                os.path.join(fold_roc_dir, f"epoch_{epoch}.csv"), index=False
            )

            if epoch % 25 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}  train_auc={t_auc:.4f}  val_auc={v_auc:.4f}  val_acc={v_acc:.4f}"
                )

    # Use final epoch predictions for pooling
    pooled_probs.extend(v_probs)
    pooled_targets.extend(v_labels_all)
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
