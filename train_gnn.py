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
import torch_cluster
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import BatchNorm, SAGEConv, global_max_pool, global_mean_pool
from torch_geometric.utils import to_undirected

# Configuration
METADATA_PATH = "metadata.csv"
DATA_DIR = "predictions/pinn_corrected"
OUTPUT_DIR = "results_gnn"
N_FOLDS = 5
BATCH_SIZE = 16

EPOCHS = 120
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
SEED = 42
K_NEIGHBORS = 16
TARGET_N = 2048
HIDDEN_DIM = 96
NUM_GNN_LAYERS = 2
DROPOUT = 0.5
EDGE_DROPOUT = 0.1
EARLY_STOP_PATIENCE = 15
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1
GRAPH_FEATURE_DIM = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Helpers


def safe_auc(targets, probs):
    try:
        return roc_auc_score(targets, probs)
    except ValueError:
        return 0.5


def normalize_points(pts):
    centroid = pts.mean(axis=0)
    pts = pts - centroid
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts


def normalize_features_log1p(feats):
    """Log1p + clip normalization — preserves absolute magnitude (unlike z-score)."""
    feats = np.clip(feats, 1e-6, None)
    feats = np.log1p(feats)
    feats = np.clip(feats, -3.0, 3.0)
    return feats.astype(np.float32)


def compute_graph_features(raw_feats, pts):
    """Compute 20 global summary statistics for graph-level feature enrichment."""
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

    # Geometric features (3)
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    distances = np.linalg.norm(centered, axis=1)
    features.extend(
        [
            np.max(distances),
            np.std(distances),
            np.max(distances) / (np.mean(distances) + 1e-6),
        ]
    )

    features = np.array(features, dtype=np.float32)
    features = np.clip(features, -10, 10)
    # (1, 20) for PyG batching
    return torch.from_numpy(features).float().unsqueeze(0)


# Focal Loss (nn.Module — the only allowed exception type)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()


# Data loading


def load_graph(path, label, k=K_NEIGHBORS, target_n=TARGET_N, augment=False):
    """Load CSV → PyG Data with graph-level features and optional augmentation."""
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]

        coords = df[["x", "y", "z"]].values.astype(np.float32)

        feat_cols = []
        for key in ["tawss", "osi", "von"]:
            match = next((c for c in df.columns if key in c), None)
            feat_cols.append(df[match].values if match else np.zeros(len(df)))
        raw_feats = np.stack(feat_cols, axis=1).astype(np.float32)

        # Graph-level features from FULL data (before subsampling)
        graph_feats = compute_graph_features(raw_feats, coords)

        # Subsample / pad
        n = len(coords)
        if n > target_n:
            idx = np.random.choice(n, target_n, replace=False)
        elif n < target_n:
            idx = np.concatenate([np.arange(n), np.random.choice(n, target_n - n, replace=True)])
        else:
            idx = np.arange(n)
        coords = coords[idx]
        feats = raw_feats[idx]

        coords = normalize_points(coords)
        feats = normalize_features_log1p(feats)

        # Node features = xyz + hemodynamics (6 channels)
        node_feats = np.concatenate([coords, feats], axis=1).astype(np.float32)

        pos = torch.tensor(coords, dtype=torch.float)
        x = torch.tensor(node_feats, dtype=torch.float)

        # --- Augmentation (train only) ---
        if augment:
            # Z-axis rotation
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            R = torch.tensor([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=torch.float)
            pos = pos @ R.T
            x = x.clone()
            x[:, :3] = pos

            # Jitter
            pos = pos + torch.clamp(0.01 * torch.randn_like(pos), -0.03, 0.03)

            # Scaling
            scale = np.random.uniform(0.9, 1.1)
            pos = pos * scale

            # Feature noise
            x[:, 3:] = x[:, 3:] + torch.clamp(0.02 * torch.randn_like(x[:, 3:]), -0.05, 0.05)

        # Build graph
        edge_index = torch_cluster.knn_graph(pos, k=k)
        edge_index = to_undirected(edge_index)

        # Edge dropout (augmentation only)
        if augment and EDGE_DROPOUT > 0 and edge_index.size(1) > 0:
            mask = torch.rand(edge_index.size(1)) > EDGE_DROPOUT
            edge_index = edge_index[:, mask]
            if edge_index.size(1) < 10:
                edge_index = torch_cluster.knn_graph(pos, k=k)
                edge_index = to_undirected(edge_index)

        data = Data(x=x, edge_index=edge_index, pos=pos, y=torch.tensor(label, dtype=torch.long))
        data.graph_features = graph_feats
        return data

    except Exception as e:
        print(f"Error loading {path}: {e}")
        pos = torch.zeros((10, 3))
        x = torch.zeros((10, 6))
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, pos=pos, y=torch.tensor(label, dtype=torch.long))
        data.graph_features = torch.zeros(1, GRAPH_FEATURE_DIM)
        return data


# GNN model (functional — no wrapper class)


def build_gnn(
    in_channels=6,
    hidden=HIDDEN_DIM,
    n_layers=NUM_GNN_LAYERS,
    dropout=DROPOUT,
    graph_feature_dim=GRAPH_FEATURE_DIM,
):
    convs = nn.ModuleList()
    bns = nn.ModuleList()
    convs.append(SAGEConv(in_channels, hidden))
    bns.append(BatchNorm(hidden))
    for _ in range(n_layers - 1):
        convs.append(SAGEConv(hidden, hidden))
        bns.append(BatchNorm(hidden))

    # Graph-feature projection
    graph_fc = nn.Linear(graph_feature_dim, hidden // 2) if graph_feature_dim > 0 else None

    # Classifier: mean+max pooling (hidden*2) + graph features (hidden//2)
    classifier_in = hidden * 2 + (hidden // 2 if graph_feature_dim > 0 else 0)
    head = nn.Sequential(
        nn.Linear(classifier_in, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 2),
    )

    parts = {"convs": convs, "bns": bns, "head": head}
    if graph_fc is not None:
        parts["graph_fc"] = graph_fc
    return nn.ModuleDict(parts)


def forward_gnn(model, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch

    for i, (conv, bn) in enumerate(zip(model["convs"], model["bns"])):
        x_new = F.relu(bn(conv(x, edge_index)))
        x_new = F.dropout(x_new, p=DROPOUT, training=model.training)
        if i > 0:
            x = x + x_new  # residual after first layer
        else:
            x = x_new

    pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

    # Fuse graph-level features
    if "graph_fc" in model and hasattr(data, "graph_features"):
        gf = F.relu(model["graph_fc"](data.graph_features))
        pooled = torch.cat([pooled, gf], dim=1)

    return model["head"](pooled)


# Build metadata label map and walk DATA_DIR
print("Scanning files ...")
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

    # Build graph datasets (augmentation for train only)
    print("  Building graphs ...")
    train_graphs = [load_graph(file_paths[i], labels[i], augment=False) for i in train_idx]
    val_graphs = [load_graph(file_paths[i], labels[i], augment=False) for i in val_idx]

    # Balanced sampling: oversample minority class
    train_labels_arr = labels[train_idx]
    n_pos = int(train_labels_arr.sum())
    n_neg = len(train_labels_arr) - n_pos
    sample_weights = np.where(train_labels_arr == 1, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
    sample_weights = sample_weights / sample_weights.sum()
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_labels_arr),
        replacement=True,
    )

    train_loader = PyGDataLoader(
        train_graphs, batch_size=BATCH_SIZE, sampler=train_sampler, drop_last=True
    )
    val_loader = PyGDataLoader(val_graphs, batch_size=BATCH_SIZE)

    # Class weights:  total / (2 * n_class)  — matches v3
    total = n_pos + n_neg
    w = torch.tensor([total / (2.0 * max(n_neg, 1)), total / (2.0 * max(n_pos, 1))], device=DEVICE)

    model = build_gnn().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = FocalLoss(alpha=w, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)

    # Best model tracking
    best_val_auc = 0.0
    patience_counter = 0
    best_v_probs = None
    best_v_labels = None
    best_metrics = None

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

        early_stop_triggered = False
        final_v_probs = []
        final_v_labels = []
        final_metrics = {}
        for epoch in range(1, EPOCHS + 1):
            # Train
            model.train()
            t_loss, t_probs, t_labels = 0.0, [], []
            for batch in train_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                logits = forward_gnn(model, batch)
                loss = criterion(logits, batch.y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                t_loss += loss.item() * batch.num_graphs
                t_probs.extend(F.softmax(logits, 1)[:, 1].detach().cpu().numpy())
                t_labels.extend(batch.y.cpu().numpy())
            scheduler.step()

            t_loss /= len(train_graphs)
            t_acc = accuracy_score(t_labels, np.array(t_probs) > 0.5)
            t_auc = safe_auc(t_labels, t_probs)
            t_pr_auc = average_precision_score(t_labels, t_probs) if len(set(t_labels)) > 1 else 0.0

            # Validate
            model.eval()
            v_loss, v_probs, v_labels = 0.0, [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(DEVICE)
                    logits = forward_gnn(model, batch)
                    v_loss += criterion(logits, batch.y).item() * batch.num_graphs
                    v_probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
                    v_labels.extend(batch.y.cpu().numpy())

            v_loss /= len(val_graphs)
            v_acc = accuracy_score(v_labels, np.array(v_probs) > 0.5)
            v_auc = safe_auc(v_labels, v_probs)
            v_pr_auc = average_precision_score(v_labels, v_probs) if len(set(v_labels)) > 1 else 0.0
            tn, fp, fn, tp = confusion_matrix(v_labels, np.array(v_probs) > 0.5).ravel()

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

            fpr, tpr, thr = roc_curve(v_labels, v_probs)
            pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
                os.path.join(fold_roc_dir, f"epoch_{epoch}.csv"), index=False
            )

            if epoch % 20 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}  train_auc={t_auc:.4f}  "
                    f"val_auc={v_auc:.4f}  val_acc={v_acc:.4f}"
                )

            # Best model tracking + early stopping condition (without breaking)
            if not early_stop_triggered:
                if v_auc > best_val_auc:
                    best_val_auc = v_auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOP_PATIENCE:
                        print(
                            f"    Early stopping condition met at epoch {epoch}. "
                            "Continuing training but locking these metrics."
                        )
                        early_stop_triggered = True
                        final_v_probs = list(v_probs)
                        final_v_labels = list(v_labels)
                        final_metrics = {
                            "val_loss": v_loss,
                            "val_acc": v_acc,
                            "val_auc": v_auc,
                            "tn": tn,
                            "fp": fp,
                            "fn": fn,
                            "tp": tp,
                        }

    # Use metrics from when early stopping triggered (or final epoch if never triggered)
    if not early_stop_triggered:
        final_v_probs = list(v_probs)
        final_v_labels = list(v_labels)
        final_metrics = {
            "val_loss": v_loss,
            "val_acc": v_acc,
            "val_auc": v_auc,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

    pooled_probs.extend(final_v_probs)
    pooled_targets.extend(final_v_labels)
    fold_summaries.append({"fold": fold + 1, **final_metrics})

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
