# Version 7 source snapshot
"""
Graph Neural Network classifier over the wall point graph.

Improvements vs v6:
- Adds edge distance as an edge feature and uses GATv2Conv, so longer
  hemodynamic interactions are weighted more accurately than pure mean-pool
  SAGEConv.
- Three-way pooling (mean + max + add) instead of just mean+max; the add
  term captures total integrated risk across the aneurysm surface.
- Graph-level features are computed on the full mesh (include_rrt flag) and
  fused after the GNN encoder with a gated projection.
- Edge-dropout augmentation plus coordinate jitter/rotation during training.
- Reuses shared common helpers for metrics, training loop, EMA.
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

try:
    import torch_cluster

    HAS_CLUSTER = True
except ImportError:
    HAS_CLUSTER = False

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
    import pointnet_pytorch.common as vc
except Exception:
    import common as vc

METADATA_PATH = os.environ.get("V7_METADATA", "metadata.csv")
DATA_DIR = os.environ.get("V7_DATA_DIR", "predictions/pinn_corrected")
OUTPUT_DIR = Path(os.environ.get("V7_OUTPUT_DIR", "results_gnn_v7"))
N_FOLDS = int(os.environ.get("V7_FOLDS", 5))
BATCH_SIZE = int(os.environ.get("V7_BATCH", 16))
EPOCHS = int(os.environ.get("V7_EPOCHS", 150))
LR = float(os.environ.get("V7_LR", 1e-3))
WEIGHT_DECAY = float(os.environ.get("V7_WD", 1e-3))
K_NEIGHBORS = int(os.environ.get("V7_KNN", 16))
TARGET_N = int(os.environ.get("V7_TARGET_N", 2048))
HIDDEN_DIM = int(os.environ.get("V7_HIDDEN", 96))
NUM_GNN_LAYERS = int(os.environ.get("V7_GNN_LAYERS", 3))
DROPOUT = float(os.environ.get("V7_DROPOUT", 0.5))
EDGE_DROPOUT = float(os.environ.get("V7_EDGE_DROPOUT", 0.1))
EARLY_STOP_PATIENCE = int(os.environ.get("V7_PATIENCE", 25))
HEADS = int(os.environ.get("V7_HEADS", 4))
GRAPH_FEATURE_DIM = 23
NODE_FEATURE_DIM = 3 + 3  # xyz + (tawss, osi, von_mises)

vc.set_seed(vc.SEED)
DEVICE = vc.get_device()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# knn_graph fallback (pure-torch) when torch_cluster is unavailable
def _pure_torch_knn(pos: torch.Tensor, k: int) -> torch.Tensor:
    dists = torch.cdist(pos, pos)
    dists.fill_diagonal_(float("inf"))
    _, idx = torch.topk(dists, k=k, dim=1, largest=False)
    src = torch.arange(pos.size(0), device=pos.device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = idx.reshape(-1)
    return torch.stack([src, dst], dim=0)


def knn_graph(pos: torch.Tensor, k: int) -> torch.Tensor:
    if HAS_CLUSTER:
        return torch_cluster.knn_graph(pos, k=k)
    return _pure_torch_knn(pos, k)


# Sample loading
def load_graph(path: str, label: int, augment: bool) -> Data:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    coords = df[["x", "y", "z"]].values.astype(np.float32)

    feat_cols = []
    for key in ("tawss", "osi", "von"):
        match = next((c for c in df.columns if key in c), None)
        feat_cols.append(df[match].values if match else np.zeros(len(df)))
    raw = np.stack(feat_cols, axis=1).astype(np.float32)

    # Graph-level features computed on the full mesh, then resample
    graph_feats = vc.summarize_global_features(raw, coords, include_rrt=False)
    idx = vc.resample_cloud(coords, TARGET_N)
    coords = coords[idx]
    feats = raw[idx]

    coords = vc.normalize_points(coords)
    feats = np.clip(np.log1p(np.clip(feats, 1e-6, None)), -3.0, 3.0).astype(np.float32)
    node_feats = np.concatenate([coords, feats], axis=1)

    pos = torch.tensor(coords, dtype=torch.float)
    x = torch.tensor(node_feats, dtype=torch.float)

    if augment:
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float)
        pos = pos @ R.T
        pos = pos + torch.clamp(0.01 * torch.randn_like(pos), -0.03, 0.03)
        pos = pos * float(np.random.uniform(0.9, 1.1))
        x = x.clone()
        x[:, :3] = pos
        x[:, 3:] = x[:, 3:] + torch.clamp(0.02 * torch.randn_like(x[:, 3:]), -0.05, 0.05)

    edge_index = knn_graph(pos, k=K_NEIGHBORS)
    edge_index = to_undirected(edge_index)
    if augment and EDGE_DROPOUT > 0 and edge_index.size(1) > 0:
        mask = torch.rand(edge_index.size(1)) > EDGE_DROPOUT
        edge_index = edge_index[:, mask]
        if edge_index.size(1) < 10:
            edge_index = to_undirected(knn_graph(pos, k=K_NEIGHBORS))

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


# Model
def build_model() -> nn.ModuleDict:
    convs = nn.ModuleList()
    bns = nn.ModuleList()
    in_ch = NODE_FEATURE_DIM
    for _ in range(NUM_GNN_LAYERS):
        convs.append(
            GATv2Conv(in_ch, HIDDEN_DIM // HEADS, heads=HEADS, edge_dim=1, dropout=DROPOUT * 0.5)
        )
        bns.append(BatchNorm(HIDDEN_DIM))
        in_ch = HIDDEN_DIM
    graph_fc = nn.Sequential(
        nn.Linear(GRAPH_FEATURE_DIM, HIDDEN_DIM),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
        nn.GELU(),
    )
    head = nn.Sequential(
        nn.Linear(HIDDEN_DIM * 3 + HIDDEN_DIM // 2, HIDDEN_DIM),
        nn.BatchNorm1d(HIDDEN_DIM),
        nn.GELU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_DIM, 2),
    )
    return nn.ModuleDict({"convs": convs, "bns": bns, "graph_fc": graph_fc, "head": head})


def forward_model(model, batch):
    x, edge_index, edge_attr, b = batch.x, batch.edge_index, batch.edge_attr, batch.batch
    for i, (conv, bn) in enumerate(zip(model["convs"], model["bns"])):
        x_new = conv(x, edge_index, edge_attr=edge_attr)
        x_new = F.gelu(bn(x_new))
        x_new = F.dropout(x_new, p=DROPOUT, training=model.training)
        x = x_new if i == 0 else x + x_new
    pooled = torch.cat(
        [global_mean_pool(x, b), global_max_pool(x, b), global_add_pool(x, b)],
        dim=1,
    )
    gf = model["graph_fc"](batch.graph_features)
    return model["head"](torch.cat([pooled, gf], dim=1))


# Label discovery
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


# Training glue
def forward_fn(model, batch, device, train: bool):
    batch = batch.to(device, non_blocking=True)
    logits = forward_model(model, batch)
    return logits, batch.y, int(batch.num_graphs)


def main():
    paths, labels = discover_labeled_cases()
    print(f"Total samples: {len(paths)}   Device: {DEVICE}  torch_cluster={HAS_CLUSTER}")

    fold_indices = vc.stratified_kfold_indices(labels, N_FOLDS)
    pooled_probs, pooled_labels, fold_summaries = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(fold_indices):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
        print("  Building graphs...")
        train_graphs = [load_graph(paths[i], int(labels[i]), augment=True) for i in tr_idx]
        val_graphs = [load_graph(paths[i], int(labels[i]), augment=False) for i in va_idx]

        y_tr = torch.tensor(labels[tr_idx], dtype=torch.long)
        sampler = vc.balanced_sampler(y_tr)
        train_loader = PyGDataLoader(
            train_graphs, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True
        )
        val_loader = PyGDataLoader(val_graphs, batch_size=BATCH_SIZE)

        model = build_model().to(DEVICE)
        weights = vc.class_weights(y_tr, DEVICE)
        criterion = vc.FocalLoss(alpha=weights, gamma=2.0, label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

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
            amp=False,  # PyG + AMP can be flaky on older GPUs
        )
        fold_summaries.append({"fold": fold + 1, **best_metrics})
        pooled_probs.extend(best_probs)
        pooled_labels.extend(best_labels)
        print(f"  Best val AUC: {best_metrics.get('val_auc', 0.0):.4f}")

    vc.write_fold_summary(OUTPUT_DIR, fold_summaries, pooled_probs, pooled_labels)


if __name__ == "__main__":
    main()
