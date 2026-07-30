#!/usr/bin/env python3
# Version 1 source snapshot
"""
gnn_rupture_classification.py

Graph Neural Network for Aneurysm Rupture Prediction
Uses hemodynamics data (TAWSS, OSI, Von Mises stress) from PINN-corrected predictions.

Constructs graphs from point clouds using k-NN connectivity and applies
GNN layers (GCN, GraphSAGE, or GAT) for classification.

Data structure expected:
- hemodynamics_aggregate.csv files with columns: x, y, z, tawss, osi, von_mises
- metadata.csv with rupture status (ruptured/unruptured)

Requirements:
- torch
- torch_geometric
- torch_scatter
- torch_sparse

Install with: pip install torch-geometric torch-scatter torch-sparse

"""

import argparse
import csv
import os
import random
import re
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=UserWarning)

# PyTorch Geometric imports
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.nn import (
        BatchNorm,
        GATConv,
        GCNConv,
        GINConv,
        SAGEConv,
        global_add_pool,
        global_max_pool,
        global_mean_pool,
        knn_graph,
        radius_graph,
    )
    from torch_geometric.utils import to_undirected

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("[WARNING] torch_geometric not installed. Install with:")
    print("  pip install torch-geometric torch-scatter torch-sparse")


# Loss Functions
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reduces loss for well-classified examples, focusing on hard cases.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter (higher = more focus on hard examples)
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        num_classes = inputs.size(1)
        if self.label_smoothing > 0:
            smooth_targets = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1.0)
            smooth_targets = (
                smooth_targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            )

        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)

        # Get probability for true class
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Apply focal weighting
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# Graph Construction Utilities
def build_knn_graph(pos: torch.Tensor, k: int = 16) -> torch.Tensor:
    """
    Build k-nearest neighbor graph from point positions.

    Args:
        pos: (N, 3) tensor of point positions
        k: number of nearest neighbors

    Returns:
        edge_index: (2, E) tensor of edges
    """
    edge_index = knn_graph(pos, k=k, loop=False)
    edge_index = to_undirected(edge_index)
    return edge_index


def build_radius_graph(pos: torch.Tensor, r: float = 0.1, max_neighbors: int = 32) -> torch.Tensor:
    """
    Build radius-based graph from point positions.

    Args:
        pos: (N, 3) tensor of point positions
        r: radius for connectivity
        max_neighbors: maximum neighbors per node

    Returns:
        edge_index: (2, E) tensor of edges
    """
    edge_index = radius_graph(pos, r=r, max_num_neighbors=max_neighbors, loop=False)
    edge_index = to_undirected(edge_index)
    return edge_index


# Dataset
class AneurysmGraphDataset(Dataset):
    """
    Dataset for aneurysm rupture prediction using graph representation.

    Each sample is converted to a PyG Data object with:
    - x: node features (tawss, osi, von_mises + optional xyz)
    - pos: node positions (x, y, z)
    - edge_index: graph connectivity
    - graph_features: global summary statistics
    - y: label
    """

    def __init__(
        self,
        file_label_pairs: List[Tuple[str, int]],
        target_n: int = 2048,
        k_neighbors: int = 16,
        use_radius: bool = False,
        radius: float = 0.1,
        augment: bool = False,
        normalize_xyz: bool = True,
        normalize_features: bool = True,
        include_xyz_as_features: bool = True,
        node_dropout: float = 0.0,
        edge_dropout: float = 0.0,
        add_graph_features: bool = True,
    ):
        """
        Args:
            file_label_pairs: list of (csv_path, label) tuples
            target_n: target number of points (sample/pad to this)
            k_neighbors: number of k-nearest neighbors for graph
            use_radius: whether to use radius-based graph instead of k-NN
            radius: radius for connectivity (if use_radius=True)
            augment: whether to apply data augmentation
            normalize_xyz: whether to normalize xyz coordinates
            normalize_features: whether to normalize hemodynamic features
            include_xyz_as_features: whether to include xyz in node features
            node_dropout: probability of dropping nodes during augmentation
            edge_dropout: probability of dropping edges during augmentation
            add_graph_features: whether to compute graph-level summary features
        """
        self.file_label_pairs = file_label_pairs
        self.target_n = target_n
        self.k_neighbors = k_neighbors
        self.use_radius = use_radius
        self.radius = radius
        self.augment = augment
        self.normalize_xyz = normalize_xyz
        self.normalize_features = normalize_features
        self.include_xyz_as_features = include_xyz_as_features
        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout
        self.add_graph_features = add_graph_features

        # Pre-load and cache data for faster training
        self.cached_data = []
        print(f"[INFO] Loading and caching {len(file_label_pairs)} graphs...")
        for idx, (path, label) in enumerate(file_label_pairs):
            data = self._load_single(path, label)
            self.cached_data.append(data)
            if (idx + 1) % 100 == 0:
                print(f"  Loaded {idx + 1}/{len(file_label_pairs)}")
        print("[INFO] Caching complete.")

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, idx: int) -> Data:
        data = self.cached_data[idx]

        if self.augment:
            data = self._augment_graph(data)

        return data

    def _load_single(self, path: str, label: int) -> Data:
        """Load a single sample and convert to PyG Data object."""
        try:
            raw_data = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return dummy data
            pos = torch.zeros((self.target_n, 3), dtype=torch.float32)
            x = torch.zeros(
                (self.target_n, 6 if self.include_xyz_as_features else 3), dtype=torch.float32
            )
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            return Data(
                x=x, pos=pos, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long)
            )

        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)

        if raw_data.shape[1] < 6:
            print(f"Warning: {path} has only {raw_data.shape[1]} columns")
            pos = torch.zeros((self.target_n, 3), dtype=torch.float32)
            x = torch.zeros(
                (self.target_n, 6 if self.include_xyz_as_features else 3), dtype=torch.float32
            )
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            return Data(
                x=x, pos=pos, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long)
            )

        # Extract coordinates and features
        pts = raw_data[:, :3].astype(np.float32)  # x, y, z
        feats = raw_data[:, 3:6].astype(np.float32)  # tawss, osi, von_mises

        # Sample or pad to target number of points
        n_points = pts.shape[0]
        if n_points < self.target_n:
            # Pad with duplicates of existing points (better than zeros for graphs)
            pad_indices = np.random.choice(n_points, self.target_n - n_points, replace=True)
            pts = np.vstack([pts, pts[pad_indices]])
            feats = np.vstack([feats, feats[pad_indices]])
        elif n_points > self.target_n:
            # Random sampling
            idx_sample = np.random.choice(n_points, self.target_n, replace=False)
            pts = pts[idx_sample]
            feats = feats[idx_sample]

        # Normalize coordinates
        if self.normalize_xyz:
            pts = self._normalize_points(pts)

        # Normalize features
        if self.normalize_features:
            feats = self._normalize_features(feats)

        # Convert to tensors
        pos = torch.from_numpy(pts).float()

        # Build node features
        if self.include_xyz_as_features:
            x = torch.cat([pos, torch.from_numpy(feats).float()], dim=1)
        else:
            x = torch.from_numpy(feats).float()

        # Build graph connectivity
        if self.use_radius:
            edge_index = build_radius_graph(pos, r=self.radius)
        else:
            edge_index = build_knn_graph(pos, k=self.k_neighbors)

        # Compute graph-level summary features (feature enrichment)
        graph_features = None
        if self.add_graph_features:
            graph_features = self._compute_graph_features(raw_data[:, 3:6], pts)

        # Create PyG Data object
        data = Data(
            x=x, pos=pos, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long), path=path
        )
        if graph_features is not None:
            data.graph_features = graph_features

        return data

    def _compute_graph_features(self, raw_feats: np.ndarray, pts: np.ndarray) -> torch.Tensor:
        """
        Compute graph-level summary statistics for feature enrichment.
        These capture global hemodynamic patterns that node-level features miss.
        """
        # Raw features: tawss, osi, von_mises (before normalization)
        tawss = raw_feats[:, 0]
        osi = raw_feats[:, 1]
        von_mises = raw_feats[:, 2]

        # Compute summary statistics
        features = []

        # TAWSS statistics (critical for rupture)
        features.extend(
            [
                np.mean(tawss),
                np.std(tawss),
                np.max(tawss),
                np.min(tawss),
                np.percentile(tawss, 95),
                np.percentile(tawss, 5),
                np.sum(tawss > np.percentile(tawss, 90)) / len(tawss),  # High TAWSS ratio
            ]
        )

        # OSI statistics (oscillatory flow indicator)
        features.extend(
            [
                np.mean(osi),
                np.std(osi),
                np.max(osi),
                np.percentile(osi, 95),
                np.sum(osi > 0.2) / len(osi),  # High OSI ratio (threshold from literature)
            ]
        )

        # Von Mises statistics (stress concentration)
        features.extend(
            [
                np.mean(von_mises),
                np.std(von_mises),
                np.max(von_mises),
                np.percentile(von_mises, 95),
                np.percentile(von_mises, 99),
            ]
        )

        # Geometric features
        centroid = np.mean(pts, axis=0)
        centered = pts - centroid
        distances = np.linalg.norm(centered, axis=1)
        features.extend(
            [
                np.max(distances),  # Max radius (size proxy)
                np.std(distances),  # Shape irregularity
                np.max(distances) / (np.mean(distances) + 1e-6),  # Aspect ratio proxy
            ]
        )

        # Log-transform and clip for stability
        features = np.array(features, dtype=np.float32)
        features = np.clip(features, -10, 10)

        # Return as 2D tensor (1, num_features) so PyG batching concatenates correctly
        return torch.from_numpy(features).float().unsqueeze(0)

    def _normalize_points(self, pts: np.ndarray) -> np.ndarray:
        """Center and scale point cloud to unit sphere."""
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        max_dist = np.max(np.linalg.norm(pts, axis=1))
        if max_dist > 0:
            pts = pts / max_dist
        return pts

    def _normalize_features(self, feats: np.ndarray) -> np.ndarray:
        """Clip-based normalization to preserve absolute feature magnitude.

        Z-score normalization removes absolute WSS magnitude differences
        between aneurysms, which is important for rupture prediction.
        Clipping stabilizes training while keeping magnitude info.
        """
        # Log-transform for heavy-tailed distributions (common in CFD)
        # Then clip to stabilize without losing magnitude differences
        feats = np.clip(feats, 1e-6, None)  # Ensure positive for log
        feats = np.log1p(feats)  # log(1+x) for stability
        feats = np.clip(feats, -3.0, 3.0)  # Clip extremes
        return feats.astype(np.float32)

    def _augment_graph(self, data: Data) -> Data:
        """Apply data augmentation to graph including node/edge dropout."""
        pos = data.pos.clone()
        x = data.x.clone()
        edge_index = data.edge_index.clone()

        # Random rotation around z-axis
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = torch.tensor([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=torch.float32)
        pos = pos @ R.T

        # Update xyz in features if included
        if self.include_xyz_as_features:
            x[:, :3] = pos

        # Random jitter on positions
        pos = pos + torch.clamp(0.01 * torch.randn_like(pos), -0.03, 0.03)

        # Random scaling
        scale = np.random.uniform(0.9, 1.1)
        pos = pos * scale

        # Feature noise
        feat_start = 3 if self.include_xyz_as_features else 0
        x[:, feat_start:] = x[:, feat_start:] + torch.clamp(
            0.02 * torch.randn_like(x[:, feat_start:]), -0.05, 0.05
        )

        # Edge dropout - randomly remove edges
        if self.edge_dropout > 0 and edge_index.size(1) > 0:
            edge_mask = torch.rand(edge_index.size(1)) > self.edge_dropout
            edge_index = edge_index[:, edge_mask]

        # If edge dropout removed too many edges, rebuild graph
        if edge_index.size(1) < 10:
            if self.use_radius:
                edge_index = build_radius_graph(pos, r=self.radius)
            else:
                edge_index = build_knn_graph(pos, k=self.k_neighbors)

        result = Data(x=x, pos=pos, edge_index=edge_index, y=data.y, path=data.path)

        # Copy graph features if present
        if hasattr(data, "graph_features"):
            result.graph_features = data.graph_features

        return result


# GNN Models
class GCNClassifier(nn.Module):
    """
    Graph Convolutional Network for graph classification.
    Uses mean+max pooling to capture both average and extreme (hotspot) values.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 96,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
        pool_type: str = "mean_max",
    ):
        super().__init__()

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        # Pooling
        self.pool_type = pool_type

        # Classifier input size depends on pooling
        if pool_type == "mean_max":
            classifier_in = hidden_channels * 2
        else:
            classifier_in = hidden_channels

        # Classifier
        self.fc1 = nn.Linear(classifier_in, hidden_channels // 2)
        self.fc_bn = nn.BatchNorm1d(hidden_channels // 2)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_channels // 2, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GCN layers with dropout
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling - mean + max for hotspot detection
        if self.pool_type == "mean_max":
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        elif self.pool_type == "max":
            x = global_max_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)

        # Classifier (skip BatchNorm if batch size is 1 to avoid error)
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.fc_bn(x)
        x = F.relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)

        return x


class GraphSAGEClassifier(nn.Module):
    """
    GraphSAGE for graph classification with improvements:
    - Mean + Max pooling to capture both average and extreme values
    - Dropout after each conv layer to prevent oversmoothing
    - Residual connections for deeper models
    - Optional graph-level feature fusion
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 96,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
        pool_type: str = "mean_max",
        aggr: str = "mean",
        use_residual: bool = True,
        graph_feature_dim: int = 0,
    ):
        super().__init__()

        self.dropout = dropout
        self.use_residual = use_residual
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers with residual connections
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(BatchNorm(hidden_channels))

        # Pooling type - mean_max concatenates both for hotspot detection
        self.pool_type = pool_type

        # Classifier input size depends on pooling and graph features
        if pool_type == "mean_max":
            classifier_in = hidden_channels * 2  # concat mean + max
        else:
            classifier_in = hidden_channels

        # Add graph-level features if provided
        self.graph_feature_dim = graph_feature_dim
        if graph_feature_dim > 0:
            self.graph_fc = nn.Linear(graph_feature_dim, hidden_channels // 2)
            classifier_in += hidden_channels // 2

        # Classifier
        self.fc1 = nn.Linear(classifier_in, hidden_channels)
        self.fc_bn = nn.BatchNorm1d(hidden_channels)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_channels, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GraphSAGE layers with dropout and residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            # Residual connection (skip first layer due to dimension mismatch)
            if self.use_residual and i > 0:
                x = x + x_new
            else:
                x = x_new

        # Global pooling - mean + max to capture hotspots
        if self.pool_type == "mean_max":
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        elif self.pool_type == "max":
            x = global_max_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)

        # Fuse graph-level features if available
        if self.graph_feature_dim > 0 and hasattr(data, "graph_features"):
            gf = self.graph_fc(data.graph_features)
            gf = F.relu(gf)
            x = torch.cat([x, gf], dim=1)

        # Classifier (skip BatchNorm if batch size is 1 to avoid error)
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.fc_bn(x)
        x = F.relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)

        return x


class GATClassifier(nn.Module):
    """
    Graph Attention Network for graph classification.
    Uses attention mechanism to weight neighbor contributions.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 48,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
        heads: int = 4,
        pool_type: str = "mean_max",
    ):
        super().__init__()

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.bns.append(BatchNorm(hidden_channels * heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
            self.bns.append(BatchNorm(hidden_channels * heads))

        # Last conv layer (single head for output)
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout
                )
            )
            self.bns.append(BatchNorm(hidden_channels))
            final_channels = hidden_channels
        else:
            final_channels = hidden_channels * heads

        # Pooling
        self.pool_type = pool_type
        if pool_type == "mean_max":
            classifier_in = final_channels * 2
        else:
            classifier_in = final_channels

        # Classifier
        self.fc1 = nn.Linear(classifier_in, hidden_channels)
        self.fc_bn = nn.BatchNorm1d(hidden_channels)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_channels, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GAT layers with dropout
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling - mean + max
        if self.pool_type == "mean_max":
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        elif self.pool_type == "max":
            x = global_max_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)

        # Classifier (skip BatchNorm if batch size is 1 to avoid error)
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.fc_bn(x)
        x = F.relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)

        return x


class GINClassifier(nn.Module):
    """
    Graph Isomorphism Network for graph classification.
    Theoretically most expressive GNN architecture.
    Uses Jumping Knowledge to aggregate multi-scale features.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 96,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
        pool_type: str = "add",
    ):
        super().__init__()

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.convs.append(GINConv(mlp1))
        self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(BatchNorm(hidden_channels))

        # Pooling - GIN typically uses sum pooling
        if pool_type == "add":
            self.pool = global_add_pool
        elif pool_type == "mean":
            self.pool = global_mean_pool
        else:
            self.pool = global_add_pool

        # Classifier with JK (Jumping Knowledge) style - concat all layer outputs
        self.fc1 = nn.Linear(hidden_channels * num_layers, hidden_channels)
        self.fc_bn = nn.BatchNorm1d(hidden_channels)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_channels, num_classes)

        self.num_layers = num_layers

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GIN layers with JK aggregation and dropout
        layer_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(self.pool(x, batch))

        # Concatenate all layer outputs (Jumping Knowledge)
        x = torch.cat(layer_outputs, dim=1)

        # Classifier (skip BatchNorm if batch size is 1 to avoid error)
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.fc_bn(x)
        x = F.relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)

        return x


class HierarchicalGNN(nn.Module):
    """
    Hierarchical GNN with graph coarsening for multi-scale learning.
    Similar concept to PointNet++ but for graphs.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Level 1: Full resolution
        self.conv1_1 = SAGEConv(in_channels, 64)
        self.conv1_2 = SAGEConv(64, 128)
        self.bn1_1 = BatchNorm(64)
        self.bn1_2 = BatchNorm(128)

        # Level 2: Coarsened (we'll use pooling + new graph)
        self.conv2_1 = SAGEConv(128, 128)
        self.conv2_2 = SAGEConv(128, 256)
        self.bn2_1 = BatchNorm(128)
        self.bn2_2 = BatchNorm(256)

        # Level 3: Global
        self.conv3 = SAGEConv(256, 512)
        self.bn3 = BatchNorm(512)

        # Classifier
        self.fc1 = nn.Linear(512 + 256 + 128, hidden_channels)
        self.fc_bn = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_channels, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch, _pos = data.x, data.edge_index, data.batch, data.pos

        # Level 1
        x = F.relu(self.bn1_1(self.conv1_1(x, edge_index)))
        x = F.relu(self.bn1_2(self.conv1_2(x, edge_index)))
        global_1 = global_mean_pool(x, batch)

        # Coarsen by random sampling (simplified version)
        # In practice, you'd use proper graph coarsening
        x = F.relu(self.bn2_1(self.conv2_1(x, edge_index)))
        x = F.relu(self.bn2_2(self.conv2_2(x, edge_index)))
        global_2 = global_mean_pool(x, batch)

        # Level 3
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        global_3 = global_mean_pool(x, batch)

        # Combine multi-scale features
        x = torch.cat([global_1, global_2, global_3], dim=1)

        # Classifier (skip BatchNorm if batch size is 1 to avoid error)
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.fc_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Data Loading & Metadata Processing
def load_metadata(metadata_csv: str) -> Dict[str, int]:
    """Load metadata and create mapping from case name to rupture label."""
    mapping = {}

    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row.get("dataset", "").strip()
            status = row.get("status", "").strip().lower()
            cut = row.get("cutToShow", "").strip()

            if not dataset or status not in ["ruptured", "unruptured"]:
                continue

            label = 1 if status == "ruptured" else 0

            key1 = f"{dataset}_{cut}"
            mapping[key1] = label
            mapping[dataset] = label

    print(f"[INFO] Loaded {len(mapping)} entries from metadata")
    return mapping


def find_hemodynamics_files(data_dir: str, metadata_csv: str) -> List[Tuple[str, int]]:
    """Find all hemodynamics_aggregate.csv files and match with labels."""
    mapping = load_metadata(metadata_csv)
    file_label_pairs = []
    unmatched = []

    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)

        if not os.path.isdir(item_path):
            continue

        csv_path = os.path.join(item_path, "hemodynamics_aggregate.csv")
        if not os.path.exists(csv_path):
            continue

        case_name = item
        label = None

        if case_name in mapping:
            label = mapping[case_name]
        else:
            base_name = re.sub(r"_cut\d+$", "", case_name)
            if base_name in mapping:
                label = mapping[base_name]

        if label is not None:
            file_label_pairs.append((csv_path, label))
        else:
            unmatched.append(case_name)

    n_ruptured = sum(1 for _, label in file_label_pairs if label == 1)
    n_unruptured = sum(1 for _, label in file_label_pairs if label == 0)

    print(f"[INFO] Found {len(file_label_pairs)} matched cases:")
    print(f"       - Ruptured: {n_ruptured}")
    print(f"       - Unruptured: {n_unruptured}")
    print(f"       - Unmatched: {len(unmatched)}")

    return file_label_pairs


def prepare_train_val_split(
    file_label_pairs: List[Tuple[str, int]],
    val_fraction: float = 0.2,
    balance_classes: bool = True,
    seed: int = 42,
) -> Tuple[List, List]:
    """Split data into training and validation sets."""
    random.seed(seed)

    ruptured = [(path, label) for path, label in file_label_pairs if label == 1]
    unruptured = [(path, label) for path, label in file_label_pairs if label == 0]

    random.shuffle(ruptured)
    random.shuffle(unruptured)

    if balance_classes:
        min_count = min(len(ruptured), len(unruptured))
        ruptured = ruptured[:min_count]
        unruptured = unruptured[:min_count]
        print(f"[INFO] Balanced classes to {min_count} samples each")

    n_val_rupt = max(1, int(len(ruptured) * val_fraction))
    n_val_unrupt = max(1, int(len(unruptured) * val_fraction))

    val_ruptured = ruptured[:n_val_rupt]
    train_ruptured = ruptured[n_val_rupt:]

    val_unruptured = unruptured[:n_val_unrupt]
    train_unruptured = unruptured[n_val_unrupt:]

    train_files = train_ruptured + train_unruptured
    val_files = val_ruptured + val_unruptured

    random.shuffle(train_files)
    random.shuffle(val_files)

    print(f"[INFO] Train: {len(train_files)} (R:{len(train_ruptured)}, U:{len(train_unruptured)})")
    print(f"[INFO] Val: {len(val_files)} (R:{len(val_ruptured)}, U:{len(val_unruptured)})")

    return train_files, val_files


# Training & Evaluation
def train_one_epoch(
    model: nn.Module,
    dataloader: PyGDataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data in dataloader:
        data = data.to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.num_graphs
        preds = logits.argmax(dim=1)
        correct += (preds == data.y).sum().item()
        total += data.num_graphs

    return running_loss / total, correct / total


def evaluate(
    model: nn.Module, dataloader: PyGDataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on validation/test set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)

            logits = model(data)
            loss = criterion(logits, data.y)

            running_loss += loss.item() * data.num_graphs
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    total = len(all_labels)
    loss = running_loss / total
    acc = (all_preds == all_labels).sum() / total

    return loss, acc, all_preds, all_labels, all_probs


def print_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray, phase: str = ""):
    """Print detailed classification metrics."""
    print(f"\n{phase} Classification Report:")
    print("-" * 50)
    print(
        classification_report(
            labels, preds, target_names=["Unruptured", "Ruptured"], zero_division=0
        )
    )

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print("  Predicted:  Unrupt  Rupt")
    print(f"  Unruptured:  {cm[0, 0]:5d}  {cm[0, 1]:5d}")
    print(f"  Ruptured:    {cm[1, 0]:5d}  {cm[1, 1]:5d}")

    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, probs)
        ap = average_precision_score(labels, probs)
        print(f"\nAUC-ROC: {auc:.4f}")
        print(f"Average Precision: {ap:.4f}")

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")


def train_kfold(
    data_dir: str,
    metadata_csv: str,
    n_folds: int = 5,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    target_n: int = 2048,
    k_neighbors: int = 16,
    model_type: str = "sage",
    hidden_channels: int = 96,
    num_layers: int = 2,
    dropout: float = 0.5,
    save_dir: str = "kfold_models",
    seed: int = 42,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    early_stopping_patience: int = 15,
    label_smoothing: float = 0.1,
    edge_dropout: float = 0.1,
):
    """
    K-Fold Cross-Validation training for more robust AUC estimates.
    Returns ensemble of models and aggregated metrics.
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("torch_geometric is required.")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Find all data files
    file_label_pairs = find_hemodynamics_files(data_dir, metadata_csv)
    if len(file_label_pairs) == 0:
        raise RuntimeError("No data files found!")

    # Balance classes
    ruptured = [(path, label) for path, label in file_label_pairs if label == 1]
    unruptured = [(path, label) for path, label in file_label_pairs if label == 0]
    min_count = min(len(ruptured), len(unruptured))
    random.shuffle(ruptured)
    random.shuffle(unruptured)
    file_label_pairs = ruptured[:min_count] + unruptured[:min_count]
    random.shuffle(file_label_pairs)

    print(f"[INFO] K-Fold CV with {n_folds} folds on {len(file_label_pairs)} samples")

    # Prepare for stratified k-fold
    paths = [p for p, _ in file_label_pairs]
    labels = np.array([label for _, label in file_label_pairs])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Store results
    fold_aucs = []
    fold_accs = []
    all_val_probs = []
    all_val_labels = []
    models = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Determine feature dimensions
    include_xyz = True
    in_channels = 6 if include_xyz else 3
    graph_feature_dim = 20  # Number of graph-level features

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'=' * 60}")

        # Create train/val splits
        train_pairs = [(paths[i], labels[i]) for i in train_idx]
        val_pairs = [(paths[i], labels[i]) for i in val_idx]

        print(f"[FOLD {fold + 1}] Train: {len(train_pairs)}, Val: {len(val_pairs)}")

        # Create datasets
        train_ds = AneurysmGraphDataset(
            train_pairs,
            target_n=target_n,
            k_neighbors=k_neighbors,
            augment=True,
            normalize_xyz=True,
            normalize_features=True,
            include_xyz_as_features=include_xyz,
            edge_dropout=edge_dropout,
            add_graph_features=True,
        )
        val_ds = AneurysmGraphDataset(
            val_pairs,
            target_n=target_n,
            k_neighbors=k_neighbors,
            augment=False,
            normalize_xyz=True,
            normalize_features=True,
            include_xyz_as_features=include_xyz,
            add_graph_features=True,
        )

        train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Create model with graph feature fusion
        if model_type.lower() in ["sage", "graphsage"]:
            model = GraphSAGEClassifier(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                num_classes=2,
                dropout=dropout,
                use_residual=True,
                graph_feature_dim=graph_feature_dim,
            )
        elif model_type.lower() == "gcn":
            model = GCNClassifier(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                num_classes=2,
                dropout=dropout,
            )
        elif model_type.lower() == "gat":
            model = GATClassifier(
                in_channels=in_channels,
                hidden_channels=hidden_channels // 2,
                num_layers=num_layers,
                num_classes=2,
                dropout=dropout,
                heads=4,
            )
        else:
            model = GraphSAGEClassifier(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                num_classes=2,
                dropout=dropout,
                use_residual=True,
                graph_feature_dim=graph_feature_dim,
            )

        model = model.to(device)

        # Setup loss with class weights
        n_ruptured = sum(1 for _, label in train_pairs if label == 1)
        n_unruptured = sum(1 for _, label in train_pairs if label == 0)
        total = n_ruptured + n_unruptured
        weights = torch.tensor(
            [total / (2 * n_unruptured), total / (2 * n_ruptured)],
            dtype=torch.float32,
            device=device,
        )

        if use_focal_loss:
            criterion = FocalLoss(alpha=weights, gamma=focal_gamma, label_smoothing=label_smoothing)
            print(f"[FOLD {fold + 1}] Using Focal Loss (gamma={focal_gamma})")
        else:
            criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

        # Optimizer with warmup
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        # Training loop
        best_val_auc = 0.0
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc, val_preds, val_labels_ep, val_probs = evaluate(
                model, val_loader, criterion, device
            )
            val_auc = (
                roc_auc_score(val_labels_ep, val_probs)
                if len(np.unique(val_labels_ep)) > 1
                else 0.5
            )
            scheduler.step()

            if epoch % 10 == 0 or val_auc > best_val_auc:
                print(
                    f"  Epoch {epoch:03d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} AUC: {val_auc:.4f}"
                )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(save_dir, f"fold{fold + 1}_best.pth"))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # Load best model and get final predictions
        model.load_state_dict(
            torch.load(os.path.join(save_dir, f"fold{fold + 1}_best.pth"), weights_only=True)
        )
        _, _, val_preds, val_labels_final, val_probs_final = evaluate(
            model, val_loader, criterion, device
        )

        fold_auc = roc_auc_score(val_labels_final, val_probs_final)
        fold_acc = (val_preds == val_labels_final).mean()

        fold_aucs.append(fold_auc)
        fold_accs.append(fold_acc)
        all_val_probs.extend(val_probs_final)
        all_val_labels.extend(val_labels_final)
        models.append(model)

        print(f"\n[FOLD {fold + 1}] Best AUC: {fold_auc:.4f}, Acc: {fold_acc:.4f}")

    # Aggregate results
    print("\n" + "=" * 60)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"AUC per fold: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"Acc per fold: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"Mean AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})")
    print(f"Mean Acc: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")

    # Overall metrics
    all_val_probs = np.array(all_val_probs)
    all_val_labels = np.array(all_val_labels)
    overall_auc = roc_auc_score(all_val_labels, all_val_probs)
    print(f"\nOverall AUC (all folds combined): {overall_auc:.4f}")

    return models, fold_aucs, fold_accs


def train_model(
    data_dir: str,
    metadata_csv: str,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    val_fraction: float = 0.2,
    num_workers: int = 0,
    target_n: int = 2048,
    k_neighbors: int = 16,
    model_type: str = "sage",
    hidden_channels: int = 96,
    num_layers: int = 2,
    dropout: float = 0.5,
    save_path: str = "best_gnn_rupture_model.pth",
    seed: int = 42,
    use_class_weights: bool = True,
    early_stopping_patience: int = 20,
    label_smoothing: float = 0.1,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    edge_dropout: float = 0.1,
):
    """Main training function for GNN models."""

    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("torch_geometric is required. Install with: pip install torch-geometric")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Find data files
    file_label_pairs = find_hemodynamics_files(data_dir, metadata_csv)

    if len(file_label_pairs) == 0:
        raise RuntimeError("No data files found! Check paths.")

    # Split data
    train_files, val_files = prepare_train_val_split(
        file_label_pairs, val_fraction=val_fraction, balance_classes=True, seed=seed
    )

    # Determine input channels (xyz + features or just features)
    include_xyz = True
    in_channels = 6 if include_xyz else 3
    graph_feature_dim = 20  # Number of graph-level summary features

    # Create datasets with edge dropout for augmentation
    print("\n[INFO] Creating training dataset...")
    train_ds = AneurysmGraphDataset(
        train_files,
        target_n=target_n,
        k_neighbors=k_neighbors,
        augment=True,
        normalize_xyz=True,
        normalize_features=True,
        include_xyz_as_features=include_xyz,
        edge_dropout=edge_dropout,
        add_graph_features=True,
    )

    print("\n[INFO] Creating validation dataset...")
    val_ds = AneurysmGraphDataset(
        val_files,
        target_n=target_n,
        k_neighbors=k_neighbors,
        augment=False,
        normalize_xyz=True,
        normalize_features=True,
        include_xyz_as_features=include_xyz,
        add_graph_features=True,
    )

    # Create dataloaders (drop_last=True avoids BatchNorm issues with single-sample batches)
    train_loader = PyGDataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = PyGDataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Using device: {device}")

    # Create model
    model_type_lower = model_type.lower()
    if model_type_lower == "gcn":
        model = GCNClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout,
        )
    elif model_type_lower == "sage" or model_type_lower == "graphsage":
        model = GraphSAGEClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout,
            use_residual=True,
            graph_feature_dim=graph_feature_dim,
        )
    elif model_type_lower == "gat":
        model = GATClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels // 2,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout,
            heads=4,
        )
    elif model_type_lower == "gin":
        model = GINClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout,
        )
    elif model_type_lower == "hierarchical":
        model = HierarchicalGNN(
            in_channels=in_channels, hidden_channels=hidden_channels, num_classes=2, dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    print(f"[INFO] Model: {model_type}")
    print(f"[INFO] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup loss - Focal Loss helps with class imbalance
    if use_class_weights:
        n_ruptured = sum(1 for _, label in train_files if label == 1)
        n_unruptured = sum(1 for _, label in train_files if label == 0)
        total = n_ruptured + n_unruptured
        weights = torch.tensor(
            [total / (2 * n_unruptured), total / (2 * n_ruptured)],
            dtype=torch.float32,
            device=device,
        )

        if use_focal_loss:
            criterion = FocalLoss(alpha=weights, gamma=focal_gamma, label_smoothing=label_smoothing)
            print(f"[INFO] Using Focal Loss (gamma={focal_gamma})")
        else:
            criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        print(f"[INFO] Class weights: Unruptured={weights[0]:.3f}, Ruptured={weights[1]:.3f}")
    else:
        if use_focal_loss:
            criterion = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Early stopping
    epochs_without_improvement = 0
    print(f"[INFO] Early stopping patience: {early_stopping_patience} epochs")
    print(f"[INFO] Label smoothing: {label_smoothing}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training loop
    best_val_auc = 0.0
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}

    print("\n" + "=" * 60)
    print("Starting GNN Training")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device
        )

        # Calculate AUC
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5

        # Update scheduler
        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        # Print progress
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}"
        )

        # Save best model and early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "history": history,
                    "model_type": model_type,
                    "hidden_channels": hidden_channels,
                    "num_layers": num_layers,
                    "in_channels": in_channels,
                },
                save_path,
            )
            print(f"  [*] Saved best model (AUC: {val_auc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n[INFO] Early stopping after {epoch} epochs")
                break

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training Complete in {elapsed / 60:.1f} minutes")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Best Val AUC-ROC: {best_val_auc:.4f}")
    print(f"Model saved to: {save_path}")
    print("=" * 60)

    # Load best model and print final metrics
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, _, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, device)
    print_metrics(val_labels, val_preds, val_probs, "Final Validation")

    return model, history


# Main
def main():
    parser = argparse.ArgumentParser(description="Train GNN for Aneurysm Rupture Prediction")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="predictions/pinn_corrected",
        help="Directory containing hemodynamics data",
    )
    parser.add_argument("--metadata", type=str, default="metadata.csv", help="Path to metadata.csv")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--target_n", type=int, default=2048, help="Target number of points per sample"
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=16,
        help="Number of k-nearest neighbors for graph construction",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sage",
        choices=["gcn", "sage", "graphsage", "gat", "gin", "hierarchical"],
        help="GNN architecture",
    )
    parser.add_argument("--hidden_channels", type=int, default=96, help="Hidden channel dimension")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of GNN layers (2 recommended to avoid oversmoothing)",
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument(
        "--early_stopping", type=int, default=20, help="Early stopping patience (epochs)"
    )
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument(
        "--focal_loss",
        action="store_true",
        default=True,
        help="Use focal loss instead of cross-entropy",
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    parser.add_argument(
        "--edge_dropout", type=float, default=0.1, help="Edge dropout rate for augmentation"
    )
    parser.add_argument(
        "--kfold", type=int, default=0, help="Number of folds for k-fold CV (0 = single split)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="best_gnn_rupture_model.pth",
        help="Path to save best model",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # K-Fold Cross-Validation or single split
    if args.kfold > 1:
        models, aucs, accs = train_kfold(
            data_dir=args.data_dir,
            metadata_csv=args.metadata,
            n_folds=args.kfold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            target_n=args.target_n,
            k_neighbors=args.k_neighbors,
            model_type=args.model,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            save_dir="kfold_models",
            seed=args.seed,
            use_focal_loss=args.focal_loss,
            focal_gamma=args.focal_gamma,
            early_stopping_patience=args.early_stopping,
            label_smoothing=args.label_smoothing,
            edge_dropout=args.edge_dropout,
        )
    else:
        # Standard single split training
        model, history = train_model(
            data_dir=args.data_dir,
            metadata_csv=args.metadata,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            target_n=args.target_n,
            k_neighbors=args.k_neighbors,
            model_type=args.model,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            save_path=args.save_path,
            seed=args.seed,
            early_stopping_patience=args.early_stopping,
            label_smoothing=args.label_smoothing,
            use_focal_loss=args.focal_loss,
            focal_gamma=args.focal_gamma,
            edge_dropout=args.edge_dropout,
        )


if __name__ == "__main__":
    main()
