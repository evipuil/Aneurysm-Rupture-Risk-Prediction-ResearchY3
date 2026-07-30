#!/usr/bin/env python3
# Version 1 source snapshot
"""
pinn_rupture_classification.py

Physics-Informed Neural Network (PINN) for Aneurysm Rupture Classification

This PINN incorporates hemodynamic physics constraints into the learning process:
1. WSS-OSI relationship: High OSI regions typically have low/oscillating TAWSS
2. Stress concentration physics: Von Mises stress correlates with wall geometry
3. Flow consistency: Spatial smoothness of hemodynamic fields

The model learns physics-aware representations that capture rupture-relevant patterns.

Data structure expected:
- hemodynamics_aggregate.csv files with columns: x, y, z, tawss, osi, von_mises
- metadata.csv with rupture status (ruptured/unruptured)

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
from scipy.spatial import cKDTree
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)


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
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = F.softmax(inputs, dim=1)
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class PhysicsInformedLoss(nn.Module):
    """
    Combined loss with physics-informed regularization terms.

    Physics constraints:
    1. WSS-OSI Consistency: High OSI implies oscillating (typically lower mean) TAWSS
    2. Spatial Smoothness: Hemodynamic fields should be spatially coherent
    3. Stress-Flow Coupling: Von Mises stress relates to flow patterns
    """

    def __init__(
        self,
        classification_loss: nn.Module,
        lambda_wss_osi: float = 0.1,
        lambda_smoothness: float = 0.05,
        lambda_stress: float = 0.05,
    ):
        super().__init__()
        self.classification_loss = classification_loss
        self.lambda_wss_osi = lambda_wss_osi
        self.lambda_smoothness = lambda_smoothness
        self.lambda_stress = lambda_stress

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        physics_losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss with physics terms.

        Args:
            logits: classification logits
            targets: ground truth labels
            physics_losses: dict of physics loss terms from the model

        Returns:
            total_loss, loss_dict
        """
        # Classification loss
        cls_loss = self.classification_loss(logits, targets)

        loss_dict = {"cls_loss": cls_loss.item()}
        total_loss = cls_loss

        if physics_losses is not None:
            if "wss_osi" in physics_losses:
                wss_osi_loss = physics_losses["wss_osi"]
                total_loss = total_loss + self.lambda_wss_osi * wss_osi_loss
                loss_dict["wss_osi_loss"] = wss_osi_loss.item()

            if "smoothness" in physics_losses:
                smooth_loss = physics_losses["smoothness"]
                total_loss = total_loss + self.lambda_smoothness * smooth_loss
                loss_dict["smoothness_loss"] = smooth_loss.item()

            if "stress_coupling" in physics_losses:
                stress_loss = physics_losses["stress_coupling"]
                total_loss = total_loss + self.lambda_stress * stress_loss
                loss_dict["stress_loss"] = stress_loss.item()

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict


# Dataset
class AneurysmPINNDataset(Dataset):
    """
    Dataset for PINN-based rupture classification.

    Includes neighbor information for computing physics-based losses
    (spatial derivatives, smoothness constraints).
    """

    def __init__(
        self,
        file_label_pairs: List[Tuple[str, int]],
        target_n: int = 2048,
        k_neighbors: int = 8,
        augment: bool = False,
        normalize_xyz: bool = True,
        normalize_features: bool = True,
        add_global_features: bool = True,
    ):
        """
        Args:
            file_label_pairs: list of (csv_path, label) tuples
            target_n: target number of points
            k_neighbors: neighbors for physics computations
            augment: whether to apply augmentation
            normalize_xyz: whether to normalize coordinates
            normalize_features: whether to normalize hemodynamic features
            add_global_features: whether to compute global summary statistics
        """
        self.file_label_pairs = file_label_pairs
        self.target_n = target_n
        self.k_neighbors = k_neighbors
        self.augment = augment
        self.normalize_xyz = normalize_xyz
        self.normalize_features = normalize_features
        self.add_global_features = add_global_features

    def __len__(self) -> int:
        return len(self.file_label_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, label = self.file_label_pairs[idx]

        # Load CSV data
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            # Return dummy data
            return {
                "xyz": torch.zeros(self.target_n, 3),
                "features": torch.zeros(self.target_n, 3),
                "raw_features": torch.zeros(self.target_n, 3),
                "neighbor_idx": torch.zeros(self.target_n, self.k_neighbors, dtype=torch.long),
                "global_features": torch.zeros(20),
                "label": torch.tensor(label, dtype=torch.long),
                "path": path,
            }

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 6:
            print(f"[WARNING] Insufficient columns in {path}")
            data = np.pad(data, ((0, 0), (0, 6 - data.shape[1])))

        # Extract coordinates and features
        pts = data[:, :3].astype(np.float32)
        feats = data[:, 3:6].astype(np.float32)  # tawss, osi, von_mises

        # Store raw features for physics computations
        raw_feats = feats.copy()

        # Compute global features before processing
        if self.add_global_features:
            global_feats = self._compute_global_features(feats, pts)
        else:
            global_feats = np.zeros(20, dtype=np.float32)

        # Sample or pad to target number of points
        n_points = pts.shape[0]
        if n_points < self.target_n:
            pad_idx = np.random.choice(n_points, self.target_n - n_points, replace=True)
            pts = np.vstack([pts, pts[pad_idx]])
            feats = np.vstack([feats, feats[pad_idx]])
            raw_feats = np.vstack([raw_feats, raw_feats[pad_idx]])
        elif n_points > self.target_n:
            idx = np.random.choice(n_points, self.target_n, replace=False)
            pts = pts[idx]
            feats = feats[idx]
            raw_feats = raw_feats[idx]

        # Compute k-nearest neighbors for physics constraints
        tree = cKDTree(pts)
        _, neighbor_idx = tree.query(pts, k=self.k_neighbors + 1)
        neighbor_idx = neighbor_idx[:, 1:]  # Exclude self

        # Normalize coordinates
        if self.normalize_xyz:
            pts = self._normalize_points(pts)

        # Normalize features (clip-based to preserve magnitude)
        if self.normalize_features:
            feats = self._normalize_features(feats)

        # Data augmentation
        if self.augment:
            pts, feats = self._augment(pts, feats)

        return {
            "xyz": torch.from_numpy(pts).float(),
            "features": torch.from_numpy(feats).float(),
            "raw_features": torch.from_numpy(raw_feats).float(),
            "neighbor_idx": torch.from_numpy(neighbor_idx).long(),
            "global_features": torch.from_numpy(global_feats).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
        }

    def _normalize_points(self, pts: np.ndarray) -> np.ndarray:
        """Center and scale point cloud to unit sphere."""
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        max_dist = np.max(np.linalg.norm(pts, axis=1))
        if max_dist > 0:
            pts = pts / max_dist
        return pts

    def _normalize_features(self, feats: np.ndarray) -> np.ndarray:
        """Clip-based normalization preserving absolute magnitude."""
        feats = np.clip(feats, 1e-6, None)
        feats = np.log1p(feats)
        feats = np.clip(feats, -3.0, 3.0)
        return feats.astype(np.float32)

    def _compute_global_features(self, raw_feats: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Compute global summary statistics for feature enrichment."""
        tawss = raw_feats[:, 0]
        osi = raw_feats[:, 1]
        von_mises = raw_feats[:, 2]

        features = []

        # TAWSS statistics
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

        # OSI statistics
        features.extend(
            [
                np.mean(osi),
                np.std(osi),
                np.max(osi),
                np.percentile(osi, 95),
                np.sum(osi > 0.2) / len(osi),
            ]
        )

        # Von Mises statistics
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
                np.max(distances),
                np.std(distances),
                np.max(distances) / (np.mean(distances) + 1e-6),
            ]
        )

        features = np.array(features, dtype=np.float32)
        features = np.clip(features, -10, 10)
        return features

    def _augment(self, pts: np.ndarray, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        # Random rotation around z-axis
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
        pts = pts @ R.T

        # Random jitter
        if np.random.random() < 0.5:
            pts += np.random.normal(0, 0.01, pts.shape).astype(np.float32)

        # Random scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            pts = pts * scale

        return pts, feats


def pinn_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for PINN dataset."""
    return {
        "xyz": torch.stack([b["xyz"] for b in batch]),
        "features": torch.stack([b["features"] for b in batch]),
        "raw_features": torch.stack([b["raw_features"] for b in batch]),
        "neighbor_idx": torch.stack([b["neighbor_idx"] for b in batch]),
        "global_features": torch.stack([b["global_features"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "path": [b["path"] for b in batch],
    }


# PINN Model Components
class PointwiseEncoder(nn.Module):
    """
    Encodes each point's features through shared MLPs.
    Similar to PointNet's feature transform.
    """

    def __init__(self, in_channels: int, hidden_channels: List[int], dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_ch = in_channels
        for i, ch in enumerate(hidden_channels):
            layers.append(nn.Conv1d(prev_ch, ch, 1))
            layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU(inplace=True))
            if i < len(hidden_channels) - 1:
                layers.append(nn.Dropout(dropout))
            prev_ch = ch

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, N) point features
        Returns:
            (B, hidden_channels[-1], N) encoded features
        """
        return self.encoder(x)


class PhysicsModule(nn.Module):
    """
    Computes physics-informed losses based on hemodynamic relationships.

    Physics constraints:
    1. WSS-OSI relationship: Regions with high OSI have oscillating flow
    2. Spatial smoothness: Fields should be locally smooth
    3. Stress-flow coupling: Von Mises relates to flow gradients
    """

    def __init__(self, feature_dim: int, k_neighbors: int = 8):
        super().__init__()
        self.k_neighbors = k_neighbors

        # Learnable physics coefficients
        self.wss_osi_weight = nn.Parameter(torch.tensor(1.0))
        self.smoothness_weight = nn.Parameter(torch.tensor(1.0))

        # Small MLP to predict expected relationships
        self.physics_mlp = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # Predict: expected_tawss_grad, expected_osi, expected_stress
        )

    def forward(
        self,
        encoded_features: torch.Tensor,
        raw_features: torch.Tensor,
        xyz: torch.Tensor,
        neighbor_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-based loss terms.

        Args:
            encoded_features: (B, D, N) encoded point features
            raw_features: (B, N, 3) raw hemodynamic features [tawss, osi, von_mises]
            xyz: (B, N, 3) point coordinates
            neighbor_idx: (B, N, K) neighbor indices

        Returns:
            Dict of physics loss terms
        """
        B, D, N = encoded_features.shape
        K = neighbor_idx.shape[2]

        # Extract raw hemodynamics
        tawss = raw_features[:, :, 0]  # (B, N)
        osi = raw_features[:, :, 1]  # (B, N)
        von_mises = raw_features[:, :, 2]  # (B, N)

        physics_losses = {}

        # 1. WSS-OSI Consistency Loss
        # High OSI should correlate with larger TAWSS variance in neighborhood
        # OSI > 0.2 indicates significant flow oscillation
        neighbor_idx.reshape(B, -1)  # (B, N*K)

        # Gather neighbor TAWSS values
        tawss.unsqueeze(2).expand(-1, -1, K)  # (B, N, K)
        neighbor_tawss = torch.gather(
            tawss.unsqueeze(1).expand(-1, N, -1), dim=2, index=neighbor_idx
        )  # (B, N, K)

        # Local TAWSS variance
        local_tawss_var = torch.var(neighbor_tawss, dim=2)  # (B, N)

        # Physics: High OSI should correlate with high local variance
        # Loss: penalize when high OSI but low variance
        high_osi_mask = (osi > 0.15).float()
        wss_osi_loss = high_osi_mask * F.relu(0.1 - local_tawss_var)
        physics_losses["wss_osi"] = wss_osi_loss.mean() * torch.abs(self.wss_osi_weight)

        # 2. Spatial Smoothness Loss
        # Encoded features should be locally smooth (physics fields are continuous)
        encoded_transposed = encoded_features.permute(0, 2, 1)  # (B, N, D)

        # Gather neighbor features
        neighbor_features = torch.gather(
            encoded_transposed.unsqueeze(2).expand(-1, -1, K, -1),
            dim=1,
            index=neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, D),
        )  # (B, N, K, D)

        center_features = encoded_transposed.unsqueeze(2)  # (B, N, 1, D)
        feature_diff = (neighbor_features - center_features).pow(2).mean(dim=(2, 3))  # (B, N)

        # Smoothness loss (penalize large local variations)
        smoothness_loss = feature_diff.mean() * torch.abs(self.smoothness_weight)
        physics_losses["smoothness"] = smoothness_loss

        # 3. Stress-Flow Coupling
        # Von Mises stress should correlate with TAWSS gradients (wall mechanics)
        # Gather neighbor positions
        xyz.unsqueeze(2).expand(-1, -1, K, -1)  # (B, N, K, 3)
        neighbor_xyz = torch.gather(
            xyz.unsqueeze(1).expand(-1, N, -1, -1),
            dim=2,
            index=neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, 3),
        )  # (B, N, K, 3)

        # Approximate TAWSS gradient magnitude
        tawss_diff = torch.abs(neighbor_tawss - tawss.unsqueeze(2))  # (B, N, K)
        xyz_dist = torch.norm(neighbor_xyz - xyz.unsqueeze(2), dim=-1).clamp(min=1e-6)  # (B, N, K)
        tawss_grad = (tawss_diff / xyz_dist).mean(dim=2)  # (B, N)

        # High stress regions should have higher flow gradients
        # Normalize for correlation
        tawss_grad_norm = (tawss_grad - tawss_grad.mean()) / (tawss_grad.std() + 1e-6)
        von_mises_norm = (von_mises - von_mises.mean()) / (von_mises.std() + 1e-6)

        # Negative correlation loss (we want positive correlation)
        stress_coupling_loss = -torch.mean(tawss_grad_norm * von_mises_norm) + 1.0
        physics_losses["stress_coupling"] = F.relu(stress_coupling_loss)

        return physics_losses


class PINNClassifier(nn.Module):
    """
    Physics-Informed Neural Network for Rupture Classification.

    Architecture:
    1. Pointwise encoder (shared MLP)
    2. Physics module for computing physics losses
    3. Global pooling (mean + max)
    4. Global feature fusion
    5. Classification head
    """

    def __init__(
        self,
        in_channels: int = 6,  # xyz + features or just features
        hidden_channels: List[int] = [64, 128, 256],
        global_feature_dim: int = 20,
        num_classes: int = 2,
        dropout: float = 0.5,
        k_neighbors: int = 8,
        use_physics: bool = True,
    ):
        super().__init__()

        self.use_physics = use_physics
        self.k_neighbors = k_neighbors

        # Pointwise encoder
        self.encoder = PointwiseEncoder(in_channels, hidden_channels, dropout=0.3)

        # Physics module
        if use_physics:
            self.physics_module = PhysicsModule(hidden_channels[-1], k_neighbors)

        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 64),
        )

        # Classifier head
        # Mean + Max pooling doubles the feature dimension
        pooled_dim = hidden_channels[-1] * 2 + 64

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass.

        Args:
            batch: dict with 'xyz', 'features', 'raw_features', 'neighbor_idx', 'global_features'

        Returns:
            logits: (B, num_classes) classification logits
            physics_losses: dict of physics loss terms (or None if not computing)
        """
        xyz = batch["xyz"]  # (B, N, 3)
        features = batch["features"]  # (B, N, 3)
        raw_features = batch["raw_features"]  # (B, N, 3)
        neighbor_idx = batch["neighbor_idx"]  # (B, N, K)
        global_features = batch["global_features"]  # (B, 20)

        B, N, _ = xyz.shape

        # Concatenate xyz and features
        x = torch.cat([xyz, features], dim=-1)  # (B, N, 6)
        x = x.permute(0, 2, 1)  # (B, 6, N)

        # Encode pointwise features
        encoded = self.encoder(x)  # (B, D, N)

        # Compute physics losses during training
        physics_losses = None
        if self.training and self.use_physics:
            physics_losses = self.physics_module(encoded, raw_features, xyz, neighbor_idx)

        # Global pooling (mean + max)
        mean_pool = encoded.mean(dim=2)  # (B, D)
        max_pool = encoded.max(dim=2)[0]  # (B, D)
        pooled = torch.cat([mean_pool, max_pool], dim=1)  # (B, 2*D)

        # Global feature fusion
        global_proj = self.global_proj(global_features)  # (B, 64)
        fused = torch.cat([pooled, global_proj], dim=1)  # (B, 2*D + 64)

        # Classify
        logits = self.classifier(fused)  # (B, num_classes)

        return logits, physics_losses


class DeepPINNClassifier(nn.Module):
    """
    Deeper PINN with residual connections and attention.
    Better for larger datasets.
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: List[int] = [64, 128, 256, 512],
        global_feature_dim: int = 20,
        num_classes: int = 2,
        dropout: float = 0.5,
        k_neighbors: int = 8,
        use_physics: bool = True,
    ):
        super().__init__()

        self.use_physics = use_physics

        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels[0], 1),
            nn.BatchNorm1d(hidden_channels[0]),
            nn.ReLU(),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_channels) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_channels[i], hidden_channels[i + 1], dropout)
            )

        # Self-attention for global context
        self.attention = SelfAttention(hidden_channels[-1], num_heads=4)

        # Physics module
        if use_physics:
            self.physics_module = PhysicsModule(hidden_channels[-1], k_neighbors)

        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 128),
        )

        # Classifier
        pooled_dim = hidden_channels[-1] * 2 + 128
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(128, num_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[Dict]]:
        xyz = batch["xyz"]
        features = batch["features"]
        raw_features = batch["raw_features"]
        neighbor_idx = batch["neighbor_idx"]
        global_features = batch["global_features"]

        B, N, _ = xyz.shape

        # Concatenate and project
        x = torch.cat([xyz, features], dim=-1).permute(0, 2, 1)
        x = self.input_proj(x)

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Self-attention
        x = self.attention(x)

        # Physics losses
        physics_losses = None
        if self.training and self.use_physics:
            physics_losses = self.physics_module(x, raw_features, xyz, neighbor_idx)

        # Pooling
        mean_pool = x.mean(dim=2)
        max_pool = x.max(dim=2)[0]
        pooled = torch.cat([mean_pool, max_pool], dim=1)

        # Global features
        global_proj = self.global_proj(global_features)
        fused = torch.cat([pooled, global_proj], dim=1)

        # Classify
        logits = self.classifier(fused)

        return logits, physics_losses


class ResidualBlock(nn.Module):
    """Residual block for point cloud processing."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1), nn.BatchNorm1d(out_channels)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        out = F.relu(out + residual)
        return out


class SelfAttention(nn.Module):
    """Multi-head self-attention for point clouds."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_proj = nn.Conv1d(channels, channels, 1)
        self.k_proj = nn.Conv1d(channels, channels, 1)
        self.v_proj = nn.Conv1d(channels, channels, 1)
        self.out_proj = nn.Conv1d(channels, channels, 1)

        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape

        q = self.q_proj(x).view(B, self.num_heads, self.head_dim, N)
        k = self.k_proj(x).view(B, self.num_heads, self.head_dim, N)
        v = self.v_proj(x).view(B, self.num_heads, self.head_dim, N)

        # Attention scores
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(B, C, N)
        out = self.out_proj(out)

        return x + out  # Residual connection


# Data Loading & Metadata Processing
def load_metadata(metadata_csv: str) -> Dict[str, int]:
    """Load metadata and create mapping from case name to rupture label."""
    mapping = {}

    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle both 'name' and 'Name' column headers
            name = row.get("name") or row.get("Name") or row.get("case_name", "")
            status = row.get("rupture_status") or row.get("Rupture_status") or row.get("status", "")

            if not name or not status:
                continue

            # Clean name
            name = name.strip()
            status = status.strip().lower()

            # Parse label
            if status in ["ruptured", "r", "1", "yes", "true"]:
                mapping[name] = 1
            elif status in ["unruptured", "u", "0", "no", "false"]:
                mapping[name] = 0

    print(f"[INFO] Loaded {len(mapping)} entries from metadata")
    return mapping


def find_hemodynamics_files(data_dir: str, metadata_csv: str) -> List[Tuple[str, int]]:
    """Find all hemodynamics_aggregate.csv files and match with labels."""
    mapping = load_metadata(metadata_csv)
    file_label_pairs = []
    unmatched = []

    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            csv_path = os.path.join(item_path, "hemodynamics_aggregate.csv")
            if os.path.exists(csv_path):
                # Try to match with metadata
                folder_name = item

                # Try different matching strategies
                label = None
                for key in mapping:
                    if key in folder_name or folder_name in key:
                        label = mapping[key]
                        break
                    # Try without _cut suffix
                    base_name = re.sub(r"_cut\d*$", "", folder_name)
                    if key in base_name or base_name in key:
                        label = mapping[key]
                        break

                if label is not None:
                    file_label_pairs.append((csv_path, label))
                else:
                    unmatched.append(folder_name)

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
        print(f"[INFO] Balanced to {min_count} samples per class")

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
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: PhysicsInformedLoss,
    device: torch.device,
) -> Tuple[float, float, Dict]:
    """Train for one epoch with physics losses."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_losses = {}

    for batch in dataloader:
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        labels = batch["label"]

        optimizer.zero_grad()

        # Forward pass
        logits, physics_losses = model(batch)

        # Compute loss
        loss, loss_dict = criterion(logits, labels, physics_losses)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Accumulate loss components
        for k, v in loss_dict.items():
            epoch_losses[k] = epoch_losses.get(k, 0) + v

    # Average losses
    for k in epoch_losses:
        epoch_losses[k] /= len(dataloader)

    return running_loss / total, correct / total, epoch_losses


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: PhysicsInformedLoss, device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on validation/test set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            labels = batch["label"]

            logits, _ = model(batch)
            loss, _ = criterion(logits, labels, None)

            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            running_loss += loss.item() * labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

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
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    target_n: int = 2048,
    model_type: str = "pinn",
    save_dir: str = "kfold_pinn_models",
    seed: int = 42,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    early_stopping_patience: int = 15,
    label_smoothing: float = 0.1,
    num_workers: int = 4,
    dropout: float = 0.5,
    use_physics: bool = True,
    lambda_physics: float = 0.1,
):
    """K-Fold Cross-Validation training."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)

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

    paths = [p for p, _ in file_label_pairs]
    labels = np.array([label for _, label in file_label_pairs])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_aucs = []
    fold_accs = []
    all_val_probs = []
    all_val_labels = []
    models = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Physics-informed training: {use_physics}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'=' * 60}")

        train_pairs = [(paths[i], labels[i]) for i in train_idx]
        val_pairs = [(paths[i], labels[i]) for i in val_idx]

        print(f"[FOLD {fold + 1}] Train: {len(train_pairs)}, Val: {len(val_pairs)}")

        train_ds = AneurysmPINNDataset(
            train_pairs,
            target_n=target_n,
            augment=True,
            normalize_xyz=True,
            normalize_features=True,
            add_global_features=True,
        )
        val_ds = AneurysmPINNDataset(
            val_pairs,
            target_n=target_n,
            augment=False,
            normalize_xyz=True,
            normalize_features=True,
            add_global_features=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=pinn_collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=pinn_collate_fn,
        )

        # Create model
        if model_type.lower() == "deep":
            model = DeepPINNClassifier(
                in_channels=6,
                global_feature_dim=20,
                num_classes=2,
                dropout=dropout,
                use_physics=use_physics,
            )
        else:
            model = PINNClassifier(
                in_channels=6,
                global_feature_dim=20,
                num_classes=2,
                dropout=dropout,
                use_physics=use_physics,
            )

        model = model.to(device)

        # Setup loss
        n_ruptured = sum(1 for _, label in train_pairs if label == 1)
        n_unruptured = sum(1 for _, label in train_pairs if label == 0)
        total = n_ruptured + n_unruptured
        weights = torch.tensor(
            [total / (2 * n_unruptured), total / (2 * n_ruptured)],
            dtype=torch.float32,
            device=device,
        )

        if use_focal_loss:
            cls_loss = FocalLoss(alpha=weights, gamma=focal_gamma, label_smoothing=label_smoothing)
            print(f"[FOLD {fold + 1}] Using Focal Loss (gamma={focal_gamma})")
        else:
            cls_loss = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

        criterion = PhysicsInformedLoss(
            cls_loss,
            lambda_wss_osi=lambda_physics,
            lambda_smoothness=lambda_physics * 0.5,
            lambda_stress=lambda_physics * 0.5,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        best_val_auc = 0.0
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc, loss_dict = train_one_epoch(
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
                physics_str = ""
                if "wss_osi_loss" in loss_dict:
                    physics_str = f" | Phys: {loss_dict.get('wss_osi_loss', 0):.4f}"
                print(
                    f"  Epoch {epoch:03d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} AUC: {val_auc:.4f}{physics_str}"
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

        # Load best model
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

    # Results
    print("\n" + "=" * 60)
    print("K-FOLD CROSS-VALIDATION RESULTS (PINN)")
    print("=" * 60)
    print(f"AUC per fold: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"Acc per fold: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"Mean AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})")
    print(f"Mean Acc: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")

    overall_auc = roc_auc_score(np.array(all_val_labels), np.array(all_val_probs))
    print(f"\nOverall AUC (all folds combined): {overall_auc:.4f}")

    return models, fold_aucs, fold_accs


def train_model(
    data_dir: str,
    metadata_csv: str,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    val_fraction: float = 0.2,
    num_workers: int = 4,
    target_n: int = 2048,
    model_type: str = "pinn",
    save_path: str = "best_pinn_rupture_model.pth",
    seed: int = 42,
    use_class_weights: bool = True,
    early_stopping_patience: int = 20,
    label_smoothing: float = 0.1,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    dropout: float = 0.5,
    use_physics: bool = True,
    lambda_physics: float = 0.1,
):
    """Main training function for PINN classifier."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    file_label_pairs = find_hemodynamics_files(data_dir, metadata_csv)

    if len(file_label_pairs) == 0:
        raise RuntimeError("No data files found! Check paths.")

    train_files, val_files = prepare_train_val_split(
        file_label_pairs, val_fraction=val_fraction, balance_classes=True, seed=seed
    )

    train_ds = AneurysmPINNDataset(
        train_files,
        target_n=target_n,
        augment=True,
        normalize_xyz=True,
        normalize_features=True,
        add_global_features=True,
    )
    val_ds = AneurysmPINNDataset(
        val_files,
        target_n=target_n,
        augment=False,
        normalize_xyz=True,
        normalize_features=True,
        add_global_features=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=pinn_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pinn_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Create model
    if model_type.lower() == "deep":
        model = DeepPINNClassifier(
            in_channels=6,
            global_feature_dim=20,
            num_classes=2,
            dropout=dropout,
            use_physics=use_physics,
        )
    else:
        model = PINNClassifier(
            in_channels=6,
            global_feature_dim=20,
            num_classes=2,
            dropout=dropout,
            use_physics=use_physics,
        )

    model = model.to(device)
    print(f"[INFO] Model: {model_type}")
    print(f"[INFO] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[INFO] Physics-informed training: {use_physics}")

    # Setup loss
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
            cls_loss = FocalLoss(alpha=weights, gamma=focal_gamma, label_smoothing=label_smoothing)
            print(f"[INFO] Using Focal Loss (gamma={focal_gamma})")
        else:
            cls_loss = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        print(f"[INFO] Class weights: Unruptured={weights[0]:.3f}, Ruptured={weights[1]:.3f}")
    else:
        if use_focal_loss:
            cls_loss = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
        else:
            cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    criterion = PhysicsInformedLoss(
        cls_loss,
        lambda_wss_osi=lambda_physics,
        lambda_smoothness=lambda_physics * 0.5,
        lambda_stress=lambda_physics * 0.5,
    )

    epochs_without_improvement = 0
    print(f"[INFO] Early stopping patience: {early_stopping_patience} epochs")
    print(f"[INFO] Physics loss weight: {lambda_physics}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_auc = 0.0
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}

    print("\n" + "=" * 60)
    print("Starting PINN Training")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, loss_dict = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device
        )
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        # Print with physics loss info
        physics_str = ""
        if "wss_osi_loss" in loss_dict:
            physics_str = f" | Phys: {loss_dict.get('wss_osi_loss', 0):.4f}"

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}{physics_str}"
        )

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
                },
                save_path,
            )
            print(f"  [*] Saved best model (AUC: {val_auc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n[INFO] Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training Complete in {elapsed / 60:.1f} minutes")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Best Val AUC-ROC: {best_val_auc:.4f}")
    print(f"Model saved to: {save_path}")
    print("=" * 60)

    # Final evaluation
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, _, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, device)
    print_metrics(val_labels, val_preds, val_probs, "Final Validation")

    return model, history


# Main
def main():
    parser = argparse.ArgumentParser(description="Train PINN for Aneurysm Rupture Prediction")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="predictions/pinn_corrected",
        help="Directory containing hemodynamics data",
    )
    parser.add_argument("--metadata", type=str, default="metadata.csv", help="Path to metadata.csv")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--target_n", type=int, default=2048, help="Target number of points per sample"
    )
    parser.add_argument(
        "--early_stopping", type=int, default=15, help="Early stopping patience (epochs)"
    )
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument(
        "--model",
        type=str,
        default="pinn",
        choices=["pinn", "deep"],
        help="Model architecture (pinn=standard, deep=deeper with attention)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="best_pinn_rupture_model.pth",
        help="Path to save best model",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--focal_loss", action="store_true", help="Use Focal Loss instead of CrossEntropyLoss"
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument(
        "--kfold",
        type=int,
        default=1,
        help="Number of folds for cross-validation (1 = single split)",
    )
    parser.add_argument(
        "--kfold_save_dir",
        type=str,
        default="kfold_pinn_models",
        help="Directory to save k-fold models",
    )
    parser.add_argument(
        "--use_physics", action="store_true", default=True, help="Use physics-informed loss terms"
    )
    parser.add_argument(
        "--no_physics", action="store_true", help="Disable physics-informed loss terms"
    )
    parser.add_argument(
        "--lambda_physics", type=float, default=0.1, help="Weight for physics loss terms"
    )

    args = parser.parse_args()

    use_physics = args.use_physics and not args.no_physics

    print("=" * 60)
    print("PINN Rupture Classification")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Focal Loss: {args.focal_loss} (gamma={args.focal_gamma})")
    print(f"Physics-Informed: {use_physics} (lambda={args.lambda_physics})")
    print(f"K-Fold: {args.kfold}")
    print(f"Dropout: {args.dropout}")
    print("=" * 60)

    if args.kfold > 1:
        models, fold_aucs, fold_accs = train_kfold(
            data_dir=args.data_dir,
            metadata_csv=args.metadata,
            n_folds=args.kfold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            target_n=args.target_n,
            model_type=args.model,
            save_dir=args.kfold_save_dir,
            seed=args.seed,
            use_focal_loss=args.focal_loss,
            focal_gamma=args.focal_gamma,
            early_stopping_patience=args.early_stopping,
            label_smoothing=args.label_smoothing,
            num_workers=args.num_workers,
            dropout=args.dropout,
            use_physics=use_physics,
            lambda_physics=args.lambda_physics,
        )
    else:
        model, history = train_model(
            data_dir=args.data_dir,
            metadata_csv=args.metadata,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            target_n=args.target_n,
            model_type=args.model,
            save_path=args.save_path,
            num_workers=args.num_workers,
            seed=args.seed,
            early_stopping_patience=args.early_stopping,
            label_smoothing=args.label_smoothing,
            use_focal_loss=args.focal_loss,
            focal_gamma=args.focal_gamma,
            dropout=args.dropout,
            use_physics=use_physics,
            lambda_physics=args.lambda_physics,
        )


if __name__ == "__main__":
    main()
