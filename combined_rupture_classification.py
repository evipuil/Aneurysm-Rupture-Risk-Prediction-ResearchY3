#!/usr/bin/env python3
# Version 1 source snapshot
"""
combined_rupture_classification.py

Combined Geometry + Hemodynamics Model for Aneurysm Rupture Classification

This model leverages both data sources:
1. Geometry (xyz) - proven strong predictor, full SO(3) augmentation
2. Hemodynamics (TAWSS, OSI, Von Mises) - additional physiological information

Architecture options:
- "geometry": Geometry-only (like the working model)
- "hemodynamics": Hemodynamics-only
- "early_fusion": Concatenate xyz + hemodynamics early
- "late_fusion": Dual-branch with late feature fusion
- "attention_fusion": Cross-attention between geometry and hemodynamics

Data structure expected:
- hemodynamics_aggregate.csv files with columns: x, y, z, tawss, osi, von_mises
- metadata.csv with rupture status (ruptured/unruptured)

"""

import argparse
import csv
import math
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
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)


# Loss Functions
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

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


# Utils: Point Cloud Operations
def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index points based on indices."""
    B, N, C = points.shape
    S = idx.shape[1]
    idx_expanded = idx.unsqueeze(-1).expand(B, S, C)
    new_points = torch.gather(points, 1, idx_expanded)
    return new_points


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest Point Sampling (FPS) algorithm."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device, dtype=torch.float32) * 1e10
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


# Dataset
class CombinedAneurysmDataset(Dataset):
    """
    Dataset for combined geometry + hemodynamics rupture prediction.

    Supports multiple input modes:
    - geometry: xyz only (with full SO(3) augmentation)
    - hemodynamics: tawss, osi, von_mises only
    - combined: xyz + hemodynamics
    """

    def __init__(
        self,
        file_label_pairs: List[Tuple[str, int]],
        target_n: int = 1024,
        mode: str = "combined",  # "geometry", "hemodynamics", "combined"
        augment: bool = False,
        normalize_xyz: bool = True,
        normalize_features: bool = True,
        add_global_features: bool = True,
    ):
        """
        Args:
            file_label_pairs: list of (csv_path, label) tuples
            target_n: target number of points
            mode: input mode - "geometry", "hemodynamics", or "combined"
            augment: whether to apply data augmentation
            normalize_xyz: whether to normalize coordinates
            normalize_features: whether to normalize hemodynamic features
            add_global_features: whether to compute global summary statistics
        """
        self.file_label_pairs = file_label_pairs
        self.target_n = target_n
        self.mode = mode
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
            return self._dummy_sample(label, path)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 6:
            print(f"[WARNING] Insufficient columns in {path}")
            data = np.pad(data, ((0, 0), (0, 6 - data.shape[1])))

        # Extract coordinates and features
        pts = data[:, :3].astype(np.float32)
        feats = data[:, 3:6].astype(np.float32)  # tawss, osi, von_mises

        # Compute global features before processing (raw values)
        if self.add_global_features:
            global_feats = self._compute_global_features(feats, pts)
        else:
            global_feats = np.zeros(23, dtype=np.float32)  # 20 + 3 extra

        # Sample or pad to target number of points
        n_points = pts.shape[0]
        if n_points < self.target_n:
            pad_idx = np.random.choice(n_points, self.target_n - n_points, replace=True)
            pts = np.vstack([pts, pts[pad_idx]])
            feats = np.vstack([feats, feats[pad_idx]])
        elif n_points > self.target_n:
            idx = np.random.choice(n_points, self.target_n, replace=False)
            pts = pts[idx]
            feats = feats[idx]

        # Normalize coordinates (center + scale to unit sphere)
        if self.normalize_xyz:
            pts = self._normalize_points(pts)

        # Normalize hemodynamic features (minimal processing to preserve magnitude)
        if self.normalize_features:
            feats = self._normalize_features(feats)

        # Data augmentation - FULL SO(3) rotation (key difference!)
        if self.augment:
            pts = self._so3_rotate(pts)
            pts = self._jitter(pts)
            # Random scaling
            scale = np.random.uniform(0.95, 1.05)
            pts = pts * scale

        return {
            "xyz": torch.from_numpy(pts).float(),
            "features": torch.from_numpy(feats).float(),
            "global_features": torch.from_numpy(global_feats).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
        }

    def _dummy_sample(self, label: int, path: str) -> Dict[str, torch.Tensor]:
        """Return dummy sample on load failure."""
        return {
            "xyz": torch.zeros(self.target_n, 3),
            "features": torch.zeros(self.target_n, 3),
            "global_features": torch.zeros(23),
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
        """
        Minimal normalization to preserve absolute magnitude.
        Uses per-sample standardization but keeps relative values.
        """
        # Per-feature normalization (preserves differences between samples better than log)
        mu = np.mean(feats, axis=0)
        sigma = np.std(feats, axis=0)
        sigma[sigma < 1e-8] = 1.0
        feats = (feats - mu) / sigma
        # Clip extremes
        feats = np.clip(feats, -3.0, 3.0)
        return feats.astype(np.float32)

    def _compute_global_features(self, raw_feats: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Compute global summary statistics for feature enrichment."""
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

        # Geometric features (6) - these are key for rupture!
        centroid = np.mean(pts, axis=0)
        centered = pts - centroid
        distances = np.linalg.norm(centered, axis=1)

        # Principal axes analysis (shape descriptors)
        try:
            cov = np.cov(pts.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            # Normalize eigenvalues
            eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-8)
        except Exception:
            eigenvalues = np.array([0.5, 0.3, 0.2])

        features.extend(
            [
                np.max(distances),  # Size proxy
                np.std(distances),  # Shape irregularity
                np.max(distances) / (np.mean(distances) + 1e-6),  # Aspect ratio proxy
                eigenvalues[0],  # Largest principal component ratio
                eigenvalues[1],  # Second principal component ratio
                eigenvalues[0] / (eigenvalues[2] + 1e-6),  # Elongation ratio
            ]
        )

        features = np.array(features, dtype=np.float32)
        # Log transform for heavy-tailed distributions, then clip
        features = np.sign(features) * np.log1p(np.abs(features))
        features = np.clip(features, -10, 10)
        return features

    @staticmethod
    def _so3_rotate(points: np.ndarray) -> np.ndarray:
        """Apply random SO(3) rotation (full 3D rotation)."""
        rx = np.random.uniform(0, 2 * np.pi)
        ry = np.random.uniform(0, 2 * np.pi)
        rz = np.random.uniform(0, 2 * np.pi)

        Rx = np.array(
            [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float32
        )

        Ry = np.array(
            [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32
        )

        Rz = np.array(
            [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float32
        )

        R = Rz @ Ry @ Rx
        return (points @ R.T).astype(np.float32)

    @staticmethod
    def _jitter(points: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
        """Add random jitter to points."""
        jittered = points + np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
        return jittered.astype(np.float32)


def combined_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function."""
    return {
        "xyz": torch.stack([b["xyz"] for b in batch]),
        "features": torch.stack([b["features"] for b in batch]),
        "global_features": torch.stack([b["global_features"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "path": [b["path"] for b in batch],
    }


# Model Components
class PointNetSetAbstraction(nn.Module):
    """Set Abstraction module for PointNet++ (simplified version that works well)."""

    def __init__(self, npoint: Optional[int], in_channel: int, mlp: List[int]):
        super().__init__()
        self.npoint = npoint

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv1d(last_ch, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_ch))
            last_ch = out_ch

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            points: (B, C, N) point features (or None)
        Returns:
            new_points: (B, mlp[-1], npoint) abstracted features
            new_xyz: (B, npoint, 3) sampled coordinates
        """
        B, N, _ = xyz.shape

        if self.npoint is not None:
            idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, idx)
        else:
            idx = None
            new_xyz = torch.mean(xyz, dim=1, keepdim=True)

        if points is None:
            feats = xyz
        else:
            feats = points.permute(0, 2, 1).contiguous()  # B, N, C

        if self.npoint is not None:
            new_feats = index_points(feats, idx)
        else:
            new_feats = feats.mean(dim=1, keepdim=True)

        x = new_feats.permute(0, 2, 1).contiguous()  # B, C, S

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            x = F.relu(bn(conv(x)))

        return x, new_xyz


# Model Architectures
class GeometryOnlyModel(nn.Module):
    """
    Geometry-only PointNet++ (mimics the working model structure).
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(npoint=512, in_channel=3, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, in_channel=128, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=None, in_channel=256, mlp=[256, 512, 1024])

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        xyz = batch["xyz"]  # B, N, 3
        B = xyz.shape[0]

        l1_pts, l1_xyz = self.sa1(xyz, None)
        l2_pts, l2_xyz = self.sa2(l1_xyz, l1_pts)
        l3_pts, l3_xyz = self.sa3(l2_xyz, l2_pts)

        x = F.adaptive_max_pool1d(l3_pts, 1).view(B, -1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)

        return x


class HemodynamicsOnlyModel(nn.Module):
    """
    Hemodynamics-only model using the same PointNet++ structure.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5, global_feature_dim: int = 23):
        super().__init__()

        # Use hemodynamics as input features
        self.sa1 = PointNetSetAbstraction(npoint=512, in_channel=3, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, in_channel=128, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=None, in_channel=256, mlp=[256, 512, 1024])

        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feature_dim, 64), nn.ReLU(), nn.Linear(64, 64)
        )

        self.fc1 = nn.Linear(1024 + 64, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        xyz = batch["xyz"]  # B, N, 3 - use for FPS sampling
        feats = batch["features"]  # B, N, 3 - hemodynamics
        global_feats = batch["global_features"]  # B, 23
        B = xyz.shape[0]

        # Use hemodynamics as input, xyz for sampling structure
        # We'll create a combined representation
        l1_pts, l1_xyz = self.sa1(xyz, feats.permute(0, 2, 1))
        l2_pts, l2_xyz = self.sa2(l1_xyz, l1_pts)
        l3_pts, l3_xyz = self.sa3(l2_xyz, l2_pts)

        x = F.adaptive_max_pool1d(l3_pts, 1).view(B, -1)

        # Add global features
        global_proj = self.global_proj(global_feats)
        x = torch.cat([x, global_proj], dim=1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)

        return x


class EarlyFusionModel(nn.Module):
    """
    Early fusion: concatenate xyz + hemodynamics at input.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5, global_feature_dim: int = 23):
        super().__init__()

        # 6 input channels: xyz (3) + hemodynamics (3)
        self.sa1 = PointNetSetAbstraction(npoint=512, in_channel=6, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, in_channel=128, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=None, in_channel=256, mlp=[256, 512, 1024])

        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feature_dim, 64), nn.ReLU(), nn.Linear(64, 64)
        )

        self.fc1 = nn.Linear(1024 + 64, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        xyz = batch["xyz"]  # B, N, 3
        feats = batch["features"]  # B, N, 3
        global_feats = batch["global_features"]  # B, 23
        B = xyz.shape[0]

        # Concatenate xyz and features
        combined = torch.cat([xyz, feats], dim=2)  # B, N, 6

        # First SA uses combined as input
        l1_pts, l1_xyz = self.sa1(xyz, combined.permute(0, 2, 1))
        l2_pts, l2_xyz = self.sa2(l1_xyz, l1_pts)
        l3_pts, l3_xyz = self.sa3(l2_xyz, l2_pts)

        x = F.adaptive_max_pool1d(l3_pts, 1).view(B, -1)

        # Add global features
        global_proj = self.global_proj(global_feats)
        x = torch.cat([x, global_proj], dim=1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)

        return x


class LateFusionModel(nn.Module):
    """
    Late fusion: separate branches for geometry and hemodynamics,
    fused at the classifier level.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5, global_feature_dim: int = 23):
        super().__init__()

        # Geometry branch (smaller, since it works well)
        self.geo_sa1 = PointNetSetAbstraction(npoint=512, in_channel=3, mlp=[64, 64, 128])
        self.geo_sa2 = PointNetSetAbstraction(npoint=128, in_channel=128, mlp=[128, 128, 256])
        self.geo_sa3 = PointNetSetAbstraction(npoint=None, in_channel=256, mlp=[256, 512])

        # Hemodynamics branch (smaller)
        self.hemo_sa1 = PointNetSetAbstraction(npoint=512, in_channel=3, mlp=[32, 32, 64])
        self.hemo_sa2 = PointNetSetAbstraction(npoint=128, in_channel=64, mlp=[64, 64, 128])
        self.hemo_sa3 = PointNetSetAbstraction(npoint=None, in_channel=128, mlp=[128, 256])

        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feature_dim, 64), nn.ReLU(), nn.Linear(64, 64)
        )

        # Fusion: 512 (geo) + 256 (hemo) + 64 (global) = 832
        self.fc1 = nn.Linear(512 + 256 + 64, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        xyz = batch["xyz"]  # B, N, 3
        feats = batch["features"]  # B, N, 3
        global_feats = batch["global_features"]  # B, 23
        B = xyz.shape[0]

        # Geometry branch
        g1_pts, g1_xyz = self.geo_sa1(xyz, None)
        g2_pts, g2_xyz = self.geo_sa2(g1_xyz, g1_pts)
        g3_pts, g3_xyz = self.geo_sa3(g2_xyz, g2_pts)
        geo_feat = F.adaptive_max_pool1d(g3_pts, 1).view(B, -1)  # B, 512

        # Hemodynamics branch
        h1_pts, h1_xyz = self.hemo_sa1(xyz, feats.permute(0, 2, 1))
        h2_pts, h2_xyz = self.hemo_sa2(h1_xyz, h1_pts)
        h3_pts, h3_xyz = self.hemo_sa3(h2_xyz, h2_pts)
        hemo_feat = F.adaptive_max_pool1d(h3_pts, 1).view(B, -1)  # B, 256

        # Global features
        global_proj = self.global_proj(global_feats)  # B, 64

        # Late fusion
        x = torch.cat([geo_feat, hemo_feat, global_proj], dim=1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)

        return x


class AttentionFusionModel(nn.Module):
    """
    Attention-based fusion: cross-attention between geometry and hemodynamics features.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5, global_feature_dim: int = 23):
        super().__init__()

        # Geometry branch
        self.geo_sa1 = PointNetSetAbstraction(npoint=256, in_channel=3, mlp=[64, 64, 128])
        self.geo_sa2 = PointNetSetAbstraction(npoint=64, in_channel=128, mlp=[128, 128, 256])

        # Hemodynamics branch
        self.hemo_sa1 = PointNetSetAbstraction(npoint=256, in_channel=3, mlp=[64, 64, 128])
        self.hemo_sa2 = PointNetSetAbstraction(npoint=64, in_channel=128, mlp=[128, 128, 256])

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=256, num_heads=4, dropout=dropout, batch_first=True
        )

        # Final SA
        self.final_sa = PointNetSetAbstraction(npoint=None, in_channel=512, mlp=[512, 512, 1024])

        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feature_dim, 64), nn.ReLU(), nn.Linear(64, 64)
        )

        # Classifier
        self.fc1 = nn.Linear(1024 + 64, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        xyz = batch["xyz"]
        feats = batch["features"]
        global_feats = batch["global_features"]
        B = xyz.shape[0]

        # Geometry branch
        g1_pts, g1_xyz = self.geo_sa1(xyz, None)
        g2_pts, g2_xyz = self.geo_sa2(g1_xyz, g1_pts)  # B, 256, 64

        # Hemodynamics branch
        h1_pts, h1_xyz = self.hemo_sa1(xyz, feats.permute(0, 2, 1))
        h2_pts, h2_xyz = self.hemo_sa2(h1_xyz, h1_pts)  # B, 256, 64

        # Cross-attention: geometry attends to hemodynamics
        geo_seq = g2_pts.permute(0, 2, 1)  # B, 64, 256
        hemo_seq = h2_pts.permute(0, 2, 1)  # B, 64, 256

        attn_out, _ = self.cross_attn(geo_seq, hemo_seq, hemo_seq)  # B, 64, 256

        # Combine
        combined = torch.cat([g2_pts, attn_out.permute(0, 2, 1)], dim=1)  # B, 512, 64

        # Dummy xyz for final SA (use geometry positions)
        final_pts, _ = self.final_sa(g2_xyz, combined)
        x = F.adaptive_max_pool1d(final_pts, 1).view(B, -1)

        # Global features
        global_proj = self.global_proj(global_feats)
        x = torch.cat([x, global_proj], dim=1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)

        return x


# Data Loading & Metadata Processing
def load_metadata(metadata_csv: str) -> Dict[str, int]:
    """Load metadata and create mapping from case name to rupture label."""
    mapping = {}

    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name") or row.get("Name") or row.get("case_name", "")
            status = row.get("rupture_status") or row.get("Rupture_status") or row.get("status", "")

            if not name or not status:
                continue

            name = name.strip()
            status = status.strip().lower()

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
                folder_name = item

                label = None
                for key in mapping:
                    if key in folder_name or folder_name in key:
                        label = mapping[key]
                        break
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

    n_val_rupt = max(1, int(math.ceil(val_fraction * len(ruptured))))
    n_val_unrupt = max(1, int(math.ceil(val_fraction * len(unruptured))))

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
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        labels = batch["label"]

        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
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
            logits = model(batch)
            loss = criterion(logits, labels)

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


def create_model(model_type: str, num_classes: int = 2, dropout: float = 0.5) -> nn.Module:
    """Create model based on type."""
    if model_type == "geometry":
        return GeometryOnlyModel(num_classes=num_classes, dropout=dropout)
    elif model_type == "hemodynamics":
        return HemodynamicsOnlyModel(num_classes=num_classes, dropout=dropout)
    elif model_type == "early_fusion":
        return EarlyFusionModel(num_classes=num_classes, dropout=dropout)
    elif model_type == "late_fusion":
        return LateFusionModel(num_classes=num_classes, dropout=dropout)
    elif model_type == "attention":
        return AttentionFusionModel(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_kfold(
    data_dir: str,
    metadata_csv: str,
    n_folds: int = 5,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    target_n: int = 1024,
    model_type: str = "late_fusion",
    save_dir: str = "kfold_combined_models",
    seed: int = 42,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    early_stopping_patience: int = 30,
    num_workers: int = 2,
    dropout: float = 0.5,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Model type: {model_type}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'=' * 60}")

        train_pairs = [(paths[i], labels[i]) for i in train_idx]
        val_pairs = [(paths[i], labels[i]) for i in val_idx]

        print(f"[FOLD {fold + 1}] Train: {len(train_pairs)}, Val: {len(val_pairs)}")

        train_ds = CombinedAneurysmDataset(
            train_pairs,
            target_n=target_n,
            mode="combined",
            augment=True,
            normalize_xyz=True,
            normalize_features=True,
            add_global_features=True,
        )
        val_ds = CombinedAneurysmDataset(
            val_pairs,
            target_n=target_n,
            mode="combined",
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
            collate_fn=combined_collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=combined_collate_fn,
        )

        model = create_model(model_type, num_classes=2, dropout=dropout)
        model = model.to(device)

        # Loss
        if use_focal_loss:
            criterion = FocalLoss(gamma=focal_gamma)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        best_val_auc = 0.0
        best_val_acc = 0.0
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

            if epoch % 20 == 0 or val_auc > best_val_auc:
                print(
                    f"  Epoch {epoch:03d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} AUC: {val_auc:.4f}"
                )

            # Track best by accuracy (like the working model)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
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

        print(f"\n[FOLD {fold + 1}] Best Acc: {fold_acc:.4f}, AUC: {fold_auc:.4f}")

    # Results
    print("\n" + "=" * 60)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"Model: {model_type}")
    print(f"AUC per fold: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"Acc per fold: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"Mean AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})")
    print(f"Mean Acc: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")

    overall_auc = roc_auc_score(np.array(all_val_labels), np.array(all_val_probs))
    print(f"\nOverall AUC (all folds combined): {overall_auc:.4f}")

    return fold_aucs, fold_accs


def train_model(
    data_dir: str,
    metadata_csv: str,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    num_workers: int = 2,
    target_n: int = 1024,
    model_type: str = "late_fusion",
    save_path: str = "best_combined_model.pth",
    seed: int = 42,
    early_stopping_patience: int = 30,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    dropout: float = 0.5,
):
    """Main training function."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    file_label_pairs = find_hemodynamics_files(data_dir, metadata_csv)

    if len(file_label_pairs) == 0:
        raise RuntimeError("No data files found!")

    train_files, val_files = prepare_train_val_split(
        file_label_pairs, val_fraction=val_fraction, balance_classes=True, seed=seed
    )

    train_ds = CombinedAneurysmDataset(
        train_files,
        target_n=target_n,
        mode="combined",
        augment=True,
        normalize_xyz=True,
        normalize_features=True,
        add_global_features=True,
    )
    val_ds = CombinedAneurysmDataset(
        val_files,
        target_n=target_n,
        mode="combined",
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
        collate_fn=combined_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=combined_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = create_model(model_type, num_classes=2, dropout=dropout)
    model = model.to(device)
    print(f"[INFO] Model: {model_type}")
    print(f"[INFO] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if use_focal_loss:
        criterion = FocalLoss(gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_val_acc = 0.0
    epochs_without_improvement = 0

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device
        )
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5

        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                },
                save_path,
            )
            print(f"  [*] Saved best model (Acc: {val_acc:.4f}, AUC: {val_auc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n[INFO] Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training Complete in {elapsed / 60:.1f} minutes")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {save_path}")
    print("=" * 60)

    # Final evaluation
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, _, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, device)
    print_metrics(val_labels, val_preds, val_probs, "Final Validation")

    return model


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Combined Geometry + Hemodynamics for Rupture Prediction"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="predictions/pinn_corrected",
        help="Directory containing hemodynamics data",
    )
    parser.add_argument("--metadata", type=str, default="metadata.csv", help="Path to metadata.csv")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--target_n", type=int, default=1024, help="Target number of points per sample"
    )
    parser.add_argument(
        "--early_stopping", type=int, default=30, help="Early stopping patience (epochs)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="late_fusion",
        choices=["geometry", "hemodynamics", "early_fusion", "late_fusion", "attention"],
        help="Model architecture",
    )
    parser.add_argument(
        "--save_path", type=str, default="best_combined_model.pth", help="Path to save best model"
    )
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--focal_loss", action="store_true", help="Use Focal Loss")
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
        default="kfold_combined_models",
        help="Directory to save k-fold models",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Combined Geometry + Hemodynamics Rupture Classification")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Points: {args.target_n}")
    print(f"Focal Loss: {args.focal_loss}")
    print(f"K-Fold: {args.kfold}")
    print("=" * 60)

    if args.kfold > 1:
        train_kfold(
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
            num_workers=args.num_workers,
            dropout=args.dropout,
        )
    else:
        train_model(
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
            use_focal_loss=args.focal_loss,
            focal_gamma=args.focal_gamma,
            dropout=args.dropout,
        )


if __name__ == "__main__":
    main()
