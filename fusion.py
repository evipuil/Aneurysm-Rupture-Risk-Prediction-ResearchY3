#!/usr/bin/env python3
# Version 2 source snapshot
"""
fusion.py

Multi-Modal Fusion Model for Aneurysm Rupture Classification

This model combines:
1. Geometry (xyz coordinates) - shape/morphology features
2. Hemodynamics (TAWSS, OSI, Von Mises) - blood flow characteristics
3. Clinical features (age, sex) - patient demographics

The clinical features (age, sex) are loaded from metadata.csv and fused
with the point cloud features at the classifier level.

Architecture options:
- "late_fusion": Dual-branch PointNet++ with clinical feature fusion
- "attention_fusion": Cross-attention between all modalities
- "clinical_only": Baseline using only age + sex (for comparison)

Data structure expected:
- hemodynamics_aggregate.csv files with columns: x, y, z, tawss, osi, von_mises
- metadata.csv with rupture status, sex, and age columns

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
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# Utils: Point Cloud Operations
def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, K]
    Return:
        new_points: indexed points data, [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
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


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_points, new_xyz, fps_idx


def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# Dataset
class FusionDataset(Dataset):
    """
    Dataset for multi-modal fusion including clinical features (age, sex).

    Combines:
    - Point cloud data (geometry + hemodynamics)
    - Clinical features from metadata (age, sex)
    """

    def __init__(
        self,
        file_label_clinical_tuples: List[Tuple[str, int, float, int]],
        target_n: int = 8192,
        augment: bool = False,
        normalize_xyz: bool = True,
        normalize_features: bool = True,
        add_global_features: bool = True,
    ):
        """
        Args:
            file_label_clinical_tuples: list of (csv_path, label, age, sex) tuples
                - age: normalized age value
                - sex: 0 for female, 1 for male
            target_n: target number of points
            augment: whether to apply data augmentation
            normalize_xyz: whether to normalize coordinates
            normalize_features: whether to normalize hemodynamic features
            add_global_features: whether to compute global summary statistics
        """
        self.data = file_label_clinical_tuples
        self.target_n = target_n
        self.augment = augment
        self.normalize_xyz = normalize_xyz
        self.normalize_features = normalize_features
        self.add_global_features = add_global_features

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, label, age, sex = self.data[idx]

        # Load CSV data
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            return self._dummy_sample(label, age, sex, path)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 6:
            print(f"[WARNING] Insufficient columns in {path}")
            data = np.pad(data, ((0, 0), (0, 6 - data.shape[1])))

        # Extract coordinates and features
        pts = data[:, :3].astype(np.float32)
        raw_feats = data[:, 3:6].astype(np.float32)  # tawss, osi, von_mises

        # Compute derived features (per-point)
        tawss = raw_feats[:, 0:1]
        osi = raw_feats[:, 1:2]
        von_mises = raw_feats[:, 2:3]

        # Derived features:
        # 1. Low TAWSS indicator (< median)
        tawss_median = np.median(tawss)
        low_tawss = (tawss < tawss_median).astype(np.float32)

        # 2. High OSI indicator (> 0.2 is typically considered high)
        high_osi = (osi > 0.2).astype(np.float32)

        # 3. TAWSS * (1 - 2*OSI) - combined hemodynamic stress indicator
        # Low TAWSS + high OSI is bad → this becomes negative
        combined_stress = tawss * (1 - 2 * osi)

        # 4. Velocity magnitude proxy from von_mises
        vm_normalized = von_mises / (np.max(von_mises) + 1e-8)

        # 5. Risk score: high OSI + low TAWSS regions
        risk_score = high_osi * low_tawss

        # Stack all features: original (3) + derived (5) = 8 features
        feats = np.hstack(
            [
                tawss,
                osi,
                von_mises,  # Original 3
                low_tawss,
                high_osi,
                combined_stress,
                vm_normalized,
                risk_score,  # Derived 5
            ]
        ).astype(np.float32)

        # Compute global features before processing (raw values)
        if self.add_global_features:
            global_feats = self._compute_global_features(raw_feats, pts)
        else:
            global_feats = np.zeros(30, dtype=np.float32)  # Updated size

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

        # Normalize coordinates
        if self.normalize_xyz:
            pts = self._normalize_points(pts)

        # Normalize hemodynamic features
        if self.normalize_features:
            feats = self._normalize_features(feats)

        # Data augmentation - FULL SO(3) rotation
        if self.augment:
            pts = self._so3_rotate(pts)
            pts = self._jitter(pts)
            scale = np.random.uniform(0.95, 1.05)
            pts = pts * scale

        # Clinical features: [age, sex]
        clinical_feats = np.array([age, float(sex)], dtype=np.float32)

        return {
            "xyz": torch.from_numpy(pts).float(),
            "features": torch.from_numpy(feats).float(),
            "global_features": torch.from_numpy(global_feats).float(),
            "clinical_features": torch.from_numpy(clinical_feats).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
        }

    def _dummy_sample(self, label: int, age: float, sex: int, path: str) -> Dict[str, torch.Tensor]:
        """Return dummy sample on load failure."""
        return {
            "xyz": torch.zeros(self.target_n, 3),
            "features": torch.zeros(self.target_n, 8),  # 3 original + 5 derived
            "global_features": torch.zeros(30),  # Updated size
            "clinical_features": torch.tensor([age, float(sex)], dtype=torch.float32),
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
        """Minimal normalization to preserve absolute magnitude."""
        mu = np.mean(feats, axis=0)
        sigma = np.std(feats, axis=0)
        sigma[sigma < 1e-8] = 1.0
        feats = (feats - mu) / sigma
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

        # Derived feature statistics (7) - matches __getitem__ derived features
        tawss_median = np.median(tawss)
        low_tawss = (tawss < tawss_median).astype(np.float32)
        high_osi = (osi > 0.2).astype(np.float32)
        combined_stress = tawss * (1 - 2 * osi)
        vm_normalized = von_mises / (np.max(von_mises) + 1e-8)
        risk_score = high_osi * low_tawss

        features.extend(
            [
                np.mean(low_tawss),  # Fraction of low TAWSS points
                np.mean(high_osi),  # Fraction of high OSI points
                np.mean(combined_stress),
                np.std(combined_stress),  # Combined stress stats
                np.mean(vm_normalized),
                np.std(vm_normalized),  # Normalized VM stats
                np.mean(risk_score),  # Risk score mean (high OSI + low TAWSS overlap)
            ]
        )

        # Geometric features (6)
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


def fusion_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for fusion dataset."""
    return {
        "xyz": torch.stack([b["xyz"] for b in batch]),
        "features": torch.stack([b["features"] for b in batch]),
        "global_features": torch.stack([b["global_features"] for b in batch]),
        "clinical_features": torch.stack([b["clinical_features"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "path": [b["path"] for b in batch],
    }


# Model Components
class PointNetSetAbstraction(nn.Module):
    """Set Abstraction module for PointNet++."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch
        self.group_all = group_all

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None):
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_points, new_xyz, _ = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )

        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0, 2, 1)  # [B, npoint, C']
        return new_points, new_xyz


# Model Architectures
class ClinicalOnlyModel(nn.Module):
    """
    Baseline model using only clinical features (age, sex).
    Useful for comparison to understand the added value of imaging.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        # Clinical features: age, sex (2 features)
        self.fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        clinical = batch["clinical_features"]  # (B, 2)
        return self.fc(clinical)


class LateFusionWithClinicalModel(nn.Module):
    """
    Late fusion model: separate branches for geometry and hemodynamics,
    plus clinical features (age, sex), fused at classifier level.
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        global_feature_dim: int = 30,
        clinical_feature_dim: int = 2,
    ):
        super().__init__()

        # Geometry branch (xyz only) - mimics working geometry model
        self.geo_sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False
        )
        self.geo_sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=131, mlp=[128, 128, 256], group_all=False
        )
        self.geo_sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=259,
            mlp=[256, 512, 1024],
            group_all=True,
        )

        # Hemodynamics branch (xyz + 8 hemodynamic features = 11 channels)
        self.hemo_sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=11, mlp=[64, 64, 128], group_all=False
        )
        self.hemo_sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=131, mlp=[128, 128, 256], group_all=False
        )
        self.hemo_sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=259,
            mlp=[256, 512, 512],
            group_all=True,
        )

        # Clinical feature encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_feature_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Fusion: geometry (1024) + hemodynamics (512) + global (128) + clinical (64) = 1728
        fusion_dim = 1024 + 512 + 128 + 64

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        xyz = batch["xyz"]  # (B, N, 3)
        features = batch["features"]  # (B, N, 3) - hemodynamics
        global_features = batch["global_features"]  # (B, 23)
        clinical_features = batch["clinical_features"]  # (B, 2)

        # Geometry branch
        geo_pts, geo_xyz = self.geo_sa1(xyz, None)
        geo_pts, geo_xyz = self.geo_sa2(geo_xyz, geo_pts)
        geo_pts, _ = self.geo_sa3(geo_xyz, geo_pts)
        geo_global = geo_pts.squeeze(1)  # (B, 1024)

        # Hemodynamics branch
        hemo_pts, hemo_xyz = self.hemo_sa1(xyz, features)
        hemo_pts, hemo_xyz = self.hemo_sa2(hemo_xyz, hemo_pts)
        hemo_pts, _ = self.hemo_sa3(hemo_xyz, hemo_pts)
        hemo_global = hemo_pts.squeeze(1)  # (B, 512)

        # Encode global features
        global_enc = self.global_encoder(global_features)  # (B, 128)

        # Encode clinical features
        clinical_enc = self.clinical_encoder(clinical_features)  # (B, 64)

        # Fusion
        fused = torch.cat([geo_global, hemo_global, global_enc, clinical_enc], dim=-1)

        return self.classifier(fused)


class AttentionFusionWithClinicalModel(nn.Module):
    """
    Attention-based fusion with clinical features.
    Cross-attention between geometry, hemodynamics, and clinical features.
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        global_feature_dim: int = 30,
        clinical_feature_dim: int = 2,
    ):
        super().__init__()

        # Geometry branch
        self.geo_sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False
        )
        self.geo_sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=131, mlp=[128, 128, 256], group_all=False
        )
        self.geo_sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=259,
            mlp=[256, 512, 512],
            group_all=True,
        )

        # Hemodynamics branch (xyz + 8 hemodynamic features = 11 channels)
        self.hemo_sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=11, mlp=[64, 64, 128], group_all=False
        )
        self.hemo_sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=131, mlp=[128, 128, 256], group_all=False
        )
        self.hemo_sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=259,
            mlp=[256, 512, 512],
            group_all=True,
        )

        # Clinical feature encoder (project to same dimension)
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Cross-attention between modalities
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=4, dropout=dropout, batch_first=True
        )

        # Fusion after attention: 3 * 512 (geo, hemo, clinical) + 256 (global) = 1792
        fusion_dim = 512 * 3 + 256

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        xyz = batch["xyz"]
        features = batch["features"]
        global_features = batch["global_features"]
        clinical_features = batch["clinical_features"]

        # Geometry branch
        geo_pts, geo_xyz = self.geo_sa1(xyz, None)
        geo_pts, geo_xyz = self.geo_sa2(geo_xyz, geo_pts)
        geo_pts, _ = self.geo_sa3(geo_xyz, geo_pts)
        geo_global = geo_pts.squeeze(1)  # (B, 512)

        # Hemodynamics branch
        hemo_pts, hemo_xyz = self.hemo_sa1(xyz, features)
        hemo_pts, hemo_xyz = self.hemo_sa2(hemo_xyz, hemo_pts)
        hemo_pts, _ = self.hemo_sa3(hemo_xyz, hemo_pts)
        hemo_global = hemo_pts.squeeze(1)  # (B, 512)

        # Clinical features
        clinical_enc = self.clinical_encoder(clinical_features)  # (B, 512)

        # Global features
        global_enc = self.global_encoder(global_features)  # (B, 256)

        # Cross-attention: stack modalities as sequence
        modalities = torch.stack([geo_global, hemo_global, clinical_enc], dim=1)  # (B, 3, 512)
        attended, _ = self.cross_attention(modalities, modalities, modalities)  # (B, 3, 512)

        # Flatten attended features
        attended_flat = attended.view(attended.size(0), -1)  # (B, 1536)

        # Fusion
        fused = torch.cat([attended_flat, global_enc], dim=-1)  # (B, 1792)

        return self.classifier(fused)


class FullFusionModel(nn.Module):
    """
    Full fusion model combining all modalities with gating mechanism.
    Learns to weight the contribution of each modality.
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        global_feature_dim: int = 30,
        clinical_feature_dim: int = 2,
    ):
        super().__init__()

        # Geometry branch
        self.geo_sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False
        )
        self.geo_sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=131, mlp=[128, 128, 256], group_all=False
        )
        self.geo_sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=259,
            mlp=[256, 512, 512],
            group_all=True,
        )

        # Hemodynamics branch (xyz + 8 hemodynamic features = 11 channels)
        self.hemo_sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=11, mlp=[64, 64, 128], group_all=False
        )
        self.hemo_sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=131, mlp=[128, 128, 256], group_all=False
        )
        self.hemo_sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=259,
            mlp=[256, 512, 512],
            group_all=True,
        )

        # Clinical feature encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Gating mechanism - learns importance of each modality
        # Input: concat of all modality features
        gate_input_dim = 512 + 512 + 256 + 256  # geo + hemo + clinical + global
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),  # 4 gates for 4 modalities
            nn.Softmax(dim=-1),
        )

        # Project all to same dimension for weighted sum
        self.geo_proj = nn.Linear(512, 256)
        self.hemo_proj = nn.Linear(512, 256)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        xyz = batch["xyz"]
        features = batch["features"]
        global_features = batch["global_features"]
        clinical_features = batch["clinical_features"]

        # Geometry branch
        geo_pts, geo_xyz = self.geo_sa1(xyz, None)
        geo_pts, geo_xyz = self.geo_sa2(geo_xyz, geo_pts)
        geo_pts, _ = self.geo_sa3(geo_xyz, geo_pts)
        geo_global = geo_pts.squeeze(1)  # (B, 512)

        # Hemodynamics branch
        hemo_pts, hemo_xyz = self.hemo_sa1(xyz, features)
        hemo_pts, hemo_xyz = self.hemo_sa2(hemo_xyz, hemo_pts)
        hemo_pts, _ = self.hemo_sa3(hemo_xyz, hemo_pts)
        hemo_global = hemo_pts.squeeze(1)  # (B, 512)

        # Clinical features
        clinical_enc = self.clinical_encoder(clinical_features)  # (B, 256)

        # Global features
        global_enc = self.global_encoder(global_features)  # (B, 256)

        # Compute gates
        concat_all = torch.cat([geo_global, hemo_global, clinical_enc, global_enc], dim=-1)
        gates = self.gate(concat_all)  # (B, 4)

        # Project to same dimension
        geo_proj = self.geo_proj(geo_global)  # (B, 256)
        hemo_proj = self.hemo_proj(hemo_global)  # (B, 256)

        # Weighted sum using gates
        fused = (
            gates[:, 0:1] * geo_proj
            + gates[:, 1:2] * hemo_proj
            + gates[:, 2:3] * clinical_enc
            + gates[:, 3:4] * global_enc
        )  # (B, 256)

        return self.classifier(fused)


# Data Loading & Metadata Processing
def load_metadata_with_clinical(metadata_csv: str) -> Dict[str, Dict]:
    """
    Load metadata including clinical features (age, sex).

    Returns:
        Dictionary mapping case name to {'label': int, 'age': float, 'sex': int}
    """
    mapping = {}
    ages = []

    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Get case name (try multiple possible column names)
            name = (
                row.get("dataset") or row.get("name") or row.get("Name") or row.get("case_name", "")
            )
            status = row.get("status") or row.get("rupture_status") or row.get("Rupture_status", "")
            sex = row.get("sex") or row.get("Sex") or row.get("gender", "")
            age = row.get("age") or row.get("Age", "")

            if not name or not status:
                continue

            name = name.strip()
            status = status.strip().lower()
            sex = sex.strip().lower() if sex else ""

            # Parse label
            if status in ["ruptured", "r", "1", "yes", "true"]:
                label = 1
            elif status in ["unruptured", "u", "0", "no", "false"]:
                label = 0
            else:
                continue

            # Parse sex (0 = female, 1 = male)
            if sex in ["female", "f", "0"]:
                sex_val = 0
            elif sex in ["male", "m", "1"]:
                sex_val = 1
            else:
                sex_val = 0  # Default to female if unknown

            # Parse age
            try:
                age_val = float(age)
                ages.append(age_val)
            except (ValueError, TypeError):
                age_val = 50.0  # Default age if missing

            mapping[name] = {"label": label, "age": age_val, "sex": sex_val}

    # Normalize ages (z-score normalization based on loaded data)
    if ages:
        mean_age = np.mean(ages)
        std_age = np.std(ages) if np.std(ages) > 0 else 1.0
        for name in mapping:
            mapping[name]["age_normalized"] = (mapping[name]["age"] - mean_age) / std_age
    else:
        for name in mapping:
            mapping[name]["age_normalized"] = 0.0

    print(f"[INFO] Loaded {len(mapping)} entries with clinical features")
    if ages:
        print(
            f"[INFO] Age stats: mean={np.mean(ages):.1f}, std={np.std(ages):.1f}, range=[{np.min(ages):.1f}, {np.max(ages):.1f}]"
        )

    return mapping


def find_hemodynamics_files_with_clinical(
    data_dir: str, metadata_csv: str
) -> List[Tuple[str, int, float, int]]:
    """
    Find hemodynamics files and match with labels + clinical features.

    Returns:
        List of (csv_path, label, age_normalized, sex) tuples
    """
    mapping = load_metadata_with_clinical(metadata_csv)
    file_tuples = []
    unmatched = []

    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            csv_path = os.path.join(item_path, "hemodynamics_aggregate.csv")
            if os.path.exists(csv_path):
                folder_name = item

                matched_data = None
                for key in mapping:
                    if key in folder_name or folder_name in key:
                        matched_data = mapping[key]
                        break
                    base_name = re.sub(r"_cut\d*$", "", folder_name)
                    if key in base_name or base_name in key:
                        matched_data = mapping[key]
                        break

                if matched_data is not None:
                    file_tuples.append(
                        (
                            csv_path,
                            matched_data["label"],
                            matched_data["age_normalized"],
                            matched_data["sex"],
                        )
                    )
                else:
                    unmatched.append(folder_name)

    n_ruptured = sum(1 for t in file_tuples if t[1] == 1)
    n_unruptured = sum(1 for t in file_tuples if t[1] == 0)
    n_male = sum(1 for t in file_tuples if t[3] == 1)
    n_female = sum(1 for t in file_tuples if t[3] == 0)

    print(f"[INFO] Found {len(file_tuples)} matched cases:")
    print(f"       - Ruptured: {n_ruptured}, Unruptured: {n_unruptured}")
    print(f"       - Male: {n_male}, Female: {n_female}")
    print(f"       - Unmatched: {len(unmatched)}")

    return file_tuples


def prepare_train_val_split(
    file_tuples: List[Tuple],
    val_fraction: float = 0.2,
    balance_classes: bool = True,
    seed: int = 42,
) -> Tuple[List, List]:
    """Split data into training and validation sets."""
    random.seed(seed)

    ruptured = [t for t in file_tuples if t[1] == 1]
    unruptured = [t for t in file_tuples if t[1] == 0]

    random.shuffle(ruptured)
    random.shuffle(unruptured)

    if balance_classes:
        min_count = min(len(ruptured), len(unruptured))
        ruptured = ruptured[:min_count]
        unruptured = unruptured[:min_count]

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

    all_labels = []
    all_probs = []

    for batch in dataloader:
        xyz = batch["xyz"].to(device)
        features = batch["features"].to(device)
        global_features = batch["global_features"].to(device)
        clinical_features = batch["clinical_features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        batch_dict = {
            "xyz": xyz,
            "features": features,
            "global_features": global_features,
            "clinical_features": clinical_features,
        }

        logits = model(batch_dict)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xyz.size(0)
        preds = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)[:, 1]
        correct += (preds == labels).sum().item()
        total += xyz.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5

    return running_loss / total, correct / total, auc


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
            xyz = batch["xyz"].to(device)
            features = batch["features"].to(device)
            global_features = batch["global_features"].to(device)
            clinical_features = batch["clinical_features"].to(device)
            labels = batch["label"].to(device)

            batch_dict = {
                "xyz": xyz,
                "features": features,
                "global_features": global_features,
                "clinical_features": clinical_features,
            }

            logits = model(batch_dict)
            loss = criterion(logits, labels)

            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            running_loss += loss.item() * xyz.size(0)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

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
        print(f"\nAUC-ROC: {auc:.4f}")

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")


def create_model(model_type: str, num_classes: int = 2, dropout: float = 0.5) -> nn.Module:
    """Create model based on type."""
    if model_type == "clinical_only":
        return ClinicalOnlyModel(num_classes, dropout)
    elif model_type == "late_fusion":
        return LateFusionWithClinicalModel(num_classes, dropout)
    elif model_type == "attention":
        return AttentionFusionWithClinicalModel(num_classes, dropout)
    elif model_type == "full_fusion":
        return FullFusionModel(num_classes, dropout)
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
    save_dir: str = "kfold_fusion_models",
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

    file_tuples = find_hemodynamics_files_with_clinical(data_dir, metadata_csv)
    if len(file_tuples) == 0:
        raise RuntimeError("No data files found!")

    # Balance classes
    ruptured = [t for t in file_tuples if t[1] == 1]
    unruptured = [t for t in file_tuples if t[1] == 0]
    min_count = min(len(ruptured), len(unruptured))
    random.shuffle(ruptured)
    random.shuffle(unruptured)
    file_tuples = ruptured[:min_count] + unruptured[:min_count]
    random.shuffle(file_tuples)

    print(f"[INFO] K-Fold CV with {n_folds} folds on {len(file_tuples)} samples")

    paths = [t[0] for t in file_tuples]
    labels = np.array([t[1] for t in file_tuples])

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

        train_tuples = [file_tuples[i] for i in train_idx]
        val_tuples = [file_tuples[i] for i in val_idx]

        print(f"[FOLD {fold + 1}] Train: {len(train_tuples)}, Val: {len(val_tuples)}")

        train_ds = FusionDataset(
            train_tuples,
            target_n=target_n,
            augment=True,
            normalize_xyz=True,
            normalize_features=True,
            add_global_features=True,
        )
        val_ds = FusionDataset(
            val_tuples,
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
            collate_fn=fusion_collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=fusion_collate_fn,
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

        # CSV logging for this fold
        csv_path = os.path.join(save_dir, f"fold{fold + 1}_training_log.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["epoch", "train_loss", "train_acc", "train_auc", "val_loss", "val_acc", "val_auc"]
            )

        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_auc = train_one_epoch(
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

            # Log to CSV
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [epoch, train_loss, train_acc, train_auc, val_loss, val_acc, val_auc]
                )

            if epoch % 20 == 0 or val_auc > best_val_auc:
                print(
                    f"  Epoch {epoch:03d} | Train Acc: {train_acc:.4f} AUC: {train_auc:.4f} | Val Acc: {val_acc:.4f} AUC: {val_auc:.4f}"
                )

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

    # Write summary CSV with k-fold results
    summary_csv_path = os.path.join(save_dir, "kfold_summary.csv")
    with open(summary_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["fold", "model_type", "val_acc", "val_auc"])
        for i, (acc, auc) in enumerate(zip(fold_accs, fold_aucs)):
            writer.writerow([i + 1, model_type, acc, auc])
        writer.writerow(["mean", model_type, np.mean(fold_accs), np.mean(fold_aucs)])
        writer.writerow(["std", model_type, np.std(fold_accs), np.std(fold_aucs)])
        writer.writerow(
            ["overall", model_type, (np.array(all_val_probs) > 0.5).mean(), overall_auc]
        )
    print(f"\nSummary saved to: {summary_csv_path}")

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
    save_path: str = "best_fusion_model.pth",
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

    file_tuples = find_hemodynamics_files_with_clinical(data_dir, metadata_csv)

    if len(file_tuples) == 0:
        raise RuntimeError("No data files found!")

    train_files, val_files = prepare_train_val_split(
        file_tuples, val_fraction=val_fraction, balance_classes=True, seed=seed
    )

    train_ds = FusionDataset(
        train_files,
        target_n=target_n,
        augment=True,
        normalize_xyz=True,
        normalize_features=True,
        add_global_features=True,
    )
    val_ds = FusionDataset(
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
        collate_fn=fusion_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=fusion_collate_fn,
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

    # CSV logging
    csv_path = save_path.replace(".pth", "_training_log.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_auc"])

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device
        )
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5

        scheduler.step()

        # Log to CSV
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, val_auc])

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
        description="Multi-Modal Fusion (Geometry + Hemodynamics + Clinical) for Rupture Prediction"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="predictions/pinn_corrected",
        help="Directory containing hemodynamics data",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="metadata.csv",
        help="Path to metadata.csv (must include sex and age columns)",
    )
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--target_n", type=int, default=8192, help="Target number of points per sample"
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=999,
        help="Early stopping patience (epochs), set high to disable",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="late_fusion",
        choices=["clinical_only", "late_fusion", "attention", "full_fusion"],
        help="Model architecture",
    )
    parser.add_argument(
        "--save_path", type=str, default="best_fusion_model.pth", help="Path to save best model"
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
        default="kfold_fusion_models",
        help="Directory to save k-fold models",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Modal Fusion Rupture Classification")
    print("(Geometry + Hemodynamics + Age + Sex)")
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
            val_fraction=0.2,
            num_workers=args.num_workers,
            target_n=args.target_n,
            model_type=args.model,
            save_path=args.save_path,
            seed=args.seed,
            early_stopping_patience=args.early_stopping,
            use_focal_loss=args.focal_loss,
            focal_gamma=args.focal_gamma,
            dropout=args.dropout,
        )


if __name__ == "__main__":
    main()
