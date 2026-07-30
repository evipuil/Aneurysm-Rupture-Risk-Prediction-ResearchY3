#!/usr/bin/env python3
# Version 2 source snapshot
"""
multichannel_pointnet_rupture.py

Multichannel PointNet++ for Aneurysm Rupture Prediction
Uses hemodynamics data (TAWSS, OSI, Von Mises stress) from PINN-corrected predictions.

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
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter (higher = more focus on hard examples)
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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


# Utility Functions: Point Cloud Operations
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
    """
    Farthest Point Sampling (FPS) algorithm.

    Args:
        xyz: (B, N, 3) tensor of point coordinates
        npoint: number of points to sample

    Returns:
        (B, npoint) tensor of sampled point indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device, dtype=torch.float32) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_ar = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_ar, farthest].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def query_ball_point(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """
    Ball query for grouping points within a radius.

    Args:
        radius: local region radius
        nsample: max sample number in local region
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) query points

    Returns:
        (B, S, nsample) group indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate squared Euclidean distance between two point sets.

    Args:
        src: (B, N, C) source points
        dst: (B, M, C) destination points

    Returns:
        (B, N, M) squared distances
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst**2, dim=-1).view(B, 1, M)
    return dist


# Dataset
class AneurysmHemodynamicsDataset(Dataset):
    """
    Dataset for aneurysm rupture prediction using hemodynamics features.

    Expects hemodynamics_aggregate.csv files with columns:
    x, y, z, tawss, osi, von_mises

    Labels:
    - 0: unruptured
    - 1: ruptured
    """

    def __init__(
        self,
        file_label_pairs: List[Tuple[str, int]],
        target_n: int = 4096,
        augment: bool = False,
        normalize_xyz: bool = True,
        normalize_features: bool = True,
        add_global_features: bool = True,
    ):
        """
        Args:
            file_label_pairs: list of (csv_path, label) tuples
            target_n: target number of points (sample/pad to this)
            augment: whether to apply data augmentation
            normalize_xyz: whether to normalize xyz coordinates
            normalize_features: whether to normalize hemodynamic features
            add_global_features: whether to compute global summary statistics
        """
        self.file_label_pairs = file_label_pairs
        self.target_n = target_n
        self.augment = augment
        self.normalize_xyz = normalize_xyz
        self.normalize_features = normalize_features
        self.add_global_features = add_global_features

    def __len__(self) -> int:
        return len(self.file_label_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str]:
        path, label = self.file_label_pairs[idx]

        # Load CSV data
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return zeros if loading fails
            pts = np.zeros((self.target_n, 3), dtype=np.float32)
            feats = np.zeros((self.target_n, 3), dtype=np.float32)
            global_feats = np.zeros(20, dtype=np.float32)
            return (
                torch.from_numpy(pts),
                torch.from_numpy(feats),
                torch.from_numpy(global_feats),
                int(label),
                path,
            )

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 6:
            print(f"Warning: {path} has only {data.shape[1]} columns, expected 6")
            pts = np.zeros((self.target_n, 3), dtype=np.float32)
            feats = np.zeros((self.target_n, 3), dtype=np.float32)
            global_feats = np.zeros(20, dtype=np.float32)
            return (
                torch.from_numpy(pts),
                torch.from_numpy(feats),
                torch.from_numpy(global_feats),
                int(label),
                path,
            )

        # Extract coordinates and features
        pts = data[:, :3].astype(np.float32)  # x, y, z
        feats = data[:, 3:6].astype(np.float32)  # tawss, osi, von_mises

        # Compute global features before any processing
        if self.add_global_features:
            global_feats = self._compute_global_features(feats, pts)
        else:
            global_feats = np.zeros(20, dtype=np.float32)

        # Sample or pad to target number of points
        n_points = pts.shape[0]
        if n_points < self.target_n:
            # Pad with duplicates (better than zeros)
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

        # Data augmentation
        if self.augment:
            pts, feats = self._augment(pts, feats)

        return (
            torch.from_numpy(pts).float(),
            torch.from_numpy(feats).float(),
            torch.from_numpy(global_feats).float(),
            int(label),
            path,
        )

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
        feats = np.clip(feats, 1e-6, None)  # Ensure positive for log
        feats = np.log1p(feats)  # log(1+x) for stability
        feats = np.clip(feats, -3.0, 3.0)  # Clip extremes
        return feats.astype(np.float32)

    def _compute_global_features(self, raw_feats: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Compute global summary statistics for feature enrichment.
        These capture hemodynamic patterns that point-level features miss.
        """
        tawss = raw_feats[:, 0]
        osi = raw_feats[:, 1]
        von_mises = raw_feats[:, 2]

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
                np.sum(osi > 0.2) / len(osi),  # High OSI ratio
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

        features = np.array(features, dtype=np.float32)
        features = np.clip(features, -10, 10)
        return features

    def _augment(self, pts: np.ndarray, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        # Random rotation around z-axis (most common for medical data)
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
        pts = pts @ R.T

        # Random jitter
        pts = pts + np.clip(0.01 * np.random.randn(*pts.shape), -0.03, 0.03).astype(np.float32)

        # Random scaling
        scale = np.random.uniform(0.9, 1.1)
        pts = pts * scale

        # Random feature noise
        feats = feats + np.clip(0.02 * np.random.randn(*feats.shape), -0.05, 0.05).astype(
            np.float32
        )

        return pts, feats


# Model Components
class PointNetSetAbstraction(nn.Module):
    """
    Set Abstraction module for PointNet++.
    Combines sampling, grouping, and PointNet layers.
    """

    def __init__(
        self,
        npoint: Optional[int],
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: List[int],
        group_all: bool = False,
    ):
        """
        Args:
            npoint: number of points to sample (None for global)
            radius: ball query radius
            nsample: number of samples in ball query
            in_channel: input feature dimension
            mlp: list of layer sizes
            group_all: whether to group all points (global feature)
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        # Build MLP layers
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3  # +3 for xyz

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            points: (B, N, C) point features

        Returns:
            new_xyz: (B, S, 3) sampled point coordinates
            new_points: (B, S, D) new point features
        """
        B, N, C = xyz.shape

        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz = xyz.view(B, 1, N, 3)
            if points is not None:
                grouped_points = points.view(B, 1, N, -1)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz
        else:
            # FPS sampling
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)

            # Ball query grouping
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx.reshape(B, -1)).reshape(
                B, self.npoint, self.nsample, 3
            )
            grouped_xyz = grouped_xyz - new_xyz.reshape(
                B, self.npoint, 1, 3
            )  # Normalize to local frame

            if points is not None:
                grouped_points = index_points(points, idx.reshape(B, -1)).reshape(
                    B, self.npoint, self.nsample, -1
                )
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz

        # (B, S, nsample, C) -> (B, C, S, nsample)
        grouped_points = grouped_points.permute(0, 3, 1, 2)

        # Apply MLP
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped_points = F.relu(bn(conv(grouped_points)))

        # Max pooling
        new_points = torch.max(grouped_points, dim=3)[0]  # (B, D, S)
        new_points = new_points.permute(0, 2, 1).contiguous()  # (B, S, D)

        return new_xyz, new_points


class PointNetPlusPlusRupture(nn.Module):
    """
    PointNet++ for aneurysm rupture classification with improvements:
    - Mean + Max pooling to capture both average and extreme (hotspot) values
    - Global feature fusion for summary statistics
    - Conditional BatchNorm for single-sample batches

    Architecture:
    - 3 Set Abstraction layers for hierarchical feature learning
    - Global pooling (mean + max)
    - Global feature fusion
    - MLP classifier
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channel: int = 3,
        global_feature_dim: int = 20,
        dropout: float = 0.5,
    ):
        """
        Args:
            num_classes: number of output classes (2 for rupture/unruptured)
            in_channel: number of input feature channels (3 for tawss, osi, von_mises)
            global_feature_dim: dimension of global summary features
            dropout: dropout rate
        """
        super().__init__()

        self.global_feature_dim = global_feature_dim

        # Set Abstraction layers
        # SA1: 4096 -> 1024 points
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.1, nsample=32, in_channel=in_channel, mlp=[32, 32, 64]
        )

        # SA2: 1024 -> 256 points
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.2, nsample=32, in_channel=64, mlp=[64, 64, 128]
        )

        # SA3: 256 -> 64 points
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.4, nsample=32, in_channel=128, mlp=[128, 128, 256]
        )

        # SA4: Global feature (64 -> 1 point)
        self.sa4 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256,
            mlp=[256, 512, 512],
            group_all=True,
        )

        # Global feature fusion layer
        if global_feature_dim > 0:
            self.global_fc = nn.Linear(global_feature_dim, 64)
            classifier_in = 512 * 2 + 64  # mean + max + global features
        else:
            classifier_in = 512 * 2  # mean + max pooling

        # Classification head
        self.fc1 = nn.Linear(classifier_in, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
        global_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, N, C) point features
            global_features: (B, global_feature_dim) global summary statistics

        Returns:
            (B, num_classes) class logits
        """
        B = xyz.shape[0]

        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Mean + Max pooling (captures both average and hotspots)
        x_reshape = l4_points.reshape(B, -1)  # (B, 512)
        x_mean = x_reshape  # Already global from SA4
        x_max = x_reshape  # SA4 uses max pooling internally
        x = torch.cat([x_mean, x_max], dim=1)  # (B, 1024)

        # Fuse global features if available
        if self.global_feature_dim > 0 and global_features is not None:
            gf = F.relu(self.global_fc(global_features))
            x = torch.cat([x, gf], dim=1)

        # Classification head (skip BatchNorm if batch size is 1)
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)

        x = self.fc3(x)

        return x


# Simplified PointNet (for faster training / baseline)
class PointNetRupture(nn.Module):
    """
    Simplified PointNet for rupture classification with improvements:
    - Mean + Max pooling
    - Global feature fusion
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channel: int = 3,
        global_feature_dim: int = 20,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.global_feature_dim = global_feature_dim

        # Feature extraction
        self.conv1 = nn.Conv1d(in_channel + 3, 64, 1)  # +3 for xyz
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, 512, 1)  # Reduced from 1024

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(512)

        # Global feature fusion
        if global_feature_dim > 0:
            self.global_fc = nn.Linear(global_feature_dim, 64)
            classifier_in = 512 * 2 + 64  # mean + max + global
        else:
            classifier_in = 512 * 2  # mean + max pooling

        # Classification head
        self.fc1 = nn.Linear(classifier_in, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(256, 128)
        self.fc_bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
        global_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = xyz.shape

        # Concatenate xyz and features
        x = torch.cat([xyz, features], dim=-1)  # (B, N, 6)
        x = x.permute(0, 2, 1)  # (B, 6, N)

        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Mean + Max pooling (captures both average and hotspots)
        x_mean = torch.mean(x, dim=2)  # (B, 512)
        x_max = torch.max(x, dim=2)[0]  # (B, 512)
        x = torch.cat([x_mean, x_max], dim=1)  # (B, 1024)

        # Fuse global features if available
        if self.global_feature_dim > 0 and global_features is not None:
            gf = F.relu(self.global_fc(global_features))
            x = torch.cat([x, gf], dim=1)

        # Classification (skip BatchNorm if batch size is 1)
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.fc_bn1(x)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.fc_bn2(x)
        x = F.relu(x)
        x = self.drop2(x)

        x = self.fc3(x)

        return x


# Data Loading & Metadata Processing
def load_metadata(metadata_csv: str) -> Dict[str, int]:
    """
    Load metadata and create mapping from case name to rupture label.

    Args:
        metadata_csv: path to metadata.csv

    Returns:
        Dictionary mapping case identifiers to labels (0=unruptured, 1=ruptured)
    """
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

            # Create multiple possible keys for matching
            # Key 1: dataset_cut (e.g., p043_HAARCREcDAAQDQcbHgANDRQM_cut1)
            key1 = f"{dataset}_{cut}"
            mapping[key1] = label

            # Key 2: just dataset (e.g., p043_HAARCREcDAAQDQcbHgANDRQM)
            mapping[dataset] = label

            # Key 3: For cases like C0001, SNF00000024, etc.
            # These are directly the dataset names

    print(f"[INFO] Loaded {len(mapping)} entries from metadata")
    return mapping


def find_hemodynamics_files(data_dir: str, metadata_csv: str) -> List[Tuple[str, int]]:
    """
    Find all hemodynamics_aggregate.csv files and match with labels.

    Args:
        data_dir: directory containing case folders (e.g., pinn_corrected/)
        metadata_csv: path to metadata.csv

    Returns:
        List of (csv_path, label) tuples
    """
    mapping = load_metadata(metadata_csv)
    file_label_pairs = []
    unmatched = []

    # Walk through directory
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)

        if not os.path.isdir(item_path):
            continue

        # Look for hemodynamics_aggregate.csv
        csv_path = os.path.join(item_path, "hemodynamics_aggregate.csv")
        if not os.path.exists(csv_path):
            continue

        # Try to match with metadata
        # Remove _cut1 suffix for matching
        case_name = item  # e.g., C0001_cut1
        label = None

        # Try direct match first
        if case_name in mapping:
            label = mapping[case_name]
        else:
            # Try without _cut suffix
            base_name = re.sub(r"_cut\d+$", "", case_name)
            if base_name in mapping:
                label = mapping[base_name]

        if label is not None:
            file_label_pairs.append((csv_path, label))
        else:
            unmatched.append(case_name)

    # Print statistics
    n_ruptured = sum(1 for _, label in file_label_pairs if label == 1)
    n_unruptured = sum(1 for _, label in file_label_pairs if label == 0)

    print(f"[INFO] Found {len(file_label_pairs)} matched cases:")
    print(f"       - Ruptured: {n_ruptured}")
    print(f"       - Unruptured: {n_unruptured}")
    print(f"       - Unmatched: {len(unmatched)}")

    if len(unmatched) > 0 and len(unmatched) <= 20:
        print(f"[DEBUG] Unmatched cases: {unmatched[:20]}")

    return file_label_pairs


def prepare_train_val_split(
    file_label_pairs: List[Tuple[str, int]],
    val_fraction: float = 0.2,
    balance_classes: bool = True,
    seed: int = 42,
) -> Tuple[List, List]:
    """
    Split data into training and validation sets.

    Args:
        file_label_pairs: list of (path, label) tuples
        val_fraction: fraction of data for validation
        balance_classes: whether to balance classes
        seed: random seed

    Returns:
        (train_files, val_files) lists
    """
    random.seed(seed)

    # Separate by class
    ruptured = [(path, label) for path, label in file_label_pairs if label == 1]
    unruptured = [(path, label) for path, label in file_label_pairs if label == 0]

    random.shuffle(ruptured)
    random.shuffle(unruptured)

    if balance_classes:
        # Use minimum of both classes
        min_count = min(len(ruptured), len(unruptured))
        ruptured = ruptured[:min_count]
        unruptured = unruptured[:min_count]
        print(f"[INFO] Balanced classes to {min_count} samples each")

    # Split each class
    n_val_rupt = max(1, int(len(ruptured) * val_fraction))
    n_val_unrupt = max(1, int(len(unruptured) * val_fraction))

    val_ruptured = ruptured[:n_val_rupt]
    train_ruptured = ruptured[n_val_rupt:]

    val_unruptured = unruptured[:n_val_unrupt]
    train_unruptured = unruptured[n_val_unrupt:]

    # Combine and shuffle
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

    for xyz, features, global_feats, labels, _ in dataloader:
        xyz = xyz.to(device)
        features = features.to(device)
        global_feats = global_feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(xyz, features, global_feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xyz.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += xyz.size(0)

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
        for xyz, features, global_feats, labels, _ in dataloader:
            xyz = xyz.to(device)
            features = features.to(device)
            global_feats = global_feats.to(device)
            labels = labels.to(device)

            logits = model(xyz, features, global_feats)
            loss = criterion(logits, labels)

            running_loss += loss.item() * xyz.size(0)
            probs = F.softmax(logits, dim=1)[:, 1]  # Probability of ruptured
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
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

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print("  Predicted:  Unrupt  Rupt")
    print(f"  Unruptured:  {cm[0, 0]:5d}  {cm[0, 1]:5d}")
    print(f"  Ruptured:    {cm[1, 0]:5d}  {cm[1, 1]:5d}")

    # AUC-ROC
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, probs)
        ap = average_precision_score(labels, probs)
        print(f"\nAUC-ROC: {auc:.4f}")
        print(f"Average Precision: {ap:.4f}")

    # Sensitivity and Specificity
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
    model_type: str = "pointnet++",
    save_dir: str = "kfold_models",
    seed: int = 42,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    early_stopping_patience: int = 15,
    label_smoothing: float = 0.1,
    num_workers: int = 4,
    dropout: float = 0.5,
):
    """
    K-Fold Cross-Validation training for more robust AUC estimates.
    Returns ensemble of models and aggregated metrics.
    """
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

    global_feature_dim = 20

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'=' * 60}")

        # Create train/val splits
        train_pairs = [(paths[i], labels[i]) for i in train_idx]
        val_pairs = [(paths[i], labels[i]) for i in val_idx]

        print(f"[FOLD {fold + 1}] Train: {len(train_pairs)}, Val: {len(val_pairs)}")

        # Create datasets
        train_ds = AneurysmHemodynamicsDataset(
            train_pairs,
            target_n=target_n,
            augment=True,
            normalize_xyz=True,
            normalize_features=True,
            add_global_features=True,
        )
        val_ds = AneurysmHemodynamicsDataset(
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
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        # Create model
        if model_type.lower() == "pointnet++":
            model = PointNetPlusPlusRupture(
                num_classes=2, in_channel=3, global_feature_dim=global_feature_dim, dropout=dropout
            )
        else:
            model = PointNetRupture(
                num_classes=2, in_channel=3, global_feature_dim=global_feature_dim, dropout=dropout
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

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        # Training loop
        best_val_auc = 0.0
        epochs_without_improvement = 0

        # CSV logging for this fold
        csv_path = os.path.join(save_dir, f"fold{fold + 1}_training_log.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_auc"])

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

            # Log to CSV
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, val_auc])

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

    # Write summary CSV with k-fold results
    summary_csv_path = os.path.join(save_dir, "kfold_summary.csv")
    with open(summary_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["fold", "val_acc", "val_auc"])
        for i, (acc, auc) in enumerate(zip(fold_accs, fold_aucs)):
            writer.writerow([i + 1, acc, auc])
        writer.writerow(["mean", np.mean(fold_accs), np.mean(fold_aucs)])
        writer.writerow(["std", np.std(fold_accs), np.std(fold_aucs)])
        writer.writerow(["overall", (all_val_probs > 0.5).mean(), overall_auc])
    print(f"\nSummary saved to: {summary_csv_path}")

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
    model_type: str = "pointnet++",
    save_path: str = "best_rupture_model.pth",
    seed: int = 42,
    use_class_weights: bool = True,
    early_stopping_patience: int = 20,
    label_smoothing: float = 0.1,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    dropout: float = 0.5,
):
    """
    Main training function with focal loss and global feature support.
    """
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

    global_feature_dim = 20

    # Create datasets
    train_ds = AneurysmHemodynamicsDataset(
        train_files,
        target_n=target_n,
        augment=True,
        normalize_xyz=True,
        normalize_features=True,
        add_global_features=True,
    )
    val_ds = AneurysmHemodynamicsDataset(
        val_files,
        target_n=target_n,
        augment=False,
        normalize_xyz=True,
        normalize_features=True,
        add_global_features=True,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Create model with global feature fusion
    if model_type.lower() == "pointnet++":
        model = PointNetPlusPlusRupture(
            num_classes=2, in_channel=3, global_feature_dim=global_feature_dim, dropout=dropout
        )
    else:
        model = PointNetRupture(
            num_classes=2, in_channel=3, global_feature_dim=global_feature_dim, dropout=dropout
        )

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

    # Early stopping tracking
    epochs_without_improvement = 0
    print(f"[INFO] Early stopping patience: {early_stopping_patience} epochs")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training loop
    best_val_auc = 0.0
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}

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

        # Log to CSV
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, val_auc])

        # Print progress
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}"
        )

        # Save best model and check early stopping
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
                print(
                    f"\n[INFO] Early stopping triggered after {epoch} epochs (no improvement for {early_stopping_patience} epochs)"
                )
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


# Inference
def predict_single(
    model: nn.Module, csv_path: str, device: torch.device, target_n: int = 4096
) -> Tuple[int, float]:
    """
    Predict rupture status for a single case.

    Args:
        model: trained model
        csv_path: path to hemodynamics_aggregate.csv
        device: torch device
        target_n: number of points to use

    Returns:
        (prediction, probability) tuple
    """
    model.eval()

    # Load data
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    pts = data[:, :3].astype(np.float32)
    feats = data[:, 3:6].astype(np.float32)

    # Sample/pad
    n = pts.shape[0]
    if n < target_n:
        pad_pts = np.zeros((target_n - n, 3), dtype=np.float32)
        pad_feats = np.zeros((target_n - n, 3), dtype=np.float32)
        pts = np.vstack([pts, pad_pts])
        feats = np.vstack([feats, pad_feats])
    elif n > target_n:
        idx = np.random.choice(n, target_n, replace=False)
        pts = pts[idx]
        feats = feats[idx]

    # Normalize
    pts = pts - np.mean(pts, axis=0)
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts = pts / max_dist

    mu = np.mean(feats, axis=0)
    sigma = np.std(feats, axis=0)
    sigma[sigma < 1e-8] = 1.0
    feats = (feats - mu) / sigma

    # Convert to tensors
    pts = torch.from_numpy(pts).float().unsqueeze(0).to(device)
    feats = torch.from_numpy(feats).float().unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits = model(pts, feats)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        prob_ruptured = probs[0, 1].item()

    return pred, prob_ruptured


# Main
def main():
    parser = argparse.ArgumentParser(description="Train PointNet++ for Aneurysm Rupture Prediction")

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
        default="pointnet++",
        choices=["pointnet", "pointnet++"],
        help="Model architecture",
    )
    parser.add_argument(
        "--save_path", type=str, default="best_rupture_model.pth", help="Path to save best model"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # New focal loss and k-fold arguments
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
        default="kfold_pointnet_models",
        help="Directory to save k-fold models",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PointNet++ Rupture Classification")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Focal Loss: {args.focal_loss} (gamma={args.focal_gamma})")
    print(f"K-Fold: {args.kfold}")
    print(f"Dropout: {args.dropout}")
    print("=" * 60)

    if args.kfold > 1:
        # K-Fold Cross-Validation
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
        )
    else:
        # Single train/val split
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
        )


if __name__ == "__main__":
    main()
