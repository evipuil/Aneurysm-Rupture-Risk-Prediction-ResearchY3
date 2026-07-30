#!/usr/bin/env python3
# Version 3 source snapshot
"""
ensemble_model.py

Ensemble model combining predictions from best-performing models:
- Geometry PointNet++ (best overall accuracy)
- Fusion Late with Clinical (best balanced performance)

Uses probability averaging and optional learned weighting.
"""

import argparse
import csv
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset


# Utils: indexing + FPS
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
    """Farthest point sampling."""
    device = xyz.device
    B, N, _ = xyz.shape
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
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn-xm)^2 + (yn-ym)^2 + (zn-zm)^2
         = sum(src**2) + sum(dst**2) - 2*src^T*dst
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Output:
        group_idx: grouped points index, [B, S, nsample]
    """
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
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    """
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], dim=-1
        )  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, new_xyz, fps_idx


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# PointNet++ Building Blocks
class PointNetSetAbstraction(nn.Module):
    """Set Abstraction layer for PointNet++."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, S, 3]
            new_points_concatenated: sample points feature data, [B, S, D']
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_points, new_xyz, _ = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )

        # new_points: (B, npoint, nsample, C+D)
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0, 2, 1)  # [B, npoint, D']
        return new_points, new_xyz


# Geometry-Only Model
class GeometryPointNetPP(nn.Module):
    """Geometry-only PointNet++ (matches geometry_pointnet.py)."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Input: xyz (3). SA1 groups xyz. in_channel = 3 (xyz) + 0 (points) = 3
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False
        )
        # SA2 input: xyz (3) + features (128) = 131
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=131, mlp=[128, 128, 256], group_all=False
        )
        # SA3 input: xyz (3) + features (256) = 259
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=259,
            mlp=[256, 512, 1024],
            group_all=True,
        )

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        xyz = batch["xyz"]  # (B, N, 3)
        B = xyz.shape[0]
        l1_pts, l1_xyz = self.sa1(xyz, None)
        l2_pts, l2_xyz = self.sa2(l1_xyz, l1_pts)
        l3_pts, _ = self.sa3(l2_xyz, l2_pts)
        x = l3_pts.view(B, 1024)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        return self.fc3(x)


# Late Fusion Model (matches fusion.py)
class LateFusionModel(nn.Module):
    """Late fusion with clinical features."""

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        global_feature_dim: int = 30,
        clinical_feature_dim: int = 2,
    ):
        super().__init__()

        # Geometry branch (xyz only)
        # SA1: in 3
        self.geo_sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False
        )
        # SA2: in 3+128=131
        self.geo_sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=131, mlp=[128, 128, 256], group_all=False
        )
        # SA3: in 3+256=259
        self.geo_sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=259,
            mlp=[256, 512, 1024],
            group_all=True,
        )

        # Hemodynamics branch (xyz + 8 features)
        # SA1: input xyz(3) + features(8) = 11?
        # Wait, forward passes features. new_points = cat(xyz_norm, features). 3 + 8 = 11.
        self.hemo_sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=11, mlp=[64, 64, 128], group_all=False
        )
        # SA2: in 3+128=131
        self.hemo_sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=131, mlp=[128, 128, 256], group_all=False
        )
        # SA3: in 3+256=259
        self.hemo_sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=259,
            mlp=[256, 512, 512],
            group_all=True,
        )

        # Clinical encoder
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

        # Fusion: geo(1024) + hemo(512) + global(128) + clinical(64) = 1728
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

    def _process_features(self, sa1, sa2, sa3, xyz, features=None):
        """Process through SA layers."""
        B = xyz.shape[0]
        pts1, xyz1 = sa1(xyz, features)
        pts2, xyz2 = sa2(xyz1, pts1)
        pts3, _ = sa3(xyz2, pts2)
        return pts3.view(B, -1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        xyz = batch["xyz"]
        features = batch["features"]
        global_features = batch["global_features"]
        clinical_features = batch["clinical_features"]

        geo_feat = self._process_features(self.geo_sa1, self.geo_sa2, self.geo_sa3, xyz)
        hemo_feat = self._process_features(
            self.hemo_sa1, self.hemo_sa2, self.hemo_sa3, xyz, features
        )
        global_enc = self.global_encoder(global_features)
        clinical_enc = self.clinical_encoder(clinical_features)

        fused = torch.cat([geo_feat, hemo_feat, global_enc, clinical_enc], dim=-1)
        return self.classifier(fused)


# Ensemble Model
class EnsembleModel(nn.Module):
    """
    Ensemble combining Geometry PointNet++ and Late Fusion models.
    Supports:
    - Simple probability averaging
    - Learned weighting
    - Stacking with meta-classifier
    """

    def __init__(
        self,
        num_classes: int = 2,
        method: str = "average",
        geo_weight: float = 0.5,
        fusion_weight: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.method = method

        # Sub-models
        self.geometry_model = GeometryPointNetPP(num_classes)
        self.fusion_model = LateFusionModel(num_classes)

        # Weights for averaging (can be learned)
        if method == "learned":
            self.geo_weight = nn.Parameter(torch.tensor(geo_weight))
            self.fusion_weight = nn.Parameter(torch.tensor(fusion_weight))
        else:
            self.register_buffer("geo_weight", torch.tensor(geo_weight))
            self.register_buffer("fusion_weight", torch.tensor(fusion_weight))

        # Meta-classifier for stacking
        if method == "stacking":
            self.meta_classifier = nn.Sequential(
                nn.Linear(num_classes * 2, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(32, num_classes),
            )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Get predictions from both models
        geo_logits = self.geometry_model(batch)
        fusion_logits = self.fusion_model(batch)

        if self.method == "average":
            # Simple probability averaging
            geo_probs = F.softmax(geo_logits, dim=-1)
            fusion_probs = F.softmax(fusion_logits, dim=-1)
            weights = F.softmax(torch.stack([self.geo_weight, self.fusion_weight]), dim=0)
            combined_probs = weights[0] * geo_probs + weights[1] * fusion_probs
            return torch.log(combined_probs + 1e-8)  # Return log-probs for cross-entropy

        elif self.method == "learned":
            # Learned weighted combination in logit space
            weights = F.softmax(torch.stack([self.geo_weight, self.fusion_weight]), dim=0)
            return weights[0] * geo_logits + weights[1] * fusion_logits

        elif self.method == "stacking":
            # Meta-classifier combines predictions
            combined = torch.cat([geo_logits, fusion_logits], dim=-1)
            return self.meta_classifier(combined)

        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def load_pretrained(self, geo_checkpoint: str, fusion_checkpoint: str):
        """Load pretrained weights for sub-models."""
        if geo_checkpoint and os.path.exists(geo_checkpoint):
            geo_state = torch.load(geo_checkpoint, map_location="cpu")
            # Handle different checkpoint formats
            if "model_state_dict" in geo_state:
                geo_state = geo_state["model_state_dict"]
            self.geometry_model.load_state_dict(geo_state, strict=False)
            print(f"[INFO] Loaded geometry model from {geo_checkpoint}")

        if fusion_checkpoint and os.path.exists(fusion_checkpoint):
            fusion_state = torch.load(fusion_checkpoint, map_location="cpu")
            if "model_state_dict" in fusion_state:
                fusion_state = fusion_state["model_state_dict"]
            self.fusion_model.load_state_dict(fusion_state, strict=False)
            print(f"[INFO] Loaded fusion model from {fusion_checkpoint}")


# Dataset
class EnsembleDataset(Dataset):
    """Dataset for ensemble model - provides both xyz and hemodynamic features."""

    def __init__(
        self,
        files_labels: List[Tuple[str, int]],
        metadata_df,
        hemo_dir: str,
        target_n: int = 8192,
        augment: bool = False,
    ):
        self.files_labels = files_labels
        self.metadata = metadata_df
        self.hemo_dir = hemo_dir
        self.target_n = target_n
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files_labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        geo_path, label = self.files_labels[idx]

        # Extract case name from path
        case_name = os.path.splitext(os.path.basename(geo_path))[0]

        # Load geometry (xyz)
        xyz = self._load_geometry(geo_path)

        # Load hemodynamic features
        features, global_features = self._load_hemodynamics(case_name)

        # Get clinical features
        clinical = self._get_clinical(case_name)

        # Augmentation
        if self.augment:
            xyz = self._augment_xyz(xyz)

        return {
            "xyz": torch.from_numpy(xyz).float(),
            "features": torch.from_numpy(features).float(),
            "global_features": torch.from_numpy(global_features).float(),
            "clinical_features": torch.tensor(clinical, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "path": geo_path,
        }

    def _load_geometry(self, path: str) -> np.ndarray:
        """Load and normalize geometry."""
        pts = np.loadtxt(path).astype(np.float32)
        pts = self._resample(pts, self.target_n)
        pts = pts - np.mean(pts, axis=0)
        max_dist = np.max(np.linalg.norm(pts, axis=1))
        if max_dist > 0:
            pts = pts / max_dist
        return pts

    def _load_hemodynamics(self, case_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load hemodynamic features from CSV."""
        csv_path = os.path.join(self.hemo_dir, case_name, "hemodynamics_aggregate.csv")

        if not os.path.exists(csv_path):
            # Return zeros if not found
            return (np.zeros((self.target_n, 8), dtype=np.float32), np.zeros(30, dtype=np.float32))

        try:
            df = pd.read_csv(csv_path)
            pts = df[["x", "y", "z"]].values.astype(np.float32)

            # Helper to handle case-sensitivity
            def get_col(df, candidates):
                for c in candidates:
                    if c in df.columns:
                        return df[c].values.astype(np.float32).reshape(-1, 1)
                raise KeyError(f"Columns not found: {candidates}")

            tawss = get_col(df, ["tawss", "TAWSS"])
            osi = get_col(df, ["osi", "OSI"])
            von_mises = get_col(df, ["von_mises", "von_Mises", "VonMises"])

            # Compute derived features
            tawss_median = np.median(tawss)
            low_tawss = (tawss < tawss_median).astype(np.float32)
            high_osi = (osi > 0.2).astype(np.float32)
            combined_stress = tawss * (1 - 2 * osi)
            vm_normalized = von_mises / (np.max(von_mises) + 1e-8)
            risk_score = high_osi * low_tawss

            # Combine features (8 total)
            raw_feats = np.hstack(
                [
                    tawss,
                    osi,
                    von_mises,
                    low_tawss,
                    high_osi,
                    combined_stress,
                    vm_normalized,
                    risk_score,
                ]
            )

            # Resample to target_n
            indices = self._get_resample_indices(len(pts), self.target_n)
            pts = pts[indices]
            feats = raw_feats[indices]

            # Normalize
            feats = self._normalize_features(feats)

            # Compute global features
            global_feats = self._compute_global_features(raw_feats, pts)

            return feats, global_feats

        except Exception as e:
            print(f"[WARN] Error loading hemodynamics for {case_name}: {e}")
            return (np.zeros((self.target_n, 8), dtype=np.float32), np.zeros(30, dtype=np.float32))

    def _get_clinical(self, case_name: str) -> List[float]:
        """Get clinical features (age, sex)."""
        try:
            row = self.metadata[self.metadata["case_name"] == case_name]
            if len(row) == 0:
                return [0.0, 0.0]
            age = float(row["age"].values[0]) / 100.0  # Normalize age
            sex = float(row["sex"].values[0])
            return [age, sex]
        except Exception:
            return [0.0, 0.0]

    def _resample(self, pts: np.ndarray, n: int) -> np.ndarray:
        """Resample points to target count."""
        current = len(pts)
        if current == n:
            return pts
        elif current > n:
            indices = np.random.choice(current, n, replace=False)
            return pts[indices]
        else:
            pad_size = n - current
            indices = np.random.choice(current, pad_size, replace=True)
            return np.vstack([pts, pts[indices]])

    def _get_resample_indices(self, current: int, target: int) -> np.ndarray:
        """Get indices for resampling."""
        if current == target:
            return np.arange(current)
        elif current > target:
            return np.random.choice(current, target, replace=False)
        else:
            pad_indices = np.random.choice(current, target - current, replace=True)
            return np.concatenate([np.arange(current), pad_indices])

    def _normalize_features(self, feats: np.ndarray) -> np.ndarray:
        """Normalize features."""
        mu = np.mean(feats, axis=0)
        sigma = np.std(feats, axis=0)
        sigma[sigma < 1e-8] = 1.0
        feats = (feats - mu) / sigma
        feats = np.clip(feats, -3.0, 3.0)
        return feats.astype(np.float32)

    def _compute_global_features(self, raw_feats: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Compute global summary statistics (30 features)."""
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

        # Derived feature statistics (7)
        tawss_median = np.median(tawss)
        low_tawss = (tawss < tawss_median).astype(np.float32)
        high_osi = (osi > 0.2).astype(np.float32)
        combined_stress = tawss * (1 - 2 * osi)
        vm_normalized = von_mises / (np.max(von_mises) + 1e-8)
        risk_score = high_osi * low_tawss

        features.extend(
            [
                np.mean(low_tawss),
                np.mean(high_osi),
                np.mean(combined_stress),
                np.std(combined_stress),
                np.mean(vm_normalized),
                np.std(vm_normalized),
                np.mean(risk_score),
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

    def _augment_xyz(self, pts: np.ndarray) -> np.ndarray:
        """Apply SO(3) rotation and jitter."""
        pts = self._so3_rotate(pts)
        pts = pts + np.clip(0.01 * np.random.randn(*pts.shape), -0.05, 0.05)
        pts = pts * np.random.uniform(0.95, 1.05)
        return pts.astype(np.float32)

    @staticmethod
    def _so3_rotate(points: np.ndarray) -> np.ndarray:
        """Apply random SO(3) rotation."""
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


# Training Functions
def evaluate(
    model: nn.Module, dataloader: DataLoader, device: torch.device, criterion: nn.Module
) -> Tuple[float, float, float, List, List, List]:
    """Evaluate model on dataloader."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            labels = batch["label"]

            logits = model(batch)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)

            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5

    return loss_sum / total, correct / total, auc, all_labels, all_preds, all_probs


def train_kfold(
    geo_dir: str = "data",
    hemo_dir: str = "predictions/pinn_corrected",
    metadata_path: str = "metadata.csv",
    n_folds: int = 5,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    target_n: int = 8192,
    ensemble_method: str = "average",
    geo_checkpoint: str = None,
    fusion_checkpoint: str = None,
    freeze_base: bool = False,
    output_dir: str = "training_logs/ensemble",
):
    """Train ensemble model with k-fold cross-validation."""

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    print(f"[INFO] Loaded metadata with {len(metadata_df)} entries")

    # Gather files
    un_list = sorted(glob.glob(os.path.join(geo_dir, "unruptured", "*.txt")))
    ru_list = sorted(glob.glob(os.path.join(geo_dir, "ruptured", "*.txt")))

    if len(un_list) == 0 or len(ru_list) == 0:
        raise RuntimeError(f"No files found in {geo_dir}")

    # Balance classes
    min_count = min(len(un_list), len(ru_list))
    un_list = un_list[:min_count]
    ru_list = ru_list[:min_count]
    print(f"[INFO] Balanced: {min_count} per class, {2 * min_count} total")

    all_files = [(p, 0) for p in un_list] + [(p, 1) for p in ru_list]
    all_labels = [label for _, label in all_files]

    # K-fold CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_val_labels = []
    all_val_probs = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_files, all_labels)):
        print(f"\n{'=' * 50}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'=' * 50}")

        train_files = [all_files[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]

        # Datasets
        train_dataset = EnsembleDataset(
            train_files, metadata_df, hemo_dir, target_n=target_n, augment=True
        )
        val_dataset = EnsembleDataset(
            val_files, metadata_df, hemo_dir, target_n=target_n, augment=False
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Model
        model = EnsembleModel(num_classes=2, method=ensemble_method)

        # Load pretrained weights if provided
        if geo_checkpoint or fusion_checkpoint:
            model.load_pretrained(geo_checkpoint, fusion_checkpoint)

        # Optionally freeze base models
        if freeze_base:
            for param in model.geometry_model.parameters():
                param.requires_grad = False
            for param in model.fusion_model.parameters():
                param.requires_grad = False
            print("[INFO] Base models frozen - only training ensemble weights")

        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_auc = 0.0
        best_state = None

        # CSV logging
        csv_path = os.path.join(output_dir, f"fold{fold_idx + 1}_training_log.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["epoch", "train_loss", "train_acc", "train_auc", "val_loss", "val_acc", "val_auc"]
            )

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_labels_all = []
            train_probs_all = []

            for batch in train_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                }
                labels = batch["label"]

                optimizer.zero_grad()
                logits = model(batch)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * labels.size(0)
                train_correct += (logits.argmax(1) == labels).sum().item()
                train_total += labels.size(0)

                probs = F.softmax(logits, dim=1)[:, 1]
                train_labels_all.extend(labels.cpu().numpy())
                train_probs_all.extend(probs.detach().cpu().numpy())

            scheduler.step()

            try:
                train_auc = roc_auc_score(train_labels_all, train_probs_all)
            except Exception:
                train_auc = 0.5

            train_acc_ep = train_correct / train_total if train_total > 0 else 0
            train_loss_ep = train_loss / train_total if train_total > 0 else 0

            # Validation
            val_loss, val_acc, val_auc, val_labels, val_preds, val_probs = evaluate(
                model, val_loader, device, criterion
            )

            # Log to CSV
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [epoch + 1, train_loss_ep, train_acc_ep, train_auc, val_loss, val_acc, val_auc]
                )

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1:3d}/{epochs}: "
                    f"Train Loss={train_loss_ep:.4f}, Acc={train_acc_ep:.4f}, AUC={train_auc:.4f} | "
                    f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, AUC={val_auc:.4f}"
                )

        # Final evaluation with best model
        if best_state is not None:
            model.load_state_dict(best_state)

        _, final_acc, final_auc, final_labels, final_preds, final_probs = evaluate(
            model, val_loader, device, criterion
        )

        fold_results.append({"fold": fold_idx + 1, "accuracy": final_acc, "auc": final_auc})

        all_val_labels.extend(final_labels)
        all_val_probs.extend(final_probs)

        # Save fold model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "fold": fold_idx + 1,
                "accuracy": final_acc,
                "auc": final_auc,
            },
            os.path.join(output_dir, f"ensemble_fold{fold_idx + 1}.pt"),
        )

        print(f"\nFold {fold_idx + 1} Results: Accuracy={final_acc:.4f}, AUC={final_auc:.4f}")
        print(
            classification_report(
                final_labels, final_preds, target_names=["Unruptured", "Ruptured"]
            )
        )

    # Overall results
    print(f"\n{'=' * 50}")
    print("OVERALL RESULTS")
    print(f"{'=' * 50}")

    mean_acc = np.mean([r["accuracy"] for r in fold_results])
    std_acc = np.std([r["accuracy"] for r in fold_results])
    mean_auc = np.mean([r["auc"] for r in fold_results])
    std_auc = np.std([r["auc"] for r in fold_results])

    try:
        overall_auc = roc_auc_score(all_val_labels, all_val_probs)
    except Exception:
        overall_auc = mean_auc

    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Overall AUC: {overall_auc:.4f}")

    # Save results
    with open(os.path.join(output_dir, "ensemble_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "accuracy", "auc"])
        writer.writeheader()
        writer.writerows(fold_results)
        writer.writerow({"fold": "Mean", "accuracy": mean_acc, "auc": mean_auc})
        writer.writerow({"fold": "Std", "accuracy": std_acc, "auc": std_auc})
        writer.writerow({"fold": "Overall", "accuracy": "", "auc": overall_auc})

    return fold_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Model Training")
    parser.add_argument("--geo_dir", type=str, default="data", help="Geometry data directory")
    parser.add_argument(
        "--hemo_dir",
        type=str,
        default="predictions/pinn_corrected",
        help="Hemodynamics data directory",
    )
    parser.add_argument("--metadata", type=str, default="metadata.csv", help="Metadata CSV path")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--target_n", type=int, default=8192, help="Target number of points")
    parser.add_argument(
        "--method",
        type=str,
        default="average",
        choices=["average", "learned", "stacking"],
        help="Ensemble method",
    )
    parser.add_argument(
        "--geo_checkpoint", type=str, default=None, help="Path to pretrained geometry model"
    )
    parser.add_argument(
        "--fusion_checkpoint", type=str, default=None, help="Path to pretrained fusion model"
    )
    parser.add_argument(
        "--freeze_base",
        action="store_true",
        help="Freeze base models (fine-tune only ensemble weights)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="training_logs/ensemble", help="Output directory"
    )

    args = parser.parse_args()

    train_kfold(
        geo_dir=args.geo_dir,
        hemo_dir=args.hemo_dir,
        metadata_path=args.metadata,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        target_n=args.target_n,
        ensemble_method=args.method,
        geo_checkpoint=args.geo_checkpoint,
        fusion_checkpoint=args.fusion_checkpoint,
        freeze_base=args.freeze_base,
        output_dir=args.output_dir,
    )
