#!/usr/bin/env python3
# Version 3 source snapshot
"""
comprehensive_rupture_model.py

Comprehensive PointNet++ model for aneurysm rupture classification that fuses
ALL available data sources:

Point-cloud features (per-point, from case folder):
  1. Geometry (x, y, z)
  2. Time-averaged hemodynamics: TAWSS, OSI, von Mises  (hemodynamics_aggregate.csv)
  3. Per-timestep velocity (u, v, w) averaged over time   (timesteps/flow_t*.csv)
  4. Per-timestep pressure (p) averaged over time          (when available)
  5. Per-timestep WSS vectors (wss_x, wss_y, wss_z) averaged over time (timesteps/wss_t*.csv)
  6. Derived: average velocity magnitude, average pressure, average WSS magnitude

Global (scalar) features per case:
  - Statistical summaries of all hemodynamic fields
  - Geometric shape descriptors (eigenvalues, aspect ratio)

Clinical features:
  - Age (z-score normalised)
  - Biological sex (one-hot: female / male)
  - Aneurysm location (one-hot encoded, 21 categories)

Directory layout expected (same as optimized_pinn output):
  data_dir/
    <case_name>/
      hemodynamics_aggregate.csv          # x,y,z,TAWSS,OSI,von_Mises  (case-insensitive)
      timesteps/
        flow_t00.csv ... flow_t09.csv     # x,y,z,[p,]u,v,w[,velocity_magnitude]
        wss_t00.csv  ... wss_t09.csv      # x,y,z,wss_x,wss_y,wss_z,wss_magnitude
        time_index.csv
      pinn_model.pt                       # (unused here)

  metadata.csv   # source,dataset,...,status,location,side,sex,age,...

Usage:
  # K-fold cross-validation (recommended)
  python comprehensive_rupture_model.py \\
      --data_dir predictions/optimized_pinn \\
      --metadata metadata.csv \\
      --kfold 5 --epochs 200 --batch_size 8

  # Single split
  python comprehensive_rupture_model.py \\
      --data_dir predictions/optimized_pinn \\
      --metadata metadata.csv \\
      --epochs 200 --batch_size 8 \\
      --save_path training_logs/comprehensive_best.pth

"""

import argparse
import csv
import glob
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

# Reproducibility
SEED = 42
# Known aneurysm locations (from metadata.csv) – used for one-hot
KNOWN_LOCATIONS: List[str] = [
    "ACA A1",
    "ACA dist",
    "AComA",
    "BA",
    "BA tip",
    "ICA bif",
    "ICA cav",
    "ICA chor",
    "ICA hypo",
    "ICA oph",
    "ICA para",
    "ICA pcom",
    "MCA M1",
    "MCA M2",
    "MCA M3",
    "MCA bif",
    "PCA P1-P2",
    "PICA",
    "PeriA",
    "SCA",
    "VA V4",
]
LOCATION_TO_IDX: Dict[str, int] = {loc: i for i, loc in enumerate(KNOWN_LOCATIONS)}
NUM_LOCATIONS = len(KNOWN_LOCATIONS)

# Clinical feature dimension: age(1) + sex(1) + location_one_hot(21) = 23
CLINICAL_DIM = 1 + 1 + NUM_LOCATIONS  # 23


# Loss – Focal Loss for class imbalance
class FocalLoss(nn.Module):
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
        ce = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        fl = (1 - pt) ** self.gamma * ce
        return fl.mean() if self.reduction == "mean" else fl.sum()


# Point-cloud utilities (shared across models in this repo)
def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
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
    new_xyz = torch.zeros(B, 1, C, device=device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# PointNet++ Set Abstraction
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
            new_points, new_xyz, _ = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        # new_points: (B, npoint, nsample, C)
        new_points = new_points.permute(0, 3, 2, 1)  # (B, C, nsample, npoint)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]  # (B, C', npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, C')
        return new_points, new_xyz


# Dataset – loads & fuses all available data for each case
# Number of per-point feature channels (excluding xyz):
#   tawss, osi, von_mises                                = 3
#   avg velocity (u_avg, v_avg, w_avg, vel_mag)           = 4
#   avg pressure (p_avg)                                  = 1  (0 if unavailable)
#   avg wss (wss_x_avg, wss_y_avg, wss_z_avg, wss_mag)   = 4  (0 if unavailable; only on wall pts → interpolated)
#   derived: low_tawss, high_osi, combined_stress,
#            vm_normalised, risk_score                     = 5
#   ──────────────────────────────────────────────────────
#   total                                                  = 17

NUM_POINT_FEATURES = 17
NUM_GLOBAL_FEATURES = 45  # statistical summaries (see _compute_global_features)


def _safe_load_csv(path: str) -> Tuple[List[str], np.ndarray]:
    """Load a CSV robustly (handles both scientific-notation and decimal)."""
    try:
        import pandas as pd

        df = pd.read_csv(path)
        # Strip any 'time=...' suffix from column names
        cols = [re.sub(r"[,=].*$", "", c.strip().split("=")[0].strip()) for c in df.columns]
        df.columns = cols
        return [c.lower() for c in cols], df.values.astype(np.float32)
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline()
        raw_cols = [c.strip() for c in header.split(",") if c.strip()]
        cols = [re.sub(r"=.*$", "", c).strip().lower() for c in raw_cols]
        data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
        return cols, data


def _col_index(cols: List[str], name: str) -> Optional[int]:
    """Case-insensitive column lookup."""
    name_l = name.lower()
    for i, c in enumerate(cols):
        if c.lower() == name_l:
            return i
    return None


class ComprehensiveDataset(Dataset):
    """
    Loads per-case:
      - hemodynamics_aggregate.csv  → tawss, osi, von_mises (on surface points)
      - timesteps/flow_t*.csv       → u, v, w, [p]  (on volumetric points, averaged over timesteps)
      - timesteps/wss_t*.csv        → wss_x, wss_y, wss_z (on wall points, averaged over timesteps)

    All arrays are resampled / interpolated onto a common point set of
    `target_n` points (from the hemodynamics file, which sits on the surface mesh).
    """

    def __init__(
        self,
        samples: List[Dict],
        target_n: int = 8192,
        augment: bool = False,
        normalize_xyz: bool = True,
        normalize_features: bool = True,
    ):
        """
        Args:
            samples: list of dicts with keys:
                'case_dir': str    – path to case folder
                'label': int       – 0 / 1
                'age': float       – normalised age
                'sex': int         – 0 female, 1 male
                'location_idx': int – index in KNOWN_LOCATIONS (-1 = unknown)
        """
        self.samples = samples
        self.target_n = target_n
        self.augment = augment
        self.normalize_xyz = normalize_xyz
        self.normalize_features = normalize_features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        case_dir = s["case_dir"]
        label = s["label"]

        # ---- 1. Load hemodynamics aggregate (primary point set) ------
        hemo_path = os.path.join(case_dir, "hemodynamics_aggregate.csv")
        try:
            hemo_cols, hemo_data = _safe_load_csv(hemo_path)
        except Exception as e:
            print(f"[WARN] Cannot load {hemo_path}: {e}")
            return self._dummy(s)

        if hemo_data.ndim == 1:
            hemo_data = hemo_data.reshape(1, -1)

        # xyz from hemodynamics (the reference point set)
        pts = hemo_data[:, :3].astype(np.float32)
        n_pts = pts.shape[0]

        # Extract hemodynamic scalars (case-insensitive)
        tawss_i = _col_index(hemo_cols, "tawss")
        osi_i = _col_index(hemo_cols, "osi")
        vm_i = _col_index(hemo_cols, "von_mises") or _col_index(hemo_cols, "von_Mises")
        tawss = hemo_data[:, tawss_i] if tawss_i is not None else np.zeros(n_pts, dtype=np.float32)
        osi = hemo_data[:, osi_i] if osi_i is not None else np.zeros(n_pts, dtype=np.float32)
        von_mises = hemo_data[:, vm_i] if vm_i is not None else np.zeros(n_pts, dtype=np.float32)

        # ---- 2. Load timestep velocity & pressure (average over t) ----
        ts_dir = os.path.join(case_dir, "timesteps")
        flow_files = sorted(glob.glob(os.path.join(ts_dir, "flow_t*.csv")))

        avg_u = np.zeros(n_pts, dtype=np.float32)
        avg_v = np.zeros(n_pts, dtype=np.float32)
        avg_w = np.zeros(n_pts, dtype=np.float32)
        avg_p = np.zeros(n_pts, dtype=np.float32)
        has_pressure = False

        if flow_files:
            u_acc, v_acc, w_acc, p_acc = [], [], [], []
            ref_xyz = None
            for ff in flow_files:
                try:
                    fcols, fdata = _safe_load_csv(ff)
                except Exception:
                    continue
                if fdata.ndim == 1:
                    fdata = fdata.reshape(1, -1)
                u_i = _col_index(fcols, "u")
                v_i = _col_index(fcols, "v")
                w_i = _col_index(fcols, "w")
                p_i = _col_index(fcols, "p")
                if u_i is None or v_i is None or w_i is None:
                    continue
                u_acc.append(fdata[:, u_i])
                v_acc.append(fdata[:, v_i])
                w_acc.append(fdata[:, w_i])
                if p_i is not None:
                    has_pressure = True
                    p_acc.append(fdata[:, p_i])
                if ref_xyz is None:
                    ref_xyz = fdata[:, :3]

            if u_acc:
                # Average across timesteps
                u_mean = np.mean(np.stack(u_acc, axis=0), axis=0).astype(np.float32)
                v_mean = np.mean(np.stack(v_acc, axis=0), axis=0).astype(np.float32)
                w_mean = np.mean(np.stack(w_acc, axis=0), axis=0).astype(np.float32)
                p_mean = (
                    np.mean(np.stack(p_acc, axis=0), axis=0).astype(np.float32) if p_acc else None
                )

                # The flow point set may differ from hemodynamics → interpolate via nearest-neighbor
                if ref_xyz is not None and ref_xyz.shape[0] != n_pts:
                    from scipy.spatial import cKDTree

                    tree = cKDTree(ref_xyz[:, :3])
                    _, nn_idx = tree.query(pts, k=1)
                    avg_u = u_mean[nn_idx]
                    avg_v = v_mean[nn_idx]
                    avg_w = w_mean[nn_idx]
                    if p_mean is not None:
                        avg_p = p_mean[nn_idx]
                else:
                    avg_u = (
                        u_mean[:n_pts]
                        if len(u_mean) >= n_pts
                        else np.pad(u_mean, (0, n_pts - len(u_mean)))
                    )
                    avg_v = (
                        v_mean[:n_pts]
                        if len(v_mean) >= n_pts
                        else np.pad(v_mean, (0, n_pts - len(v_mean)))
                    )
                    avg_w = (
                        w_mean[:n_pts]
                        if len(w_mean) >= n_pts
                        else np.pad(w_mean, (0, n_pts - len(w_mean)))
                    )
                    if p_mean is not None:
                        avg_p = (
                            p_mean[:n_pts]
                            if len(p_mean) >= n_pts
                            else np.pad(p_mean, (0, n_pts - len(p_mean)))
                        )

        vel_mag = np.sqrt(avg_u**2 + avg_v**2 + avg_w**2)

        # ---- 3. Load timestep WSS (average over t) -------------------
        wss_files = sorted(glob.glob(os.path.join(ts_dir, "wss_t*.csv")))
        avg_wss_x = np.zeros(n_pts, dtype=np.float32)
        avg_wss_y = np.zeros(n_pts, dtype=np.float32)
        avg_wss_z = np.zeros(n_pts, dtype=np.float32)
        avg_wss_mag = np.zeros(n_pts, dtype=np.float32)

        if wss_files:
            wx_acc, wy_acc, wz_acc = [], [], []
            wss_ref_xyz = None
            for wf in wss_files:
                try:
                    wcols, wdata = _safe_load_csv(wf)
                except Exception:
                    continue
                if wdata.ndim == 1:
                    wdata = wdata.reshape(1, -1)
                wxi = _col_index(wcols, "wss_x")
                wyi = _col_index(wcols, "wss_y")
                wzi = _col_index(wcols, "wss_z")
                if wxi is None or wyi is None or wzi is None:
                    continue
                wx_acc.append(wdata[:, wxi])
                wy_acc.append(wdata[:, wyi])
                wz_acc.append(wdata[:, wzi])
                if wss_ref_xyz is None:
                    wss_ref_xyz = wdata[:, :3]

            if wx_acc:
                wx_mean = np.mean(np.stack(wx_acc, axis=0), axis=0).astype(np.float32)
                wy_mean = np.mean(np.stack(wy_acc, axis=0), axis=0).astype(np.float32)
                wz_mean = np.mean(np.stack(wz_acc, axis=0), axis=0).astype(np.float32)

                # WSS is on wall points (different count) → NN interpolation to hemo points
                if wss_ref_xyz is not None and wss_ref_xyz.shape[0] != n_pts:
                    from scipy.spatial import cKDTree

                    tree = cKDTree(wss_ref_xyz[:, :3])
                    _, nn_idx = tree.query(pts, k=1)
                    avg_wss_x = wx_mean[nn_idx]
                    avg_wss_y = wy_mean[nn_idx]
                    avg_wss_z = wz_mean[nn_idx]
                else:
                    avg_wss_x = (
                        wx_mean[:n_pts]
                        if len(wx_mean) >= n_pts
                        else np.pad(wx_mean, (0, n_pts - len(wx_mean)))
                    )
                    avg_wss_y = (
                        wy_mean[:n_pts]
                        if len(wy_mean) >= n_pts
                        else np.pad(wy_mean, (0, n_pts - len(wy_mean)))
                    )
                    avg_wss_z = (
                        wz_mean[:n_pts]
                        if len(wz_mean) >= n_pts
                        else np.pad(wz_mean, (0, n_pts - len(wz_mean)))
                    )
                avg_wss_mag = np.sqrt(avg_wss_x**2 + avg_wss_y**2 + avg_wss_z**2)

        # ---- 4. Derived per-point features ---------------------------
        tawss_median = np.median(tawss) if np.any(tawss) else 0.0
        low_tawss = (tawss < tawss_median).astype(np.float32)
        high_osi = (osi > 0.2).astype(np.float32)
        combined_stress = tawss * (1 - 2 * osi)
        vm_max = np.max(von_mises) if np.any(von_mises) else 1e-8
        vm_normalised = von_mises / (vm_max + 1e-8)
        risk_score = high_osi * low_tawss

        # ---- 5. Stack all per-point features (17 channels) -----------
        feats = np.stack(
            [
                tawss,
                osi,
                von_mises,  # 3  hemodynamics
                avg_u,
                avg_v,
                avg_w,
                vel_mag,  # 4  velocity
                avg_p,  # 1  pressure
                avg_wss_x,
                avg_wss_y,
                avg_wss_z,
                avg_wss_mag,  # 4  wss vectors
                low_tawss,
                high_osi,
                combined_stress,  # 3  derived
                vm_normalised,
                risk_score,  # 2  derived
            ],
            axis=-1,
        ).astype(np.float32)  # (N, 17)

        # ---- 6. Global features (statistical summary) ----------------
        global_feats = self._compute_global_features(
            tawss,
            osi,
            von_mises,
            avg_u,
            avg_v,
            avg_w,
            vel_mag,
            avg_p,
            avg_wss_mag,
            pts,
            has_pressure,
        )

        # ---- 7. Sample / pad to target_n points ---------------------
        if n_pts < self.target_n:
            pad_idx = np.random.choice(n_pts, self.target_n - n_pts, replace=True)
            pts = np.vstack([pts, pts[pad_idx]])
            feats = np.vstack([feats, feats[pad_idx]])
        elif n_pts > self.target_n:
            sel = np.random.choice(n_pts, self.target_n, replace=False)
            pts = pts[sel]
            feats = feats[sel]

        # ---- 8. Normalise -------------------------------------------
        if self.normalize_xyz:
            centroid = np.mean(pts, axis=0)
            pts = pts - centroid
            max_dist = np.max(np.linalg.norm(pts, axis=1))
            if max_dist > 0:
                pts = pts / max_dist

        if self.normalize_features:
            mu = np.mean(feats, axis=0)
            sigma = np.std(feats, axis=0)
            sigma[sigma < 1e-8] = 1.0
            feats = np.clip((feats - mu) / sigma, -5.0, 5.0).astype(np.float32)

        # ---- 9. Augmentation ----------------------------------------
        if self.augment:
            pts = self._so3_rotate(pts)
            pts = self._jitter(pts)
            pts = pts * np.random.uniform(0.95, 1.05)

        # ---- 10. Clinical features -----------------------------------
        loc_onehot = np.zeros(NUM_LOCATIONS, dtype=np.float32)
        loc_idx = s["location_idx"]
        if 0 <= loc_idx < NUM_LOCATIONS:
            loc_onehot[loc_idx] = 1.0
        clinical = np.concatenate(
            [
                np.array([s["age"], float(s["sex"])], dtype=np.float32),
                loc_onehot,
            ]
        )  # (23,)

        return {
            "xyz": torch.from_numpy(pts).float(),
            "features": torch.from_numpy(feats).float(),
            "global_features": torch.from_numpy(global_feats).float(),
            "clinical": torch.from_numpy(clinical).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "path": case_dir,
        }

    def _dummy(self, s):
        loc_onehot = np.zeros(NUM_LOCATIONS, dtype=np.float32)
        loc_idx = s["location_idx"]
        if 0 <= loc_idx < NUM_LOCATIONS:
            loc_onehot[loc_idx] = 1.0
        clinical = np.concatenate(
            [
                np.array([s["age"], float(s["sex"])], dtype=np.float32),
                loc_onehot,
            ]
        )
        return {
            "xyz": torch.zeros(self.target_n, 3),
            "features": torch.zeros(self.target_n, NUM_POINT_FEATURES),
            "global_features": torch.zeros(NUM_GLOBAL_FEATURES),
            "clinical": torch.from_numpy(clinical).float(),
            "label": torch.tensor(s["label"], dtype=torch.long),
            "path": s["case_dir"],
        }

    @staticmethod
    def _compute_global_features(
        tawss,
        osi,
        von_mises,
        avg_u,
        avg_v,
        avg_w,
        vel_mag,
        avg_p,
        avg_wss_mag,
        pts,
        has_pressure,
    ) -> np.ndarray:
        """45-dimensional statistical summary vector."""
        feats = []

        def _stats5(arr):
            """mean, std, max, p5, p95"""
            if len(arr) == 0 or np.all(arr == 0):
                return [0.0] * 5
            return [
                float(np.mean(arr)),
                float(np.std(arr)),
                float(np.max(arr)),
                float(np.percentile(arr, 5)),
                float(np.percentile(arr, 95)),
            ]

        # TAWSS (5) + fraction of low-TAWSS (1) + fraction high-TAWSS (1) = 7
        feats.extend(_stats5(tawss))
        tawss_med = np.median(tawss) if np.any(tawss) else 0
        feats.append(float(np.mean(tawss < tawss_med)))
        feats.append(float(np.mean(tawss > np.percentile(tawss, 90))) if np.any(tawss) else 0.0)

        # OSI (5) + fraction high-OSI (1) = 6
        feats.extend(_stats5(osi))
        feats.append(float(np.mean(osi > 0.2)))

        # Von Mises (5)
        feats.extend(_stats5(von_mises))

        # Velocity magnitude (5)
        feats.extend(_stats5(vel_mag))

        # Pressure (5) – zeros if not available
        feats.extend(_stats5(avg_p))

        # WSS magnitude (5)
        feats.extend(_stats5(avg_wss_mag))

        # Derived cross-feature (3)
        combined_stress = tawss * (1 - 2 * osi)
        feats.extend(
            [
                float(np.mean(combined_stress)),
                float(np.std(combined_stress)),
                float(np.mean((osi > 0.2) & (tawss < tawss_med))),
            ]
        )

        # Geometric shape descriptors (6)
        centroid = np.mean(pts, axis=0)
        centered = pts - centroid
        dists = np.linalg.norm(centered, axis=1)
        try:
            cov = np.cov(pts.T)
            evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            evals = evals / (evals.sum() + 1e-8)
        except Exception:
            evals = np.array([0.5, 0.3, 0.2])
        feats.extend(
            [
                float(np.max(dists)),
                float(np.std(dists)),
                float(np.max(dists) / (np.mean(dists) + 1e-6)),
                float(evals[0]),
                float(evals[1]),
                float(evals[0] / (evals[2] + 1e-6)),
            ]
        )

        # has_pressure flag (1)
        feats.append(1.0 if has_pressure else 0.0)

        arr = np.array(feats, dtype=np.float32)  # 7+6+5+5+5+5+3+6+1 = 43 … pad to 45
        if len(arr) < NUM_GLOBAL_FEATURES:
            arr = np.pad(arr, (0, NUM_GLOBAL_FEATURES - len(arr)))
        return np.sign(arr) * np.log1p(np.abs(arr))

    @staticmethod
    def _so3_rotate(points):
        rx, ry, rz = [np.random.uniform(0, 2 * np.pi) for _ in range(3)]
        Rx = np.array(
            [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float32
        )
        Ry = np.array(
            [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32
        )
        Rz = np.array(
            [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float32
        )
        return (points @ (Rz @ Ry @ Rx).T).astype(np.float32)

    @staticmethod
    def _jitter(points, sigma=0.01, clip=0.05):
        return (points + np.clip(sigma * np.random.randn(*points.shape), -clip, clip)).astype(
            np.float32
        )


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "xyz": torch.stack([b["xyz"] for b in batch]),
        "features": torch.stack([b["features"] for b in batch]),
        "global_features": torch.stack([b["global_features"] for b in batch]),
        "clinical": torch.stack([b["clinical"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "path": [b["path"] for b in batch],
    }


# Model
class ComprehensiveRuptureModel(nn.Module):
    """
    Three-branch PointNet++ with clinical-feature gating:
      Branch A – Geometry only        (xyz, 3 channels)  → 1024-d
      Branch B – Full hemodynamics     (xyz + 17 features → 20 in_channel) → 512-d
      Branch C – Clinical + global     (23 + 45 = 68)     → 128-d
    Fusion: concat → gated weighting → classifier
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        feat_dim: int = NUM_POINT_FEATURES,
        global_dim: int = NUM_GLOBAL_FEATURES,
        clinical_dim: int = CLINICAL_DIM,
    ):
        super().__init__()

        # ---- Branch A: geometry (xyz only) ---------------------------
        self.geo_sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128]
        )
        self.geo_sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256]
        )
        self.geo_sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )

        # ---- Branch B: hemodynamics (xyz + feat_dim features) --------
        hemo_in = feat_dim + 3  # grouped_xyz_norm(3) + feat_dim
        self.hemo_sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=hemo_in, mlp=[64, 64, 128]
        )
        self.hemo_sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256]
        )
        self.hemo_sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 512],
            group_all=True,
        )

        # ---- Branch C: clinical + global features --------------------
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim + global_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # ---- Cross-attention over the three branch embeddings --------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=512, num_heads=4, dropout=dropout, batch_first=True
        )

        # Project geometry to 512 for attention
        self.geo_proj = nn.Linear(1024, 512)
        # Project clinical to 512 for attention
        self.clin_proj = nn.Linear(128, 512)

        # ---- Gating --------------------------------------------------
        gate_in = 512 * 3  # three branches after attention
        self.gate = nn.Sequential(
            nn.Linear(gate_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
            nn.Softmax(dim=-1),
        )

        # ---- Classifier (after gated fusion → 512) -------------------
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        xyz = batch["xyz"]  # (B, N, 3)
        feats = batch["features"]  # (B, N, 17)
        gf = batch["global_features"]  # (B, 45)
        clin = batch["clinical"]  # (B, 23)

        # Branch A – geometry
        g1, g1x = self.geo_sa1(xyz, None)
        g2, g2x = self.geo_sa2(g1x, g1)
        g3, _ = self.geo_sa3(g2x, g2)
        geo_vec = g3.squeeze(1)  # (B, 1024)

        # Branch B – hemodynamics
        h1, h1x = self.hemo_sa1(xyz, feats)
        h2, h2x = self.hemo_sa2(h1x, h1)
        h3, _ = self.hemo_sa3(h2x, h2)
        hemo_vec = h3.squeeze(1)  # (B, 512)

        # Branch C – clinical + global
        cg = torch.cat([clin, gf], dim=-1)  # (B, 68)
        clin_vec = self.clinical_encoder(cg)  # (B, 128)

        # Project all to 512 for cross-attention
        geo_512 = F.relu(self.geo_proj(geo_vec))  # (B, 512)
        clin_512 = F.relu(self.clin_proj(clin_vec))  # (B, 512)
        hemo_512 = hemo_vec  # already (B, 512)

        tokens = torch.stack([geo_512, hemo_512, clin_512], dim=1)  # (B, 3, 512)
        attended, _ = self.cross_attn(tokens, tokens, tokens)  # (B, 3, 512)

        # Gated fusion
        flat = attended.reshape(attended.size(0), -1)  # (B, 1536)
        gates = self.gate(flat)  # (B, 3)
        fused = (
            gates[:, 0:1] * attended[:, 0]
            + gates[:, 1:2] * attended[:, 1]
            + gates[:, 2:3] * attended[:, 2]
        )  # (B, 512)

        return self.classifier(fused)


# Metadata loading
def load_metadata(metadata_csv: str) -> Dict[str, Dict]:
    """
    Returns mapping: case_name → {label, age, age_norm, sex, location, location_idx}
    """
    mapping: Dict[str, Dict] = {}
    ages = []

    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("dataset") or row.get("name") or "").strip()
            status = (row.get("status") or row.get("rupture_status") or "").strip().lower()
            sex_raw = (row.get("sex") or row.get("Sex") or "").strip().lower()
            age_raw = row.get("age") or row.get("Age") or ""
            location = (row.get("location") or "").strip()

            if not name or status not in ("ruptured", "unruptured"):
                continue

            label = 1 if status == "ruptured" else 0
            sex_val = 1 if sex_raw in ("male", "m", "1") else 0

            try:
                age_val = float(age_raw)
            except (ValueError, TypeError):
                age_val = 55.0  # fallback

            ages.append(age_val)
            loc_idx = LOCATION_TO_IDX.get(location, -1)

            mapping[name] = {
                "label": label,
                "age": age_val,
                "sex": sex_val,
                "location": location,
                "location_idx": loc_idx,
            }

    # Z-score normalise age
    if ages:
        mu, sigma = float(np.mean(ages)), float(np.std(ages))
        if sigma < 1e-6:
            sigma = 1.0
        for k in mapping:
            mapping[k]["age_norm"] = (mapping[k]["age"] - mu) / sigma
    else:
        for k in mapping:
            mapping[k]["age_norm"] = 0.0

    print(
        f"[META] Loaded {len(mapping)} entries "
        f"(ruptured={sum(1 for v in mapping.values() if v['label'] == 1)}, "
        f"unruptured={sum(1 for v in mapping.values() if v['label'] == 0)})"
    )
    return mapping


def discover_cases(data_dir: str, metadata_csv: str) -> List[Dict]:
    """
    Walk `data_dir` for case folders that have hemodynamics_aggregate.csv,
    match to metadata for labels + clinical features.
    """
    meta = load_metadata(metadata_csv)
    samples: List[Dict] = []
    unmatched = []

    for item in sorted(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item)
        if not os.path.isdir(item_path):
            continue
        hemo = os.path.join(item_path, "hemodynamics_aggregate.csv")
        if not os.path.isfile(hemo):
            continue

        # Try to match folder name against metadata 'dataset' key
        matched = None
        folder = item  # e.g. "C0032_cut1" or "p114_FxQV..._cut1"
        for key, val in meta.items():
            # key is the 'dataset' field, e.g. "p114_FxQVDRUaDBQUCQMdFRQJCRAK"
            # folder may have an extra prefix or "_cutN" suffix
            if key in folder or folder in key:
                matched = val
                break
            # Try stripping _cutN
            folder_base = re.sub(r"_cut\d+$", "", folder)
            if key in folder_base or folder_base in key:
                matched = val
                break

        if matched is None:
            unmatched.append(folder)
            continue

        samples.append(
            {
                "case_dir": item_path,
                "label": matched["label"],
                "age": matched["age_norm"],
                "sex": matched["sex"],
                "location_idx": matched["location_idx"],
            }
        )

    n_r = sum(1 for s in samples if s["label"] == 1)
    n_u = sum(1 for s in samples if s["label"] == 0)
    print(f"[DATA] Matched {len(samples)} cases (R={n_r}, U={n_u}), unmatched={len(unmatched)}")
    if unmatched:
        print(f"       Unmatched folders: {unmatched[:10]}{'...' if len(unmatched) > 10 else ''}")
    return samples


# Training helpers
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []

    for batch in loader:
        for k in ("xyz", "features", "global_features", "clinical", "label"):
            batch[k] = batch[k].to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch["label"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = batch["label"].size(0)
        loss_sum += loss.item() * bs
        preds = logits.argmax(1)
        correct += (preds == batch["label"]).sum().item()
        total += bs
        all_labels.extend(batch["label"].cpu().numpy())
        all_probs.extend(F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5
    return loss_sum / total, correct / total, auc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, total = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        for k in ("xyz", "features", "global_features", "clinical", "label"):
            batch[k] = batch[k].to(device)
        logits = model(batch)
        loss = criterion(logits, batch["label"])

        bs = batch["label"].size(0)
        loss_sum += loss.item() * bs
        total += bs
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(batch["label"].cpu().numpy())
        all_probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    acc = (all_preds == all_labels).mean()
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5
    return loss_sum / total, acc, auc, all_preds, all_labels, all_probs


def print_metrics(labels, preds, probs, header=""):
    print(f"\n{header} Classification Report:")
    print("-" * 55)
    print(
        classification_report(
            labels, preds, target_names=["Unruptured", "Ruptured"], zero_division=0
        )
    )
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print("  Predicted:   Unrupt  Rupt")
    print(f"  Unruptured:  {cm[0, 0]:5d}  {cm[0, 1]:5d}")
    print(f"  Ruptured:    {cm[1, 0]:5d}  {cm[1, 1]:5d}")
    if len(np.unique(labels)) > 1:
        print(f"\nAUC-ROC: {roc_auc_score(labels, probs):.4f}")
    tn, fp, fn, tp = cm.ravel()
    print(f"Sensitivity: {tp / (tp + fn):.4f}" if (tp + fn) else "Sensitivity: N/A")
    print(f"Specificity: {tn / (tn + fp):.4f}" if (tn + fp) else "Specificity: N/A")


# K-Fold CV
def train_kfold(
    data_dir: str,
    metadata_csv: str,
    n_folds: int = 5,
    epochs: int = 200,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    target_n: int = 8192,
    save_dir: str = "kfold_comprehensive",
    seed: int = 42,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    early_stopping: int = 40,
    num_workers: int = 2,
    dropout: float = 0.5,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    samples = discover_cases(data_dir, metadata_csv)
    if not samples:
        raise RuntimeError("No matched cases found.")

    # Balance classes
    rupt = [s for s in samples if s["label"] == 1]
    unrupt = [s for s in samples if s["label"] == 0]
    min_c = min(len(rupt), len(unrupt))
    random.shuffle(rupt)
    random.shuffle(unrupt)
    samples = rupt[:min_c] + unrupt[:min_c]
    random.shuffle(samples)
    print(f"[CV] Balanced to {min_c} per class, total {len(samples)}")

    labels_arr = np.array([s["label"] for s in samples])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_aucs, fold_accs = [], []
    all_val_probs, all_val_labels = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CV] Device: {device}")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(samples)), labels_arr)):
        print(f"\n{'=' * 60}\nFOLD {fold + 1}/{n_folds}\n{'=' * 60}")
        tr_samples = [samples[i] for i in tr_idx]
        va_samples = [samples[i] for i in va_idx]
        print(f"  Train={len(tr_samples)}, Val={len(va_samples)}")

        tr_ds = ComprehensiveDataset(tr_samples, target_n=target_n, augment=True)
        va_ds = ComprehensiveDataset(va_samples, target_n=target_n, augment=False)
        tr_loader = DataLoader(
            tr_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        va_loader = DataLoader(
            va_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        model = ComprehensiveRuptureModel(num_classes=2, dropout=dropout).to(device)
        criterion = FocalLoss(gamma=focal_gamma) if use_focal_loss else nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        best_auc, best_acc, patience_ctr = 0.0, 0.0, 0

        log_csv = os.path.join(save_dir, f"fold{fold + 1}_log.csv")
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "train_acc", "train_auc", "val_loss", "val_acc", "val_auc"]
            )

        for epoch in range(1, epochs + 1):
            t_loss, t_acc, t_auc = train_one_epoch(model, tr_loader, optimizer, criterion, device)
            v_loss, v_acc, v_auc, _, _, _ = evaluate(model, va_loader, criterion, device)
            scheduler.step()

            with open(log_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch, t_loss, t_acc, t_auc, v_loss, v_acc, v_auc])

            improved = v_auc > best_auc or (v_auc == best_auc and v_acc > best_acc)
            if improved:
                best_auc, best_acc = v_auc, v_acc
                patience_ctr = 0
                torch.save(model.state_dict(), os.path.join(save_dir, f"fold{fold + 1}_best.pth"))
            else:
                patience_ctr += 1

            if epoch % 20 == 0 or improved:
                print(
                    f"  E{epoch:03d} | Train {t_acc:.3f}/{t_auc:.3f} | "
                    f"Val {v_acc:.3f}/{v_auc:.3f} {'*' if improved else ''}"
                )

            if patience_ctr >= early_stopping:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Reload best
        model.load_state_dict(
            torch.load(os.path.join(save_dir, f"fold{fold + 1}_best.pth"), weights_only=True)
        )
        _, _, fold_auc, preds, labels, probs = evaluate(model, va_loader, criterion, device)
        fold_acc = float((preds == labels).mean())
        fold_aucs.append(fold_auc)
        fold_accs.append(fold_acc)
        all_val_probs.extend(probs)
        all_val_labels.extend(labels)
        print(f"\n  [FOLD {fold + 1}] Acc={fold_acc:.4f}  AUC={fold_auc:.4f}")
        print_metrics(labels, preds, probs, f"FOLD {fold + 1}")

    # Summary
    print("\n" + "=" * 60)
    print("K-FOLD CROSS-VALIDATION RESULTS  (Comprehensive Model)")
    print("=" * 60)
    print(f"AUC per fold: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"Acc per fold: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"Mean Acc: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    overall_auc = roc_auc_score(all_val_labels, all_val_probs)
    print(f"Overall AUC: {overall_auc:.4f}")

    summary_csv = os.path.join(save_dir, "kfold_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "val_acc", "val_auc"])
        for i, (a, u) in enumerate(zip(fold_accs, fold_aucs)):
            w.writerow([i + 1, a, u])
        w.writerow(["mean", np.mean(fold_accs), np.mean(fold_aucs)])
        w.writerow(["std", np.std(fold_accs), np.std(fold_aucs)])
        w.writerow(["overall", float((np.array(all_val_probs) > 0.5).mean()), overall_auc])
    print(f"Summary → {summary_csv}")
    return fold_aucs, fold_accs


# Single-split training
def train_single(
    data_dir: str,
    metadata_csv: str,
    epochs: int = 200,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    target_n: int = 8192,
    val_fraction: float = 0.2,
    save_path: str = "comprehensive_best.pth",
    seed: int = 42,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    early_stopping: int = 40,
    num_workers: int = 2,
    dropout: float = 0.5,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    samples = discover_cases(data_dir, metadata_csv)
    if not samples:
        raise RuntimeError("No matched cases found.")

    # Balance & split
    rupt = [s for s in samples if s["label"] == 1]
    unrupt = [s for s in samples if s["label"] == 0]
    min_c = min(len(rupt), len(unrupt))
    random.shuffle(rupt)
    random.shuffle(unrupt)
    rupt, unrupt = rupt[:min_c], unrupt[:min_c]

    n_val = max(1, int(math.ceil(val_fraction * min_c)))
    tr_samples = rupt[n_val:] + unrupt[n_val:]
    va_samples = rupt[:n_val] + unrupt[:n_val]
    random.shuffle(tr_samples)
    random.shuffle(va_samples)
    print(f"[SPLIT] Train={len(tr_samples)}, Val={len(va_samples)}")

    tr_ds = ComprehensiveDataset(tr_samples, target_n=target_n, augment=True)
    va_ds = ComprehensiveDataset(va_samples, target_n=target_n, augment=False)
    tr_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComprehensiveRuptureModel(num_classes=2, dropout=dropout).to(device)
    print(f"[MODEL] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = FocalLoss(gamma=focal_gamma) if use_focal_loss else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    best_acc, best_auc, patience_ctr = 0.0, 0.0, 0

    log_csv = save_path.replace(".pth", "_log.csv")
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "train_auc", "val_loss", "val_acc", "val_auc"]
        )

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        t_loss, t_acc, t_auc = train_one_epoch(model, tr_loader, optimizer, criterion, device)
        v_loss, v_acc, v_auc, _, _, _ = evaluate(model, va_loader, criterion, device)
        scheduler.step()

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, t_loss, t_acc, t_auc, v_loss, v_acc, v_auc])

        improved = v_auc > best_auc or (v_auc == best_auc and v_acc > best_acc)
        if improved:
            best_acc, best_auc = v_acc, v_auc
            patience_ctr = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": v_acc,
                    "val_auc": v_auc,
                },
                save_path,
            )
            print(
                f"  E{epoch:03d} | Train {t_acc:.3f}/{t_auc:.3f} | "
                f"Val {v_acc:.3f}/{v_auc:.3f} [SAVED]"
            )
        else:
            patience_ctr += 1
            if epoch % 20 == 0:
                print(
                    f"  E{epoch:03d} | Train {t_acc:.3f}/{t_auc:.3f} | Val {v_acc:.3f}/{v_auc:.3f}"
                )

        if patience_ctr >= early_stopping:
            print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - t0
    print(
        f"\nTraining done in {elapsed / 60:.1f} min.  Best Val Acc={best_acc:.4f}  AUC={best_auc:.4f}"
    )

    # Final eval
    ckpt = torch.load(save_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, _, _, preds, labels, probs = evaluate(model, va_loader, criterion, device)
    print_metrics(labels, preds, probs, "Final Validation")


# CLI
def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive PointNet++ Rupture Classification "
        "(Geometry + Flow + Hemodynamics + Clinical)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="predictions/optimized_pinn",
        help="Root dir containing case folders",
    )
    parser.add_argument("--metadata", type=str, default="metadata.csv", help="Path to metadata.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--target_n", type=int, default=8192, help="Points per sample")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--early_stopping", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--focal_loss", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--kfold", type=int, default=1, help="K-fold CV folds (1 = single split)")
    parser.add_argument("--kfold_save_dir", type=str, default="training_logs/kfold_comprehensive")
    parser.add_argument("--save_path", type=str, default="training_logs/comprehensive_best.pth")

    args = parser.parse_args()

    print("=" * 60)
    print("  COMPREHENSIVE RUPTURE CLASSIFICATION")
    print("  Geometry + Velocity + Pressure + WSS + TAWSS/OSI/VM")
    print("  + Clinical (Age, Sex, Location)")
    print("=" * 60)
    print(f"  Data dir:     {args.data_dir}")
    print(f"  Metadata:     {args.metadata}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Points:       {args.target_n}")
    print(f"  K-Fold:       {args.kfold}")
    print(f"  Focal Loss:   {args.focal_loss}")
    print(f"  Dropout:      {args.dropout}")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    if args.kfold > 1:
        train_kfold(
            data_dir=args.data_dir,
            metadata_csv=args.metadata,
            n_folds=args.kfold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            target_n=args.target_n,
            save_dir=args.kfold_save_dir,
            seed=args.seed,
            use_focal_loss=args.focal_loss,
            focal_gamma=args.focal_gamma,
            early_stopping=args.early_stopping,
            num_workers=args.num_workers,
            dropout=args.dropout,
        )
    else:
        train_single(
            data_dir=args.data_dir,
            metadata_csv=args.metadata,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            target_n=args.target_n,
            save_path=args.save_path,
            seed=args.seed,
            use_focal_loss=args.focal_loss,
            focal_gamma=args.focal_gamma,
            early_stopping=args.early_stopping,
            num_workers=args.num_workers,
            dropout=args.dropout,
        )


if __name__ == "__main__":
    main()
