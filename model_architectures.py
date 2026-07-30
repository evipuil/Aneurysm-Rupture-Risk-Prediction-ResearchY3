# Version 12 source snapshot
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import (
        BatchNorm,
        GATv2Conv,
        global_add_pool,
        global_max_pool,
        global_mean_pool,
    )

    HAS_PYG = True
except Exception:
    HAS_PYG = False


# Point Encoder (PointNet++ / PointNeXt)


class PointSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False, dropout=0.0):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.dropout = dropout
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch
        self.need_proj = in_channel != mlp[-1]
        if self.need_proj:
            self.proj = nn.Conv2d(in_channel, mlp[-1], 1)
            self.proj_bn = nn.BatchNorm2d(mlp[-1])

    def _index_points(self, points, idx):
        B = points.shape[0]
        view_shape = [1] * idx.ndim
        view_shape[0] = B
        batch_indices = torch.arange(B, device=points.device).view(view_shape).expand_as(idx)
        return points[batch_indices, idx]

    def _square_distance(self, src, dst):
        return torch.cdist(src, dst).pow(2)

    def farthest_point_sample(self, xyz, npoint):
        B, N, _ = xyz.shape
        device = xyz.device
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.full((B, N), 1e10, device=device)
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_ar = torch.arange(B, dtype=torch.long, device=device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_ar, farthest].unsqueeze(1)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)
            mask = dist < distance
            distance = torch.where(mask, dist, distance)
            farthest = torch.max(distance, dim=-1).indices
        return centroids

    def query_ball_point(self, radius, nsample, xyz, new_xyz):
        B, N, _ = xyz.shape
        S = new_xyz.shape[1]
        device = xyz.device
        group_idx = torch.arange(N, device=device).view(1, 1, N).expand(B, S, N).clone()
        sqrdists = self._square_distance(new_xyz, xyz)
        group_idx[sqrdists > radius**2] = N
        group_idx, _ = torch.sort(group_idx, dim=-1)
        group_idx = group_idx[:, :, :nsample]
        first = group_idx[:, :, :1].expand(-1, -1, nsample)
        mask = group_idx == N
        group_idx = torch.where(mask, first, group_idx)
        return group_idx

    def sample_and_group(self, npoint, radius, nsample, xyz, points):
        B, _, C = xyz.shape
        fps_idx = self.farthest_point_sample(xyz, npoint)
        new_xyz = self._index_points(xyz, fps_idx)
        idx = self.query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = self._index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C)
        if points is not None:
            grouped_points = self._index_points(points, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm
        return new_points, new_xyz

    def sample_and_group_all(self, xyz, points):
        B, N, C = xyz.shape
        new_xyz = torch.zeros(B, 1, C, device=xyz.device)
        grouped_xyz = xyz.view(B, 1, N, C)
        if points is not None:
            new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        else:
            new_points = grouped_xyz
        return new_points, new_xyz

    def forward(self, xyz, points=None):
        if self.group_all:
            new_points, new_xyz = self.sample_and_group_all(xyz, points)
        else:
            new_points, new_xyz = self.sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        x = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            x = bn(conv(x))
            x = F.gelu(x)
            if self.dropout > 0 and self.training:
                x = F.dropout2d(x, p=self.dropout, training=True)
        if self.need_proj:
            sc = self.proj_bn(self.proj(new_points.permute(0, 3, 2, 1)))
            x = F.gelu(x + sc)
        new_points = torch.max(x, dim=2).values
        return new_points.permute(0, 2, 1), new_xyz


class PointNeXtSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False, dropout=0.0):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.dropout = dropout
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch
        self.need_proj = in_channel != mlp[-1]
        if self.need_proj:
            self.proj = nn.Conv2d(in_channel, mlp[-1], 1)
            self.proj_bn = nn.BatchNorm2d(mlp[-1])

    def _index_points(self, points, idx):
        B = points.shape[0]
        view_shape = [1] * idx.ndim
        view_shape[0] = B
        batch_indices = torch.arange(B, device=points.device).view(view_shape).expand_as(idx)
        return points[batch_indices, idx]

    def forward(self, xyz, points=None):
        # Simplified PointNeXt variant: uses point-wise layers with GELU
        if points is None:
            raise ValueError("PointNeXt requires both xyz and feature points")

        B, N, C = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B, 1, C, device=xyz.device)
            grouped_xyz_norm = xyz.view(B, 1, N, C)
            grouped_points = points.view(B, 1, N, -1)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            x = new_points.permute(0, 3, 2, 1)
            for conv, bn in zip(self.mlp_convs, self.mlp_bns):
                x = bn(conv(x))
                x = F.gelu(x)
                if self.dropout > 0 and self.training:
                    x = F.dropout2d(x, p=self.dropout, training=True)
            new_points = torch.max(x, dim=2).values
            return new_points.permute(0, 2, 1), new_xyz

        npoint = self.npoint if self.npoint else N
        sample_idx = torch.randperm(N, device=xyz.device)[:npoint]
        new_xyz = xyz[:, sample_idx, :]

        dists = torch.cdist(new_xyz, xyz)
        _, group_idx = torch.topk(dists, min(self.nsample, N), dim=-1, largest=False)
        grouped_xyz = self._index_points(xyz, group_idx)
        grouped_points = self._index_points(points, group_idx)

        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)

        x = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            x = bn(conv(x))
            x = F.gelu(x)
            if self.dropout > 0 and self.training:
                x = F.dropout2d(x, p=self.dropout, training=True)
        new_points = torch.max(x, dim=2).values
        return new_points.permute(0, 2, 1), new_xyz


class PointEncoder(nn.Module):
    """Encoder for point cloud data (PointNet++ or PointNeXt backbone)."""

    def __init__(
        self,
        in_channel: int = 0,
        embed_dim: int = 256,
        backbone: str = "pointnet2",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.backbone = backbone
        self.dropout = dropout

        if backbone == "pointnet2":
            self.Layer1 = PointSetAbstraction(
                npoint=1024,
                radius=0.1,
                nsample=32,
                in_channel=in_channel + 3,
                mlp=[64, 64, 128],
                dropout=dropout,
            )
            self.Layer2 = PointSetAbstraction(
                npoint=256,
                radius=0.2,
                nsample=32,
                in_channel=128 + 3,
                mlp=[128, 128, 256],
                dropout=dropout,
            )
            self.Layer3 = PointSetAbstraction(
                npoint=None,
                radius=None,
                nsample=None,
                in_channel=256 + 3,
                mlp=[256, 512, embed_dim],
                group_all=True,
                dropout=dropout,
            )
        elif backbone == "pointnext":
            self.Layer1 = PointNeXtSetAbstraction(
                npoint=1024,
                radius=0.1,
                nsample=32,
                in_channel=in_channel + 3,
                mlp=[64, 64, 128],
                dropout=dropout,
            )
            self.Layer2 = PointNeXtSetAbstraction(
                npoint=256,
                radius=0.2,
                nsample=32,
                in_channel=128 + 3,
                mlp=[128, 128, 256],
                dropout=dropout,
            )
            self.Layer3 = PointNeXtSetAbstraction(
                npoint=None,
                radius=None,
                nsample=None,
                in_channel=256 + 3,
                mlp=[256, 512, embed_dim],
                group_all=True,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, xyz, points=None):
        new_points, new_xyz = self.Layer1(xyz, points)
        new_points, new_xyz = self.Layer2(new_xyz, new_points)
        new_points, new_xyz = self.Layer3(new_xyz, new_points)
        feat = new_points.squeeze(-2)
        aux_loss = 0.0
        return feat, aux_loss


# Clinical Encoder


class ClinicalEncoder(nn.Module):
    def __init__(self, clinical_dim: int, embed_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(clinical_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, clinical):
        feat = self.mlp(clinical)
        aux_loss = 0.0
        return feat, aux_loss


# Branch Fusion Classifier


class BranchFusionClassifier(nn.Module):
    def __init__(
        self, branches: List[dict], embed_dim: int = 256, dropout: float = 0.3, num_classes: int = 2
    ):
        super().__init__()
        self.branches = nn.ModuleDict()
        for b in branches:
            self.branches[b["name"]] = b["module"]

        total_embed = embed_dim * len(branches)
        self.fusion_head = nn.Sequential(
            nn.Linear(total_embed, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, xyz=None, flow=None, clinical=None):
        embeds = []
        aux_losses = []
        for name, module in self.branches.items():
            if name == "geometry":
                token, aux = module(xyz)
            elif name == "flow":
                token, aux = module(xyz, flow)
            elif name == "clinical":
                token, aux = module(clinical)
            else:
                continue
            embeds.append(token)
            aux_losses.append(aux)

        fused = torch.cat(embeds, dim=1)
        logits = self.fusion_head(fused)
        aux_loss_total = sum(aux_losses) if aux_losses else 0.0
        return logits, aux_loss_total


# Graph Encoder (GNN via torch_geometric)


class GraphEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        embed_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
        input_dim: int = 14,
    ):
        super().__init__()
        if not HAS_PYG:
            raise RuntimeError("torch_geometric is required for GraphEncoder")

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gat_layers = nn.ModuleList(
            [
                GATv2Conv(hidden_dim, hidden_dim, heads=4, dropout=dropout),
                GATv2Conv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout),
            ]
        )
        self.bn_layers = nn.ModuleList([BatchNorm(hidden_dim * 4), BatchNorm(hidden_dim * 4)])
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 * 3, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        for gat, bn in zip(self.gat_layers, self.bn_layers):
            x = gat(x, data.edge_index)
            x = bn(x)
            x = F.gelu(x)

        # Global pooling
        x_add = global_add_pool(x, data.batch)
        x_max = global_max_pool(x, data.batch)
        x_mean = global_mean_pool(x, data.batch)
        x_pooled = torch.cat([x_add, x_max, x_mean], dim=1)

        feat = self.pool_mlp(x_pooled)
        logits = self.head(feat)
        return logits, 0.0
