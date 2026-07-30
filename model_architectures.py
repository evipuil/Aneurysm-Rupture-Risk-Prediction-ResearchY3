# Version 14 source snapshot
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from torch_geometric.nn import (
        GATv2Conv,
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
        self.use_xyz_as_features = backbone == "pointnext" and in_channel == 0
        first_layer_in_channel = 6 if self.use_xyz_as_features else in_channel + 3

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
                in_channel=first_layer_in_channel,
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
        if points is None and self.use_xyz_as_features:
            points = xyz
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


class GatedFlowFusionClassifier(nn.Module):
    """Geometry-preserving fusion where flow contributes as a learned residual.

    The geometry branch stays as the backbone representation. The flow branch is
    projected into the same embedding space, then a learned gate decides how
    much flow residual to add. This reduces overfitting when simulated flow
    channels are noisy or partly redundant with morphology.
    """

    def __init__(
        self,
        branches: List[dict],
        embed_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
        flow_branch_dropout: float = 0.25,
    ):
        super().__init__()
        self.branches = nn.ModuleDict()
        for b in branches:
            self.branches[b["name"]] = b["module"]
        if "geometry" not in self.branches or "flow" not in self.branches:
            raise ValueError("GatedFlowFusionClassifier requires geometry and flow branches")

        self.flow_branch_dropout = float(flow_branch_dropout)
        self.flow_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.flow_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.geo_flow_norm = nn.LayerNorm(embed_dim)

        has_clinical = "clinical" in self.branches
        head_in = embed_dim * (2 if has_clinical else 1)
        self.fusion_head = nn.Sequential(
            nn.Linear(head_in, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, xyz=None, flow=None, clinical=None):
        geo_token, geo_aux = self.branches["geometry"](xyz)
        flow_token, flow_aux = self.branches["flow"](xyz, flow)

        if self.training and self.flow_branch_dropout > 0:
            keep_prob = max(1e-6, 1.0 - self.flow_branch_dropout)
            keep = (
                torch.rand(flow_token.shape[0], 1, device=flow_token.device) < keep_prob
            ).float()
            flow_token = flow_token * keep / keep_prob

        flow_delta = self.flow_proj(flow_token)
        flow_gate = self.flow_gate(torch.cat([geo_token, flow_token], dim=1))
        geo_flow = self.geo_flow_norm(geo_token + flow_gate * flow_delta)

        embeds = [geo_flow]
        aux_losses = [geo_aux, flow_aux]
        if "clinical" in self.branches:
            clinical_token, clinical_aux = self.branches["clinical"](clinical)
            embeds.append(clinical_token)
            aux_losses.append(clinical_aux)

        fused = torch.cat(embeds, dim=1)
        logits = self.fusion_head(fused)
        aux_loss_total = sum(aux_losses) if aux_losses else 0.0
        return logits, aux_loss_total


class ClinicalAnchoredLateFusionClassifier(nn.Module):
    """Late-fuse supervised geometry-clinical and geometry-flow heads.

    The clinical head remains the anchor. A single learned flow weight is
    constrained to a conservative range so complementary flow information can
    improve ranking without replacing the stronger geometry-clinical signal.
    """

    def __init__(
        self,
        branches: List[dict],
        embed_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
        max_flow_weight: float = 0.5,
        initial_flow_weight: float = 0.25,
    ):
        super().__init__()
        self.branches = nn.ModuleDict({b["name"]: b["module"] for b in branches})
        required = {"geometry", "flow", "clinical"}
        if not required.issubset(self.branches):
            raise ValueError(
                "ClinicalAnchoredLateFusionClassifier requires geometry, flow, and clinical branches"
            )

        def pair_head():
            return nn.Sequential(
                nn.Linear(embed_dim * 2, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

        self.geometry_clinical_head = pair_head()
        self.geometry_flow_head = pair_head()
        self.max_flow_weight = float(max_flow_weight)
        ratio = min(max(initial_flow_weight / max(self.max_flow_weight, 1e-6), 1e-4), 1.0 - 1e-4)
        self.flow_weight_logit = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        )

    def current_flow_weight(self):
        return self.max_flow_weight * torch.sigmoid(self.flow_weight_logit)

    def forward(self, xyz=None, flow=None, clinical=None):
        geometry_token, _ = self.branches["geometry"](xyz)
        flow_token, _ = self.branches["flow"](xyz, flow)
        clinical_token, _ = self.branches["clinical"](clinical)

        base_logits = self.geometry_clinical_head(
            torch.cat([geometry_token, clinical_token], dim=1)
        )
        flow_logits = self.geometry_flow_head(torch.cat([geometry_token, flow_token], dim=1))
        base_margin = base_logits[:, 1] - base_logits[:, 0]
        flow_margin = flow_logits[:, 1] - flow_logits[:, 0]
        flow_weight = self.current_flow_weight()
        final_margin = (1.0 - flow_weight) * base_margin + flow_weight * flow_margin
        logits = torch.stack([-0.5 * final_margin, 0.5 * final_margin], dim=1)
        return logits, {
            "geometry_clinical": base_logits,
            "geometry_flow": flow_logits,
            "flow_weight": flow_weight,
        }


# Voxel CNN Encoder


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class VoxelConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class VoxelResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = VoxelConvBlock(in_channels, out_channels, stride=stride, dropout=dropout)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.gelu(x + residual)
        return self.dropout(x)


class VoxelCNNClassifier(nn.Module):
    """3D CNN for voxelized geometry and hemodynamic fields."""

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 24,
        dropout: float = 0.25,
        num_classes: int = 2,
    ):
        super().__init__()
        self.stem = VoxelConvBlock(in_channels, base_channels, dropout=dropout * 0.5)
        self.stage1 = VoxelResidualBlock(base_channels, base_channels, dropout=dropout * 0.5)
        self.stage2 = VoxelResidualBlock(
            base_channels, base_channels * 2, stride=2, dropout=dropout
        )
        self.stage3 = VoxelResidualBlock(
            base_channels * 2, base_channels * 4, stride=2, dropout=dropout
        )
        self.stage4 = VoxelResidualBlock(
            base_channels * 4, base_channels * 8, stride=2, dropout=dropout
        )
        pooled_channels = base_channels * 8 * 2
        self.head = nn.Sequential(
            nn.Linear(pooled_channels, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, voxels, flow=None, clinical=None):
        x = self.stem(voxels)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        pooled = torch.cat(
            [
                F.adaptive_avg_pool3d(x, 1).flatten(1),
                F.adaptive_max_pool3d(x, 1).flatten(1),
            ],
            dim=1,
        )
        return self.head(pooled), 0.0


# Graph Encoder (GNN via torch_geometric)


class GraphEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        embed_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
        input_dim: int = 14,
        num_heads: int = 2,
        num_layers: int = 3,
        checkpoint_layers: bool = True,
    ):
        super().__init__()
        if not HAS_PYG:
            raise RuntimeError("torch_geometric is required for GraphEncoder")
        self.checkpoint_layers = bool(checkpoint_layers)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.gat_layers = nn.ModuleList(
            [
                GATv2Conv(
                    hidden_dim,
                    hidden_dim,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=4,
                    add_self_loops=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.gat_layers])
        self.node_fuse = nn.Sequential(
            nn.Linear(hidden_dim * (num_layers + 1), hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is None:
            src, dst = data.edge_index
            delta = data.pos[dst] - data.pos[src]
            edge_attr = torch.cat([delta, torch.norm(delta, dim=1, keepdim=True)], dim=1)
        states = [x]
        for gat, norm in zip(self.gat_layers, self.norm_layers):

            def layer_forward(
                node_features,
                edge_index,
                edge_features,
                gat_layer=gat,
                norm_layer=norm,
            ):
                updated = gat_layer(node_features, edge_index, edge_attr=edge_features)
                return F.gelu(norm_layer(updated)) + node_features

            if self.checkpoint_layers and self.training and x.requires_grad:
                x = checkpoint(
                    layer_forward,
                    x,
                    data.edge_index,
                    edge_attr,
                    use_reentrant=False,
                )
            else:
                x = layer_forward(x, data.edge_index, edge_attr)
            states.append(x)

        x = self.node_fuse(torch.cat(states, dim=1))

        # Compute graph moments in FP32. Under float16 autocast, x * x can
        # overflow and make E[x^2] - E[x]^2 evaluate to inf - inf = NaN.
        pooled_x = x.float()
        x_max = global_max_pool(pooled_x, data.batch)
        x_mean = global_mean_pool(pooled_x, data.batch)
        x_mean_sq = global_mean_pool(pooled_x.square(), data.batch)
        x_std = torch.sqrt(torch.clamp(x_mean_sq - x_mean * x_mean, min=1e-8))
        x_pooled = torch.cat([x_mean, x_max, x_std], dim=1)

        feat = self.pool_mlp(x_pooled)
        logits = self.head(feat)
        return logits, feat
