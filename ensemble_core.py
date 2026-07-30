# Version 7 source snapshot
"""
Shared ensemble trainer used by train_ensemble.py and train_ensemble_rrt.py.

Fuses four branches:
  - Geometry PointNet++ (xyz only)
  - Flow PointNet++ (xyz + derived hemodynamics, optionally including RRT)
  - Clinical MLP (age, sex, location one-hot)
  - Global summary MLP (per-case statistical vector)

Improvements vs v6:
- Gated attention fusion over the four branch embeddings.
- Preloading + per-fold fold-consistent normalization (same as v6 but using
  shared helpers so the logic stays consistent across variants).
- EMA weights for validation + best-AUC checkpointing via common.run_fold_training.
- Label smoothing + focal loss option; AMP safe if USE_AMP is enabled.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
    import importlib.util

    _local_v7 = Path(_pointnet_root) / "common.py"
    if _local_v7.exists():
        _spec = importlib.util.spec_from_file_location("common_local", str(_local_v7))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        vc = _mod
    else:
        try:
            import pointnet_pytorch.common as vc
        except Exception:
            import common as vc
except Exception:
    try:
        import pointnet_pytorch.common as vc
    except Exception:
        import common as vc


# Model
class GatedFusion(nn.Module):
    """Four-branch gated attention fusion. Each branch is projected to a common
    dim and scaled by a softmax gate learned from their concatenation."""

    def __init__(self, dims, fused_dim: int = 512):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, fused_dim) for d in dims])
        self.gate = nn.Sequential(
            nn.Linear(sum(dims), len(dims)),
            nn.Softmax(dim=-1),
        )

    def forward(self, inputs):
        proj = [F.gelu(p(x)) for p, x in zip(self.projs, inputs)]
        concat = torch.cat(inputs, dim=-1)
        w = self.gate(concat)
        out = sum(w[:, i : i + 1] * proj[i] for i in range(len(inputs)))
        return out


def build_ensemble(hemo_channels: int, global_feature_dim: int, clinical_dim: int) -> nn.ModuleDict:
    # Geometry branch (xyz only)
    geo = nn.ModuleDict(
        {
            "sa1": vc.PointNetSetAbstraction(
                512, 0.2, 32, in_channel=3, mlp=[64, 64, 128], dropout=0.1
            ),
            "sa2": vc.PointNetSetAbstraction(
                128, 0.4, 64, in_channel=131, mlp=[128, 128, 256], dropout=0.1
            ),
            "sa3": vc.PointNetSetAbstraction(
                None, None, None, in_channel=259, mlp=[256, 512, 1024], group_all=True, dropout=0.1
            ),
        }
    )
    # Flow branch (xyz + hemo features)
    flow = nn.ModuleDict(
        {
            "sa1": vc.PointNetSetAbstraction(
                512, 0.2, 32, in_channel=3 + hemo_channels, mlp=[64, 64, 128], dropout=0.1
            ),
            "sa2": vc.PointNetSetAbstraction(
                128, 0.4, 64, in_channel=131, mlp=[128, 128, 256], dropout=0.1
            ),
            "sa3": vc.PointNetSetAbstraction(
                None, None, None, in_channel=259, mlp=[256, 512], group_all=True, dropout=0.1
            ),
        }
    )
    clinical_enc = nn.Sequential(
        nn.Linear(clinical_dim, 64),
        nn.BatchNorm1d(64),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(64, 64),
        nn.BatchNorm1d(64),
        nn.GELU(),
        nn.Dropout(0.3),
    )
    global_fc = nn.Sequential(
        nn.Linear(global_feature_dim, 64),
        nn.BatchNorm1d(64),
        nn.GELU(),
        nn.Dropout(0.3),
    )
    fuse = GatedFusion([1024, 512, 64, 64], fused_dim=512)
    head = nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2),
    )
    return nn.ModuleDict(
        {
            "geo": geo,
            "flow": flow,
            "clinical_enc": clinical_enc,
            "global_fc": global_fc,
            "fuse": fuse,
            "head": head,
        }
    )


def forward_ensemble(model, xyz, feats, clinical, global_feats):
    B = xyz.shape[0]
    g1, g1_xyz = model["geo"]["sa1"](xyz, None)
    g2, g2_xyz = model["geo"]["sa2"](g1_xyz, g1)
    g3, _ = model["geo"]["sa3"](g2_xyz, g2)
    geo_feat = g3.view(B, 1024)

    f1, f1_xyz = model["flow"]["sa1"](xyz, feats)
    f2, f2_xyz = model["flow"]["sa2"](f1_xyz, f1)
    f3, _ = model["flow"]["sa3"](f2_xyz, f2)
    flow_feat = f3.view(B, 512)

    clin_feat = model["clinical_enc"](clinical)
    gf = model["global_fc"](global_feats)

    fused = model["fuse"]([geo_feat, flow_feat, clin_feat, gf])
    return model["head"](fused)


# Sample loader and fold-level normalization
def load_sample(path: str, label: int, include_rrt: bool, target_n: int):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    try:
        pts = df[["x", "y", "z"]].values.astype(np.float32)
    except KeyError:
        pts = df.iloc[:, :3].values.astype(np.float32)

    raw_cols = []
    for key in ("tawss", "osi", "von"):
        match = next((c for c in df.columns if key in c), None)
        raw_cols.append(df[match].values if match else np.zeros(len(df)))
    raw = np.stack(raw_cols, axis=1).astype(np.float32)

    global_feats = vc.summarize_global_features(raw, pts, include_rrt=include_rrt)
    derived = vc.derive_hemo_channels(raw, include_rrt=include_rrt)

    idx = vc.resample_cloud(pts, target_n)
    pts = pts[idx]
    derived = derived[idx]
    return (
        torch.tensor(pts, dtype=torch.float32),
        torch.tensor(derived, dtype=torch.float32),
        torch.tensor(global_feats, dtype=torch.float32),
        torch.tensor(label, dtype=torch.long),
    )


def _apply_stats(
    tensor: torch.Tensor, mu: np.ndarray, std: np.ndarray, clip: float = 3.0
) -> torch.Tensor:
    arr = tensor.cpu().numpy()
    out = (arr - mu) / std
    if clip > 0:
        out = np.clip(out, -clip, clip)
    return torch.tensor(out, dtype=torch.float32)


def fold_normalize(train_list, val_list):
    """Compute train-side normalization stats and apply to both splits."""
    t_pts = np.concatenate([p[0].cpu().numpy() for p in train_list], axis=0)
    t_feats = np.concatenate([p[1].cpu().numpy() for p in train_list], axis=0)
    t_gf = np.stack([p[2].cpu().numpy() for p in train_list], axis=0)

    pts_mu, pts_std = t_pts.mean(0), t_pts.std(0) + 1e-8
    feats_mu, feats_std = t_feats.mean(0), t_feats.std(0)
    feats_std[feats_std < 1e-8] = 1.0
    gf_mu, gf_std = t_gf.mean(0), t_gf.std(0)
    gf_std[gf_std < 1e-8] = 1.0

    def norm(samples):
        out = []
        for pts, feats, gf, lab in samples:
            out.append(
                (
                    _apply_stats(pts, pts_mu, pts_std, clip=0),
                    _apply_stats(feats, feats_mu, feats_std),
                    _apply_stats(gf, gf_mu, gf_std),
                    lab,
                )
            )
        return out

    return norm(train_list), norm(val_list)


def augment_points(pts: torch.Tensor) -> torch.Tensor:
    arr = vc.so3_rotate(pts.cpu().numpy())
    arr = vc.jitter(arr) * float(np.random.uniform(0.95, 1.05))
    return torch.tensor(arr, dtype=torch.float32)


# Metadata discovery with clinical columns + per-case file paths
def discover_dataframe(metadata_path: str, data_dir: str) -> pd.DataFrame:
    df = vc.discover_cases(data_dir, metadata_path)
    return vc.prepare_clinical_columns(df)


# Training entry point
def train_ensemble(
    output_dir: Path,
    metadata_path: str,
    data_dir: str,
    include_rrt: bool,
    n_folds: int = 5,
    batch_size: int = 8,
    epochs: int = 200,
    lr: float = 5e-4,
    weight_decay: float = 1e-3,
    target_n: int = 4096,
    early_stop_patience: int = 40,
    label_smoothing: float = 0.1,
    use_amp: bool = False,
):
    vc.set_seed(vc.SEED)
    device = vc.get_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = discover_dataframe(metadata_path, data_dir)
    print(f"Valid samples: {len(df)}")
    all_locations = sorted(df["location"].unique())
    loc_to_idx = {loc: i for i, loc in enumerate(all_locations)}
    targets = df["target"].values

    hemo_channels = 9 if include_rrt else 8
    global_dim = 28 if include_rrt else 23
    clinical_dim = 2 + len(all_locations)

    print(
        f"  include_rrt={include_rrt}  hemo_channels={hemo_channels}  global_dim={global_dim}  clinical_dim={clinical_dim}"
    )
    print("Preloading samples into memory...")
    sample_cache = {}
    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"  Preloaded {i}/{len(df)}")
        try:
            sample_cache[row["filepath"]] = load_sample(
                row["filepath"], int(row["target"]), include_rrt, target_n
            )
        except Exception as exc:
            print(f"  [warn] failed to preload {row['filepath']}: {exc}")
            sample_cache[row["filepath"]] = (
                torch.zeros(target_n, 3),
                torch.zeros(target_n, hemo_channels),
                torch.zeros(global_dim),
                torch.tensor(int(row["target"]), dtype=torch.long),
            )
    print(f"Preloading complete: {len(sample_cache)} samples")

    fold_indices = vc.stratified_kfold_indices(targets, n_folds)
    pooled_probs, pooled_labels, fold_summaries = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(fold_indices):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)

        clin_train, stats = vc.encode_clinical(train_df, all_locations, loc_to_idx)
        clin_val, _ = vc.encode_clinical(val_df, all_locations, loc_to_idx, train_stats=stats)

        train_raw = [sample_cache[row["filepath"]] for _, row in train_df.iterrows()]
        val_raw = [sample_cache[row["filepath"]] for _, row in val_df.iterrows()]
        train_norm, val_norm = fold_normalize(train_raw, val_raw)

        tr_pts = torch.stack([t[0] for t in train_norm])
        tr_feats = torch.stack([t[1] for t in train_norm])
        tr_gf = torch.stack([t[2] for t in train_norm])
        tr_labels = torch.stack([t[3] for t in train_norm])
        va_pts = torch.stack([t[0] for t in val_norm])
        va_feats = torch.stack([t[1] for t in val_norm])
        va_gf = torch.stack([t[2] for t in val_norm])
        va_labels = torch.stack([t[3] for t in val_norm])

        sampler = vc.balanced_sampler(tr_labels)
        train_loader = DataLoader(
            TensorDataset(tr_pts, tr_feats, tr_gf, clin_train, tr_labels),
            batch_size=batch_size,
            sampler=sampler,
            drop_last=len(tr_labels) > batch_size,
        )
        val_loader = DataLoader(
            TensorDataset(va_pts, va_feats, va_gf, clin_val, va_labels),
            batch_size=batch_size,
            shuffle=False,
        )

        model = build_ensemble(hemo_channels, global_dim, clinical_dim).to(device)
        weights = vc.class_weights(tr_labels, device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        def forward_fn(model, batch, dev, train: bool):
            pts, feats, gf, clin, yb = batch
            pts = pts.to(dev, non_blocking=True)
            feats = feats.to(dev, non_blocking=True)
            gf = gf.to(dev, non_blocking=True)
            clin = clin.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            if train:
                # Light augmentation: re-rotate/jitter each batch to regularize geometry branch
                aug_pts = torch.stack(
                    [augment_points(pts[i].cpu()) for i in range(pts.shape[0])]
                ).to(dev)
                pts = aug_pts
            logits = forward_ensemble(model, pts, feats, clin, gf)
            return logits, yb, pts.size(0)

        best_metrics, best_probs, best_labels = vc.run_fold_training(
            forward_fn=forward_fn,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
            fold_dir=output_dir,
            fold=fold,
            use_ema=True,
            grad_clip=1.0,
            amp=use_amp,
        )
        fold_summaries.append({"fold": fold + 1, **best_metrics})
        pooled_probs.extend(best_probs)
        pooled_labels.extend(best_labels)
        print(f"  Best val AUC: {best_metrics.get('val_auc', 0.0):.4f}")

    vc.write_fold_summary(output_dir, fold_summaries, pooled_probs, pooled_labels)
