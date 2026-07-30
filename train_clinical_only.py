# Version 7 source snapshot
"""
Clinical-only (age, sex, location) MLP with stratified k-fold.

Improvements vs v6:
- Shared helpers (metadata, encoding, training loop, pooled summary).
- Uses EMA weights for validation: smoother metrics on tiny clinical tables.
- Tracks the best-AUC snapshot instead of the first early-stop-trigger snapshot.
- Cosine warm restarts + AdamW with decoupled weight decay.
- Label smoothing in the criterion to reduce overconfidence on the rupture class.
"""

import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
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

METADATA_PATH = os.environ.get("V7_METADATA", "metadata.csv")
DATA_DIR = os.environ.get("V7_DATA_DIR", "predictions/pinn_corrected")
OUTPUT_DIR = Path(os.environ.get("V7_OUTPUT_DIR", "results_clinical_v7"))
N_FOLDS = int(os.environ.get("V7_FOLDS", 5))
BATCH_SIZE = int(os.environ.get("V7_BATCH", 32))
EPOCHS = int(os.environ.get("V7_EPOCHS", 200))
LR = float(os.environ.get("V7_LR", 1e-3))
WEIGHT_DECAY = float(os.environ.get("V7_WD", 1e-4))
EARLY_STOP_PATIENCE = int(os.environ.get("V7_PATIENCE", 40))
LABEL_SMOOTH = float(os.environ.get("V7_LS", 0.05))

vc.set_seed(vc.SEED)
DEVICE = vc.get_device()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_mlp(in_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 128),
        nn.BatchNorm1d(128),
        nn.GELU(),
        nn.Dropout(0.4),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.GELU(),
        nn.Linear(32, 2),
    )


def forward_fn(model, batch, device, train: bool):
    xb, yb = batch
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    logits = model(xb)
    return logits, yb, xb.size(0)


def main():
    df = vc.discover_cases(DATA_DIR, METADATA_PATH)
    df = vc.prepare_clinical_columns(df)
    print(f"Valid samples: {len(df)}")

    all_locations = sorted(df["location"].unique())
    loc_to_idx = {loc: i for i, loc in enumerate(all_locations)}
    print(f"Locations ({len(all_locations)}): {all_locations}")

    targets = df["target"].values
    fold_indices = vc.stratified_kfold_indices(targets, N_FOLDS)
    pooled_probs, pooled_labels, fold_summaries = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(fold_indices):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)

        X_train, stats = vc.encode_clinical(train_df, all_locations, loc_to_idx)
        X_val, _ = vc.encode_clinical(val_df, all_locations, loc_to_idx, train_stats=stats)
        y_train = torch.tensor(targets[tr_idx], dtype=torch.long)
        y_val = torch.tensor(targets[va_idx], dtype=torch.long)

        sampler = vc.balanced_sampler(y_train)
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, sampler=sampler
        )
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

        model = build_mlp(X_train.shape[1]).to(DEVICE)
        weights = vc.class_weights(y_train, DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTH)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2)

        best_metrics, best_probs, best_labels = vc.run_fold_training(
            forward_fn=forward_fn,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            epochs=EPOCHS,
            early_stop_patience=EARLY_STOP_PATIENCE,
            fold_dir=OUTPUT_DIR,
            fold=fold,
            use_ema=True,
            grad_clip=1.0,
            amp=False,
        )
        fold_summaries.append({"fold": fold + 1, **best_metrics})
        pooled_probs.extend(best_probs)
        pooled_labels.extend(best_labels)
        print(f"  Best val AUC: {best_metrics.get('val_auc', 0.0):.4f}")

    vc.write_fold_summary(OUTPUT_DIR, fold_summaries, pooled_probs, pooled_labels)


if __name__ == "__main__":
    main()
