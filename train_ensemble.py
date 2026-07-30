# Version 9 source snapshot
"""
train_ensemble.py

Stacked ensemble trainer for the v9 flow+geometry pipeline.
Uses the base v9 fusion model to produce logits, then trains a small
stacking head on top of those logits plus global features.

This script reuses the Version 9 flow-geometry data and training helpers.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from train_flow_geometry import (
    BEST_METRIC as FLOW_BEST_METRIC,
)
from train_flow_geometry import (
    DEVICE,
    GLOBAL_FEATURE_DIM,
    LABEL_SMOOTHING,
    apply_global_stats,
    build_loader_from_tensors,
    build_model,
    collect_samples,
    discover_labeled_cases,
    fit_global_stats,
    forward_model,
    run_fold_training,
    set_seed,
    stratified_fold_splits,
    synthetic_case,
    write_fold_summary,
)

SEED = int(os.environ.get("V9E_SEED", 42))
BASE_FUSION = os.environ.get("V9E_BASE_FUSION", "late").lower()
METADATA_PATH = os.environ.get("V9E_METADATA", "metadata.csv")
DATA_DIR = os.environ.get("V9E_DATA_DIR", "predictions/pinn_corrected")
OUTPUT_ROOT = Path(os.environ.get("V9E_OUTPUT_DIR", "results_v9_ensemble"))
N_FOLDS = int(os.environ.get("V9E_FOLDS", 5))
BATCH_SIZE = int(os.environ.get("V9E_BATCH", 8))
EPOCHS = int(os.environ.get("V9E_EPOCHS", 200))
LR = float(os.environ.get("V9E_LR", 1e-4))
WEIGHT_DECAY = float(os.environ.get("V9E_WD", 1e-4))
EARLY_STOP_PATIENCE = int(os.environ.get("V9E_PATIENCE", 40))
USE_AMP = os.environ.get("V9E_AMP", "1").lower() in {"1", "true", "yes"}
DRY_RUN = os.environ.get("DRY_RUN", "0") in {"1", "true", "yes"}

set_seed(SEED)


class StackingHead(nn.Module):
    def __init__(self, logit_dim=2, global_dim=GLOBAL_FEATURE_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(logit_dim + global_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, logits, global_feats):
        return self.fc(torch.cat([logits, global_feats], dim=-1))


class FlowStackingEnsemble(nn.Module):
    def __init__(self, base_fusion: str = BASE_FUSION):
        super().__init__()
        self.base_fusion = base_fusion
        self.base_model = build_model(base_fusion)
        self.stack_head = StackingHead(logit_dim=2, global_dim=GLOBAL_FEATURE_DIM)

    def forward(self, xyz, per_point_feats, global_feats):
        base_logits = forward_model(
            self.base_model, xyz, per_point_feats, global_feats, fusion_mode=self.base_fusion
        )
        return self.stack_head(base_logits, global_feats)


def forward_fn_ensemble(model, batch, device, train: bool):
    xb, fb, gb, yb = batch
    xb = xb.to(device, non_blocking=True)
    fb = fb.to(device, non_blocking=True)
    gb = gb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    return model(xb, fb, gb), yb, xb.size(0)


def main():
    global METADATA_PATH, DATA_DIR, OUTPUT_ROOT, N_FOLDS, BATCH_SIZE, EPOCHS
    global LR, WEIGHT_DECAY, EARLY_STOP_PATIENCE, USE_AMP, BASE_FUSION

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-fusion", default=BASE_FUSION, choices=["late", "attention", "early"]
    )
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN)
    parser.add_argument("--metadata-path", default=METADATA_PATH)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--amp", action="store_true", default=USE_AMP)
    args = parser.parse_args()
    METADATA_PATH = args.metadata_path
    DATA_DIR = args.data_dir
    OUTPUT_ROOT = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT
    N_FOLDS = args.folds
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    EARLY_STOP_PATIENCE = args.patience
    USE_AMP = args.amp
    BASE_FUSION = args.base_fusion

    output_dir = OUTPUT_ROOT / f"stack_{BASE_FUSION}"
    print(f"Building v9 stacking ensemble with base_fusion={BASE_FUSION} on {DEVICE}")
    print(
        f"Training config: folds={N_FOLDS} epochs={EPOCHS} batch_size={BATCH_SIZE} lr={LR} "
        f"wd={WEIGHT_DECAY} patience={EARLY_STOP_PATIENCE} label_smoothing={LABEL_SMOOTHING} "
        f"best_metric={FLOW_BEST_METRIC}"
    )

    if args.dry_run:
        model = FlowStackingEnsemble(base_fusion=BASE_FUSION).to(DEVICE)
        xyz, per_point, global_feats = synthetic_case(batch_size=2)
        with torch.no_grad():
            output = model(xyz.to(DEVICE), per_point.to(DEVICE), global_feats.to(DEVICE))
        print("Dry-run forward pass successful; output shape:", output.shape)
        return

    paths, labels = discover_labeled_cases()
    labels = np.asarray(labels).astype(int)
    if len(paths) == 0:
        raise RuntimeError(
            f"No labeled cases found under {DATA_DIR} using metadata {METADATA_PATH}"
        )

    print(f"Total samples: {len(paths)}   Device: {DEVICE}")
    class_counts = np.bincount(labels, minlength=2)
    print(f"Class counts: unruptured={int(class_counts[0])}, ruptured={int(class_counts[1])}")

    fold_indices, n_splits_eff = stratified_fold_splits(labels, N_FOLDS, SEED)
    pooled_probs, pooled_labels, fold_summaries = [], [], []
    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        print(f"\n--- Fold {fold + 1}/{n_splits_eff} ---")
        tr_x, tr_f, tr_g, tr_y = collect_samples(paths[train_idx], labels[train_idx], augment=True)
        va_x, va_f, va_g, va_y = collect_samples(paths[val_idx], labels[val_idx], augment=False)
        global_mean, global_std = fit_global_stats(tr_g)
        tr_g = apply_global_stats(tr_g, global_mean, global_std)
        va_g = apply_global_stats(va_g, global_mean, global_std)
        train_loader = build_loader_from_tensors(tr_x, tr_f, tr_g, tr_y, shuffle=True)
        val_loader = build_loader_from_tensors(va_x, va_f, va_g, va_y, shuffle=False)

        model = FlowStackingEnsemble(base_fusion=BASE_FUSION).to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
        best_metrics, best_probs, best_labels = run_fold_training(
            forward_fn=forward_fn_ensemble,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            epochs=EPOCHS,
            early_stop_patience=EARLY_STOP_PATIENCE,
            fold_dir=output_dir,
            fold=fold,
            use_ema=True,
            grad_clip=1.0,
            amp=USE_AMP,
            selection_metric=FLOW_BEST_METRIC,
        )
        fold_summaries.append({"fold": fold + 1, **best_metrics})
        pooled_probs.extend(best_probs)
        pooled_labels.extend(best_labels)
        print(f"  Best val AUC: {best_metrics.get('val_auc', 0.0):.4f}")

    write_fold_summary(output_dir, fold_summaries, pooled_probs, pooled_labels)
    print(f"Training complete. Results written to {output_dir}")


if __name__ == "__main__":
    main()
