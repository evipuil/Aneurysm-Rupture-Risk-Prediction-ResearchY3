# Version 14 source snapshot
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from base_trainer import (
    FLOW_CHANNELS,
    TORCH_AVAILABLE,
    append_prediction_rows,
    augment_flow_tensor,
    build_case_cache,
    build_epoch_row,
    build_point_loaders,
    compute_clinical_categories,
    compute_point_fold_tensors,
    discover_cases,
    evaluate_tensor_model,
    get_optimizer,
    get_scheduler,
    load_checkpoint_weights,
    make_cv_splits,
    predict_tensor_model,
    set_seed,
    write_epoch_log,
)
from model_architectures import ClinicalAnchoredLateFusionClassifier, ClinicalEncoder, PointEncoder

if not TORCH_AVAILABLE:
    print("ERROR: PyTorch is required", file=sys.stderr)
    sys.exit(1)

SEED, CV_SEED = 42, 42
METADATA_PATH, DATA_DIR = "metadata.csv", "flow_data/full_accuracy2_copy"
OUTPUT_ROOT = Path("results_V14_suite")
BACKBONE = "pointnext"
N_FOLDS, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY = 5, 6, 220, 3e-4, 2e-4
TARGET_N = 4096
EARLY_STOP_PATIENCE, USE_AMP, LABEL_SMOOTHING = 35, True, 0.05
AUX_LOSS_WEIGHT, DROPOUT, EMBED_DIM = 0.15, 0.30, 256
FLOW_CHANNEL_DROPOUT = float(os.environ.get("v14_FLOW_CHANNEL_DROPOUT", 0.05))
FLOW_NOISE_STD = float(os.environ.get("v14_FLOW_NOISE_STD", 0.01))
MAX_FLOW_WEIGHT = float(os.environ.get("v14_MAX_FLOW_WEIGHT", 0.50))
INITIAL_FLOW_WEIGHT = float(os.environ.get("v14_INITIAL_FLOW_WEIGHT", 0.25))
BASE_AUX_WEIGHT = float(os.environ.get("v14_BASE_AUX_WEIGHT", 0.35))
FLOW_AUX_WEIGHT = float(os.environ.get("v14_FLOW_AUX_WEIGHT", 0.20))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_geometry_flow_clinical_experiment(df: pd.DataFrame, output_dir: Path):
    print(f"Training GEOMETRY+FLOW+CLINICAL model at {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    categories = compute_clinical_categories(df)
    cache = build_case_cache(df)
    cv_splits, split_strategy = make_cv_splits(df, N_FOLDS, CV_SEED)
    print(f"  CV split: {split_strategy}")
    fold_metrics = []
    prediction_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n=== FOLD {fold_idx + 1}/{N_FOLDS} ===")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_xyz, train_flow, train_clin, train_labels, val_xyz, val_flow, val_clin, val_labels = (
            compute_point_fold_tensors(train_df, val_df, cache, categories, target_n=TARGET_N)
        )

        train_loader, val_loader = build_point_loaders(
            (train_xyz, train_flow, train_clin, train_labels),
            (val_xyz, val_flow, val_clin, val_labels),
            batch_size=BATCH_SIZE,
        )

        clin_dim = train_clin.shape[1]
        geo_encoder = PointEncoder(
            in_channel=0, embed_dim=EMBED_DIM, backbone=BACKBONE, dropout=DROPOUT
        )
        flow_encoder = PointEncoder(
            in_channel=FLOW_CHANNELS, embed_dim=EMBED_DIM, backbone=BACKBONE, dropout=DROPOUT
        )
        clin_encoder = ClinicalEncoder(clin_dim, embed_dim=EMBED_DIM, dropout=DROPOUT)

        classifier = ClinicalAnchoredLateFusionClassifier(
            [
                {"name": "geometry", "module": geo_encoder},
                {"name": "flow", "module": flow_encoder},
                {"name": "clinical", "module": clin_encoder},
            ],
            embed_dim=EMBED_DIM,
            dropout=DROPOUT,
            max_flow_weight=MAX_FLOW_WEIGHT,
            initial_flow_weight=INITIAL_FLOW_WEIGHT,
        )
        classifier = classifier.to(DEVICE)

        optimizer = get_optimizer(classifier, LR, WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        scheduler = get_scheduler(optimizer, EPOCHS, len(train_loader))

        best_auc, patience_counter = 0.0, 0
        best_model_path = output_dir / f"fold_{fold_idx}_best.pt"
        epoch_rows = []
        hyperparams = {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "target_n": TARGET_N,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "aux_loss_weight": AUX_LOSS_WEIGHT,
            "dropout": DROPOUT,
            "embed_dim": EMBED_DIM,
            "flow_channels": FLOW_CHANNELS,
            "flow_channel_dropout": FLOW_CHANNEL_DROPOUT,
            "flow_noise_std": FLOW_NOISE_STD,
            "max_flow_weight": MAX_FLOW_WEIGHT,
            "initial_flow_weight": INITIAL_FLOW_WEIGHT,
            "base_aux_weight": BASE_AUX_WEIGHT,
            "flow_aux_weight": FLOW_AUX_WEIGHT,
            "fusion": "clinical_anchored_late_fusion",
            "backbone": BACKBONE,
        }

        for epoch in range(EPOCHS):
            classifier.train()
            train_loss = 0.0
            n_train = 0

            for xb, fb, cb, yb in train_loader:
                xb, fb, cb, yb = xb.to(DEVICE), fb.to(DEVICE), cb.to(DEVICE), yb.to(DEVICE)
                fb = augment_flow_tensor(fb, FLOW_CHANNEL_DROPOUT, FLOW_NOISE_STD)

                with torch.amp.autocast("cuda", enabled=USE_AMP and DEVICE.type == "cuda"):
                    logits, auxiliary = classifier(xb, fb, cb)
                    ce_loss = criterion(logits, yb)
                    base_loss = criterion(auxiliary["geometry_clinical"], yb)
                    flow_loss = criterion(auxiliary["geometry_flow"], yb)
                    loss = ce_loss + BASE_AUX_WEIGHT * base_loss + FLOW_AUX_WEIGHT * flow_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item() * len(yb)
                n_train += len(yb)

            val_metrics = evaluate_tensor_model(classifier, val_loader, criterion, DEVICE, USE_AMP)
            val_auc = val_metrics.get("auc", 0.0)
            epoch_rows.append(
                build_epoch_row(
                    "geometry_flow_clinical_pointnext",
                    fold_idx,
                    epoch + 1,
                    train_loss / max(1, n_train),
                    val_metrics,
                    optimizer,
                    hyperparams,
                )
            )

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}: val_auc={val_auc:.4f}")

            if val_auc > best_auc:
                best_auc, patience_counter = val_auc, 0
                torch.save(classifier.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    break
        write_epoch_log(output_dir, fold_idx, epoch_rows)

        classifier.load_state_dict(load_checkpoint_weights(best_model_path, DEVICE))
        val_metrics, labels, probs, preds = predict_tensor_model(
            classifier, val_loader, criterion, DEVICE, USE_AMP
        )
        val_metrics["flow_weight"] = float(classifier.current_flow_weight().detach().cpu())
        append_prediction_rows(prediction_rows, fold_idx, val_df, labels, probs, preds)
        val_metrics["split_strategy"] = split_strategy
        fold_metrics.append(val_metrics)

    pd.DataFrame(fold_metrics).to_csv(output_dir / "fold_summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "pooled_predictions.csv", index=False)
    print("\nGeometry+Flow+Clinical training complete.")


def main():
    global TARGET_N
    parser = argparse.ArgumentParser(description="Train full fusion model (V14 modular)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--metadata-path", default=METADATA_PATH)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-n", type=int, default=TARGET_N)
    args = parser.parse_args()

    set_seed(args.seed)
    TARGET_N = args.target_n
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else OUTPUT_ROOT / f"geometry_flow_clinical_pointnext_seed_{args.seed}"
    )

    df = discover_cases(args.data_dir, args.metadata_path)
    if len(df) == 0:
        print(f"ERROR: No samples found under {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Training GEOMETRY+FLOW+CLINICAL model (seed={args.seed})")
    run_geometry_flow_clinical_experiment(df, output_dir)


if __name__ == "__main__":
    main()
