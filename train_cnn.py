# Version 14 source snapshot
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from base_trainer import (
    FLOW_CHANNELS,
    TORCH_AVAILABLE,
    append_prediction_rows,
    build_case_cache,
    build_epoch_row,
    build_point_loaders,
    compute_voxel_fold_tensors,
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

if not TORCH_AVAILABLE:
    print("ERROR: PyTorch is required", file=sys.stderr)
    sys.exit(1)

from model_architectures import VoxelCNNClassifier

SEED = 42
CV_SEED = 42
METADATA_PATH = "metadata.csv"
DATA_DIR = "flow_data/full_accuracy2_copy"
OUTPUT_ROOT = Path("results_V14_suite")
N_FOLDS = 5
BATCH_SIZE = 4
EPOCHS = 220
LR = 2e-4
WEIGHT_DECAY = 2e-4
GRID_SIZE = 24
EARLY_STOP_PATIENCE = 35
USE_AMP = True
LABEL_SMOOTHING = 0.05
AUX_LOSS_WEIGHT = 0.0
DROPOUT = 0.25
BASE_CHANNELS = 24

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_cnn_experiment(
    df: pd.DataFrame,
    output_dir: Path,
    seed: int = SEED,
    n_folds: int = N_FOLDS,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    grid_size: int = GRID_SIZE,
    base_channels: int = BASE_CHANNELS,
    dropout: float = DROPOUT,
    include_flow: bool = True,
):
    """Train a 3D voxel CNN on voxelized geometry, optionally with hemodynamic fields."""
    input_channels = FLOW_CHANNELS + 1 if include_flow else 1
    model_name = "cnn_voxel_flow" if include_flow else "cnn_voxel_geometry"
    mode_label = "geometry + flow" if include_flow else "geometry only"
    print(f"Training VOXEL CNN model ({mode_label}) at {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    cache = build_case_cache(df)
    cv_splits, split_strategy = make_cv_splits(df, n_folds, CV_SEED)
    print(f"  CV split: {split_strategy}")
    fold_metrics = []
    prediction_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n=== FOLD {fold_idx + 1}/{len(cv_splits)} ===")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_vox, train_flow, train_clin, train_labels, val_vox, val_flow, val_clin, val_labels = (
            compute_voxel_fold_tensors(
                train_df, val_df, cache, grid_size=grid_size, include_flow=include_flow
            )
        )

        train_loader, val_loader = build_point_loaders(
            (train_vox, train_flow, train_clin, train_labels),
            (val_vox, val_flow, val_clin, val_labels),
            batch_size=batch_size,
        )

        model = VoxelCNNClassifier(
            in_channels=input_channels,
            base_channels=base_channels,
            dropout=dropout,
        ).to(DEVICE)
        optimizer = get_optimizer(model, lr, WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        scheduler = get_scheduler(optimizer, epochs, len(train_loader))

        best_auc = -1.0
        patience_counter = 0
        best_model_path = output_dir / f"fold_{fold_idx}_best.pt"
        epoch_rows = []
        hyperparams = {
            "batch_size": batch_size,
            "epochs": epochs,
            "grid_size": grid_size,
            "input_channels": input_channels,
            "include_flow": int(include_flow),
            "base_channels": base_channels,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "aux_loss_weight": AUX_LOSS_WEIGHT,
            "dropout": dropout,
        }

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            n_train = 0

            for xb, _fb, _cb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                with torch.amp.autocast("cuda", enabled=USE_AMP and DEVICE.type == "cuda"):
                    logits, aux_loss = model(xb)
                    ce_loss = criterion(logits, yb)
                    loss = ce_loss + AUX_LOSS_WEIGHT * aux_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += float(loss.item()) * len(yb)
                n_train += len(yb)

            val_metrics = evaluate_tensor_model(model, val_loader, criterion, DEVICE, USE_AMP)
            val_auc = val_metrics.get("auc", 0.0)
            epoch_rows.append(
                build_epoch_row(
                    model_name,
                    fold_idx,
                    epoch + 1,
                    train_loss / max(1, n_train),
                    val_metrics,
                    optimizer,
                    hyperparams,
                )
            )

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(
                    f"  Epoch {epoch:3d}: "
                    f"train_loss={train_loss / max(1, n_train):.4f} "
                    f"val_auc={val_auc:.4f} "
                    f"val_loss={val_metrics.get('loss', 0.0):.4f}"
                )

            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"  Early stop at epoch {epoch + 1}")
                    break

        write_epoch_log(output_dir, fold_idx, epoch_rows)

        model.load_state_dict(load_checkpoint_weights(best_model_path, DEVICE))
        val_metrics, labels, probs, preds = predict_tensor_model(
            model, val_loader, criterion, DEVICE, USE_AMP
        )
        append_prediction_rows(prediction_rows, fold_idx, val_df, labels, probs, preds)
        val_metrics["split_strategy"] = split_strategy
        fold_metrics.append(val_metrics)
        print(f"  Best fold AUC: {best_auc:.4f}")

    pd.DataFrame(fold_metrics).to_csv(output_dir / "fold_summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "pooled_predictions.csv", index=False)
    print(f"\nVoxel CNN training complete. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train voxel CNN rupture model (V14 modular)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--metadata-path", default=METADATA_PATH)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--base-channels", type=int, default=BASE_CHANNELS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument(
        "--no-flow",
        action="store_true",
        help="Use occupancy-only voxel grids without hemodynamic channels",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    include_flow = not args.no_flow
    model_name = "cnn_voxel_flow" if include_flow else "cnn_voxel_geometry"
    output_dir = (
        Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"{model_name}_seed_{args.seed}"
    )

    df = discover_cases(args.data_dir, args.metadata_path)
    if len(df) == 0:
        print(f"ERROR: No samples found under {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Training VOXEL CNN model ({'geometry + flow' if include_flow else 'geometry only'}, seed={args.seed})"
    )
    print(f"  Samples: {len(df)}, Ruptured: {(df['target'] == 1).sum()}")
    print(f"  Grid: {args.grid_size}^3, Channels: {FLOW_CHANNELS + 1 if include_flow else 1}")

    run_cnn_experiment(
        df,
        output_dir,
        seed=args.seed,
        n_folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grid_size=args.grid_size,
        base_channels=args.base_channels,
        dropout=args.dropout,
        include_flow=include_flow,
    )


if __name__ == "__main__":
    main()
