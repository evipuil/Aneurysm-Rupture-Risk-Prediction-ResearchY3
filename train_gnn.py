# Version 12 source snapshot
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_trainer import (
    TORCH_AVAILABLE,
    StratifiedKFold,
    build_epoch_row,
    discover_cases,
    load_checkpoint_weights,
    load_graph_case,
    set_seed,
    write_epoch_log,
)
from model_architectures import GraphEncoder

if not TORCH_AVAILABLE:
    print("ERROR: PyTorch is required", file=sys.stderr)
    sys.exit(1)

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader

    HAS_PYG = True
except ImportError:
    HAS_PYG = False

SEED, CV_SEED = 42, 42
METADATA_PATH, DATA_DIR = "metadata.csv", "predictions/pinn_corrected"
OUTPUT_ROOT = Path("results_V12_suite")
N_FOLDS, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY = 5, 6, 220, 3e-4, 2e-4
EARLY_STOP_PATIENCE, DROPOUT, EMBED_DIM = 35, 0.30, 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_gnn_experiment(df: pd.DataFrame, output_dir: Path):
    """Train graph neural network model."""
    if not HAS_PYG:
        raise RuntimeError("torch_geometric is required for GNN training")

    print(f"Training GNN model at {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["target"])):
        print(f"\n=== FOLD {fold_idx + 1}/{N_FOLDS} ===")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_graphs = [
            load_graph_case(row["filepath"], int(row["target"]), augment=True)
            for _, row in train_df.iterrows()
        ]
        val_graphs = [
            load_graph_case(row["filepath"], int(row["target"]), augment=False)
            for _, row in val_df.iterrows()
        ]

        train_loader = PyGDataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = PyGDataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)

        model = GraphEncoder(hidden_dim=128, embed_dim=EMBED_DIM, dropout=DROPOUT).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        best_auc, patience_counter = 0.0, 0
        best_model_path = output_dir / f"fold_{fold_idx}_best.pt"
        epoch_rows = []
        hyperparams = {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "embed_dim": EMBED_DIM,
        }

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            n_train = 0
            for batch in train_loader:
                batch = batch.to(DEVICE)
                logits, _ = model(batch)
                loss = criterion(logits, batch.y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * int(batch.y.numel())
                n_train += int(batch.y.numel())

            model.eval()
            with torch.no_grad():
                val_preds, val_labels = [], []
                val_loss = 0.0
                n_val = 0
                for batch in val_loader:
                    batch = batch.to(DEVICE)
                    logits, _ = model(batch)
                    loss = criterion(logits, batch.y)
                    val_loss += loss.item() * int(batch.y.numel())
                    n_val += int(batch.y.numel())
                    val_preds.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
                    val_labels.extend(batch.y.cpu().numpy())

                from base_trainer import classification_report_dict

                val_metrics = classification_report_dict(val_labels, val_preds)
                val_metrics["loss"] = val_loss / max(1, n_val)
                val_auc = val_metrics.get("auc", 0.0)
            epoch_rows.append(
                build_epoch_row(
                    "gnn",
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
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    break
        write_epoch_log(output_dir, fold_idx, epoch_rows)

        model.load_state_dict(load_checkpoint_weights(best_model_path, DEVICE))
        model.eval()
        with torch.no_grad():
            val_preds, val_labels = [], []
            for batch in val_loader:
                batch = batch.to(DEVICE)
                logits, _ = model(batch)
                val_preds.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())

            from base_trainer import classification_report_dict

            val_metrics = classification_report_dict(val_labels, val_preds)

        fold_metrics.append(val_metrics)

    pd.DataFrame(fold_metrics).to_csv(output_dir / "fold_summary.csv", index=False)
    print("\nGNN training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train GNN rupture model (v12 modular)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"gnn_seed_{args.seed}"

    df = discover_cases(DATA_DIR, METADATA_PATH)
    if len(df) == 0:
        print("ERROR: No samples found", file=sys.stderr)
        sys.exit(1)

    print(f"Training GNN model (seed={args.seed})")
    print(f"  Samples: {len(df)}, Ruptured: {(df['target'] == 1).sum()}")
    run_gnn_experiment(df, output_dir)


if __name__ == "__main__":
    main()
