# Version 5 source snapshot
import csv
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

# Configuration
METADATA_PATH = "metadata.csv"
DATA_DIR = "predictions/pinn_corrected"
OUTPUT_DIR = "results_clinical_age_sex"
N_FOLDS = 5
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
EARLY_STOP_PATIENCE = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Build MLP (smaller — only 2 input features)


def build_clinical_mlp(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 2),
    )


# Feature engineering (fit on train, transform val)


def process_features(df_slice, train_stats=None):
    ages = df_slice["age"].values.astype(np.float32)
    mu = train_stats["age_mean"] if train_stats else ages.mean()
    sigma = train_stats["age_std"] if train_stats else ages.std()
    ages = (ages - mu) / (sigma + 1e-6)

    sexes = df_slice["sex_enc"].values.astype(np.float32)

    X = np.column_stack([ages, sexes])
    return torch.tensor(X, dtype=torch.float32), {"age_mean": float(mu), "age_std": float(sigma)}


def safe_auc(targets, probs):
    try:
        return roc_auc_score(targets, probs)
    except ValueError:
        return 0.5


# Load metadata and build lookup from case keys to row indices
df = pd.read_csv(METADATA_PATH)
meta_keys = {}
for idx, row in df.iterrows():
    ds = str(row.get("dataset", "")).strip()
    vid = str(row.get("vesselFileID", "")).strip()
    cut = str(row.get("cutToShow", "cut1")).strip() if pd.notna(row.get("cutToShow")) else "cut1"
    for key in [ds, vid]:
        if key:
            meta_keys[f"{key}_{cut}"] = idx
            meta_keys[key] = idx

valid_indices = []
seen = set()
for folder in sorted(os.listdir(DATA_DIR)):
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    if not os.path.exists(os.path.join(folder_path, "hemodynamics_aggregate.csv")):
        continue
    matched_idx = None
    if folder in meta_keys:
        matched_idx = meta_keys[folder]
    else:
        base = re.sub(r"_cut\d+$", "", folder)
        if base in meta_keys:
            matched_idx = meta_keys[base]
    if matched_idx is not None and matched_idx not in seen:
        seen.add(matched_idx)
        valid_indices.append(matched_idx)

df = df.loc[valid_indices].reset_index(drop=True)
print(f"Valid samples: {len(df)}")

df["target"] = (df["status"] == "ruptured").astype(int)
df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
df["sex_enc"] = df["sex"].map({"female": 0, "male": 1}).fillna(0).astype(float)
targets = df["target"].values

# K-fold training
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
pooled_probs, pooled_targets = [], []
fold_summaries = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df, targets)):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    y_train = torch.tensor(targets[train_idx], dtype=torch.long)
    y_val = torch.tensor(targets[val_idx], dtype=torch.long)

    X_train, stats = process_features(train_df)
    X_val, _ = process_features(val_df, train_stats=stats)

    # Inverse-frequency class weights
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    w = torch.tensor([1.0 / max(n_neg, 1), 1.0 / max(n_pos, 1)])
    w = (w / w.sum() * 2.0).to(DEVICE)

    # Balanced sampling: oversample minority class
    sample_weights = torch.where(y_train == 1, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
    sample_weights = sample_weights / sample_weights.sum()
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(y_train),
        replacement=True,
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, sampler=train_sampler
    )
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = build_clinical_mlp(X_train.shape[1]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=w)

    # Early-stopping trigger tracking (save model at trigger epoch, not best spike)
    best_val_auc = -1.0
    patience_counter = 0
    early_stop_v_probs = None
    early_stop_v_labels = None
    early_stop_metrics = None
    early_stop_epoch = None

    fold_csv = os.path.join(OUTPUT_DIR, f"fold_{fold + 1}_metrics.csv")
    fold_roc_dir = os.path.join(OUTPUT_DIR, f"fold_{fold + 1}_roc")
    os.makedirs(fold_roc_dir, exist_ok=True)

    with open(fold_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "train_auc",
                "train_pr_auc",
                "val_loss",
                "val_acc",
                "val_auc",
                "val_pr_auc",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        )
        early_stop_announced = False
        for epoch in range(1, EPOCHS + 1):
            # Train
            model.train()
            t_loss, t_probs, t_labels = 0.0, [], []
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                t_loss += loss.item() * xb.size(0)
                t_probs.extend(F.softmax(logits, 1)[:, 1].detach().cpu().numpy())
                t_labels.extend(yb.cpu().numpy())
            scheduler.step()

            t_loss /= len(train_loader.dataset)
            t_acc = accuracy_score(t_labels, np.array(t_probs) > 0.5)
            t_auc = safe_auc(t_labels, t_probs)
            t_pr_auc = average_precision_score(t_labels, t_probs) if len(set(t_labels)) > 1 else 0.0

            # Validate
            model.eval()
            v_loss, v_probs, v_labels = 0.0, [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb)
                    v_loss += criterion(logits, yb).item() * xb.size(0)
                    v_probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
                    v_labels.extend(yb.cpu().numpy())

            v_loss /= len(val_loader.dataset)
            v_acc = accuracy_score(v_labels, np.array(v_probs) > 0.5)
            v_auc = safe_auc(v_labels, v_probs)
            v_pr_auc = average_precision_score(v_labels, v_probs) if len(set(v_labels)) > 1 else 0.0
            tn, fp, fn, tp = confusion_matrix(
                v_labels, np.array(v_probs) > 0.5, labels=[0, 1]
            ).ravel()

            writer.writerow(
                [
                    epoch,
                    f"{t_loss:.6f}",
                    f"{t_acc:.4f}",
                    f"{t_auc:.4f}",
                    f"{t_pr_auc:.4f}",
                    f"{v_loss:.6f}",
                    f"{v_acc:.4f}",
                    f"{v_auc:.4f}",
                    f"{v_pr_auc:.4f}",
                    tn,
                    fp,
                    fn,
                    tp,
                ]
            )
            # Per-epoch ROC curve data
            fpr, tpr, thr = roc_curve(v_labels, v_probs)
            pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
                os.path.join(fold_roc_dir, f"epoch_{epoch}.csv"), index=False
            )

            if epoch % 25 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}  train_auc={t_auc:.4f}  val_auc={v_auc:.4f}  val_acc={v_acc:.4f}"
                )

            # Track when early stopping WOULD trigger (run all epochs for full logs)
            if v_auc > best_val_auc:
                best_val_auc = v_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE and not early_stop_announced:
                    print(
                        f"    Early-stop patience reached at epoch {epoch}; continuing to log all remaining epochs."
                    )
                    early_stop_announced = True
                if patience_counter >= EARLY_STOP_PATIENCE and early_stop_v_probs is None:
                    early_stop_v_probs = list(v_probs)
                    early_stop_v_labels = list(v_labels)
                    early_stop_metrics = {
                        "val_loss": v_loss,
                        "val_acc": v_acc,
                        "val_auc": v_auc,
                        "tn": tn,
                        "fp": fp,
                        "fn": fn,
                        "tp": tp,
                    }
                    early_stop_epoch = epoch
                    state = model.state_dict()
                    torch.save(
                        state, os.path.join(OUTPUT_DIR, f"early_stop_model_fold_{fold + 1}.pt")
                    )
                    # Legacy-compatible filename now points to early-stop snapshot.
                    torch.save(state, os.path.join(OUTPUT_DIR, f"best_model_fold_{fold + 1}.pt"))

    # Use early-stop-trigger epoch predictions for pooling; fallback to final epoch if never triggered.
    if early_stop_v_probs is not None:
        pooled_probs.extend(early_stop_v_probs)
        pooled_targets.extend(early_stop_v_labels)
        fold_summaries.append({"fold": fold + 1, **early_stop_metrics})
        print(
            f"  Using early-stop epoch {early_stop_epoch} metrics  val_auc={early_stop_metrics['val_auc']:.4f}"
        )
    else:
        torch.save(
            model.state_dict(), os.path.join(OUTPUT_DIR, f"early_stop_model_fold_{fold + 1}.pt")
        )
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"best_model_fold_{fold + 1}.pt"))
        pooled_probs.extend(v_probs)
        pooled_targets.extend(v_labels)
        fold_summaries.append(
            {
                "fold": fold + 1,
                "val_loss": v_loss,
                "val_acc": v_acc,
                "val_auc": v_auc,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

# Fold-averaged summary CSV
sdf = pd.DataFrame(fold_summaries)
avg = sdf.drop(columns=["fold"]).mean()
std = sdf.drop(columns=["fold"]).std()
sdf = pd.concat(
    [
        sdf,
        pd.DataFrame([{"fold": "mean", **avg.to_dict()}]),
        pd.DataFrame([{"fold": "std", **std.to_dict()}]),
    ],
    ignore_index=True,
)
sdf.to_csv(os.path.join(OUTPUT_DIR, "fold_averages.csv"), index=False)

# Pooled ROC & confusion matrix
pooled_targets = np.array(pooled_targets)
pooled_probs = np.array(pooled_probs)

fpr, tpr, thr = roc_curve(pooled_targets, pooled_probs)
pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
    os.path.join(OUTPUT_DIR, "pooled_roc.csv"), index=False
)

cm = confusion_matrix(pooled_targets, pooled_probs > 0.5)
tn, fp, fn, tp = cm.ravel()
pd.DataFrame({"tn": [tn], "fp": [fp], "fn": [fn], "tp": [tp]}).to_csv(
    os.path.join(OUTPUT_DIR, "pooled_cm.csv"), index=False
)

print(
    f"\nPooled AUC: {safe_auc(pooled_targets, pooled_probs):.4f}  "
    f"Pooled Acc: {accuracy_score(pooled_targets, pooled_probs > 0.5):.4f}"
)
