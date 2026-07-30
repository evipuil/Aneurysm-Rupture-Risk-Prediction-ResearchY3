#!/usr/bin/env python3
# Version 1 source snapshot
"""
train_pointnetpp.py

Balanced, normalized, SO(3)-augmented PointNet++ training with folder-level validation split.
Expect folder structure:
  data/
    unruptured/
      *.txt
    ruptured/
      *.txt

Each txt is "x y z" with 1024 rows (or will be cropped/padded).
"""

import glob
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset


# Utils: indexing + FPS
def index_points(points, idx):
    # points: B x N x C
    # idx: B x S
    B, N, C = points.shape
    S = idx.shape[1]
    idx_expanded = idx.unsqueeze(-1).expand(B, S, C)  # B x S x C
    new_points = torch.gather(points, 1, idx_expanded)
    return new_points  # B x S x C


def farthest_point_sample(xyz, npoint):
    # xyz: B x N x 3
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device, dtype=torch.float32) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_ar = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_ar, farthest].unsqueeze(1)  # B x 1 x 3
        dist = torch.sum((xyz - centroid) ** 2, -1)  # B x N
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids  # B x npoint


# Dataset
class AneurysmDataset(Dataset):
    def __init__(self, files_labels, augment=False):
        self.files_labels = files_labels
        self.augment = augment

    def __len__(self):
        return len(self.files_labels)

    def __getitem__(self, idx):
        path, label = self.files_labels[idx]
        pts = np.loadtxt(path).astype(np.float32)
        if pts.shape[0] < 1024:
            pad = np.zeros((1024 - pts.shape[0], 3), dtype=np.float32)
            pts = np.vstack([pts, pad])
        else:
            pts = pts[:1024, :]

        # normalize
        pts = pts - np.mean(pts, axis=0)
        max_dist = np.max(np.linalg.norm(pts, axis=1))
        if max_dist > 0:
            pts = pts / max_dist

        if self.augment:
            pts = self.so3_rotate(pts)
            pts = self.jitter(pts)
            s = np.random.uniform(0.95, 1.05)
            pts = pts * s

        return torch.from_numpy(pts).float(), int(label)

    @staticmethod
    def jitter(points, sigma=0.01, clip=0.05):
        jittered = points + np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
        return jittered.astype(np.float32)

    @staticmethod
    def so3_rotate(points):
        rx = np.random.uniform(0, 2 * np.pi)
        ry = np.random.uniform(0, 2 * np.pi)
        rz = np.random.uniform(0, 2 * np.pi)
        Rx = np.array(
            [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float32
        )
        Ry = np.array(
            [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32
        )
        Rz = np.array(
            [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float32
        )
        R = Rz @ Ry @ Rx
        return (points @ R.T).astype(np.float32)


# Set Abstraction (simple hierarchical)
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.mlp_convs = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv1d(last_ch, out_ch, 1))
            self.mlp_convs.append(nn.BatchNorm1d(out_ch))
            last_ch = out_ch

    def forward(self, xyz, points=None):
        B, N, _ = xyz.shape
        if self.npoint is not None:
            idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, idx)
        else:
            idx = None
            new_xyz = torch.mean(xyz, dim=1, keepdim=True)

        if points is None:
            feats = xyz
        else:
            feats = points.permute(0, 2, 1).contiguous()

        if self.npoint is not None:
            new_feats = index_points(feats, idx)
        else:
            new_feats = feats.mean(dim=1, keepdim=True)

        x = new_feats.permute(0, 2, 1).contiguous()
        for layer in self.mlp_convs:
            x = F.relu(layer(x)) if isinstance(layer, nn.Conv1d) else layer(x)
        return x, new_xyz


# Model
class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, in_channel=3, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, in_channel=128, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=None, in_channel=256, mlp=[256, 512, 1024])

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        B, N, _ = xyz.shape
        l1_pts, l1_xyz = self.sa1(xyz, None)
        l2_pts, l2_xyz = self.sa2(l1_xyz, l1_pts)
        l3_pts, l3_xyz = self.sa3(l2_xyz, l2_pts)
        x = F.adaptive_max_pool1d(l3_pts, 1).view(B, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        return x


# Prepare balanced train/val
def prepare_balanced_train_val(root_dir, val_fraction=0.2, seed=42):
    un_list = sorted(glob.glob(os.path.join(root_dir, "unruptured", "*.txt")))
    ru_list = sorted(glob.glob(os.path.join(root_dir, "ruptured", "*.txt")))
    if len(un_list) == 0 or len(ru_list) == 0:
        raise RuntimeError("No files found — check folder structure under root_dir")

    min_count = min(len(un_list), len(ru_list))
    un_list = un_list[:min_count]
    ru_list = ru_list[:min_count]
    print(f"[INFO] Balanced: using {min_count} per class -> {2 * min_count} total")

    random.seed(seed)
    un_shuf = un_list.copy()
    ru_shuf = ru_list.copy()
    random.shuffle(un_shuf)
    random.shuffle(ru_shuf)

    n_val = max(1, int(math.ceil(val_fraction * min_count)))
    un_val = un_shuf[:n_val]
    un_train = un_shuf[n_val:]
    ru_val = ru_shuf[:n_val]
    ru_train = ru_shuf[n_val:]

    train_files = [(p, 0) for p in un_train] + [(p, 1) for p in ru_train]
    val_files = [(p, 0) for p in un_val] + [(p, 1) for p in ru_val]

    random.shuffle(train_files)
    random.shuffle(val_files)

    print(
        f"[INFO] Train size: {len(train_files)}, Val size: {len(val_files)} (per-class val={n_val})"
    )
    return train_files, val_files


# Training & Validation
def evaluate(model, dataloader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_labels = []
    all_probs = []
    all_preds = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for points, labels in dataloader:
            points, labels = points.to(device), labels.to(device)
            logits = model(points)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * points.size(0)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += points.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5

    return loss_sum / total, correct / total, auc, all_labels, all_preds, all_probs


def train_kfold(
    root_dir="data",
    n_folds=5,
    epochs=100,
    batch_size=16,
    lr=1e-3,
    num_workers=2,
    seed=42,
    save_dir="kfold_geometry_models",
):
    """
    K-Fold Cross-Validation training for robust AUC estimates.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)

    # Load all files
    un_list = sorted(glob.glob(os.path.join(root_dir, "unruptured", "*.txt")))
    ru_list = sorted(glob.glob(os.path.join(root_dir, "ruptured", "*.txt")))
    if len(un_list) == 0 or len(ru_list) == 0:
        raise RuntimeError("No files found — check folder structure under root_dir")

    # Balance classes
    min_count = min(len(un_list), len(ru_list))
    random.shuffle(un_list)
    random.shuffle(ru_list)
    un_list = un_list[:min_count]
    ru_list = ru_list[:min_count]

    # Create file_label_pairs
    file_label_pairs = [(p, 0) for p in un_list] + [(p, 1) for p in ru_list]
    random.shuffle(file_label_pairs)

    print(
        f"[INFO] K-Fold CV with {n_folds} folds on {len(file_label_pairs)} samples ({min_count} per class)"
    )

    paths = [p for p, _ in file_label_pairs]
    labels = np.array([label for _, label in file_label_pairs])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_aucs = []
    fold_accs = []
    all_val_probs = []
    all_val_labels = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'=' * 60}")

        train_pairs = [(paths[i], labels[i]) for i in train_idx]
        val_pairs = [(paths[i], labels[i]) for i in val_idx]

        print(f"[FOLD {fold + 1}] Train: {len(train_pairs)}, Val: {len(val_pairs)}")

        train_ds = AneurysmDataset(train_pairs, augment=True)
        val_ds = AneurysmDataset(val_pairs, augment=False)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            pin_memory=True,
        )

        model = PointNetPlusPlus(num_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss, running_correct, running_total = 0.0, 0, 0

            for points, labels_batch in train_loader:
                points, labels_batch = points.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                logits = model(points)
                loss = criterion(logits, labels_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * points.size(0)
                preds = logits.argmax(dim=1)
                running_correct += (preds == labels_batch).sum().item()
                running_total += points.size(0)

            running_loss / running_total
            train_acc = running_correct / running_total

            val_loss, val_acc, val_auc, _, _, _ = evaluate(model, val_loader, device)
            scheduler.step()

            if epoch % 20 == 0 or val_acc > best_val_acc:
                print(
                    f"  Epoch {epoch:03d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} AUC: {val_auc:.4f}"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(save_dir, f"fold{fold + 1}_best.pth"))

        # Load best model and get final predictions
        model.load_state_dict(
            torch.load(os.path.join(save_dir, f"fold{fold + 1}_best.pth"), weights_only=True)
        )
        _, _, fold_auc, val_labels_fold, val_preds_fold, val_probs_fold = evaluate(
            model, val_loader, device
        )

        fold_acc = np.mean(np.array(val_preds_fold) == np.array(val_labels_fold))

        fold_aucs.append(fold_auc)
        fold_accs.append(fold_acc)
        all_val_probs.extend(val_probs_fold)
        all_val_labels.extend(val_labels_fold)

        print(f"\n[FOLD {fold + 1}] Best Acc: {fold_acc:.4f}, AUC: {fold_auc:.4f}")

    # Aggregate results
    print("\n" + "=" * 60)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"AUC per fold: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"Acc per fold: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"Mean AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})")
    print(f"Mean Acc: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")

    overall_auc = roc_auc_score(all_val_labels, all_val_probs)
    print(f"\nOverall AUC (all folds combined): {overall_auc:.4f}")

    return fold_aucs, fold_accs


def train_main(
    root_dir="data",
    epochs=100,
    batch_size=16,
    lr=1e-3,
    val_fraction=0.2,
    num_workers=2,
    seed=42,
    save_path="best_pointnetpp.pth",
):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_files, val_files = prepare_balanced_train_val(root_dir, val_fraction, seed)
    train_ds, val_ds = (
        AneurysmDataset(train_files, augment=True),
        AneurysmDataset(val_files, augment=False),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNetPlusPlus(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # --- LR Scheduler ---
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.5
    )  # decay LR by 0.5 every 30 epochs

    best_val_acc = 0.0
    best_val_auc = 0.0
    since = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(points)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * points.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += points.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc, val_auc, _, _, _ = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}"
        )

        # step scheduler
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                },
                save_path,
            )
            print(f"  [Saved best model with val_acc={val_acc:.4f}, val_auc={val_auc:.4f}]")

    elapsed = time.time() - since
    print(
        f"Training complete in {elapsed / 60:.2f} minutes. Best val acc: {best_val_acc:.4f}, Best val AUC: {best_val_auc:.4f}"
    )
    print(f"Best model saved to {save_path}")

    # Final evaluation with detailed metrics
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, val_acc, val_auc, val_labels, val_preds, val_probs = evaluate(
        model, val_loader, device
    )

    print("\n" + "=" * 50)
    print("Final Validation Metrics")
    print("=" * 50)
    print(
        classification_report(
            val_labels, val_preds, target_names=["Unruptured", "Ruptured"], zero_division=0
        )
    )
    cm = confusion_matrix(val_labels, val_preds)
    print("Confusion Matrix:")
    print("  Predicted:  Unrupt  Rupt")
    print(f"  Unruptured:  {cm[0, 0]:5d}  {cm[0, 1]:5d}")
    print(f"  Ruptured:    {cm[1, 0]:5d}  {cm[1, 1]:5d}")
    print(f"\nFinal AUC-ROC: {val_auc:.4f}")

    # Sensitivity and Specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")


# Run
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PointNet++ for Aneurysm Rupture (Geometry Only)")
    parser.add_argument("--root_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--num_workers", type=int, default=2, help="Data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_path", type=str, default="best_pointnetpp.pth", help="Model save path"
    )
    parser.add_argument("--kfold", type=int, default=0, help="K-fold CV (0 = single split)")
    parser.add_argument(
        "--kfold_save_dir", type=str, default="kfold_geometry_models", help="K-fold save dir"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PointNet++ Geometry-Only Rupture Classification")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"K-Fold: {args.kfold if args.kfold > 0 else 'Single split'}")
    print("=" * 60)

    if args.kfold > 1:
        train_kfold(
            root_dir=args.root_dir,
            n_folds=args.kfold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            seed=args.seed,
            save_dir=args.kfold_save_dir,
        )
    else:
        train_main(
            root_dir=args.root_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_fraction=args.val_fraction,
            num_workers=args.num_workers,
            seed=args.seed,
            save_path=args.save_path,
        )
