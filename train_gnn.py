# Version 14 source snapshot
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

from base_trainer import (
    TORCH_AVAILABLE,
    append_prediction_rows,
    build_epoch_row,
    classification_report_dict,
    discover_cases,
    jitter,
    load_checkpoint_weights,
    load_graph_case,
    make_cv_splits,
    set_seed,
    so3_rotate,
    write_epoch_log,
)
from model_architectures import GraphEncoder

if not TORCH_AVAILABLE:
    print("ERROR: PyTorch is required", file=sys.stderr)
    sys.exit(1)

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader

    HAS_PYG = True
except Exception:
    HAS_PYG = False


SEED = 42
METADATA_PATH = "metadata.csv"
DATA_DIR = "flow_data/full_accuracy2_copy"
OUTPUT_ROOT = Path("results_V14_suite")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_dtype():
    if (
        DEVICE.type == "cuda"
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return torch.float16


def _autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=_amp_dtype())


def _finite_forward(model, batch, criterion, use_amp: bool, context: str):
    """Run one forward pass, retrying in FP32 if autocast becomes unstable."""
    with _autocast(use_amp):
        logits, _ = model(batch)
        loss = criterion(logits, batch.y)

    if torch.isfinite(logits).all() and torch.isfinite(loss):
        return logits, loss, False

    if use_amp:
        del logits, loss
        with _autocast(False):
            logits, _ = model(batch)
            loss = criterion(logits, batch.y)
        if torch.isfinite(logits).all() and torch.isfinite(loss):
            return logits, loss, True

    nonfinite_inputs = int((~torch.isfinite(batch.x)).sum().item())
    raise FloatingPointError(
        f"Non-finite GNN output during {context}; input graph contains "
        f"{nonfinite_inputs} non-finite node values. Re-run without --amp and, "
        "if the problem persists, reduce --lr."
    )


def _positive_probability(logits: torch.Tensor) -> torch.Tensor:
    margins = logits[:, 1].float() - logits[:, 0].float()
    return torch.sigmoid(margins)


def _load_cached_graph(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class LazyGraphDataset(Dataset):
    """Disk-backed graph dataset with a small bounded RAM cache."""

    CACHE_VERSION = 2

    def __init__(
        self,
        frame: pd.DataFrame,
        cache_dir: Path,
        include_flow: bool,
        k: int,
        target_n: int,
        augment: bool,
        seed: int,
        ram_cache_size: int,
        knn_chunk_size: int,
    ):
        self.frame = frame.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.include_flow = bool(include_flow)
        self.k = int(k)
        self.target_n = int(target_n)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.ram_cache_size = max(0, int(ram_cache_size))
        self.knn_chunk_size = max(1, int(knn_chunk_size))
        self._ram_cache: OrderedDict[str, object] = OrderedDict()

    def __len__(self) -> int:
        return len(self.frame)

    def _cache_key(self, row) -> str:
        source = Path(str(row["filepath"])).resolve()
        stat = source.stat()
        descriptor = "|".join(
            [
                str(self.CACHE_VERSION),
                str(source),
                str(stat.st_size),
                str(stat.st_mtime_ns),
                str(int(row["target"])),
                str(int(self.include_flow)),
                str(self.k),
                str(self.target_n),
                str(self.seed),
            ]
        )
        return hashlib.sha1(descriptor.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        directory = self.cache_dir / key[:2]
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{key}.pt"

    def _build_graph(self, row, key: str, cache_path: Path):
        graph_seed = int(key[:16], 16) % (2**32)
        graph = load_graph_case(
            row["filepath"],
            int(row["target"]),
            augment=False,
            include_flow=self.include_flow,
            k=self.k,
            target_n=self.target_n,
            make_undirected=False,
            store_edge_attr=False,
            knn_chunk_size=self.knn_chunk_size,
            rng=np.random.default_rng(graph_seed),
        )
        graph.edge_index = graph.edge_index.to(torch.int32)
        temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(graph, temporary)
        os.replace(temporary, cache_path)
        return graph

    def _base_graph(self, row):
        key = self._cache_key(row)
        if key in self._ram_cache:
            graph = self._ram_cache.pop(key)
            self._ram_cache[key] = graph
            return graph

        cache_path = self._cache_path(key)
        graph = (
            _load_cached_graph(cache_path)
            if cache_path.exists()
            else self._build_graph(row, key, cache_path)
        )
        graph.edge_index = graph.edge_index.long()
        if self.ram_cache_size > 0:
            self._ram_cache[key] = graph
            while len(self._ram_cache) > self.ram_cache_size:
                self._ram_cache.popitem(last=False)
        return graph

    def __getitem__(self, index: int):
        row = self.frame.iloc[int(index)]
        graph = self._base_graph(row).clone()
        if self.augment:
            xyz = graph.pos.numpy().copy()
            xyz = so3_rotate(jitter(xyz, sigma=0.003, clip=0.02)).astype(np.float32)
            graph.pos = torch.from_numpy(xyz)
            graph.x = graph.x.clone()
            graph.x[:, :3] = graph.pos
        for name in ("x", "pos"):
            tensor = getattr(graph, name)
            if not torch.isfinite(tensor).all():
                count = int((~torch.isfinite(tensor)).sum().item())
                raise ValueError(
                    f"Graph for {row['filepath']} contains {count} non-finite {name} values. "
                    "Remove its cached graph and inspect the source CSV."
                )
        return graph


class FocalCrossEntropy(nn.Module):
    def __init__(
        self, gamma: float = 1.5, weight: torch.Tensor | None = None, label_smoothing: float = 0.03
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("class_weight", weight)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.softmax(logits, dim=1).gather(1, targets.view(-1, 1)).squeeze(1).clamp_min(1e-6)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = np.sqrt(counts.sum() / (2.0 * counts))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def _train_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(labels.astype(int), minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    sample_weights = 1.0 / counts[labels.astype(int)]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def _evaluate(model, loader, criterion, use_amp: bool, pin_memory: bool):
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    total_loss = 0.0
    n_items = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(DEVICE, non_blocking=pin_memory)
            logits, loss, used_fp32_fallback = _finite_forward(
                model,
                batch,
                criterion,
                use_amp=use_amp,
                context=f"evaluation batch {batch_idx}",
            )
            if used_fp32_fallback:
                print(f"  Warning: evaluation batch {batch_idx} required an FP32 retry")
            probs = _positive_probability(logits)
            total_loss += float(loss.item()) * int(batch.y.numel())
            n_items += int(batch.y.numel())
            labels.extend(batch.y.detach().cpu().numpy().astype(int).tolist())
            scores.extend(probs.detach().cpu().numpy().astype(float).tolist())
            del batch, logits, loss, probs
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    if not np.isfinite(scores_array).all():
        raise FloatingPointError(
            "Evaluation produced non-finite probabilities before metric calculation"
        )
    metrics = classification_report_dict(labels_array, scores_array)
    metrics["loss"] = total_loss / max(1, n_items)
    return metrics, labels_array, scores_array


def run_gnn_experiment(df: pd.DataFrame, output_dir: Path, args) -> None:
    if not HAS_PYG:
        raise RuntimeError("torch_geometric is required for GNN training")

    include_flow = not args.no_flow
    model_name = "gnn_flow_geometry" if include_flow else "gnn_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    cv_splits, split_strategy = make_cv_splits(df, args.folds, args.cv_seed)
    print(f"Training {model_name} at {output_dir}")
    print(f"  Device: {DEVICE}; CV split: {split_strategy}")
    print(
        f"  Physical batch: {args.batch_size}; gradient accumulation: {args.accumulation_steps}; "
        f"effective batch: {args.batch_size * args.accumulation_steps}"
    )

    graph_cache_dir = (
        Path(args.graph_cache_dir) if args.graph_cache_dir else output_dir / "graph_cache"
    )
    print(f"  Disk graph cache: {graph_cache_dir}")

    fold_metrics: list[dict] = []
    pooled_rows: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n=== FOLD {fold_idx + 1}/{len(cv_splits)} ===")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_graphs = LazyGraphDataset(
            train_df,
            cache_dir=graph_cache_dir,
            include_flow=include_flow,
            k=args.k,
            target_n=args.target_n,
            augment=True,
            seed=args.seed,
            ram_cache_size=args.ram_cache_size,
            knn_chunk_size=args.knn_chunk_size,
        )
        val_graphs = LazyGraphDataset(
            val_df,
            cache_dir=graph_cache_dir,
            include_flow=include_flow,
            k=args.k,
            target_n=args.target_n,
            augment=False,
            seed=args.seed,
            ram_cache_size=args.ram_cache_size,
            knn_chunk_size=args.knn_chunk_size,
        )

        train_labels = train_df["target"].to_numpy(dtype=int)
        loader_kwargs = {
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
        }
        if args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 1
        train_loader = PyGDataLoader(
            train_graphs,
            batch_size=args.batch_size,
            sampler=_train_sampler(train_labels),
            shuffle=False,
            **loader_kwargs,
        )
        val_loader = PyGDataLoader(
            val_graphs,
            batch_size=args.eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        )

        input_dim = int(train_graphs[0].x.shape[1])
        model = GraphEncoder(
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            dropout=args.dropout,
            input_dim=input_dim,
            num_heads=args.gat_heads,
            num_layers=args.gat_layers,
            checkpoint_layers=args.gradient_checkpointing,
        ).to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=max(1, math.ceil(len(train_loader) / args.accumulation_steps)),
            pct_start=0.12,
            div_factor=10.0,
            final_div_factor=100.0,
        )
        criterion = FocalCrossEntropy(
            gamma=args.focal_gamma,
            weight=_class_weights(train_labels),
            label_smoothing=args.label_smoothing,
        )

        use_amp = bool(args.amp and DEVICE.type == "cuda")
        scaler_enabled = bool(use_amp and _amp_dtype() == torch.float16)
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        if fold_idx == 0:
            precision = str(_amp_dtype()).replace("torch.", "") if use_amp else "float32"
            print(f"  Compute precision: {precision}; GradScaler: {scaler_enabled}")
        best_auc = -float("inf")
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = output_dir / f"fold_{fold_idx}_best.pt"
        epoch_rows: list[dict] = []
        hyperparams = {
            "batch_size": args.batch_size,
            "accumulation_steps": args.accumulation_steps,
            "effective_batch_size": args.batch_size * args.accumulation_steps,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "hidden_dim": args.hidden_dim,
            "embed_dim": args.embed_dim,
            "include_flow": int(include_flow),
            "input_dim": input_dim,
            "target_n": args.target_n,
            "knn_k": args.k,
            "gat_heads": args.gat_heads,
            "gat_layers": args.gat_layers,
            "gradient_checkpointing": int(args.gradient_checkpointing),
            "focal_gamma": args.focal_gamma,
        }

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        consecutive_skipped_updates = 0
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            n_train = 0
            optimizer.zero_grad(set_to_none=True)
            n_batches = len(train_loader)
            try:
                for batch_idx, batch in enumerate(train_loader):
                    batch = batch.to(DEVICE, non_blocking=args.pin_memory)
                    group_start = (batch_idx // args.accumulation_steps) * args.accumulation_steps
                    group_size = min(args.accumulation_steps, n_batches - group_start)
                    logits, raw_loss, used_fp32_fallback = _finite_forward(
                        model,
                        batch,
                        criterion,
                        use_amp=use_amp,
                        context=f"fold {fold_idx + 1}, epoch {epoch + 1}, batch {batch_idx}",
                    )
                    if used_fp32_fallback:
                        print(
                            f"  Warning: fold {fold_idx + 1}, epoch {epoch + 1}, "
                            f"batch {batch_idx} required an FP32 retry"
                        )
                    loss = raw_loss / group_size
                    scaler.scale(loss).backward()
                    train_loss += float(raw_loss.item()) * int(batch.y.numel())
                    n_train += int(batch.y.numel())

                    should_step = (
                        batch_idx + 1
                    ) % args.accumulation_steps == 0 or batch_idx + 1 == n_batches
                    if should_step:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.grad_clip
                        )
                        if not torch.isfinite(grad_norm):
                            consecutive_skipped_updates += 1
                            optimizer.zero_grad(set_to_none=True)
                            scaler.update()
                            print(
                                f"  Warning: skipped non-finite gradient update "
                                f"({consecutive_skipped_updates} consecutive)"
                            )
                            if consecutive_skipped_updates >= 5:
                                raise FloatingPointError(
                                    "Five consecutive GNN updates had non-finite gradients. "
                                    "Re-run without --amp and reduce --lr."
                                )
                            del batch, logits, raw_loss, loss
                            continue
                        scaler.step(optimizer)
                        scaler.update()
                        consecutive_skipped_updates = 0
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                    del batch, logits, raw_loss, loss
            except torch.cuda.OutOfMemoryError as exc:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA ran out of memory despite memory-safe batching. Re-run with "
                    "--batch-size 1 --eval-batch-size 1 --ram-cache-size 0; if needed, "
                    "reduce --k before reducing --target-n."
                ) from exc

            val_metrics, _, _ = _evaluate(
                model,
                val_loader,
                criterion,
                use_amp=use_amp,
                pin_memory=args.pin_memory,
            )
            val_auc = float(val_metrics.get("auc", 0.0))
            val_loss = float(val_metrics["loss"])
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

            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1:3d}: val_auc={val_auc:.4f}, val_loss={val_loss:.4f}")

            improved = val_auc > best_auc + 1e-4 or (
                np.isclose(val_auc, best_auc, atol=1e-4) and val_loss < best_val_loss
            )
            if improved:
                best_auc = val_auc
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        write_epoch_log(output_dir, fold_idx, epoch_rows)
        model.load_state_dict(load_checkpoint_weights(best_model_path, DEVICE))
        val_metrics, val_labels, val_scores = _evaluate(
            model,
            val_loader,
            criterion,
            use_amp=use_amp,
            pin_memory=args.pin_memory,
        )
        val_metrics["split_strategy"] = split_strategy
        val_metrics["best_epoch_auc"] = best_auc
        fold_metrics.append(val_metrics)

        val_predictions = (val_scores > 0.5).astype(int)
        append_prediction_rows(
            pooled_rows, fold_idx, val_df, val_labels, val_scores, val_predictions
        )

        if DEVICE.type == "cuda":
            peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"  Peak CUDA memory: {peak_gib:.2f} GiB")
        del (
            train_loader,
            val_loader,
            train_graphs,
            val_graphs,
            model,
            optimizer,
            scheduler,
            criterion,
            scaler,
        )
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    fold_frame = pd.DataFrame(fold_metrics)
    pooled_frame = pd.DataFrame(pooled_rows)
    fold_frame.to_csv(output_dir / "fold_summary.csv", index=False)
    pooled_frame.to_csv(output_dir / "pooled_predictions.csv", index=False)

    pooled_metrics = classification_report_dict(pooled_frame["label"], pooled_frame["prob"])
    pd.DataFrame([pooled_metrics]).to_csv(output_dir / "pooled_metrics.csv", index=False)
    pd.DataFrame(
        [
            [pooled_metrics["tn"], pooled_metrics["fp"]],
            [pooled_metrics["fn"], pooled_metrics["tp"]],
        ],
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    ).to_csv(output_dir / "pooled_confusion_matrix.csv")
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"\nGNN training complete: {output_dir}")
    print(f"Pooled AUROC: {pooled_metrics['auc']:.4f}; AUPRC: {pooled_metrics['pr_auc']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train optimized V14 GNN rupture model")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument("--metadata-path", default=METADATA_PATH)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--no-flow", action="store_true", help="Use xyz-only node features for geometry-only GNN"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=735,
        help="Fail unless this many known-status cases are discovered; use 0 to disable the check.",
    )
    parser.add_argument("--target-n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1, help="Physical GPU batch size")
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--gat-heads", type=int, default=2)
    parser.add_argument("--gat-layers", type=int, default=3)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CUDA autocast (BF16 when supported). FP32 is the stability-first default.",
    )
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ram-cache-size", type=int, default=4)
    parser.add_argument("--graph-cache-dir", default=None)
    parser.add_argument("--knn-chunk-size", type=int, default=512)
    args = parser.parse_args()

    for name in ("batch_size", "accumulation_steps", "eval_batch_size", "gat_heads", "gat_layers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.num_workers < 0 or args.ram_cache_size < 0:
        parser.error("--num-workers and --ram-cache-size cannot be negative")

    set_seed(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    include_flow = not args.no_flow
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else OUTPUT_ROOT
        / f"{'gnn_flow_geometry_optimized' if include_flow else 'gnn_geometry_optimized'}_seed_{args.seed}"
    )
    df = discover_cases(args.data_dir, args.metadata_path)
    if len(df) == 0:
        print(f"ERROR: No samples found under {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    if args.expected_cases > 0 and len(df) != args.expected_cases:
        raise RuntimeError(
            f"Expected {args.expected_cases} known-status cases, but discovered {len(df)} under "
            f"{args.data_dir}. Use flow_data/full_accuracy2_copy for the 735-case cohort, "
            "or set --expected-cases 0 only for an intentional subset run."
        )

    print(
        f"Samples: {len(df)}; ruptured: {int((df['target'] == 1).sum())}; unruptured: {int((df['target'] == 0).sum())}"
    )
    run_gnn_experiment(df, output_dir, args)


if __name__ == "__main__":
    main()
