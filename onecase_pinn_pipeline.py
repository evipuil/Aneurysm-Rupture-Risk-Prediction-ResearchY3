# Version 7 source snapshot
"""
Run the v7 PINN pipeline on exactly one VTP case and keep per-epoch losses
(physics, inlet, wall, etc.) for plotting.

This is a thin wrapper around `pinn_pipeline.py` so model/training behavior
stays consistent with your existing pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pinn_pipeline as base


def str2bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def configure_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base.OUTPUT_DIR = str(output_dir)
    base.TRAINING_LOG_DIR = output_dir / base.TRAINING_LOG_SUBDIR
    base.TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PINN on one VTP case and save epoch losses")
    parser.add_argument("--vtp-file", required=True, help="Path to one .vtp file")
    parser.add_argument(
        "--output-dir", default="predictions/pinn_onecase", help="Directory for outputs"
    )
    parser.add_argument("--unsteady", default="1", help="1/0 (or true/false)")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epoch count for this run"
    )
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "warm_restarts", "none"],
        default="cosine",
        help="LR scheduler for one-case training (default: cosine)",
    )
    parser.add_argument(
        "--loss-ref-ema",
        type=float,
        default=0.98,
        help="EMA factor for adaptive loss references in [0, 0.9999] (default: 0.98)",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Override `EARLY_STOP_PATIENCE` for this run (int epochs)",
    )
    parser.add_argument(
        "--save-checkpoint-every",
        type=int,
        default=None,
        help="Periodic checkpoint interval in epochs (0 disables)",
    )
    parser.add_argument(
        "--fixed-collocation",
        default="0",
        help="Use fixed collocation batch to reduce sampling variance (1/0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible runs",
    )
    args = parser.parse_args()

    vtp_path = Path(args.vtp_file)
    if not vtp_path.exists():
        raise FileNotFoundError(f"VTP file not found: {vtp_path}")

    output_dir = Path(args.output_dir)
    configure_outputs(output_dir)

    unsteady = str2bool(args.unsteady)
    base.SCHEDULER_KIND = str(args.scheduler).strip().lower()
    base.LOSS_REF_EMA = float(args.loss_ref_ema)
    if args.early_stop_patience is not None:
        base.EARLY_STOP_PATIENCE = int(args.early_stop_patience)
    if args.save_checkpoint_every is not None:
        base.SAVE_CHECKPOINT_EVERY = int(args.save_checkpoint_every)
    base.FIXED_COLLOCATION = str2bool(args.fixed_collocation)
    if args.seed is not None:
        base.SEED = int(args.seed)
        try:
            base.np.random.seed(base.SEED)
        except Exception:
            pass
        try:
            base.torch.manual_seed(base.SEED)
        except Exception:
            pass
    if args.epochs is not None:
        if unsteady:
            base.UNSTEADY_EPOCHS = int(args.epochs)
        else:
            base.STEADY_EPOCHS = int(args.epochs)

    base.logger.info(
        "One-case PINN run | case=%s | unsteady=%s | output=%s | scheduler=%s | loss_ref_ema=%.4f",
        vtp_path.name,
        unsteady,
        output_dir,
        base.SCHEDULER_KIND,
        base.LOSS_REF_EMA,
    )

    summary = base.process_case(vtp_path, output_dir, unsteady)

    # Helpful pointers for plotting.
    case_name = vtp_path.stem
    epoch_loss_csv = output_dir / base.TRAINING_LOG_SUBDIR / f"{case_name}_epoch_losses.csv"
    base.logger.info("Epoch loss CSV: %s", epoch_loss_csv)
    base.logger.info("Columns include: epoch,total_loss,physics_loss,wall_loss,inlet_loss")
    base.logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
