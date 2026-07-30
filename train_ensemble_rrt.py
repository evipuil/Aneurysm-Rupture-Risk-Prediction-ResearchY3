# Version 7 source snapshot
"""
Ensemble classifier with Relative Residence Time (RRT) as an extra per-point
and global feature.

See ensemble_core.train_ensemble for implementation details.
"""

import os
from pathlib import Path

from ensemble_core import train_ensemble


def main():
    train_ensemble(
        output_dir=Path(os.environ.get("V7_OUTPUT_DIR", "results_ensemble_rrt_v7")),
        metadata_path=os.environ.get("V7_METADATA", "metadata.csv"),
        data_dir=os.environ.get("V7_DATA_DIR", "predictions/pinn_corrected"),
        include_rrt=True,
        n_folds=int(os.environ.get("V7_FOLDS", 5)),
        batch_size=int(os.environ.get("V7_BATCH", 8)),
        epochs=int(os.environ.get("V7_EPOCHS", 200)),
        lr=float(os.environ.get("V7_LR", 5e-4)),
        weight_decay=float(os.environ.get("V7_WD", 1e-3)),
        target_n=int(os.environ.get("V7_TARGET_N", 4096)),
        early_stop_patience=int(os.environ.get("V7_PATIENCE", 40)),
        label_smoothing=float(os.environ.get("V7_LS", 0.1)),
        use_amp=os.environ.get("V7_AMP", "1").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    main()
