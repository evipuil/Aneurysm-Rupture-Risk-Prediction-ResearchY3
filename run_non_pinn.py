# Version 12 source snapshot
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
TRAINER_DIR = SCRIPT_DIR
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results_V12_suite"
DEFAULT_FEATURE_OUTPUT = PROJECT_ROOT / "results_v12_feature_extraction"


def run_step(name: str, command: list[str], continue_on_error: bool = False) -> bool:
    print(f"\n=== {name} ===")
    print(" ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode == 0:
        print(f"Completed {name}")
        return True

    print(f"FAILED {name} with exit code {result.returncode}", file=sys.stderr)
    if not continue_on_error:
        raise SystemExit(result.returncode)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Run all v12 non-PINN trainers, then feature extraction"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-path", default=str(PROJECT_ROOT / "metadata.csv"))
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "predictions" / "pinn_corrected"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--feature-output-dir", default=str(DEFAULT_FEATURE_OUTPUT))
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    output_root = Path(args.output_root)
    feature_output_dir = Path(args.feature_output_dir)

    trainers = [
        (
            "Geometry",
            TRAINER_DIR / "train_geometry.py",
            output_root / f"geometry_pointnet2_seed_{args.seed}",
        ),
        (
            "Flow+Geometry",
            TRAINER_DIR / "train_flow_geometry.py",
            output_root / f"flow_geometry_seed_{args.seed}",
        ),
        ("Clinical", TRAINER_DIR / "train_clinical.py", output_root / f"clinical_seed_{args.seed}"),
        (
            "Geometry+Clinical",
            TRAINER_DIR / "train_geometry_clinical.py",
            output_root / f"geometry_clinical_seed_{args.seed}",
        ),
        (
            "Geometry+Flow+Clinical",
            TRAINER_DIR / "train_geometry_flow_clinical.py",
            output_root / f"geometry_flow_clinical_seed_{args.seed}",
        ),
        ("GNN", TRAINER_DIR / "train_gnn.py", output_root / f"gnn_seed_{args.seed}"),
    ]

    completed = []
    failed = []
    for name, script_path, output_dir in trainers:
        ok = run_step(
            name,
            [
                python,
                str(script_path),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(output_dir),
            ],
            continue_on_error=args.continue_on_error,
        )
        (completed if ok else failed).append(name)

    feature_ok = run_step(
        "Feature Extraction",
        [
            python,
            str(TRAINER_DIR / "feature_extraction.py"),
            "--metadata-path",
            str(args.metadata_path),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(feature_output_dir),
            "--seed",
            str(args.seed),
        ],
        continue_on_error=args.continue_on_error,
    )
    (completed if feature_ok else failed).append("Feature Extraction")

    print("\n=== v12 non-PINN run complete ===")
    print("Completed:", ", ".join(completed) if completed else "none")
    if failed:
        print("Failed:", ", ".join(failed), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
