# Version 14 source snapshot
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
TRAINER_DIR = SCRIPT_DIR
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results_V14_suite"
DEFAULT_FEATURE_OUTPUT = PROJECT_ROOT / "results_v14_feature_extraction"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all V14 non-PINN trainers, then feature extraction"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-path", default=str(PROJECT_ROOT / "metadata.csv"))
    parser.add_argument(
        "--data-dir", default=str(PROJECT_ROOT / "flow_data" / "full_accuracy2_copy")
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--feature-output-dir", default=str(DEFAULT_FEATURE_OUTPUT))
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    output_root = Path(args.output_root)
    feature_output_dir = Path(args.feature_output_dir)
    metadata_path = Path(args.metadata_path)
    data_dir = Path(args.data_dir)

    if not TRAINER_DIR.exists():
        raise SystemExit(f"Trainer directory not found: {TRAINER_DIR}")
    if not metadata_path.exists():
        raise SystemExit(f"Metadata file not found: {metadata_path}")
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    print("Running V14 non-PINN pipeline")
    print(f"  project root: {PROJECT_ROOT}")
    print(f"  trainer dir: {TRAINER_DIR}")
    print(f"  python: {python}")
    print(f"  seed: {args.seed}")
    print(f"  metadata: {metadata_path}")
    print(f"  data: {data_dir}")
    print(f"  output: {output_root}")
    print(f"  feature output: {feature_output_dir}")

    trainers = [
        (
            "Geometry PointNeXt",
            TRAINER_DIR / "train_geometry.py",
            output_root / f"geometry_pointnext_seed_{args.seed}",
            ["--backbone", "pointnext"],
        ),
        (
            "Flow+Geometry PointNeXt",
            TRAINER_DIR / "train_flow_geometry.py",
            output_root / f"flow_geometry_pointnext_seed_{args.seed}",
            [],
        ),
        (
            "Clinical",
            TRAINER_DIR / "train_clinical.py",
            output_root / f"clinical_seed_{args.seed}",
            [],
        ),
        (
            "Geometry+Clinical PointNeXt",
            TRAINER_DIR / "train_geometry_clinical.py",
            output_root / f"geometry_clinical_pointnext_seed_{args.seed}",
            [],
        ),
        (
            "Geometry+Flow+Clinical PointNeXt",
            TRAINER_DIR / "train_geometry_flow_clinical.py",
            output_root / f"geometry_flow_clinical_pointnext_seed_{args.seed}",
            [],
        ),
        (
            "Voxel CNN + flow",
            TRAINER_DIR / "train_cnn.py",
            output_root / f"cnn_voxel_flow_seed_{args.seed}",
            [],
        ),
        (
            "GNN geometry",
            TRAINER_DIR / "train_gnn.py",
            output_root / f"gnn_geometry_seed_{args.seed}",
            ["--no-flow"],
        ),
    ]

    completed = []
    failed = []
    for name, script_path, output_dir, extra_args in trainers:
        ok = run_step(
            name,
            [
                python,
                str(script_path),
                "--seed",
                str(args.seed),
                "--metadata-path",
                str(metadata_path),
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output_dir),
            ]
            + extra_args,
            continue_on_error=args.continue_on_error,
        )
        (completed if ok else failed).append(name)

    feature_ok = run_step(
        "Feature Extraction",
        [
            python,
            str(TRAINER_DIR / "feature_extraction.py"),
            "--metadata-path",
            str(metadata_path),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(feature_output_dir),
            "--seed",
            str(args.seed),
        ],
        continue_on_error=args.continue_on_error,
    )
    (completed if feature_ok else failed).append("Feature Extraction")

    print("\n=== V14 non-PINN run complete ===")
    print("Completed:", ", ".join(completed) if completed else "none")
    if failed:
        print("Failed:", ", ".join(failed), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
