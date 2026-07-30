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


def run_step(name: str, command: list[str], continue_on_error: bool) -> bool:
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode == 0:
        print(f"Completed {name}", flush=True)
        return True
    print(f"FAILED {name} with exit code {result.returncode}", file=sys.stderr, flush=True)
    if not continue_on_error:
        raise SystemExit(result.returncode)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V14 multimodal comparison trainers locally.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-path", default=str(PROJECT_ROOT / "metadata.csv"))
    parser.add_argument(
        "--data-dir", default=str(PROJECT_ROOT / "flow_data" / "full_accuracy2_copy")
    )
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results_V14_suite"))
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    metadata_path = Path(args.metadata_path)
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)

    if not metadata_path.is_file():
        raise SystemExit(f"Metadata file not found: {metadata_path}")
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    trainer_names = (
        "train_clinical.py",
        "train_flow_geometry.py",
        "train_geometry_clinical.py",
        "train_geometry_flow_clinical.py",
        "train_cnn.py",
    )
    missing_trainers = [name for name in trainer_names if not (TRAINER_DIR / name).is_file()]
    if missing_trainers:
        raise SystemExit(f"Missing trainer scripts: {', '.join(missing_trainers)}")

    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    print("Running V14 multimodal local CUDA pipeline", flush=True)
    print(f"  python: {python}", flush=True)
    print(f"  metadata: {metadata_path}", flush=True)
    print(f"  data: {data_dir}", flush=True)
    print(f"  output: {output_root}", flush=True)

    trainers = [
        ("Clinical baseline", "train_clinical.py", output_root / f"clinical_seed_{args.seed}", []),
        (
            "Flow + geometry PointNeXt",
            "train_flow_geometry.py",
            output_root / f"flow_geometry_pointnext_seed_{args.seed}",
            [],
        ),
        (
            "Geometry + clinical PointNeXt",
            "train_geometry_clinical.py",
            output_root / f"geometry_clinical_pointnext_seed_{args.seed}",
            [],
        ),
        (
            "Geometry + flow + clinical PointNeXt",
            "train_geometry_flow_clinical.py",
            output_root / f"geometry_flow_clinical_pointnext_seed_{args.seed}",
            [],
        ),
        ("Voxel CNN + flow", "train_cnn.py", output_root / f"cnn_voxel_flow_seed_{args.seed}", []),
    ]

    completed: list[str] = []
    failed: list[str] = []
    for name, script_name, output_dir, extra_args in trainers:
        command = [
            python,
            "-u",
            str(TRAINER_DIR / script_name),
            "--seed",
            str(args.seed),
            "--metadata-path",
            str(metadata_path),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            *extra_args,
        ]
        ok = run_step(name, command, args.continue_on_error)
        (completed if ok else failed).append(name)

    print("\n=== V14 multimodal local run complete ===", flush=True)
    print("Completed:", ", ".join(completed) if completed else "none", flush=True)
    if failed:
        print("Failed:", ", ".join(failed), file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
