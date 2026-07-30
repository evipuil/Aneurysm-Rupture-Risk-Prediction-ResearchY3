#!/bin/bash
# Version 8 source snapshot
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output=output_v8.txt
#SBATCH --error=error_v8.txt

set -euo pipefail

# Determine script dir and project root; export POINTNET_ROOT for Python bootstraps
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
POINTNET_ROOT="${POINTNET_ROOT:-}"
if [ -z "$POINTNET_ROOT" ]; then
	CUR="$SCRIPT_DIR"
	while [ "$CUR" != "/" ] && [ ! -d "$CUR/pointnet_pytorch" ]; do
		CUR="$( dirname "$CUR" )"
	done
	if [ -d "$CUR/pointnet_pytorch" ]; then
		POINTNET_ROOT="$CUR/pointnet_pytorch"
	else
		POINTNET_ROOT="$SCRIPT_DIR/.."
	fi
fi
export POINTNET_ROOT

PY="${PYTHON:-python}"
cd "$SCRIPT_DIR"

echo "Running v8 geometry training"
"$PY" train_geometry.py

echo "Running v8 flow+geometry training (PointNeXt)"
"$PY" train_flow_geometry.py

echo "Running v8 ensemble training"
"$PY" train_ensemble.py

echo "Training PINN diagnosis model"
"$PY" diagnose_from_pinn.py

echo "Computing post-hoc feature rankings"
"$PY" feature_extraction_posthoc.py

echo "All v8 jobs finished"
