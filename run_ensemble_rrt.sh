#!/bin/bash
# Version 5 source snapshot
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output=output_ensemble_rrt.txt
#SBATCH --error=error_ensemble_rrt.txt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Optional overrides:
#   EPOCHS=1 N_FOLDS=2 OUTPUT_DIR=results_ensemble_rrt_smoke bash run_ensemble_rrt.sh
export EPOCHS="${EPOCHS:-200}"
export N_FOLDS="${N_FOLDS:-5}"
export OUTPUT_DIR="${OUTPUT_DIR:-results_ensemble_rrt}"

DEFAULT_VENV_PY="${SCRIPT_DIR}/.venv/Scripts/python.exe"
if [[ -z "${PYTHON_BIN:-}" ]]; then
	if [[ -x "${DEFAULT_VENV_PY}" ]]; then
		PYTHON_BIN="${DEFAULT_VENV_PY}"
	else
		PYTHON_BIN="python"
	fi
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/train_ensemble_rrt.py"
