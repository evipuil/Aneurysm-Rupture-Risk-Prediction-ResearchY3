#!/bin/bash
# Version 7 source snapshot
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output=output_v7_clinical.txt
#SBATCH --error=error_v7_clinical.txt

source ~/.bashrc

# Optional environment activation. If conda is unavailable/broken on this node,
# provide POINTNET_PYTHON (absolute path to env python) when submitting.
if command -v conda >/dev/null 2>&1; then
	conda activate pointnet || true
fi

PYTHON_BIN="${POINTNET_PYTHON:-python}"

"${PYTHON_BIN}" train_clinical_only.py
"${PYTHON_BIN}" train_clinical_age_sex.py
