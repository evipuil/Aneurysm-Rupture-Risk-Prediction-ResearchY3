#!/bin/bash
# Version 14 source snapshot
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output=output14_flow_optimized.txt
#SBATCH --error=error14_flow_optimized.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="${SCRIPT_DIR}"
fi

if [[ -f "${PROJECT_ROOT}/train_flow_geometry.py" ]]; then
    TRAINER_DIR="${PROJECT_ROOT}"
elif [[ -f "${SCRIPT_DIR}/train_flow_geometry.py" ]]; then
    TRAINER_DIR="${SCRIPT_DIR}"
else
    echo "ERROR: Could not find Version 14 flow trainers from project root ${PROJECT_ROOT}" >&2
    exit 1
fi

SEED="${1:-42}"
METADATA_PATH="${2:-${PROJECT_ROOT}/metadata.csv}"
DATA_DIR="${3:-${PROJECT_ROOT}/flow_data/full_accuracy2_copy}"
OUTPUT_ROOT="${4:-${PROJECT_ROOT}/results_V14_suite}"
PYTHON_BIN="${PYTHON:-python}"

export PYTHONUNBUFFERED=1
export v14_FLOW_CLIP="${v14_FLOW_CLIP:-5.0}"
export v14_FLOW_BRANCH_DROPOUT="${v14_FLOW_BRANCH_DROPOUT:-0.20}"
export v14_FLOW_CHANNEL_DROPOUT="${v14_FLOW_CHANNEL_DROPOUT:-0.10}"
export v14_FLOW_NOISE_STD="${v14_FLOW_NOISE_STD:-0.02}"

if [[ ! -f "${METADATA_PATH}" ]]; then
    echo "ERROR: metadata file not found: ${METADATA_PATH}" >&2
    exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: data directory not found: ${DATA_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "Running V14 optimized flow models"
echo "  project root: ${PROJECT_ROOT}"
echo "  trainer dir: ${TRAINER_DIR}"
echo "  python: ${PYTHON_BIN}"
echo "  seed: ${SEED}"
echo "  metadata: ${METADATA_PATH}"
echo "  data: ${DATA_DIR}"
echo "  output: ${OUTPUT_ROOT}"
echo "  v14_FLOW_CLIP=${v14_FLOW_CLIP}"
echo "  v14_FLOW_BRANCH_DROPOUT=${v14_FLOW_BRANCH_DROPOUT}"
echo "  v14_FLOW_CHANNEL_DROPOUT=${v14_FLOW_CHANNEL_DROPOUT}"
echo "  v14_FLOW_NOISE_STD=${v14_FLOW_NOISE_STD}"

"${PYTHON_BIN}" "${TRAINER_DIR}/train_flow_geometry.py" \
    --seed "${SEED}" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/flow_geometry_pointnext_seed_${SEED}"

"${PYTHON_BIN}" "${TRAINER_DIR}/train_geometry_flow_clinical.py" \
    --seed "${SEED}" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/geometry_flow_clinical_pointnext_seed_${SEED}"

echo "Optimized V14 flow model training complete"
