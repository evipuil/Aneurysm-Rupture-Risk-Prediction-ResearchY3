#!/bin/bash
# Version 13 source snapshot
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output=output13.txt
#SBATCH --error=error13.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

if [[ -f "${PROJECT_ROOT}/train_geometry.py" ]]; then
    TRAINER_DIR="${PROJECT_ROOT}"
elif [[ -f "${PROJECT_ROOT}/train_geometry.py" ]]; then
    TRAINER_DIR="${PROJECT_ROOT}/version13"
elif [[ -f "${SCRIPT_DIR}/train_geometry.py" ]]; then
    TRAINER_DIR="${SCRIPT_DIR}"
else
    echo "ERROR: Could not find train_geometry.py from project root ${PROJECT_ROOT}" >&2
    exit 1
fi

SEED="${1:-42}"
METADATA_PATH="${2:-${PROJECT_ROOT}/metadata.csv}"
DATA_DIR="${3:-${PROJECT_ROOT}/predictions/pinn_corrected}"
OUTPUT_ROOT="${4:-${PROJECT_ROOT}/results_V13_suite}"
FEATURE_OUTPUT_DIR="${5:-${PROJECT_ROOT}/results_v13_feature_extraction}"

echo "Running V13 non-PINN pipeline"
echo "  project root: ${PROJECT_ROOT}"
echo "  trainer dir: ${TRAINER_DIR}"
echo "  seed: ${SEED}"
echo "  metadata: ${METADATA_PATH}"
echo "  data: ${DATA_DIR}"
echo "  output: ${OUTPUT_ROOT}"
echo "  feature output: ${FEATURE_OUTPUT_DIR}"

python "${TRAINER_DIR}/train_geometry.py" \
    --seed "${SEED}" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/geometry_pointnet2_seed_${SEED}"

python "${TRAINER_DIR}/train_flow_geometry.py" \
    --seed "${SEED}" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/flow_geometry_seed_${SEED}"

python "${TRAINER_DIR}/train_clinical.py" \
    --seed "${SEED}" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/clinical_seed_${SEED}"

python "${TRAINER_DIR}/train_geometry_clinical.py" \
    --seed "${SEED}" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/geometry_clinical_seed_${SEED}"

python "${TRAINER_DIR}/train_geometry_flow_clinical.py" \
    --seed "${SEED}" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/geometry_flow_clinical_seed_${SEED}"

python "${TRAINER_DIR}/train_gnn.py" \
    --seed "${SEED}" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/gnn_seed_${SEED}"

python "${TRAINER_DIR}/feature_extraction.py" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${FEATURE_OUTPUT_DIR}" \
    --seed "${SEED}"

echo "V13 non-PINN pipeline complete"
