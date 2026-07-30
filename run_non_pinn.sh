#!/bin/bash
# Version 14 source snapshot
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output=output14.txt
#SBATCH --error=error14.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="${SCRIPT_DIR}"
fi

if [[ -f "${PROJECT_ROOT}/train_geometry.py" ]]; then
    TRAINER_DIR="${PROJECT_ROOT}"
elif [[ -f "${SCRIPT_DIR}/train_geometry.py" ]]; then
    TRAINER_DIR="${SCRIPT_DIR}"
else
    echo "ERROR: Could not find train_geometry.py from project root ${PROJECT_ROOT}" >&2
    exit 1
fi

SEED="${1:-42}"
METADATA_PATH="${2:-${PROJECT_ROOT}/metadata.csv}"
DATA_DIR="${3:-${PROJECT_ROOT}/flow_data/full_accuracy2_copy}"
OUTPUT_ROOT="${4:-${PROJECT_ROOT}/results_V14_suite}"
FEATURE_OUTPUT_DIR="${5:-${PROJECT_ROOT}/results_v14_feature_extraction}"
PYTHON_BIN="${PYTHON:-python}"

if [[ ! -f "${METADATA_PATH}" ]]; then
    echo "ERROR: metadata file not found: ${METADATA_PATH}" >&2
    exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: data directory not found: ${DATA_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${FEATURE_OUTPUT_DIR}"
export PYTHONUNBUFFERED=1

echo "Running V14 non-PINN pipeline"
echo "  project root: ${PROJECT_ROOT}"
echo "  trainer dir: ${TRAINER_DIR}"
echo "  python: ${PYTHON_BIN}"
echo "  seed: ${SEED}"
echo "  metadata: ${METADATA_PATH}"
echo "  data: ${DATA_DIR}"
echo "  output: ${OUTPUT_ROOT}"
echo "  feature output: ${FEATURE_OUTPUT_DIR}"

run_step() {
    local name="$1"
    shift
    echo
    echo "=== ${name} ==="
    echo "$*"
    "$@"
}

run_step "Geometry PointNeXt" "${PYTHON_BIN}" "${TRAINER_DIR}/train_geometry.py" \
    --seed "${SEED}" \
    --backbone pointnext \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/geometry_pointnext_seed_${SEED}"

run_step "Flow + geometry PointNeXt" "${PYTHON_BIN}" "${TRAINER_DIR}/train_flow_geometry.py" \
    --seed "${SEED}" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/flow_geometry_pointnext_seed_${SEED}"

run_step "Clinical" "${PYTHON_BIN}" "${TRAINER_DIR}/train_clinical.py" \
    --seed "${SEED}" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/clinical_seed_${SEED}"

run_step "Geometry + clinical PointNeXt" "${PYTHON_BIN}" "${TRAINER_DIR}/train_geometry_clinical.py" \
    --seed "${SEED}" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/geometry_clinical_pointnext_seed_${SEED}"

run_step "Geometry + flow + clinical PointNeXt" "${PYTHON_BIN}" "${TRAINER_DIR}/train_geometry_flow_clinical.py" \
    --seed "${SEED}" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/geometry_flow_clinical_pointnext_seed_${SEED}"

run_step "Voxel CNN + flow" "${PYTHON_BIN}" "${TRAINER_DIR}/train_cnn.py" \
    --seed "${SEED}" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/cnn_voxel_flow_seed_${SEED}"

run_step "GNN geometry" "${PYTHON_BIN}" "${TRAINER_DIR}/train_gnn.py" \
    --seed "${SEED}" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/gnn_geometry_seed_${SEED}" \
    --no-flow

run_step "Feature extraction" "${PYTHON_BIN}" "${TRAINER_DIR}/feature_extraction.py" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${FEATURE_OUTPUT_DIR}" \
    --seed "${SEED}"

echo
echo "V14 non-PINN pipeline complete"
