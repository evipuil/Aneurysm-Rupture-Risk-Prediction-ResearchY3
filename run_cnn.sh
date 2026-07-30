#!/bin/bash
# Version 14 source snapshot
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output=output14_cnn.txt
#SBATCH --error=error14_cnn.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="${SCRIPT_DIR}"
fi

if [[ -f "${PROJECT_ROOT}/train_cnn.py" ]]; then
    TRAINER_DIR="${PROJECT_ROOT}"
elif [[ -f "${SCRIPT_DIR}/train_cnn.py" ]]; then
    TRAINER_DIR="${SCRIPT_DIR}"
else
    echo "ERROR: Could not find train_cnn.py from project root ${PROJECT_ROOT}" >&2
    exit 1
fi

SEED="${1:-42}"
METADATA_PATH="${2:-${PROJECT_ROOT}/metadata.csv}"
DATA_DIR="${3:-${PROJECT_ROOT}/flow_data/full_accuracy2_copy}"
OUTPUT_ROOT="${4:-${PROJECT_ROOT}/results_V14_suite}"
PYTHON_BIN="${PYTHON:-python}"

shift $(( $# < 4 ? $# : 4 ))
EXTRA_ARGS=("$@")

if [[ ! -f "${METADATA_PATH}" ]]; then
    echo "ERROR: metadata file not found: ${METADATA_PATH}" >&2
    exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: data directory not found: ${DATA_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
export PYTHONUNBUFFERED=1

echo "Running V14 CNN training"
echo "  project root: ${PROJECT_ROOT}"
echo "  trainer dir: ${TRAINER_DIR}"
echo "  python: ${PYTHON_BIN}"
echo "  seed: ${SEED}"
echo "  metadata: ${METADATA_PATH}"
echo "  data: ${DATA_DIR}"
echo "  output root: ${OUTPUT_ROOT}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "  extra trainer args: ${EXTRA_ARGS[*]}"
fi

run_step() {
    local name="$1"
    shift
    echo
    echo "=== ${name} ==="
    echo "$*"
    "$@"
}

run_step "Voxel CNN + flow" "${PYTHON_BIN}" "${TRAINER_DIR}/train_cnn.py" \
    --seed "${SEED}" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/cnn_voxel_flow_seed_${SEED}" \
    "${EXTRA_ARGS[@]}"

run_step "Voxel CNN geometry only" "${PYTHON_BIN}" "${TRAINER_DIR}/train_cnn.py" \
    --seed "${SEED}" \
    --metadata-path "${METADATA_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/cnn_voxel_geometry_seed_${SEED}" \
    --no-flow \
    "${EXTRA_ARGS[@]}"

echo
echo "V14 CNN training complete"
