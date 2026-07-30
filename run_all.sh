#!/bin/bash
# Version 9 source snapshot
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output=output_v9.txt
#SBATCH --error=error_v9.txt

# Runs full training for every v9 model combination and writes a separate output tree per run.
set -euo pipefail

# Determine script dir and project root; export POINTNET_ROOT for Python bootstraps
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )}"
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

resolve_output_root() {
  if [[ -n "${V9_OUTPUT_DIR:-}" ]]; then
    printf '%s' "$V9_OUTPUT_DIR"
    return 0
  fi

  local candidate
  for candidate in "${SCRATCH:-}" "${SLURM_TMPDIR:-}" "${HOME:-}" "$SCRIPT_DIR"; do
    [[ -n "$candidate" ]] || continue
    if mkdir -p "$candidate/results_v9_flow_geometry" 2>/dev/null; then
      printf '%s' "$candidate/results_v9_flow_geometry"
      return 0
    fi
  done

  printf '%s' "$PWD/results_v9_flow_geometry"
}

FLOW_MODES=(late attention early)
ENSEMBLE_MODES=(late attention early)
OUTPUT_ROOT="$(resolve_output_root)"
mkdir -p "$OUTPUT_ROOT"

AMP_ARGS=()
if [[ "${V9_AMP:-1}" == "1" || "${V9_AMP:-1}" =~ ^(true|TRUE|yes|YES)$ ]]; then
  AMP_ARGS+=(--amp)
fi

EXTRA_ARGS=()
if [[ "${V9_DRY_RUN:-0}" == "1" || "${V9_DRY_RUN:-0}" =~ ^(true|TRUE|yes|YES)$ ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

for m in "${FLOW_MODES[@]}"; do
  RUN_DIR="$OUTPUT_ROOT/flow_${m}"
  mkdir -p "$RUN_DIR"
  printf '\n=== FULL TRAINING: FUSION_MODE=%s ===\n' "$m"
  FUSION_MODE="$m" V9_OUTPUT_DIR="$RUN_DIR" "$PY" "$SCRIPT_DIR/train_flow_geometry.py" \
    --fusion "$m" \
    --metadata-path "${V9_METADATA:-metadata.csv}" \
    --data-dir "${V9_DATA_DIR:-predictions/pinn_corrected}" \
    --output-dir "$RUN_DIR" \
    --folds "${V9_FOLDS:-5}" \
    --batch-size "${V9_BATCH:-8}" \
    --epochs "${V9_EPOCHS:-200}" \
    --lr "${V9_LR:-1e-4}" \
    --weight-decay "${V9_WD:-1e-4}" \
    --patience "${V9_PATIENCE:-40}" \
    "${AMP_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$RUN_DIR/train_${m}.log"
done

ENSEMBLE_AMP_ARGS=()
if [[ "${V9E_AMP:-1}" == "1" || "${V9E_AMP:-1}" =~ ^(true|TRUE|yes|YES)$ ]]; then
  ENSEMBLE_AMP_ARGS+=(--amp)
fi

for m in "${ENSEMBLE_MODES[@]}"; do
  RUN_DIR="$OUTPUT_ROOT/ensemble_${m}"
  mkdir -p "$RUN_DIR"
  printf '\n=== FULL TRAINING: STACKING ENSEMBLE (base_fusion=%s) ===\n' "$m"
  V9E_OUTPUT_DIR="$RUN_DIR" "$PY" "$SCRIPT_DIR/train_ensemble.py" \
    --base-fusion "$m" \
    --metadata-path "${V9E_METADATA:-metadata.csv}" \
    --data-dir "${V9E_DATA_DIR:-predictions/pinn_corrected}" \
    --output-dir "$RUN_DIR" \
    --folds "${V9E_FOLDS:-5}" \
    --batch-size "${V9E_BATCH:-8}" \
    --epochs "${V9E_EPOCHS:-200}" \
    --lr "${V9E_LR:-1e-4}" \
    --weight-decay "${V9E_WD:-1e-4}" \
    --patience "${V9E_PATIENCE:-40}" \
    "${ENSEMBLE_AMP_ARGS[@]}" \
    2>&1 | tee "$RUN_DIR/train_stack_${m}.log"
done

printf '\nAll v9 full-training runs completed successfully.\n'
