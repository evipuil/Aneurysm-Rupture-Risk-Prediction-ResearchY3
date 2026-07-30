#!/bin/bash
# Version 10 source snapshot
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output=output_v10.txt
#SBATCH --error=error_v10.txt

# V10: train the multibranch model across multiple seeds and optionally
# aggregate the seed-level pooled predictions into a final ensemble.
set -euo pipefail

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
  if [[ -n "${V10_OUTPUT_DIR:-}" ]]; then
    printf '%s' "$V10_OUTPUT_DIR"
    return 0
  fi

  local candidate
  for candidate in "${SCRATCH:-}" "${SLURM_TMPDIR:-}" "${HOME:-}" "$SCRIPT_DIR"; do
    [[ -n "$candidate" ]] || continue
    if mkdir -p "$candidate/results_v10_multibranch" 2>/dev/null; then
      printf '%s' "$candidate/results_v10_multibranch"
      return 0
    fi
  done

  printf '%s' "$SCRIPT_DIR/results_v10_multibranch"
}

OUTPUT_ROOT="$(resolve_output_root)"
mkdir -p "$OUTPUT_ROOT"

IFS=' ' read -r -a SEEDS <<< "${V10_SEEDS:-42 1337 2024}"
AMP_ARGS=()
if [[ "${V10_AMP:-1}" == "1" || "${V10_AMP:-1}" =~ ^(true|TRUE|yes|YES)$ ]]; then
  AMP_ARGS+=(--amp)
fi

EXTRA_ARGS=()
if [[ "${V10_DRY_RUN:-0}" == "1" || "${V10_DRY_RUN:-0}" =~ ^(true|TRUE|yes|YES)$ ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

SEED_DIRS=()
for seed in "${SEEDS[@]}"; do
  RUN_DIR="$OUTPUT_ROOT/seed_${seed}"
  mkdir -p "$RUN_DIR"
  printf '\n=== FULL TRAINING: V10 seed=%s ===\n' "$seed"
  "$PY" "$SCRIPT_DIR/train_ensemble.py" \
    --seed "$seed" \
    --cv-seed "${V10_CV_SEED:-42}" \
    --metadata-path "${V10_METADATA:-metadata.csv}" \
    --data-dir "${V10_DATA_DIR:-predictions/pinn_corrected}" \
    --output-dir "$RUN_DIR" \
    --folds "${V10_FOLDS:-5}" \
    --batch-size "${V10_BATCH:-6}" \
    --epochs "${V10_EPOCHS:-220}" \
    --lr "${V10_LR:-3e-4}" \
    --weight-decay "${V10_WD:-2e-4}" \
    --target-n "${V10_TARGET_N:-4096}" \
    --patience "${V10_PATIENCE:-35}" \
    "${AMP_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$RUN_DIR/train_seed_${seed}.log"
  SEED_DIRS+=("$RUN_DIR")
done

if [[ "${V10_DRY_RUN:-0}" != "1" && ( "${V10_RUN_AGG:-1}" == "1" || "${V10_RUN_AGG:-1}" =~ ^(true|TRUE|yes|YES)$ ) ]]; then
  FINAL_DIR="$OUTPUT_ROOT/seed_ensemble"
  mkdir -p "$FINAL_DIR"
  printf '\n=== AGGREGATING SEED ENSEMBLE ===\n'
  "$PY" "$SCRIPT_DIR/aggregate_predictions.py" \
    --output-dir "$FINAL_DIR" \
    --seed-dirs "${SEED_DIRS[@]}"
fi

printf '\nAll v10 runs completed successfully.\n'
