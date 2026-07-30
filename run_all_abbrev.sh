#!/bin/bash
# Version 11 source snapshot
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output=output_v11_abbrev.txt
#SBATCH --error=error_v11_abbrev.txt

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )}"
cd "$SCRIPT_DIR"
PY="${PYTHON:-python}"

resolve_output_root() {
  if [[ -n "${V11_OUTPUT_DIR:-}" ]]; then
    printf '%s' "$V11_OUTPUT_DIR"
    return 0
  fi

  local candidate
  for candidate in "${SCRATCH:-}" "${SLURM_TMPDIR:-}" "${HOME:-}" "$SCRIPT_DIR"; do
    [[ -n "$candidate" ]] || continue
    if mkdir -p "$candidate/results_v11_suite" 2>/dev/null; then
      printf '%s' "$candidate/results_v11_suite"
      return 0
    fi
  done

  printf '%s' "$SCRIPT_DIR/results_v11_suite"
}

OUTPUT_ROOT="$(resolve_output_root)"
mkdir -p "$OUTPUT_ROOT"

IFS=' ' read -r -a SEEDS <<< "${V11_SEEDS:-42}"

# Ensemble models (geometry_clinical, geometry_flow_clinical) + GNN
IFS=' ' read -r -a RUN_SPECS <<< "${V11_RUNS:-geometry_clinical|pointnet2 geometry_clinical|pointnext geometry_flow_clinical|pointnet2 geometry_flow_clinical|pointnext gnn|none}"

AMP_ARGS=()
if [[ "${V11_AMP:-1}" == "1" || "${V11_AMP:-1}" =~ ^(true|TRUE|yes|YES)$ ]]; then
  AMP_ARGS+=(--amp)
fi

EXTRA_ARGS=()
if [[ "${V11_DRY_RUN:-0}" == "1" || "${V11_DRY_RUN:-0}" =~ ^(true|TRUE|yes|YES)$ ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

for spec in "${RUN_SPECS[@]}"; do
  IFS='|' read -r MODEL BACKBONE <<< "$spec"
  if [[ -z "$MODEL" ]]; then
    continue
  fi

  if [[ "$BACKBONE" == "none" ]]; then
    RUN_NAME="$MODEL"
  else
    RUN_NAME="${MODEL}_${BACKBONE}"
  fi

  RUN_ROOT="$OUTPUT_ROOT/$RUN_NAME"
  mkdir -p "$RUN_ROOT"
  SEED_DIRS=()

  for seed in "${SEEDS[@]}"; do
    RUN_DIR="$RUN_ROOT/seed_${seed}"
    mkdir -p "$RUN_DIR"
    printf '\n=== V11 TRAINING: model=%s backbone=%s seed=%s ===\n' "$MODEL" "$BACKBONE" "$seed"

    CMD=("$PY" "$SCRIPT_DIR/train_models.py" --model "$MODEL" --seed "$seed" --cv-seed "${V11_CV_SEED:-42}" --metadata-path "${V11_METADATA:-metadata.csv}" --data-dir "${V11_DATA_DIR:-predictions/pinn_corrected}" --output-dir "$RUN_DIR" --folds "${V11_FOLDS:-5}" --batch-size "${V11_BATCH:-6}" --epochs "${V11_EPOCHS:-220}" --lr "${V11_LR:-3e-4}" --weight-decay "${V11_WD:-2e-4}" --target-n "${V11_TARGET_N:-4096}" --patience "${V11_PATIENCE:-35}")
    if [[ "$BACKBONE" != "none" ]]; then
      CMD+=(--backbone "$BACKBONE")
    fi
    CMD+=("${AMP_ARGS[@]}")
    CMD+=("${EXTRA_ARGS[@]}")

    "${CMD[@]}" 2>&1 | tee "$RUN_DIR/train_${MODEL}_${BACKBONE}_seed_${seed}.log"
    SEED_DIRS+=("$RUN_DIR")
  done

  if [[ "${V11_DRY_RUN:-0}" != "1" && ( "${V11_RUN_AGG:-1}" == "1" || "${V11_RUN_AGG:-1}" =~ ^(true|TRUE|yes|YES)$ ) ]]; then
    FINAL_DIR="$RUN_ROOT/seed_ensemble"
    mkdir -p "$FINAL_DIR"
    printf '\n=== V11 AGGREGATING: model=%s backbone=%s ===\n' "$MODEL" "$BACKBONE"
    "$PY" "$SCRIPT_DIR/aggregate_predictions.py" --output-dir "$FINAL_DIR" --seed-dirs "${SEED_DIRS[@]}"
  fi
done

printf '\nAll abbreviated v11 runs (ensemble + GNN) completed successfully.\n'
