#!/bin/bash
# Version 2 source snapshot
# train1.sh - Training Script 1: Ensemble Model
#
# This script runs the Ensemble Model training.
# Designed to be run in parallel with other training scripts on separate GPUs.
#
# Usage:
#   ./train1.sh              # Run with default settings
#   ./train1.sh --single     # Run with single train/val split
#
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --nodelist=gpu04
#SBATCH --output=output1.txt
#SBATCH --error=error1.txt

# Configuration
EPOCHS=200
BATCH_SIZE=8
KFOLD=5
NUM_WORKERS=2
SEED=42
DATA_DIR="predictions/pinn_corrected"
GEOMETRY_DATA_DIR="data"
METADATA="metadata.csv"
TARGET_N=8192

# Parse command line arguments
SINGLE_SPLIT=false
QUICK_TEST=false

for arg in "$@"; do
    case $arg in
        --single)
            SINGLE_SPLIT=true
            KFOLD=1
            shift
            ;;
        --quick)
            QUICK_TEST=true
            EPOCHS=50
            shift
            ;;
        *)
            ;;
    esac
done

echo "=============================================================="
echo "       TRAIN 1: ENSEMBLE MODEL"
echo "=============================================================="
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "K-Fold: $KFOLD"
echo "=============================================================="
echo ""

# Create output directory for logs
mkdir -p training_logs

# Helper function to check if training is complete
check_if_trained() {
    local save_path="$1"
    local kfold_dir="$2"

    if [ "$SINGLE_SPLIT" = true ]; then
        # Check if model file exists
        if [ -f "$save_path" ]; then
            echo "  [SKIP] Already trained: $save_path"
            return 0
        fi
    else
        # Check if kfold_summary.csv exists (indicates completed k-fold training)
        if [ -f "$kfold_dir/kfold_summary.csv" ]; then
            echo "  [SKIP] Already trained: $kfold_dir/kfold_summary.csv"
            return 0
        fi
    fi
    return 1
}
# 3. Ensemble Model (Geometry + Fusion Late)
echo ""
echo "=============================================================="
echo "3. Training Ensemble Model (Geometry + Fusion Late)"
echo "=============================================================="

if check_if_trained "training_logs/ensemble/ensemble_fold5.pt" "training_logs/ensemble"; then
    : # Skip - already trained
elif [ "$SINGLE_SPLIT" = true ]; then
    python -u ensemble_model.py \
        --geo_dir "$GEOMETRY_DATA_DIR" \
        --hemo_dir "$DATA_DIR" \
        --metadata "$METADATA" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr 1e-4 \
        --target_n $TARGET_N \
        --method "average" \
        --output_dir "training_logs/ensemble"
else
    python -u ensemble_model.py \
        --geo_dir "$GEOMETRY_DATA_DIR" \
        --hemo_dir "$DATA_DIR" \
        --metadata "$METADATA" \
        --n_folds $KFOLD \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr 1e-4 \
        --target_n $TARGET_N \
        --method "average" \
        --output_dir "training_logs/ensemble"
fi
