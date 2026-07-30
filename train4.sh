#!/bin/bash
# Version 2 source snapshot
# train4.sh - Training Script 4: Geometry-only PointNet++
#
# This script runs the Geometry-only PointNet++ Model training.
# Designed to be run in parallel with other training scripts on separate GPUs.
#
# Usage:
#   ./train4.sh              # Run with default settings
#   ./train4.sh --single     # Run with single train/val split
#
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --nodelist=gpu07
#SBATCH --output=output.txt
#SBATCH --error=error.txt

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
echo "       TRAIN 4: GEOMETRY-ONLY POINTNET++"
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
# 1. Geometry-only PointNet++ (best baseline)
echo ""
echo "=============================================================="
echo "1. Training Geometry-only PointNet++"
echo "=============================================================="

if check_if_trained "training_logs/geometry_pointnet.pth" "training_logs/kfold_geometry"; then
    : # Skip - already trained
elif [ "$SINGLE_SPLIT" = true ]; then
    python -u geometry_pointnet.py \
        --root_dir "$GEOMETRY_DATA_DIR" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr 1e-4 \
        --target_n $TARGET_N \
        --val_fraction 0.2 \
        --num_workers $NUM_WORKERS \
        --seed $SEED \
        --save_path "training_logs/geometry_pointnet.pth"
else
    python -u geometry_pointnet.py \
        --root_dir "$GEOMETRY_DATA_DIR" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr 1e-4 \
        --target_n $TARGET_N \
        --num_workers $NUM_WORKERS \
        --seed $SEED \
        --kfold $KFOLD \
        --kfold_save_dir "training_logs/kfold_geometry"
fi
