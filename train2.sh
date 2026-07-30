#!/bin/bash
# Version 3 source snapshot
# train2.sh - Training Script 2: Late Fusion Model
#
# This script runs the Late Fusion Model (Geometry + Hemodynamics + Clinical) training.
# Designed to be run in parallel with other training scripts on separate GPUs.
#
# Usage:
#   ./train2.sh              # Run with default settings
#   ./train2.sh --single     # Run with single train/val split
#
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --nodelist=gpu06
#SBATCH --output=output2.txt
#SBATCH --error=error2.txt

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
echo "       TRAIN 2: LATE FUSION MODEL (WITH CLINICAL)"
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
# 4. Multi-Modal Fusion (Geometry + Hemodynamics + Age + Sex)
echo ""
echo "=============================================================="
echo "4. Training Late Fusion with Clinical Features"
echo "=============================================================="

if check_if_trained "training_logs/fusion_late.pth" "training_logs/kfold_fusion_late"; then
    : # Skip - already trained
elif [ "$SINGLE_SPLIT" = true ]; then
    python -u fusion.py \
        --data_dir "$DATA_DIR" \
        --metadata "$METADATA" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr 1e-4 \
        --target_n $TARGET_N \
        --model "late_fusion" \
        --dropout 0.5 \
        --num_workers $NUM_WORKERS \
        --seed $SEED \
        --save_path "training_logs/fusion_late.pth"
else
    python -u fusion.py \
        --data_dir "$DATA_DIR" \
        --metadata "$METADATA" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr 1e-4 \
        --target_n $TARGET_N \
        --model "late_fusion" \
        --dropout 0.5 \
        --num_workers $NUM_WORKERS \
        --seed $SEED \
        --kfold $KFOLD \
        --kfold_save_dir "training_logs/kfold_fusion_late"
fi
