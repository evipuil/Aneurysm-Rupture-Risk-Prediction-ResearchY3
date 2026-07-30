#!/bin/bash
# Version 3 source snapshot
# train.sh - Run all rupture classification models
#
# This script runs all model architectures for aneurysm rupture classification:
# 1. Geometry-only PointNet++ (best baseline) - runs first to verify data
# 2. GNN (Graph Neural Network)
# 3. Ensemble Model (Geometry + Fusion Late combined)
# 4. Fusion Late with Clinical Features
# 5. Clinical Only Baseline
#
# Training metrics are logged to CSV files for each run.
#
# Usage:
#   ./train.sh              # Run all models with 5-fold CV
#   ./train.sh --single     # Run all models with single train/val split
#   ./train.sh --quick      # Quick test with fewer epochs
#
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
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
echo "       ANEURYSM RUPTURE CLASSIFICATION TRAINING"
echo "=============================================================="
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "K-Fold CV: $KFOLD"
echo "Quick Test: $QUICK_TEST"
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
# 1. Geometry-only PointNet++ (best baseline) - RUN FIRST TO VERIFY DATA
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
# 2. GNN (Graph Neural Network)
echo ""
echo "=============================================================="
echo "2. Training GNN (Graph Neural Network)"
echo "=============================================================="

if check_if_trained "training_logs/gnn_model.pth" "training_logs/kfold_gnn"; then
    : # Skip - already trained
elif [ "$SINGLE_SPLIT" = true ]; then
    python -u gnn_rupture_classification.py \
        --data_dir "$DATA_DIR" \
        --metadata "$METADATA" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr 1e-4 \
        --target_n $TARGET_N \
        --model "sage" \
        --k_neighbors 16 \
        --hidden_channels 96 \
        --num_layers 2 \
        --dropout 0.5 \
        --focal_loss \
        --seed $SEED \
        --save_path "training_logs/gnn_model.pth"
else
    python -u gnn_rupture_classification.py \
        --data_dir "$DATA_DIR" \
        --metadata "$METADATA" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr 1e-4 \
        --target_n $TARGET_N \
        --model "sage" \
        --k_neighbors 16 \
        --hidden_channels 96 \
        --num_layers 2 \
        --dropout 0.5 \
        --focal_loss \
        --seed $SEED \
        --kfold $KFOLD \
        --kfold_save_dir "training_logs/kfold_gnn"
fi
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
# 4. Multichannel PointNet++ (hemodynamics features) - COMMENTED OUT
# echo ""
# echo "=============================================================="
# echo "4. Training Multichannel PointNet++ (Hemodynamics)"
# echo "=============================================================="
#
# if check_if_trained "training_logs/multichannel_pointnetpp.pth" "training_logs/kfold_multichannel"; then
#     : # Skip - already trained
# elif [ "$SINGLE_SPLIT" = true ]; then
#     python multichannel_pointnet_rupture.py \
#         --data_dir "$DATA_DIR" \
#         --metadata "$METADATA" \
#         --epochs $EPOCHS \
#         --batch_size $BATCH_SIZE \
#         --lr 1e-3 \
#         --target_n 2048 \
#         --model "pointnet++" \
#         --focal_loss \
#         --dropout 0.5 \
#         --early_stopping 30 \
#         --num_workers $NUM_WORKERS \
#         --seed $SEED \
#         --save_path "training_logs/multichannel_pointnetpp.pth"
# else
#     python multichannel_pointnet_rupture.py \
#         --data_dir "$DATA_DIR" \
#         --metadata "$METADATA" \
#         --epochs $EPOCHS \
#         --batch_size $BATCH_SIZE \
#         --lr 1e-3 \
#         --target_n 2048 \
#         --model "pointnet++" \
#         --focal_loss \
#         --dropout 0.5 \
#         --early_stopping 30 \
#         --num_workers $NUM_WORKERS \
#         --seed $SEED \
#         --kfold $KFOLD \
#         --kfold_save_dir "training_logs/kfold_multichannel"
# fi
# 3. Combined Models (Geometry + Hemodynamics)
# 3a. Late Fusion
# echo ""
# echo "=============================================================="
# echo "3a. Training Combined - Late Fusion"
# echo "=============================================================="
#
# if check_if_trained "training_logs/combined_late_fusion.pth" "training_logs/kfold_late_fusion"; then
#     : # Skip - already trained
# elif [ "$SINGLE_SPLIT" = true ]; then
#     python combined_rupture_classification.py \
#         --data_dir "$DATA_DIR" \
#         --metadata "$METADATA" \
#         --epochs $EPOCHS \
#         --batch_size $BATCH_SIZE \
#         --lr 1e-3 \
#         --target_n 1024 \
#         --model "late_fusion" \
#         --dropout 0.5 \
#         --early_stopping 30 \
#         --num_workers $NUM_WORKERS \
#         --seed $SEED \
#         --save_path "training_logs/combined_late_fusion.pth"
# else
#     python combined_rupture_classification.py \
#         --data_dir "$DATA_DIR" \
#         --metadata "$METADATA" \
#         --epochs $EPOCHS \
#         --batch_size $BATCH_SIZE \
#         --lr 1e-3 \
#         --target_n 1024 \
#         --model "late_fusion" \
#         --dropout 0.5 \
#         --early_stopping 30 \
#         --num_workers $NUM_WORKERS \
#         --seed $SEED \
#         --kfold $KFOLD \
#         --kfold_save_dir "training_logs/kfold_late_fusion"
# fi

# 3b. Attention Fusion
# echo ""
# echo "=============================================================="
# echo "3b. Training Combined - Attention Fusion"
# echo "=============================================================="
#
# if check_if_trained "training_logs/combined_attention.pth" "training_logs/kfold_attention"; then
#     : # Skip - already trained
# elif [ "$SINGLE_SPLIT" = true ]; then
#     python combined_rupture_classification.py \
#         --data_dir "$DATA_DIR" \
#         --metadata "$METADATA" \
#         --epochs $EPOCHS \
#         --batch_size $BATCH_SIZE \
#         --lr 1e-3 \
#         --target_n 1024 \
#         --model "attention" \
#         --dropout 0.5 \
#         --early_stopping 30 \
#         --num_workers $NUM_WORKERS \
#         --seed $SEED \
#         --save_path "training_logs/combined_attention.pth"
# else
#     python combined_rupture_classification.py \
#         --data_dir "$DATA_DIR" \
#         --metadata "$METADATA" \
#         --epochs $EPOCHS \
#         --batch_size $BATCH_SIZE \
#         --lr 1e-3 \
#         --target_n 1024 \
#         --model "attention" \
#         --dropout 0.5 \
#         --early_stopping 30 \
#         --num_workers $NUM_WORKERS \
#         --seed $SEED \
#         --kfold $KFOLD \
#         --kfold_save_dir "training_logs/kfold_attention"
# fi

# 3c. Early Fusion
# echo ""
# echo "=============================================================="
# echo "3c. Training Combined - Early Fusion"
# echo "=============================================================="
#
# if check_if_trained "training_logs/combined_early_fusion.pth" "training_logs/kfold_early_fusion"; then
#     : # Skip - already trained
# elif [ "$SINGLE_SPLIT" = true ]; then
#     python combined_rupture_classification.py \
#         --data_dir "$DATA_DIR" \
#         --metadata "$METADATA" \
#         --epochs $EPOCHS \
#         --batch_size $BATCH_SIZE \
#         --lr 1e-3 \
#         --target_n 1024 \
#         --model "early_fusion" \
#         --dropout 0.5 \
#         --early_stopping 30 \
#         --num_workers $NUM_WORKERS \
#         --seed $SEED \
#         --save_path "training_logs/combined_early_fusion.pth"
# else
#     python combined_rupture_classification.py \
#         --data_dir "$DATA_DIR" \
#         --metadata "$METADATA" \
#         --epochs $EPOCHS \
#         --batch_size $BATCH_SIZE \
#         --lr 1e-3 \
#         --target_n 1024 \
#         --model "early_fusion" \
#         --dropout 0.5 \
#         --early_stopping 30 \
#         --num_workers $NUM_WORKERS \
#         --seed $SEED \
#         --kfold $KFOLD \
#         --kfold_save_dir "training_logs/kfold_early_fusion"
# fi
# 4. PINN (Physics-Informed Neural Network) - COMMENTED OUT (underperforming)
# echo ""
# echo "=============================================================="
# echo "4. Training PINN (Physics-Informed Neural Network)"
# echo "=============================================================="
#
# if check_if_trained "training_logs/pinn_model.pth" "training_logs/kfold_pinn"; then
#     : # Skip - already trained
# elif [ "$SINGLE_SPLIT" = true ]; then
#     python pinn_rupture_classification.py \
#         --data_dir "$DATA_DIR" \
#         --metadata "$METADATA" \
#         --epochs $EPOCHS \
#         --batch_size $BATCH_SIZE \
#         --lr 1e-4 \
#         --target_n $TARGET_N \
#         --model "pinn" \
#         --use_physics \
#         --lambda_physics 0.1 \
#         --focal_loss \
#         --dropout 0.5 \
#         --num_workers $NUM_WORKERS \
#         --seed $SEED \
#         --save_path "training_logs/pinn_model.pth"
# else
#     python pinn_rupture_classification.py \
#         --data_dir "$DATA_DIR" \
#         --metadata "$METADATA" \
#         --epochs $EPOCHS \
#         --batch_size $BATCH_SIZE \
#         --lr 1e-4 \
#         --target_n $TARGET_N \
#         --model "pinn" \
#         --use_physics \
#         --lambda_physics 0.1 \
#         --focal_loss \
#         --dropout 0.5 \
#         --num_workers $NUM_WORKERS \
#         --seed $SEED \
#         --kfold $KFOLD \
#         --kfold_save_dir "training_logs/kfold_pinn"
# fi
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

# Attention Fusion with Clinical - COMMENTED OUT (underperforming)
# echo "  5b. Attention Fusion with Clinical Features..."
# if check_if_trained "training_logs/fusion_attention.pth" "training_logs/kfold_fusion_attention"; then
#     : # Skip - already trained
# elif [ "$SINGLE_SPLIT" = true ]; then
#     python fusion.py \
#         --data_dir "$DATA_DIR" \
#         --metadata "$METADATA" \
#         --epochs $EPOCHS \
#         --batch_size $BATCH_SIZE \
#         --lr 1e-4 \
#         --target_n $TARGET_N \
#         --model "attention" \
#         --dropout 0.5 \
#         --num_workers $NUM_WORKERS \
#         --seed $SEED \
#         --save_path "training_logs/fusion_attention.pth"
# else
#     python fusion.py \
#         --data_dir "$DATA_DIR" \
#         --metadata "$METADATA" \
#         --epochs $EPOCHS \
#         --batch_size $BATCH_SIZE \
#         --lr 1e-4 \
#         --target_n $TARGET_N \
#         --model "attention" \
#         --dropout 0.5 \
#         --num_workers $NUM_WORKERS \
#         --seed $SEED \
#         --kfold $KFOLD \
#         --kfold_save_dir "training_logs/kfold_fusion_attention"
# fi

# 5c. Full Fusion with Gating - COMMENTED OUT (underperforming)
# echo "  5c. Full Fusion with Gating Mechanism..."
# if check_if_trained "training_logs/fusion_full.pth" "training_logs/kfold_fusion_full"; then
#     : # Skip - already trained
# elif [ "$SINGLE_SPLIT" = true ]; then
#     python fusion.py \
#         --data_dir "$DATA_DIR" \
#         --metadata "$METADATA" \
#         --epochs $EPOCHS \
#         --batch_size $BATCH_SIZE \
#         --lr 1e-4 \
#         --target_n $TARGET_N \
#         --model "full_fusion" \
#         --dropout 0.5 \
#         --num_workers $NUM_WORKERS \
#         --seed $SEED \
#         --save_path "training_logs/fusion_full.pth"
# else
#     python fusion.py \
#         --data_dir "$DATA_DIR" \
#         --metadata "$METADATA" \
#         --epochs $EPOCHS \
#         --batch_size $BATCH_SIZE \
#         --lr 1e-4 \
#         --target_n $TARGET_N \
#         --model "full_fusion" \
#         --dropout 0.5 \
#         --num_workers $NUM_WORKERS \
#         --seed $SEED \
#         --kfold $KFOLD \
#         --kfold_save_dir "training_logs/kfold_fusion_full"
# fi
# 5. Clinical Only (baseline reference)
echo ""
echo "=============================================================="
echo "5. Training Clinical Only Baseline (Age + Sex only)"
echo "=============================================================="

if check_if_trained "training_logs/clinical_only.pth" "training_logs/kfold_clinical_only"; then
    : # Skip - already trained
elif [ "$SINGLE_SPLIT" = true ]; then
    python -u fusion.py \
        --data_dir "$DATA_DIR" \
        --metadata "$METADATA" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr 1e-4 \
        --target_n $TARGET_N \
        --model "clinical_only" \
        --dropout 0.5 \
        --num_workers $NUM_WORKERS \
        --seed $SEED \
        --save_path "training_logs/clinical_only.pth"
else
    python -u fusion.py \
        --data_dir "$DATA_DIR" \
        --metadata "$METADATA" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr 1e-4 \
        --target_n $TARGET_N \
        --model "clinical_only" \
        --dropout 0.5 \
        --num_workers $NUM_WORKERS \
        --seed $SEED \
        --kfold $KFOLD \
        --kfold_save_dir "training_logs/kfold_clinical_only"
fi
# Summary
echo ""
echo "=============================================================="
echo "                    TRAINING COMPLETE"
echo "=============================================================="
echo ""
echo "Training logs saved to: training_logs/"
echo ""
echo "Configuration:"
echo "  - Points per sample: $TARGET_N"
echo "  - Epochs: $EPOCHS (no early stopping)"
echo "  - Learning rate: 1e-4"
echo "  - Batch size: $BATCH_SIZE"
echo ""
echo "Active Models (in training order):"
echo "  1. Geometry-only PointNet++ ($TARGET_N points)"
echo "  2. GNN (GraphSAGE) ($TARGET_N points)"
echo "  3. Ensemble Model (Geometry + Fusion Late combined)"
echo "  4. Fusion Late (Geometry + Hemodynamics + Clinical)"
echo "  5. Clinical Only Baseline (reference)"
echo ""
echo "Commented Out (underperforming):"
echo "  - Multichannel PointNet++ (hemodynamics only)"
echo "  - Combined models (Late, Attention, Early Fusion)"
echo "  - PINN"
echo "  - Attention/Full Fusion with Clinical"
echo ""
echo "=============================================================="
