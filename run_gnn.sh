#!/bin/bash
# Version 7 source snapshot
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output=output_v7_gnn.txt
#SBATCH --error=error_v7_gnn.txt

source ~/.bashrc
conda activate pointnet

python train_gnn.py
