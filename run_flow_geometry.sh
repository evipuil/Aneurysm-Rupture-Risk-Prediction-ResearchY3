#!/bin/bash
# Version 4 source snapshot
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output=output_flow_geometry.txt
#SBATCH --error=error_flow_geometry.txt

python train_flow_geometry.py
