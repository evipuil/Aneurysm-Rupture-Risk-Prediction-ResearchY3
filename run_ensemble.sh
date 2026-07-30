#!/bin/bash
# Version 5 source snapshot
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output=output_ensemble.txt
#SBATCH --error=error_ensemble.txt

python train_ensemble.py
