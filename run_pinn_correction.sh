#!/bin/bash
# Version 4 source snapshot
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output=output_pinn.txt
#SBATCH --error=error_pinn.txt

python pinn_correction.py
