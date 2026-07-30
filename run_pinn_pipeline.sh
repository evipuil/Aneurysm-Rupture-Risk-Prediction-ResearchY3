#!/bin/bash
# Version 7 source snapshot
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output=output_pinn_pipeline.txt
#SBATCH --error=error_pinn_pipeline.txt

source ~/.bashrc
conda activate pointnet

python pinn_pipeline.py
