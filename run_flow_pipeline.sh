#!/bin/bash
# Version 6 source snapshot
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output=output_flow_simulation.txt
#SBATCH --error=error_flow_simulation.txt

python flow_simulation_pipeline.py
