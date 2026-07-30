#!/bin/bash
# Version 7 source snapshot
# submit_all.sh
# This script submits all v7 scripts to the Slurm queue.

mkdir -p logs

SCRIPTS=(
    "pinn_pipeline.py"
    "train_clinical_only.py"
    "train_clinical_age_sex.py"
    "train_geometry.py"
    "train_flow_geometry.py"
    "train_gnn.py"
    "train_ensemble.py"
    "train_ensemble_rrt.py"
)

for script in "${SCRIPTS[@]}"; do
    job_name=$(basename "$script" .py)

    echo "Submitting $script..."

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=logs/$job_name_%j.out
#SBATCH --error=logs/$job_name_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate pointnet

python $script
EOT

done

echo "All v7 scripts submitted!"
