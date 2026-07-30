#!/bin/bash
# Version 6 source snapshot
# Submit the Version 6 training jobs to Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

SCRIPTS=(
    "train_clinical_only.py"
    "train_clinical_age_sex.py"
    "train_geometry.py"
    "train_gnn.py"
    "pinn_correction.py"
)

# You can adjust Slurm parameters (time, partition, GPUs, memory, environment) as needed.
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

# Load environment (adjust if needed)
source ~/.bashrc
conda activate pointnet

# Run from the snapshot directory so normalized filenames resolve correctly.
cd "${SCRIPT_DIR}"
python "$script"
EOT

done

echo "All training scripts submitted!"
