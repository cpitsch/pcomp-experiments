#!/usr/bin/zsh

### Job Parameters
#SBATCH --job-name=pcomp_random_splits_permutation_cf
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --partition=c23ms
#SBATCH --output=random_splits_output.txt

### Program Code
cd ~/pcomp-experiments/road_traffic_random_splits
source ../.venv/bin/activate

python run.py --cores ${SLURM_JOB_CPUS_PER_NODE}
