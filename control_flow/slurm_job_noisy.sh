#!/usr/bin/zsh

### Job Parameters
#SBATCH --job-name=pcomp_noisy_cf
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=c23ms
#SBATCH --output=control_flow_output.txt

### Program Code
cd ~/pcomp-experiments/control_flow
source ../.venv/bin/activate

python run_noisy.py --cores ${SLURM_JOB_CPUS_PER_NODE}
