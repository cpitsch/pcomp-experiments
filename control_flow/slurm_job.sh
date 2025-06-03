#!/usr/bin/zsh

### Job Parameters
#SBATCH --job-name=pcomp_control_flow_all
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --partition=c23ms
#SBATCH --output=control_flow_output.txt

### Program Code
cd ~/pcomp-experiments/control_flow
source ../.venv/bin/activate

python run.py --cores ${SLURM_JOB_CPUS_PER_NODE}
