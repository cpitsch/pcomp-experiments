#!/usr/bin/zsh

### Job Parameters
#SBATCH --job-name=pcomp_control_flow_runs
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=c23ms
#SBATCH --output=control_flow_output.txt

### Program Code
cd ~/jupyterlab/pcomp_synthetic_log_runs
source .venv/bin/activate
