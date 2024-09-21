#!/usr/bin/zsh

### Job Parameters
#SBATCH --array=1..10
#SBATCH --job-name=pcomp_synthetic_log_runs
#SBATCH --time=09:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=devel

### Program Code
cd ~/jupyterlab/pcomp_synthetic_log_runs
source .venv/bin/activate

echo $SLURM_ARRAY_TASK_ID
echo $SLURM_JOB_CPUS_PER_NODE

python run_synthetic_road_traffic_logs.py --seed ${SLURM_ARRAY_TASK_ID} --cores ${SLURM_JOB_CPUS_PER_NODE} > outputs/${SLURM_ARRAY_TASK_ID}.txt

## Saved for later
#SBATCH --output=pcomp_stdout.txt
