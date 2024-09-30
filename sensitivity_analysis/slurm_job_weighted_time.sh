#!/usr/bin/zsh

### Job Parameters
#SBATCH --array=1-2
#SBATCH --job-name=pcomp_synthetic_log_runs_weighted_time
#SBATCH --time=09:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=c23ms
#SBATCH --output=some_file_weighted_time.txt

### Program Code
cd ~/jupyterlab/pcomp_synthetic_log_runs
source .venv/bin/activate

python run_synthetic_road_traffic_logs_weighted_time.py --seed ${SLURM_ARRAY_TASK_ID} --cores ${SLURM_JOB_CPUS_PER_NODE} > weighted_time_outputs/${SLURM_ARRAY_TASK_ID}.txt
