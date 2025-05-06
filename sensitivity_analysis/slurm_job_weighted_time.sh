#!/usr/bin/zsh

### Job Parameters
#SBATCH --array=1-5
#SBATCH --job-name=pcomp_sensitivity_wt
#SBATCH --time=09:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=c23ms
#SBATCH --output=pcomp_sensitivity_wt.txt

### Program Code
cd ~/pcomp-experiments/sensitivity_analysis
source ../.venv/bin/activate

python run_synthetic_road_traffic_logs.py --seed ${SLURM_ARRAY_TASK_ID} --cores ${SLURM_JOB_CPUS_PER_NODE} --weighted-time > weighted_time_outputs/${SLURM_ARRAY_TASK_ID}.txt
