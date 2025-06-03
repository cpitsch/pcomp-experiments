#!/usr/bin/zsh

### Job Parameters
#SBATCH --array=1-5
#SBATCH --job-name=pcomp_sensitivity
#SBATCH --time=09:00:00
#SBATCH --cpus-per-task=32
#SBATCH --partition=c23mm
#SBATCH --output=pcomp_sensitivity.txt

### Program Code
cd ~/pcomp-experiments/sensitivity_analysis
source ../.venv/bin/activate

mkdir outputs

python run_synthetic_road_traffic_logs.py --seed ${SLURM_ARRAY_TASK_ID} --cores ${SLURM_JOB_CPUS_PER_NODE} > outputs/${SLURM_ARRAY_TASK_ID}.txt
