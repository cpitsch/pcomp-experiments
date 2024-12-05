This experiment analyzes the Type I error rate (or FPR) of the two approaches. This is
done by applying them to many negative comparison instances created by splitting the Road
Traffic Fine Management event log into two random halves.

The dataset is created using EventLogMutator with the configuration file: `event_log_mutator --pipeline road_traffic_random_splits.toml`.

- `run_{technique}_{dimension}.py` runs the experiment for the technique ("permutation" or "bootstrap") and the dimension ("time" or control flow "cf")
  - `sbatch slurm_job_{technique}_{dimension}.sh` creates a SLURM job to run this experiment
- `generate_figures.py` takes the experiment results and draws for each significance level,
the achieved Type I error rate and compares it to the expected value of Type I error rate = &alpha;
