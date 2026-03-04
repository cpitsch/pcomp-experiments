Repository for the Evaluation of the pcomp process hypothesis testing technique

- `control_flow` contains scripts for the control flow evaluation, i.e., running experiments and analyzing results, generating figures and tables
- `mimic` contains SQL queries, a processing script, and jupyter notebooks for the analysis
- `road_traffic_random_splits` contains the scripts and EventLogMutator configuration file for the Type I Error Rate analysis.
    - Also a script for figure generation
- `sensitivity_analysis` contains the scripts for the timed-control-flow sensitivity analysis, as well as scripts for generating the figures and tables in the thesis. 

## Reproducibility

The experiments reported in the ICPM 2025 paper. Are based on the following versions of the relevant projects:

| Repository                                                                    | Used Version                                                                                                                                                                       |
|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [pcomp-experiments](https://github.com/cpitsch/pcomp-experiments) (this repo) | [`a4aba59`](https://github.com/cpitsch/pcomp-experiments/tree/a4aba59cbf4190c28156b019c0ca8b954e48b2e5) (Tag [`ICPM25`](https://github.com/cpitsch/pcomp-experiments/tree/ICPM25)) |
| [pcomp](https://github.com/cpitsch/pcomp)                                     | [`76f16a4`](https://github.com/cpitsch/pcomp/tree/76f16a43f76b1cd428d9a4be6c9fe1ed251aff4d) (Tag [`ICPM25`](https://github.com/cpitsch/pcomp/tree/ICPM25))                         |
| [event_log_mutator](https://github.com/cpitsch/event_log_mutator)             | [`d2afb17`](https://github.com/cpitsch/event_log_mutator/tree/d2afb17d5d59bf8fe174a79fcc642a3cc2c8db74)                                                                            |

- Since then, pipeline files have been updated to reflect a new version of the `event_log_mutator`
    - In theory, the results should be identical except that the log paths are slightly different in the new `event_log_mutator` version


#### Master's Thesis

For the Master's Thesis, the versions are as follows:

- `pcomp`: [`9aaac79`](https://github.com/cpitsch/pcomp/tree/9aaac7994ed0e026d63df216f56649de73a4692a)
