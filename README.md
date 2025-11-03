Repository for the Evaluation of the pcomp process hypothesis testing technique

- `control_flow` contains scripts for the control flow evaluation, i.e., running experiments and analyzing results, generating figures and tables
- `mimic` contains SQL queries, a processing script, and jupyter notebooks for the analysis
- `road_traffic_random_splits` contains the scripts and EventLogMutator configuration file for the Type I Error Rate analysis.
    - Also a script for figure generation
- `sensitivity_analysis` contains the scripts for the timed-control-flow sensitivity analysis, as well as scripts for generating the figures and tables in the thesis. 

- All experiments were performed on the version of the [pcomp](https://github.com/cpitsch/pcomp) project with the commit hash:
    - [`76f16a43f76b1cd428d9a4be6c9fe1ed251aff4d`](https://github.com/cpitsch/pcomp/tree/76f16a43f76b1cd428d9a4be6c9fe1ed251aff4d) (ICPM 2025)
    - [`9aaac7994ed0e026d63df216f56649de73a4692a`](https://github.com/cpitsch/pcomp/tree/9aaac7994ed0e026d63df216f56649de73a4692a) (Thesis)
