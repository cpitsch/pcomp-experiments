import pickle
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer
from typing import Any, Literal

import pandas as pd
from mpire.pool import WorkerPool
from pcomp.emd.comparators.bootstrap import ControlFlowBootstrapComparator
from pcomp.utils import import_log

LOGS_BASE_DIR = Path("random_split_logs")
OUTPUT_BASE_PATH = Path("results", "bootstrap_cf")

WEIGHTED_TIME_COST = True

DIST_SIZE = 10_000
SEED = 1337
SIGNIFICANCE_LEVEL = 0.05

parser = ArgumentParser()
parser.add_argument(
    "-c",
    "--cores",
    type=int,
    help="The number of cores to use for multiprocessing",
    required=True,
)
parser.add_argument(
    "-v",
    "--verbose",
    help="Enable verbose instance execution (instance-specific progress bars)",
    action="store_true",
)
args = parser.parse_args()


@dataclass
class EventLogSetting:
    path: Path

    @property
    def seed(self) -> int:
        # Example Path
        # road_traffic_random_splits/PartialOrderCreator/LogSplitter_frac_0.5_seed_<seed>
        return int(self.path.name.split("_")[-1])

    @property
    def log_paths(self) -> tuple[Path, Path]:
        return (self.path / "log_1.xes.gz", self.path / "log_2.xes.gz")

    def get_logs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        path_1, path_2 = self.log_paths
        return (
            import_log(path_1.as_posix(), variant="rustxes"),
            import_log(path_2.as_posix(), variant="rustxes"),
        )

    @property
    def identifier(self) -> str:
        return str(self.seed)


@dataclass
class Instance:
    log: EventLogSetting
    weighted_time_cost: bool

    @property
    def path(self) -> Path:
        return OUTPUT_BASE_PATH / self.log.identifier

    @property
    def pickle_path(self) -> Path:
        return self.path / "result.pkl"

    @property
    def technique_name(self) -> str:
        return "bootstrap_test"

    def get_logs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.log.get_logs()

    def get_comparator(self, verbose: bool = True) -> ControlFlowBootstrapComparator:
        return ControlFlowBootstrapComparator(
            *self.get_logs(),
            bootstrapping_dist_size=DIST_SIZE,
            seed=SEED,
            verbose=verbose,
        )

    def load_pickle(self) -> ControlFlowBootstrapComparator:
        if not self.pickle_path.exists():
            raise ValueError(
                "Attempt to load non-existent pickle file:", self.pickle_path.as_posix()
            )
        with open(self.pickle_path, "rb") as f:
            return pickle.load(f)

    def to_result_row(
        self, pval: float, logs_emd: float, duration_seconds: float
    ) -> dict[str, Any]:
        detection = pval < SIGNIFICANCE_LEVEL
        log_has_drift = False
        is_correct = detection == log_has_drift

        return {
            "technique": "Bootstrap Test (Control Flow)",
            "log_seed": self.log.seed,
            "log_path": self.log.path.as_posix(),
            "has_diff": log_has_drift,
            "is_no_diff_log": True,
            "pickle_path": self.pickle_path.as_posix(),
            "pval": pval,
            "logs_emd": logs_emd,
            "detection": detection,
            "correct": is_correct,
            "classification_class": get_classification_class(detection, log_has_drift),
            "duration": duration_seconds,
        }

    def run_and_save_results(self, verbose: bool = True) -> dict[str, Any]:
        start_time = default_timer()
        comparator = self.get_comparator(verbose)
        res = comparator.compare()
        end_time = default_timer()

        # Save results
        self.path.mkdir(parents=True)
        with open(self.pickle_path, "wb") as f:
            pickle.dump(res, f)

        return self.to_result_row(res.pvalue, res.logs_emd, end_time - start_time)


def get_classification_class(
    detection: bool, has_diff: bool
) -> Literal["TP", "TN", "FP", "FN"]:
    is_correct = detection == has_diff
    is_positive = detection

    first_part = "T" if is_correct else "F"
    second_part = "P" if is_positive else "N"
    return first_part + second_part  # type: ignore


def get_all_log_base_paths() -> list[Path]:
    return [path.parent for path in LOGS_BASE_DIR.rglob("log_1.xes.gz")]


def get_all_log_instances() -> list[EventLogSetting]:
    return [EventLogSetting(path) for path in get_all_log_base_paths()]


def get_all_instances() -> list[Instance]:
    return [
        Instance(log_instance, WEIGHTED_TIME_COST)
        for log_instance in get_all_log_instances()
    ]


def run_instance(instance: Instance) -> dict[str, Any]:
    return instance.run_and_save_results(verbose=args.verbose)


def main():
    start_time = default_timer()
    instances = get_all_instances()
    instances = [
        instance for instance in instances if not instance.pickle_path.exists()
    ]
    instances = sorted(instances, key=lambda instance: instance.log.seed)

    OUTPUT_BASE_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(instances)} comparisons")

    with WorkerPool(min(len(instances), args.cores), start_method="spawn") as p:
        results = p.map(run_instance, instances, progress_bar=True)
    df = pd.DataFrame(results)
    SUMMARY_PATH = OUTPUT_BASE_PATH / "summary.csv"
    if SUMMARY_PATH.exists():
        old_df = pd.read_csv(SUMMARY_PATH)
        df = pd.concat([old_df, df])
    df.to_csv(OUTPUT_BASE_PATH / "summary.csv", index=False)

    print(f"Elapsed Time: {default_timer() - start_time}")


if __name__ == "__main__":
    main()
