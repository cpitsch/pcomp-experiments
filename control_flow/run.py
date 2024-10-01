import pickle
from abc import ABC, abstractmethod, abstractproperty
from argparse import ArgumentParser
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from timeit import default_timer
from typing import Any, Generic, Literal, TypeVar, cast

import pandas as pd
from mpire.pool import WorkerPool
from pcomp.emd.comparators.bootstrap import ControlFlowBootstrapComparator
from pcomp.emd.comparators.bootstrap.bootstrap_comparator import (
    BootstrapTestComparisonResult,
)
from pcomp.emd.comparators.permutation_test import (
    ControlFlowPermutationComparator,
    PermutationTestComparisonResult,
)
from pm4py import read_xes

parser = ArgumentParser()
parser.add_argument(
    "-c",
    "--cores",
    type=int,
    help="The number of cores to use for multiprocessing.",
    required=True,
)
args = parser.parse_args()

LOGS_BASE_PATH = Path("testing_logs")


OUTPUT_BASE_PATH = Path("control_flow_results")

SIGNIFICANCE_LEVEL = 0.05
DIST_SIZE = 10_000
SEED = 1337
MP_CORES = 16


OSTOVAR_BASE_PATH = LOGS_BASE_PATH / "ostovar"
# The ranges of the event logs that don't contain a drift
# OSTOVAR_BEHAVIOR_RANGES = [(0, 999), (1000, 1999), (2000, 2999)]
OSTOVAR_BEHAVIOR_RANGES = [(1, 1000), (1001, 2000), (2001, 2999)]
OSTOVAR_DRIFT_COMPARISON_RANGES: list[tuple[tuple[int, int], tuple[int, int]]] = [
    (range_1, range_2)
    for range_1, range_2 in zip(OSTOVAR_BEHAVIOR_RANGES, OSTOVAR_BEHAVIOR_RANGES[1:])
]
# Divide the behavior range in half and compare the halves
OSTOVAR_NO_DRIFT_COMPARISON_RANGES: list[tuple[tuple[int, int], tuple[int, int]]] = [
    (
        (start, floor((end + start) / 2)),
        (ceil((end + start) / 2), end),
    )
    for start, end in OSTOVAR_BEHAVIOR_RANGES
]


BOSE_BASE_PATH = LOGS_BASE_PATH / "bose"
# The ranges of the event log that dont contain a drift
BOSE_BEHAVIOR_RANGES = [
    (1, 1200),
    (1201, 2400),
    (2401, 3600),
    (3601, 4800),
    (4801, 6000),
]
BOSE_DRIFT_COMPARISON_RANGES: list[tuple[tuple[int, int], tuple[int, int]]] = [
    (range_1, range_2)
    for range_1, range_2 in zip(BOSE_BEHAVIOR_RANGES, BOSE_BEHAVIOR_RANGES[1:])
]
# Divide the behavior range in half and compare the halves
BOSE_NO_DRIFT_COMPARISON_RANGES: list[tuple[tuple[int, int], tuple[int, int]]] = [
    (
        (start, floor((end + start) / 2)),
        (ceil((end + start) / 2), end),
    )
    for start, end in BOSE_BEHAVIOR_RANGES
]


CERAVOLO_BASE_PATH = LOGS_BASE_PATH / Path("ceravolo")
# The ranges of the event log that dont contain a drift
CERAVOLO_BEHAVIOR_RANGES = [(0, 499), (500, 999)]
CERAVOLO_DRIFT_COMPARISON_RANGES: list[tuple[tuple[int, int], tuple[int, int]]] = [
    (range_1, range_2)
    for range_1, range_2 in zip(CERAVOLO_BEHAVIOR_RANGES, CERAVOLO_BEHAVIOR_RANGES[1:])
]
# Divide the behavior range in half and compare the halves
CERAVOLO_NO_DRIFT_COMPARISON_RANGES: list[tuple[tuple[int, int], tuple[int, int]]] = [
    (
        (start, floor((end + start) / 2)),
        (ceil((end + start) / 2), end),
    )
    for start, end in CERAVOLO_BEHAVIOR_RANGES
]


@dataclass
class LogInstance:
    path: Path
    has_drift: bool
    log_1_range: tuple[int, int]
    log_2_range: tuple[int, int]

    def get_logs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        log = read_xes(self.path.as_posix(), variant="rustxes")
        return (
            cast(
                pd.DataFrame,
                log[
                    log["case:concept:name"]
                    .astype(int)
                    .isin(range(self.log_1_range[0], self.log_1_range[1] + 1))
                ],
            ),
            cast(
                pd.DataFrame,
                log[
                    log["case:concept:name"]
                    .astype(int)
                    .isin(range(self.log_2_range[0], self.log_2_range[1] + 1))
                ],
            ),
        )

    @property
    def source(self) -> str:
        return self.path.relative_to(LOGS_BASE_PATH).parents[0].name

    @property
    def identifier(self) -> str:
        return f"{self.source}_{self.path.name.split('.')[0]}_{self.log_1_range[0]}-{self.log_1_range[1]}_{self.log_2_range[0]}-{self.log_2_range[1]}"


def get_ostovar_log_paths() -> list[Path]:
    return list(OSTOVAR_BASE_PATH.rglob("*.xes.gz"))


def get_ostovar_log_instances() -> list[LogInstance]:
    return [
        LogInstance(path, has_drift, *comparison_range)
        for comparison_ranges, has_drift in [
            (OSTOVAR_DRIFT_COMPARISON_RANGES, True),
            (OSTOVAR_NO_DRIFT_COMPARISON_RANGES, False),
        ]
        for comparison_range in comparison_ranges
        for path in get_ostovar_log_paths()
    ]


def get_bose_log_paths() -> list[Path]:
    return list(BOSE_BASE_PATH.rglob("*.xes.gz"))


def get_bose_log_instances() -> list[LogInstance]:
    bose_log_paths = get_bose_log_paths()
    assert len(bose_log_paths) == 1  # Should only be the normal bose log
    return [
        LogInstance(path, has_drift, *comparison_range)
        for comparison_ranges, has_drift in [
            (BOSE_DRIFT_COMPARISON_RANGES, True),
            (BOSE_NO_DRIFT_COMPARISON_RANGES, False),
        ]
        for comparison_range in comparison_ranges
        for path in bose_log_paths
    ]


def get_ceravolo_log_paths() -> list[Path]:
    return list(CERAVOLO_BASE_PATH.rglob("*.xes.gz"))


def get_ceravolo_log_instances() -> list[LogInstance]:
    return [
        LogInstance(path, has_drift, *comparison_range)
        for comparison_ranges, has_drift in [
            (CERAVOLO_DRIFT_COMPARISON_RANGES, True),
            (CERAVOLO_NO_DRIFT_COMPARISON_RANGES, False),
        ]
        for comparison_range in comparison_ranges
        for path in get_ceravolo_log_paths()
    ]


T = TypeVar("T")
R = TypeVar("R")


@dataclass
class Instance(ABC, Generic[T, R]):
    log_instance: LogInstance

    @property
    def path(self) -> Path:
        return OUTPUT_BASE_PATH / self.log_instance.identifier

    @property
    def pickle_path(self) -> Path:
        return self.path / "result.pkl"

    @abstractproperty
    def technique_name(self) -> str: ...

    def get_logs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.log_instance.get_logs()

    @abstractmethod
    def get_comparator(self, verbose: bool = True) -> T: ...

    def load_pickle(
        self,
    ) -> R:
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
        log_has_drift = self.log_instance.has_drift
        is_correct = detection == log_has_drift

        return {
            "technique": self.technique_name,
            "log_source": self.log_instance.source,
            "log_path": self.log_instance.path.as_posix(),
            "has_diff": log_has_drift,
            "pval": pval,
            "logs_emd": logs_emd,
            "detection": detection,
            "correct": is_correct,
            "classification_class": get_classification_class(detection, log_has_drift),
            "duration": duration_seconds,
        }

    @abstractmethod
    def run_and_save_results(self, verbose: bool = True) -> dict[str, Any]: ...


@dataclass
class PermutationTestInstance(
    Instance[ControlFlowPermutationComparator, PermutationTestComparisonResult]
):
    @property
    def technique_name(self) -> str:
        return "Permutation Test"

    def get_comparator(self, verbose: bool = True) -> ControlFlowPermutationComparator:
        return ControlFlowPermutationComparator(
            *self.get_logs(), distribution_size=DIST_SIZE, seed=SEED, verbose=verbose
        )

    def run_and_save_results(self, verbose: bool = True) -> dict[str, Any]:
        start_time = default_timer()
        comparator = self.get_comparator(verbose)
        res = comparator.compare()
        end_time = default_timer()

        # Save results
        if not self.path.exists():
            self.path.mkdir(parents=True)
        with open(self.pickle_path, "wb") as f:
            pickle.dump(res, f)

        return self.to_result_row(res.pvalue, res.logs_emd, end_time - start_time)


@dataclass
class BootstrapTestInstance(
    Instance[ControlFlowBootstrapComparator, BootstrapTestComparisonResult]
):
    @property
    def technique_name(self) -> str:
        return "Bootstrap Test"

    def get_comparator(self, verbose: bool = True) -> ControlFlowBootstrapComparator:
        return ControlFlowBootstrapComparator(
            *self.get_logs(),
            bootstrapping_dist_size=DIST_SIZE,
            seed=SEED,
            verbose=verbose,
        )

    def run_and_save_results(self, verbose: bool = True) -> dict[str, Any]:
        start_time = default_timer()
        comparator = self.get_comparator(verbose)
        res = comparator.compare()
        end_time = default_timer()

        # Save results
        if not self.path.exists():
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
    return first_part + second_part


def run_instance(instance: Instance) -> dict[str, Any]:
    return instance.run_and_save_results(verbose=True)


def run():
    all_log_instances = (
        get_ostovar_log_instances()
        + get_bose_log_instances()
        + get_ceravolo_log_instances()
    )

    permutation_test_instances: list[Instance] = [
        PermutationTestInstance(log_instance) for log_instance in all_log_instances
    ]
    bootstrap_instances: list[Instance] = [
        BootstrapTestInstance(log_instance) for log_instance in all_log_instances
    ]

    all_instances: list[Instance] = permutation_test_instances + bootstrap_instances

    start_time = default_timer()
    with WorkerPool(args.cores) as p:
        results = p.map(run_instance, all_instances, progress_bar=True)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_BASE_PATH / "summary.csv", index=False)

    print(f"Elapsed Time: {default_timer() - start_time}")


if __name__ == "__main__":
    run()
