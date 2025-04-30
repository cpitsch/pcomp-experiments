import pickle
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from timeit import default_timer
from typing import Any, Generic, Literal, TypeVar, cast

import pandas as pd
import yaml
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
from pydantic import BaseModel

LOGS_BASE_PATH = Path("testing_logs")


YAML_PATH = Path("log_settings.yaml")
OUTPUT_BASE_PATH = Path("control_flow_results")

SIGNIFICANCE_LEVEL = 0.05
DIST_SIZE = 10_000
SEED = 1337


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
        return self.path.relative_to(LOGS_BASE_PATH).parts[0]

    @property
    def identifier(self) -> str:
        return f"{self.path.name.split('.')[0]}_{self.log_1_range[0]}-{self.log_1_range[1]}_{self.log_2_range[0]}-{self.log_2_range[1]}"

    @property
    def noise_level(self) -> int:
        if self.path.parent.name == "bose":
            return 0
        else:
            return int(self.path.parent.name)


class LogSetting(BaseModel):
    identifier: str
    base_path: Path
    behavior_ranges: list[tuple[int, int]]

    def to_log_instances(self) -> list[LogInstance]:
        log_paths = list((LOGS_BASE_PATH / self.base_path).rglob("*.xes.gz"))
        drift_comparison_ranges: list[tuple[tuple[int, int], tuple[int, int]]] = [
            (range_1, range_2)
            for range_1, range_2 in zip(self.behavior_ranges, self.behavior_ranges[1:])
        ]

        # Divide the behavior range in half and compare the halves
        no_drift_comparison_ranges: list[tuple[tuple[int, int], tuple[int, int]]] = [
            (
                (start, floor((end + start) / 2)),
                (ceil((end + start) / 2), end),
            )
            for start, end in self.behavior_ranges
        ]

        return [
            LogInstance(path, has_drift, *comparison_range)
            for comparison_ranges, has_drift in [
                (drift_comparison_ranges, True),
                (no_drift_comparison_ranges, False),
            ]
            for comparison_range in comparison_ranges
            for path in log_paths
        ]


def parse_yaml_to_log_settings(yaml_path: Path) -> list[LogSetting]:
    yml = yaml.safe_load(yaml_path.read_text())
    return [LogSetting.model_validate(v | {"identifier": k}) for k, v in yml.items()]


def get_all_log_instances() -> list[LogInstance]:
    log_settings = parse_yaml_to_log_settings(YAML_PATH)
    return [
        log_instance
        for log_setting in log_settings
        for log_instance in log_setting.to_log_instances()
    ]


T = TypeVar("T")  # Comparator Type
R = TypeVar("R")  # Result Type


@dataclass
class Instance(ABC, Generic[T, R]):
    log_instance: LogInstance

    @property
    def path(self) -> Path:
        """Base path for outputs saved by this instance"""
        return (
            OUTPUT_BASE_PATH
            / self.technique_name  # WARN: ADDED THIS SO BOOTSTRAP AND PERMUTATION TEST WOULDNT OVERWRITE EACHOTHER
            / self.log_instance.path.relative_to(LOGS_BASE_PATH).parent
            / self.log_instance.identifier
        )

    # @property
    # def path(self) -> Path:
    #     return OUTPUT_BASE_PATH / self.log_instance.identifier

    @property
    def pickle_path(self) -> Path:
        """Save path for the result pickle"""
        return self.path / "result.pkl"

    @property
    @abstractmethod
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
            "log_name": self.log_instance.path.name.split(".")[0],
            "noise_level": self.log_instance.noise_level,
            "has_diff": log_has_drift,
            "pval": pval,
            "logs_emd": logs_emd,
            "detection": detection,
            "correct": is_correct,
            "classification_class": get_classification_class(detection, log_has_drift),
            "duration": duration_seconds,
            "pickle_path": self.pickle_path.as_posix(),
        }

    # TODO: Isnt this identical for both techniques
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


def get_all_test_instances() -> list[Instance]:
    all_log_instances = get_all_log_instances()

    permutation_test_instances: list[Instance] = [
        PermutationTestInstance(log_instance) for log_instance in all_log_instances
    ]
    bootstrap_instances: list[Instance] = [
        BootstrapTestInstance(log_instance) for log_instance in all_log_instances
    ]

    return permutation_test_instances + bootstrap_instances


def run_instance(instance: Instance) -> dict[str, Any]:
    return instance.run_and_save_results(verbose=True)


def run():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--cores",
        type=int,
        help="The number of cores to use for multiprocessing.",
        required=True,
    )
    args = parser.parse_args()

    all_instances = get_all_test_instances()
    all_instances = [
        instance for instance in all_instances if not instance.pickle_path.exists()
    ]

    start_time = default_timer()
    with WorkerPool(args.cores) as p:
        results = p.map(run_instance, all_instances, progress_bar=True)
    df = pd.DataFrame(results)

    csv_path = OUTPUT_BASE_PATH / "summary.csv"
    if csv_path.exists():
        print(f"{csv_path} already exists! Appending new data to old dataframe")
        old_df = pd.read_csv(csv_path)
        df = pd.concat([old_df, df])
    df.to_csv(OUTPUT_BASE_PATH / "summary.csv", index=False)

    print(f"Elapsed Time: {default_timer() - start_time}")


if __name__ == "__main__":
    run()
