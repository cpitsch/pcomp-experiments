"""
This module contains helper functions for the analysis
"""

import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pcomp.binning import KMeans_Binner
from pcomp.emd.comparators.permutation_test.levenshtein.numerical_column_levenshtein import (
    NumericalColumn_Levenshtein_PermutationComparator,
)
from pcomp.utils import split_log_cases
from pcomp.utils.constants import DEFAULT_NAME_KEY, DEFAULT_TRACEID_KEY
from pm4py import read_xes
from scipy.stats import mannwhitneyu

NUM_MP_CORES = 6


SEED = 1337

PICKLE_ROOT = Path("analysis_results")
LOW_HEMO_PICKLE_ROOT = PICKLE_ROOT / "low_hemoglobin"
HIGH_HEMO_PICKLE_ROOT = PICKLE_ROOT / "high_hemoglobin"


def split_log_categorical(
    log: pd.DataFrame, split_column: str, class_1_value: Any, class_2_value: Any
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an event log based on a categorical attribute

    Args:
        log (pd.DataFrame): The event log.
        split_column (str): The column to split on.
        class_1_value (Any): The class for the first event log output.
        class_2_value (Any): The class for the second event log output.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The event logs
    """
    log_1: pd.DataFrame = log[log[split_column] == class_1_value]  # type: ignore
    log_2: pd.DataFrame = log[log[split_column] == class_2_value]  # type: ignore
    return log_1, log_2


def split_log_continuous(
    log: pd.DataFrame, split_column: str, under: Any, above: Any | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an event log based on a continuous variable

    Args:
        log (pd.DataFrame): The event log.
        split_column (str): The column to split on.
        under (Any): The threshold to use for the first event log (retains with <).
        above (Any | None, optional): The threshold to use for the sefcond event log
            (retains with >=). Defaults to `under`.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The event logs
    """
    if above is None:
        above = under
    log_1: pd.DataFrame = log[log[split_column] < under]  # type: ignore
    log_2: pd.DataFrame = log[log[split_column] >= above]  # type: ignore
    return log_1, log_2


def split_log_random(
    log: pd.DataFrame, seed: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an event log into two random halves

    Args:
        log (pd.DataFrame): The event logs to split.
        seed (int | None, optional): The seed to use for sampling. Defaults to None (no seed).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The event logs
    """
    log_1, log_2 = split_log_cases(log, 0.5, seed=seed)
    return log_1, log_2


def run_a_split(
    log_1: pd.DataFrame,
    log_2: pd.DataFrame,
) -> NumericalColumn_Levenshtein_PermutationComparator:
    comparator = NumericalColumn_Levenshtein_PermutationComparator(
        log_1,
        log_2,
        column="comparison_value",
        distribution_size=10_000,
        multiprocess_cores=NUM_MP_CORES,
        weighted_time_cost=True,
        binner_factory=KMeans_Binner,
        binner_args={
            "k": 3,
        },
        seed=SEED,
    )
    _ = comparator.compare()

    print("P-Value:", comparator.pval)

    comparator.plot_result().show()

    return comparator


def get_event_log() -> pd.DataFrame:
    return read_xes("gi_bleeding.xes.gz", variant="rustxes")  # type: ignore


def run_categorical_split(
    split_column: str,
    class_1_value: Any,
    class_2_value: Any,
    log: pd.DataFrame | None = None,
) -> NumericalColumn_Levenshtein_PermutationComparator:
    if log is None:
        log = get_event_log()
    log_1, log_2 = split_log_categorical(
        log, split_column, class_1_value, class_2_value
    )
    return run_a_split(log_1, log_2)


def run_continuous_split(
    split_column: str,
    under: Any,
    over: Any | None = None,
    log: pd.DataFrame | None = None,
) -> NumericalColumn_Levenshtein_PermutationComparator:
    if log is None:
        log = get_event_log()
    log_1, log_2 = split_log_continuous(log, split_column, under, over)
    return run_a_split(log_1, log_2)


def run_random_split(
    log: pd.DataFrame | None = None,
) -> NumericalColumn_Levenshtein_PermutationComparator:
    if log is None:
        log = get_event_log()
    log_1, log_2 = split_log_random(log, seed=SEED)
    return run_a_split(log_1, log_2)


def save_pickle(obj, filename, root: Path = PICKLE_ROOT):
    if not root.exists():
        root.mkdir(exist_ok=True, parents=True)
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def save_plot(
    comparator: NumericalColumn_Levenshtein_PermutationComparator, root_path: Path
):
    fig = comparator.plot_result()
    if not root_path.exists():
        root_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{root_path}/plot.pdf", bbox_inches="tight")
    fig.savefig(f"{root_path}/plot.png", bbox_inches="tight")
    fig.savefig(f"{root_path}/plot.svg", bbox_inches="tight")
    plt.close(fig)


### Explain Disparities
@dataclass
class DisparityExplainer:
    log_1_name: str
    log_2_name: str

    mean_trace_length_1: float
    mean_trace_length_2: float
    trace_length_pval: float

    mean_num_transfusions_1: float
    mean_num_transfusions_2: float
    num_transfusions_pval: float

    mean_transfusion_amount_1: float
    mean_transfusion_amount_2: float
    transfusion_amount_pval: float

    mean_hemoglobin_level_1: float
    mean_hemoglobin_level_2: float
    hemoglobin_level_pval: float

    mean_num_hemoglobin_measurements_1: float
    mean_num_hemoglobin_measurements_2: float
    num_hemoglobin_measurements_pval: float

    mortality_rate_1: float
    mortality_rate_2: float

    mean_age_1: float
    mean_age_2: float
    age_pval: float

    @classmethod
    def from_logs(
        cls,
        log_1: pd.DataFrame,
        log_2: pd.DataFrame,
        log_1_name: str = "Log 1",
        log_2_name: str = "Log 2",
    ) -> "DisparityExplainer":
        return DisparityExplainer(
            log_1_name,
            log_2_name,
            get_mean_trace_length(log_1),
            get_mean_trace_length(log_2),
            mannwhitneyu(
                _get_trace_length_series(log_1), _get_trace_length_series(log_2)
            ).pvalue,
            get_mean_num_transfusions(log_1),
            get_mean_num_transfusions(log_2),
            mannwhitneyu(
                _get_num_transfusions_series(log_1), _get_num_transfusions_series(log_2)
            ).pvalue,
            get_mean_transfusion_amount(log_1),
            get_mean_transfusion_amount(log_2),
            mannwhitneyu(
                _get_transfusion_amount_series(log_1),
                _get_transfusion_amount_series(log_2),
            ).pvalue,
            get_mean_hemoglobin_level(log_1),
            get_mean_hemoglobin_level(log_2),
            mannwhitneyu(
                _get_hemoglobin_level_series(log_1), _get_hemoglobin_level_series(log_2)
            ).pvalue,
            get_mean_num_hemoglobin_measurements(log_1),
            get_mean_num_hemoglobin_measurements(log_2),
            mannwhitneyu(
                _get_num_hemoglobin_measurements_series(log_1),
                _get_num_hemoglobin_measurements_series(log_2),
            ).pvalue,
            get_mortality_rate(log_1),
            get_mortality_rate(log_2),
            get_mean_age(log_1),
            get_mean_age(log_2),
            mannwhitneyu(_get_age_series(log_1), _get_age_series(log_2)).pvalue,
        )

    def print(self):
        print("Mean Trace Length")
        print(f"\t{self.log_1_name}: {self.mean_trace_length_1}")
        print(f"\t{self.log_2_name}: {self.mean_trace_length_2}")
        print(f"p={self.trace_length_pval:.3f}")

        # Transfusion Counts
        print("Mean number of Transfusions")
        print(f"\t{self.log_1_name}: {self.mean_num_transfusions_1}")
        print(f"\t{self.log_2_name}: {self.mean_num_transfusions_2}")
        print(f"p={self.num_transfusions_pval:.3f}")

        # Mean hemoglobin value
        print("Mean Hemoglobin Value")
        print(f"\t{self.log_1_name}: {self.mean_hemoglobin_level_1}")
        print(f"\t{self.log_2_name}: {self.mean_hemoglobin_level_2}")
        print(f"p={self.hemoglobin_level_pval:.3f}")

        print("Mean number of Hemoglobin Measurements")
        print(f"\t{self.log_1_name}: {self.mean_num_hemoglobin_measurements_1}")
        print(f"\t{self.log_2_name}: {self.mean_num_hemoglobin_measurements_2}")
        print(f"p={self.num_hemoglobin_measurements_pval:.3f}")

        print("Mortality Rate")
        print(f"\t{self.log_1_name}: {self.mortality_rate_1*100:.2f}%")
        print(f"\t{self.log_2_name}: {self.mortality_rate_2*100:.2f}%")

        print("Mean Age")
        print(f"\t{self.log_1_name}: {self.mean_age_1:.2f}")
        print(f"\t{self.log_2_name}: {self.mean_age_2:.2f}")
        print(f"p={self.age_pval:.3f}")

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Metric": "Mean Trace Length",
                    self.log_1_name: self.mean_trace_length_1,
                    self.log_2_name: self.mean_trace_length_2,
                    "Percent Change": f"{_get_percent_change(self.mean_trace_length_1, self.mean_trace_length_2):.2f}%",
                    "P-Value": self.trace_length_pval,
                },
                {
                    "Metric": "Mean Number of Transfusions",
                    self.log_1_name: self.mean_num_transfusions_1,
                    self.log_2_name: self.mean_num_transfusions_2,
                    "Percent Change": f"{_get_percent_change(self.mean_num_transfusions_1, self.mean_num_transfusions_2):.2f}%",
                    "P-Value": self.num_transfusions_pval,
                },
                {
                    "Metric": "Mean Transfusion Amount",
                    self.log_1_name: self.mean_transfusion_amount_1,
                    self.log_2_name: self.mean_transfusion_amount_2,
                    "Percent Change": f"{_get_percent_change(self.mean_transfusion_amount_1, self.mean_transfusion_amount_2):.2f}%",
                    "P-Value": self.transfusion_amount_pval,
                },
                {
                    "Metric": "Mean Hemoglobin Level",
                    self.log_1_name: self.mean_hemoglobin_level_1,
                    self.log_2_name: self.mean_hemoglobin_level_2,
                    "Percent Change": f"{_get_percent_change(self.mean_hemoglobin_level_1, self.mean_hemoglobin_level_2):.2f}%",
                    "P-Value": self.hemoglobin_level_pval,
                },
                {
                    "Metric": "Mean number of Hemoglobin Measurements",
                    self.log_1_name: self.mean_num_hemoglobin_measurements_1,
                    self.log_2_name: self.mean_num_hemoglobin_measurements_2,
                    "Percent Change": f"{_get_percent_change(self.mean_num_hemoglobin_measurements_1, self.mean_num_hemoglobin_measurements_2,):.2f}%",
                    "P-Value": self.num_hemoglobin_measurements_pval,
                },
                {
                    "Metric": "Mortality Rate",
                    self.log_1_name: f"{self.mortality_rate_1*100:.2f}%",
                    self.log_2_name: f"{self.mortality_rate_2*100:.2f}%",
                    "Percent Change": f"{_get_percent_change(self.mortality_rate_1, self.mortality_rate_2):.2f}%",
                    "P-Value": np.nan,
                },
                {
                    "Metric": "Mean Age",
                    self.log_1_name: f"{self.mean_age_1:.2f}",
                    self.log_2_name: f"{self.mean_age_2:.2f}",
                    "Percent Change": f"{_get_percent_change(self.mean_age_1, self.mean_age_2):.2f}%",
                    "P-Value": self.age_pval,
                },
            ]
        )


def _get_percent_change(before: float, after: float) -> float:
    return (100 * after / before) - 100


def _get_num_transfusions_series(log: pd.DataFrame) -> pd.Series:
    return log.groupby(DEFAULT_TRACEID_KEY).apply(
        lambda group_df: (group_df["event_type"] == "blood_transfusion").sum()
    )  # type: ignore


def get_num_transfusions(log: pd.DataFrame) -> list[float]:
    return _get_num_transfusions_series(log).tolist()


def get_mean_num_transfusions(log: pd.DataFrame) -> float:
    return _get_num_transfusions_series(log).mean()  # type: ignore


def _get_transfusion_amount_series(log: pd.DataFrame) -> pd.Series:
    return log[log["event_type"] == "blood_transfusion"]["amount"]  # type: ignore


def get_mean_transfusion_amount(log: pd.DataFrame) -> float:
    return _get_transfusion_amount_series(log).mean()  # type: ignore


def _get_hemoglobin_level_series(log: pd.DataFrame) -> float:
    return log[~log["hemoglobin_value"].isna()]["hemoglobin_value"]  # type: ignore


def get_mean_hemoglobin_level(log: pd.DataFrame) -> float:
    return _get_hemoglobin_level_series(log).mean()  # type: ignore


def _get_num_hemoglobin_measurements_series(log: pd.DataFrame) -> pd.Series:
    return log[~log["hemoglobin_value"].isna()].groupby(DEFAULT_TRACEID_KEY).size()  # type: ignore


def get_mean_num_hemoglobin_measurements(log: pd.DataFrame) -> float:
    return _get_num_hemoglobin_measurements_series(log).mean()  # type: ignore


def _get_trace_length_series(log: pd.DataFrame) -> pd.Series:
    return log.groupby(DEFAULT_TRACEID_KEY).size()  # type: ignore


def get_mean_trace_length(log: pd.DataFrame) -> float:
    return _get_trace_length_series(log).mean()  # type: ignore


def get_mortality_rate(log: pd.DataFrame) -> float:
    num_deaths = (
        log.groupby(DEFAULT_TRACEID_KEY)
        .apply(
            lambda group: (group[DEFAULT_NAME_KEY] == "DEATH").any(),
            include_groups=False,
        )
        .sum()
    )
    return num_deaths / log[DEFAULT_TRACEID_KEY].nunique()


def _get_age_series(log: pd.DataFrame) -> pd.Series:
    return log.groupby(DEFAULT_TRACEID_KEY).apply(lambda group: group.iloc[0]["anchor_age"], include_groups=False)  # type: ignore


def get_mean_age(log: pd.DataFrame) -> float:
    return _get_age_series(log).mean()  # type: ignore


def explain_disparities(
    comparator: NumericalColumn_Levenshtein_PermutationComparator,
    log_1_name: str = "Log 1",
    log_2_name: str = "Log 2",
):
    # transfusion_counts_1 = comparator.log_1[DEFAULT_NAME_KEY].value_counts(
    #     normalize=True
    # )
    # transfusion_counts_2 = comparator.log_2[DEFAULT_NAME_KEY].value_counts(
    #     normalize=True
    # )
    #
    # # Distribution of transfusion types
    # print("Distribution of Transfusion Types")
    # # print(f"\t{log_1_name}:\n", transfusion_counts_1, "\n")
    # # print(f"\t{log_2_name}:\n", transfusion_counts_2, "\n")
    # # print(f"\tDifference:\n", transfusion_counts_1 - transfusion_counts_2)
    # print(transfusion_counts_1 - transfusion_counts_2, "\n")
    #
    # print("Mean Trace Length")
    # print(f"\t{log_1_name}: {get_mean_trace_length(comparator.log_1)}")
    # print(f"\t{log_2_name}: {get_mean_trace_length(comparator.log_2)}")
    #
    # # Transfusion Counts
    # print("Mean number of Transfusions")
    # print(f"\t{log_1_name}: {get_mean_num_transfusions(comparator.log_1)}")
    # print(f"\t{log_2_name}: {get_mean_num_transfusions(comparator.log_2)}")
    #
    # # Mean hemoglobin value
    # print("Mean Hemoglobin Value")
    # print(f"\t{log_1_name}: {get_mean_hemoglobin_level(comparator.log_1)}")
    # print(f"\t{log_2_name}: {get_mean_hemoglobin_level(comparator.log_2)}")
    #
    # print("Mean number of Hemoglobin Measurements")
    # print(f"\t{log_1_name}: {get_mean_num_hemoglobin_measurements(comparator.log_1)}")
    # print(f"\t{log_2_name}: {get_mean_num_hemoglobin_measurements(comparator.log_2)}")
    #
    # print("Mortality Rate")
    # print(f"\t{log_1_name}: {get_mortality_rate(comparator.log_1)*100:.2f}%")
    # print(f"\t{log_2_name}: {get_mortality_rate(comparator.log_2)*100:.2f}%")
    #
    # print("Mean Age")
    # print(f"\t{log_1_name}: {get_mean_age(comparator.log_1):.2f}")
    # print(f"\t{log_2_name}: {get_mean_age(comparator.log_2):.2f}")

    return DisparityExplainer.from_logs(
        comparator.log_1, comparator.log_2, log_1_name, log_2_name
    ).as_dataframe()


def explain_disparities_df(
    comparator: NumericalColumn_Levenshtein_PermutationComparator,
    log_1_name: str = "Log 1",
    log_2_name: str = "Log 2",
) -> pd.DataFrame:
    return DisparityExplainer.from_logs(
        comparator.log_1, comparator.log_2, log_1_name, log_2_name
    ).as_dataframe()


def investigate_common_variants(
    comparator: NumericalColumn_Levenshtein_PermutationComparator,
    log_1_name: str = "Log 1",
    log_2_name: str = "Log 2",
    top_n: int = 3,
):
    c1 = Counter(comparator.behavior_1)
    c2 = Counter(comparator.behavior_2)

    top_1 = c1.most_common(top_n)
    top_2 = c2.most_common(top_n)

    print(log_1_name)
    for trace, count in top_1:
        print(f"\t({100*count/c1.total():.2f}): {trace}")
    print(log_2_name)
    for trace, count in top_2:
        print(f"\t({100*count/c2.total():.2f}): {trace}")


def split_log_hemoglobin() -> tuple[pd.DataFrame, pd.DataFrame]:
    _log = get_event_log()
    low_hemo_caseids = (
        _log[_log["hemoglobin_value"] < 7]["case:concept:name"].unique().tolist()  # type: ignore
    )
    return (
        _log[_log["case:concept:name"].isin(low_hemo_caseids)],  # Low
        _log[~_log["case:concept:name"].isin(low_hemo_caseids)],  # Not Low
    )  # type: ignore


def get_low_hemoglobin_log() -> pd.DataFrame:
    return split_log_hemoglobin()[0]


def get_high_hemoglobin_log() -> pd.DataFrame:
    return split_log_hemoglobin()[1]
