"""
This module contains helper functions for the analysis
"""

import pickle
from collections import Counter
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Literal
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pcomp.binning import KMeans_Binner
from pcomp.emd.comparators.permutation_test.levenshtein.numerical_column_levenshtein import (
    NumericalColumn_Levenshtein_PermutationComparator,
)
from pcomp.emd.emd import BinnedServiceTimeTrace
from pcomp.utils import split_log_cases
from pcomp.utils.constants import (
    DEFAULT_NAME_KEY,
    DEFAULT_TIMESTAMP_KEY,
    DEFAULT_TRACEID_KEY,
)
from pm4py import read_xes
from scipy.stats import chi2_contingency, ks_2samp

NUM_MP_CORES = 6


SEED = 1337

OUTPUT_ROOT = Path("analysis_results")
HemoglobinGroup = Literal["low", "normal"]


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

    print("Cohort 1 Size:", log_1[DEFAULT_TRACEID_KEY].nunique())
    print("Cohort 2 Size:", log_2[DEFAULT_TRACEID_KEY].nunique())
    print("P-Value:", comparator.pval)

    comparator.plot_result()

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


def save_pickle(obj: Any, category: str, hemoglobin_group: HemoglobinGroup):
    path = OUTPUT_ROOT / category / f"{hemoglobin_group}_hemoglobin"
    if not path.exists():
        path.mkdir(parents=True)
    with open(path / "result.pkl", "wb") as f:
        pickle.dump(obj, f)


def save_low_hemoglobin_pickle(obj: Any, category: str):
    save_pickle(obj, category, "low")


def save_high_hemoglobin_pickle(obj: Any, category: str):
    save_pickle(obj, category, "normal")


def save_plot(
    comparator: NumericalColumn_Levenshtein_PermutationComparator,
    category: str,
    hemoglobin_group: HemoglobinGroup,
):
    path = OUTPUT_ROOT / category / f"{hemoglobin_group}_hemoglobin"
    fig = comparator.plot_result()
    set_font_sizes(fig.gca())
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path}/plot.pdf", bbox_inches="tight")
    fig.savefig(f"{path}/plot.png", bbox_inches="tight")
    fig.savefig(f"{path}/plot.svg", bbox_inches="tight")
    plt.close(fig)


def set_font_sizes(ax: Axes):
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.get_xaxis().get_label().set_fontsize(14)
    ax.get_yaxis().get_label().set_fontsize(14)


def save_low_hemoglobin_plot(
    comparator: NumericalColumn_Levenshtein_PermutationComparator, category: str
):
    save_plot(comparator, category, "low")


def save_high_hemoglobin_plot(
    comparator: NumericalColumn_Levenshtein_PermutationComparator, category: str
):
    save_plot(comparator, category, "normal")


### Explain Disparities
@dataclass
class DisparityExplainer:
    log_1_name: str
    log_2_name: str

    trace_lengths_1: pd.Series  # [int]
    trace_lengths_2: pd.Series  # [int]
    trace_length_pval: float

    num_transfusions_1: pd.Series  # [int]
    num_transfusions_2: pd.Series  # [int]
    num_transfusions_pval: float

    transfusion_amounts_1: pd.Series  # [float]
    transfusion_amounts_2: pd.Series  # [float]
    transfusion_amount_pval: float

    hemoglobin_levels_1: pd.Series  # [float]
    hemoglobin_levels_2: pd.Series  # [float]
    hemoglobin_level_pval: float

    num_hemoglobin_measurements_1: pd.Series  # [int]
    num_hemoglobin_measurements_2: pd.Series  # [int]
    num_hemoglobin_measurements_pval: float

    mortality_rate_1: float
    mortality_rate_2: float
    mortality_pvalue: float

    m_times_since_last_measurement_1: pd.Series
    t_times_since_last_measurement_1: pd.Series
    m_times_since_last_measurement_2: pd.Series
    t_times_since_last_measurement_2: pd.Series
    m_time_since_last_measurement_pvalue: float
    t_time_since_last_measurement_pvalue: float

    # hemoglobin_reaction_times_1: list[float]
    # hemoglobin_reaction_times_2: list[float]
    # reaction_times_pval: float
    #
    # low_hemoglobin_reaction_times_1: list[float]
    # low_hemoglobin_reaction_times_2: list[float]
    # low_hemoglobin_reaction_times_pval: float

    ages_1: pd.Series  # [int]
    ages_2: pd.Series  # [int]
    age_pval: float

    @classmethod
    def from_logs(
        cls,
        log_1: pd.DataFrame,
        log_2: pd.DataFrame,
        log_1_name: str = "Log 1",
        log_2_name: str = "Log 2",
    ) -> "DisparityExplainer":
        # reaction_times_1 = get_low_hemoglobin_reaction_times(log_1, None)
        # reaction_times_2 = get_low_hemoglobin_reaction_times(log_2, None)
        #
        # low_reaction_times_1 = get_low_hemoglobin_reaction_times(log_1, 7.0, True)
        # low_reaction_times_2 = get_low_hemoglobin_reaction_times(log_2, 7.0, True)
        # # For high hemoglobin cases, there are, of course, no low hemoglobin measurements
        # low_reaction_times_empty = (
        #     len(low_reaction_times_1) == 0 or len(low_reaction_times_2) == 0
        # )

        return DisparityExplainer(
            log_1_name,
            log_2_name,
            _get_trace_length_series(log_1),
            _get_trace_length_series(log_2),
            ks_2samp(
                _get_trace_length_series(log_1), _get_trace_length_series(log_2)
            ).pvalue,  # type: ignore
            _get_num_transfusions_series(log_1),
            _get_num_transfusions_series(log_2),
            ks_2samp(
                _get_num_transfusions_series(log_1), _get_num_transfusions_series(log_2)
            ).pvalue,  # type: ignore
            _get_transfusion_amount_series(log_1),
            _get_transfusion_amount_series(log_2),
            ks_2samp(
                _get_transfusion_amount_series(log_1),
                _get_transfusion_amount_series(log_2),
            ).pvalue,  # type: ignore
            _get_hemoglobin_level_series(log_1),
            _get_hemoglobin_level_series(log_2),
            ks_2samp(
                _get_hemoglobin_level_series(log_1), _get_hemoglobin_level_series(log_2)
            ).pvalue,  # type: ignore
            _get_num_hemoglobin_measurements_series(log_1),
            _get_num_hemoglobin_measurements_series(log_2),
            ks_2samp(
                _get_num_hemoglobin_measurements_series(log_1),
                _get_num_hemoglobin_measurements_series(log_2),
            ).pvalue,  # type: ignore
            get_mortality_rate(log_1),
            get_mortality_rate(log_2),
            get_mortality_pvalue(log_1, log_2),
            # m_times_since_last_measurement_1 := log_1.loc[
            #     log_1["event_type"] == "hemoglobin_measurement",
            #     "time_since_last_measurement",
            # ],
            # t_times_since_last_measurement_1 := log_1.loc[
            #     log_1["event_type"] == "blood_transfusion",
            #     "time_since_last_measurement",
            # ],
            # m_times_since_last_measurement_2 := log_2.loc[
            #     log_2["event_type"] == "hemoglobin_measurement",
            #     "time_since_last_measurement",
            # ],
            # t_times_since_last_measurement_2 := log_2.loc[
            #     log_2["event_type"] == "blood_transfusion",
            #     "time_since_last_measurement",
            # ],
            m_times_since_last_measurement_1 := log_1.loc[
                log_1["event_type"] == "hemoglobin_measurement", :
            ]
            .groupby("case:concept:name")["time_since_last_measurement"]
            .mean(),
            t_times_since_last_measurement_1 := log_1.loc[
                log_1["event_type"] == "blood_transfusion", :
            ]
            .groupby("case:concept:name")["time_since_last_measurement"]
            .mean(),
            m_times_since_last_measurement_2 := log_2.loc[
                log_2["event_type"] == "hemoglobin_measurement", :
            ]
            .groupby("case:concept:name")["time_since_last_measurement"]
            .mean(),
            t_times_since_last_measurement_2 := log_2.loc[
                log_2["event_type"] == "blood_transfusion", :
            ]
            .groupby("case:concept:name")["time_since_last_measurement"]
            .mean(),
            ks_2samp(
                m_times_since_last_measurement_1, m_times_since_last_measurement_2
            ).pvalue,  # type: ignore
            ks_2samp(
                t_times_since_last_measurement_1, t_times_since_last_measurement_2
            ).pvalue,  # type: ignore
            # reaction_times_1,
            # reaction_times_2,
            # ks_2samp(reaction_times_1, reaction_times_2).pvalue,  # type: ignore
            # low_reaction_times_1,
            # low_reaction_times_2,
            # (
            #     float("nan")
            #     if low_reaction_times_empty
            #     else ks_2samp(low_reaction_times_1, low_reaction_times_2).pvalue  # type: ignore
            # ),
            _get_age_series(log_1),
            _get_age_series(log_2),
            ks_2samp(_get_age_series(log_1), _get_age_series(log_2)).pvalue,  # type: ignore
        )

    def print(self):
        warn(
            "The `print` function for DisparityExplainer is outdated. Print self.as_dataframe() instead."
        )
        print("Mean/Median Trace Length")
        print(
            f"\t{self.log_1_name}: {self.trace_lengths_1.mean()}/{self.trace_lengths_1.median()}"
        )
        print(
            f"\t{self.log_2_name}: {self.trace_lengths_2.mean()}/{self.trace_lengths_2.median()}"
        )
        print(f"p={self.trace_length_pval:.3f}")

        # Transfusion Counts
        print("Mean/Median number of Transfusions")
        print(
            f"\t{self.log_1_name}: {self.num_transfusions_1.mean()}/{self.num_transfusions_1.median()}"
        )
        print(
            f"\t{self.log_2_name}: {self.num_transfusions_2.mean()}/{self.num_transfusions_2.median()}"
        )
        print(f"p={self.num_transfusions_pval:.3f}")

        # Mean hemoglobin value
        print("Mean Hemoglobin Value")
        print(
            f"\t{self.log_1_name}: {self.hemoglobin_levels_1.mean()}/{self.hemoglobin_levels_1.median()}"
        )
        print(
            f"\t{self.log_2_name}: {self.hemoglobin_levels_2.mean()}/{self.hemoglobin_levels_2.median()}"
        )
        print(f"p={self.hemoglobin_level_pval:.3f}")

        print("Mean/Median number of Hemoglobin Measurements")
        print(
            f"\t{self.log_1_name}: {self.num_hemoglobin_measurements_1.mean()}/{self.num_hemoglobin_measurements_1.median()}"
        )
        print(
            f"\t{self.log_2_name}: {self.num_hemoglobin_measurements_2.mean()}/{self.num_hemoglobin_measurements_2.median()}"
        )
        print(f"p={self.num_hemoglobin_measurements_pval:.3f}")

        print("Mortality Rate")
        print(f"\t{self.log_1_name}: {self.mortality_rate_1 * 100:.2f}%")
        print(f"\t{self.log_2_name}: {self.mortality_rate_2 * 100:.2f}%")

        print("Mean Age")
        print(f"\t{self.log_1_name}: {self.ages_1.mean():.2f}")
        print(f"\t{self.log_2_name}: {self.ages_2.mean():.2f}")
        print(f"p={self.age_pval:.3f}")

    def display_df(self) -> pd.DataFrame:
        return self.as_dataframe(fillna="")

    def as_dataframe(self, fillna: Any | None = None) -> pd.DataFrame:
        # Excluding bonferroni correction on some things like age, hemoglobin_level, transfusion amount
        # Because, we only include the information that is extracted by the approach, to validate the results. So these measures are not considered
        # Also, because they are a) Not that interesting (amount, level) or b) Are guaranteed to be significant in some comparisons (hemoglobin level different between male/female, etc.)
        bonferroni_correction_divisor = 5
        significance_level = 0.05
        effective_significance_level = (
            significance_level / bonferroni_correction_divisor
        )

        df = pd.DataFrame(
            [
                {
                    "Metric": "Trace Length",
                    f"{self.log_1_name} Mean": self.trace_lengths_1.mean(),
                    f"{self.log_2_name} Mean": self.trace_lengths_2.mean(),
                    "%Δ Mean": f"{_get_percent_change(self.trace_lengths_1.mean(), self.trace_lengths_2.mean()):.2f}%",  # type: ignore
                    f"{self.log_1_name} Median": self.trace_lengths_1.median(),
                    f"{self.log_2_name} Median": self.trace_lengths_2.median(),
                    "%Δ Median": f"{_get_percent_change(self.trace_lengths_1.median(), self.trace_lengths_2.median()):.2f}%",  # type: ignore
                    "P-Value": self.trace_length_pval,
                    # Instead, represented by the number of transfusions and measurements
                    # f"Significant (Bonferroni {bonferroni_correction_divisor})": self.trace_length_pval
                    # <= effective_significance_level,
                },
                {
                    "Metric": "Number of Transfusions",
                    f"{self.log_1_name} Mean": self.num_transfusions_1.mean(),
                    f"{self.log_2_name} Mean": self.num_transfusions_2.mean(),
                    "%Δ Mean": f"{_get_percent_change(self.num_transfusions_1.mean(), self.num_transfusions_2.mean()):.2f}%",  # type: ignore
                    f"{self.log_1_name} Median": self.num_transfusions_1.median(),
                    f"{self.log_2_name} Median": self.num_transfusions_2.median(),
                    "%Δ Median": f"{_get_percent_change(self.num_transfusions_1.median(), self.num_transfusions_2.median()):.2f}%",  # type: ignore
                    "P-Value": self.num_transfusions_pval,
                    f"Significant (Bonferroni {bonferroni_correction_divisor})": self.num_transfusions_pval
                    <= effective_significance_level,
                },
                {
                    "Metric": "Transfusion Amount",
                    f"{self.log_1_name} Mean": self.transfusion_amounts_1.mean(),
                    f"{self.log_2_name} Mean": self.transfusion_amounts_2.mean(),
                    "%Δ Mean": f"{_get_percent_change(self.transfusion_amounts_1.mean(), self.transfusion_amounts_2.mean()):.2f}%",  # type: ignore
                    f"{self.log_1_name} Median": self.transfusion_amounts_1.median(),
                    f"{self.log_2_name} Median": self.transfusion_amounts_2.median(),
                    "%Δ Median": f"{_get_percent_change(self.transfusion_amounts_1.median(), self.transfusion_amounts_2.median()):.2f}%",  # type: ignore
                    "P-Value": self.transfusion_amount_pval,
                },
                {
                    "Metric": "Hemoglobin Level",
                    f"{self.log_1_name} Mean": self.hemoglobin_levels_1.mean(),
                    f"{self.log_2_name} Mean": self.hemoglobin_levels_2.mean(),
                    "%Δ Mean": f"{_get_percent_change(self.hemoglobin_levels_1.mean(), self.hemoglobin_levels_2.mean()):.2f}%",  # type: ignore
                    f"{self.log_1_name} Median": self.hemoglobin_levels_1.median(),
                    f"{self.log_2_name} Median": self.hemoglobin_levels_2.median(),
                    "%Δ Median": f"{_get_percent_change(self.hemoglobin_levels_1.median(), self.hemoglobin_levels_2.median()):.2f}%",  # type: ignore
                    "P-Value": self.hemoglobin_level_pval,
                },
                {
                    "Metric": "Number of Hemoglobin Measurements",
                    f"{self.log_1_name} Mean": self.num_hemoglobin_measurements_1.mean(),
                    f"{self.log_2_name} Mean": self.num_hemoglobin_measurements_2.mean(),
                    "%Δ Mean": f"{_get_percent_change(self.num_hemoglobin_measurements_1.mean(), self.num_hemoglobin_measurements_2.mean()):.2f}%",  # type: ignore
                    f"{self.log_1_name} Median": self.num_hemoglobin_measurements_1.median(),
                    f"{self.log_2_name} Median": self.num_hemoglobin_measurements_2.median(),
                    "%Δ Median": f"{_get_percent_change(self.num_hemoglobin_measurements_1.median(), self.num_hemoglobin_measurements_2.median()):.2f}%",  # type: ignore
                    "P-Value": self.num_hemoglobin_measurements_pval,
                    f"Significant (Bonferroni {bonferroni_correction_divisor})": self.num_hemoglobin_measurements_pval
                    <= effective_significance_level,
                },
                {
                    "Metric": "Age",
                    f"{self.log_1_name} Mean": f"{self.ages_1.mean():.2f}",
                    f"{self.log_2_name} Mean": f"{self.ages_2.mean():.2f}",
                    "%Δ Mean": f"{_get_percent_change(self.ages_1.mean(), self.ages_2.mean()):.2f}%",  # type: ignore
                    f"{self.log_1_name} Median": f"{self.ages_1.median():.2f}",
                    f"{self.log_2_name} Median": f"{self.ages_2.median():.2f}",
                    "%Δ Median": f"{_get_percent_change(self.ages_1.median(), self.ages_2.median()):.2f}%",  # type: ignore
                    "P-Value": self.age_pval,
                },
                {
                    "Metric": "Time Since Last Measurement (Measurements)",
                    f"{self.log_1_name} Mean": f"{self.m_times_since_last_measurement_1.mean():.2f}",
                    f"{self.log_2_name} Mean": f"{self.m_times_since_last_measurement_2.mean():.2f}",
                    "%Δ Mean": f"{_get_percent_change(self.m_times_since_last_measurement_1.mean(), self.m_times_since_last_measurement_2.mean()):.2f}%",  # type: ignore
                    f"{self.log_1_name} Median": f"{self.m_times_since_last_measurement_1.median():.2f}",
                    f"{self.log_2_name} Median": f"{self.m_times_since_last_measurement_2.median():.2f}",
                    "%Δ Median": f"{_get_percent_change(self.m_times_since_last_measurement_1.median(), self.m_times_since_last_measurement_2.median()):.2f}%",  # type: ignore
                    "P-Value": self.m_time_since_last_measurement_pvalue,
                    f"Significant (Bonferroni {bonferroni_correction_divisor})": self.m_time_since_last_measurement_pvalue
                    <= effective_significance_level,
                },
                {
                    "Metric": "Time Since Last Measurement (Transfusions)",
                    f"{self.log_1_name} Mean": f"{self.t_times_since_last_measurement_1.mean():.2f}",
                    f"{self.log_2_name} Mean": f"{self.t_times_since_last_measurement_2.mean():.2f}",
                    "%Δ Mean": f"{_get_percent_change(self.t_times_since_last_measurement_1.mean(), self.t_times_since_last_measurement_2.mean()):.2f}%",  # type: ignore
                    f"{self.log_1_name} Median": f"{self.t_times_since_last_measurement_1.median():.2f}",
                    f"{self.log_2_name} Median": f"{self.t_times_since_last_measurement_2.median():.2f}",
                    "%Δ Median": f"{_get_percent_change(self.t_times_since_last_measurement_1.median(), self.t_times_since_last_measurement_2.median()):.2f}%",  # type: ignore
                    "P-Value": self.t_time_since_last_measurement_pvalue,
                    f"Significant (Bonferroni {bonferroni_correction_divisor})": self.t_time_since_last_measurement_pvalue
                    <= effective_significance_level,
                },
                # {
                #     "Metric": "Hemoglobin Measurement Reaction Time",
                #     f"{self.log_1_name} Mean": f"{mean(self.hemoglobin_reaction_times_1):.2f}",
                #     f"{self.log_2_name} Mean": f"{mean(self.hemoglobin_reaction_times_2):.2f}",
                #     "%Δ Mean": f"{_get_percent_change(mean(self.hemoglobin_reaction_times_1), mean(self.hemoglobin_reaction_times_2)):.2f}%",  # type: ignore
                #     f"{self.log_1_name} Median": f"{median(self.hemoglobin_reaction_times_1):.2f}",
                #     f"{self.log_2_name} Median": f"{median(self.hemoglobin_reaction_times_2):.2f}",
                #     "%Δ Median": f"{_get_percent_change(median(self.hemoglobin_reaction_times_1), median(self.hemoglobin_reaction_times_2)):.2f}%",  # type: ignore
                #     "P-Value": self.reaction_times_pval,
                # },
                # (
                #     {
                #         "Metric": "Low Hemoglobin Reaction Time",
                #         f"{self.log_1_name} Mean": f"{mean(self.low_hemoglobin_reaction_times_1):.2f}",
                #         f"{self.log_2_name} Mean": f"{mean(self.low_hemoglobin_reaction_times_2):.2f}",
                #         "%Δ Mean": f"{_get_percent_change(mean(self.low_hemoglobin_reaction_times_1), mean(self.low_hemoglobin_reaction_times_2)):.2f}%",  # type: ignore
                #         f"{self.log_1_name} Median": f"{median(self.low_hemoglobin_reaction_times_1):.2f}",
                #         f"{self.log_2_name} Median": f"{median(self.low_hemoglobin_reaction_times_2):.2f}",
                #         "%Δ Median": f"{_get_percent_change(median(self.low_hemoglobin_reaction_times_1), median(self.low_hemoglobin_reaction_times_2)):.2f}%",  # type: ignore
                #         "P-Value": self.low_hemoglobin_reaction_times_pval,
                #         # f"Significant (Bonferroni {bonferroni_correction_divisor})": self.low_hemoglobin_reaction_times_pval
                #         # <= effective_significance_level,
                #     }
                #     if not math.isnan(self.low_hemoglobin_reaction_times_pval)
                #     else {
                #         "Metric": "Low Hemoglobin Reaction Time",
                #         f"{self.log_1_name} Mean": str(float("nan")),
                #         f"{self.log_2_name} Mean": str(float("nan")),
                #         "%Δ Mean": str(float("nan")),  # type: ignore
                #         f"{self.log_1_name} Median": str(float("nan")),
                #         f"{self.log_2_name} Median": str(float("nan")),
                #         "%Δ Median": str(float("nan")),  # type: ignore
                #         "P-Value": self.low_hemoglobin_reaction_times_pval,
                #     }
                # ),
                {
                    "Metric": "Mortality Rate",
                    self.log_1_name: f"{self.mortality_rate_1 * 100:.2f}%",
                    self.log_2_name: f"{self.mortality_rate_2 * 100:.2f}%",
                    "%Δ": f"{_get_percent_change(self.mortality_rate_1, self.mortality_rate_2):.2f}%",
                    "P-Value": self.mortality_pvalue,
                    f"Significant (Bonferroni {bonferroni_correction_divisor})": self.mortality_pvalue
                    <= effective_significance_level,
                },
            ]
        )
        if fillna is not None:
            df = df.fillna(fillna)
        return df


def _get_percent_change(before: float, after: float) -> float:
    if before == 0:
        return float("nan")
    else:
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


def _get_hemoglobin_level_series(log: pd.DataFrame) -> pd.Series:
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


def get_num_deaths(log: pd.DataFrame) -> float:
    return (
        log.groupby(DEFAULT_TRACEID_KEY)
        .apply(
            lambda group: (group[DEFAULT_NAME_KEY] == "DEATH").any(),
            include_groups=False,
        )
        .sum()
    )


def get_mortality_pvalue(log_1: pd.DataFrame, log_2: pd.DataFrame) -> float:
    deaths_1 = get_num_deaths(log_1)
    non_deaths_1 = log_1[DEFAULT_TRACEID_KEY].nunique() - deaths_1

    deaths_2 = get_num_deaths(log_2)
    non_deaths_2 = log_2[DEFAULT_TRACEID_KEY].nunique() - deaths_2

    contingency_matrix = np.array([[deaths_1, non_deaths_1], [deaths_2, non_deaths_2]])
    return float(chi2_contingency(contingency_matrix).pvalue)


def get_mortality_rate(log: pd.DataFrame) -> float:
    num_deaths = get_num_deaths(log)
    return num_deaths / log[DEFAULT_TRACEID_KEY].nunique()


def _get_age_series(log: pd.DataFrame) -> pd.Series:
    return log.groupby(DEFAULT_TRACEID_KEY).apply(
        lambda group: group.iloc[0]["anchor_age"], include_groups=False
    )  # type: ignore


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
    ).display_df()

    # return pd.concat(
    #     [
    #         DisparityExplainer.from_logs(
    #             comparator.log_1, comparator.log_2, log_1_name, log_2_name
    #         ).display_df(),
    #         explain_disparities_binned(comparator),
    #     ]
    # )


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
        print(f"\t({100 * count / c1.total():.2f}): {trace}")
    print(log_2_name)
    for trace, count in top_2:
        print(f"\t({100 * count / c2.total():.2f}): {trace}")


def find_common_variant_mismatches(
    comparator: NumericalColumn_Levenshtein_PermutationComparator, top_n: int = 3
) -> tuple[set[BinnedServiceTimeTrace], set[BinnedServiceTimeTrace]]:
    c1 = Counter(comparator.behavior_1)
    c2 = Counter(comparator.behavior_2)

    top_1 = c1.most_common(top_n)
    top_2 = c2.most_common(top_n)

    variants_1 = {variant for variant, _ in top_1}
    variants_2 = {variant for variant, _ in top_2}

    return (variants_1 - variants_2, variants_2 - variants_1)


def find_common_variant_mismatches_cf(
    comparator: NumericalColumn_Levenshtein_PermutationComparator, top_n: int = 3
) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]]]:
    # Only consider the control flow
    c1 = Counter(
        [
            tuple(activity for activity, _ in variant)
            for variant in comparator.behavior_1
        ]
    )
    c2 = Counter(
        [
            tuple(activity for activity, _ in variant)
            for variant in comparator.behavior_2
        ]
    )

    top_1 = c1.most_common(top_n)
    top_2 = c2.most_common(top_n)

    variants_1 = {variant for variant, _ in top_1}
    variants_2 = {variant for variant, _ in top_2}

    return (variants_1 - variants_2, variants_2 - variants_1)


def get_low_hemoglobin_reactions(
    log: pd.DataFrame,
    low_hemoglobin_threshold: float | None = 7.0,
    allow_between: bool = False,
) -> tuple[list[tuple[pd.Series, pd.Series]], int]:
    def get_reactions_for_case(
        case_log: pd.DataFrame,
    ) -> tuple[list[tuple[pd.Series, pd.Series]], int]:
        # Retun 1) List of event, reaction pairs, and 2) number of measurements without reaction
        case_log = case_log.sort_values(by=[DEFAULT_NAME_KEY, DEFAULT_TIMESTAMP_KEY])
        measurements: pd.DataFrame = case_log[
            case_log["event_type"] == "hemoglobin_measurement"
        ]  # type: ignore
        if low_hemoglobin_threshold is not None:
            low_measurements: pd.DataFrame = measurements[
                measurements["hemoglobin_value"] < low_hemoglobin_threshold
            ]  # type: ignore
        else:
            low_measurements = measurements.copy()
        transfusions: pd.DataFrame = case_log[
            case_log["event_type"] == "blood_transfusion"
        ]  # type: ignore

        def get_reaction(row: pd.Series) -> pd.Series | None:
            timestamp = row[DEFAULT_TIMESTAMP_KEY]
            reaction_df: pd.DataFrame = transfusions[
                transfusions[DEFAULT_TIMESTAMP_KEY] > timestamp
            ]  # type: ignore
            if not reaction_df.empty:
                reaction = reaction_df.iloc[0]
                reaction_timestamp = reaction[DEFAULT_TIMESTAMP_KEY]
                # Only count as reaction if there was no measurement in between
                if (
                    allow_between
                    or not measurements[
                        (measurements[DEFAULT_TIMESTAMP_KEY] < reaction_timestamp)
                        & (timestamp < measurements[DEFAULT_TIMESTAMP_KEY])
                    ].empty
                ):
                    return reaction
                else:
                    return None

            else:
                return None

        reactions = [(row, get_reaction(row)) for _, row in low_measurements.iterrows()]
        len_before = len(reactions)
        reactions = [
            (evt, reaction) for evt, reaction in reactions if reaction is not None
        ]
        num_non_reactions = len_before - len(reactions)
        return reactions, num_non_reactions

    # def get_case_reactions(case_log: pd.DataFrame) -> list[tuple[pd.Series, pd.Series]]:
    #     res: list[tuple[pd.Series, pd.Series]] = []
    #     num_unanswered_measurements = 0
    #     for i, row in case_log.iterrows():
    #         if cast(bool, row["is_low_hemoglobin"]):
    #             next_transfusion = case_log.iloc[i + 1 :][  # type: ignore
    #                 case_log["event_type"] == "blood_transfusion"
    #             ]
    #             if not next_transfusion.empty:
    #                 transfusion = next_transfusion.iloc[0]
    #                 res.append((row, transfusion))
    #             else:
    #                 num_unanswered_measurements += 1
    #     return res

    # for i, row in log.sort_values(by=["case:concept:name", "time:timestamp"]).groupby
    return reduce(
        lambda x, y: (x[0] + y[0], x[1] + y[1]),
        [
            get_reactions_for_case(case_df)
            for _, case_df in log.groupby(by=DEFAULT_TRACEID_KEY)
        ],
    )


def get_low_hemoglobin_reaction_times(
    log: pd.DataFrame,
    low_hemoglobin_threshold: float | None = 7.0,
    allow_between: bool = False,
) -> list[float]:
    reactions, non_reactions = get_low_hemoglobin_reactions(
        log, low_hemoglobin_threshold, allow_between
    )
    print(f"Non-Reactions: {non_reactions}")
    return [
        (
            transfusion[DEFAULT_TIMESTAMP_KEY] - measurement[DEFAULT_TIMESTAMP_KEY]
        ).total_seconds()  # type: ignore
        for measurement, transfusion in reactions
    ]


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
