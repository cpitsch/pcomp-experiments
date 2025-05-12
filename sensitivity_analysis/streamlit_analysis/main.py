import pickle as pkl
from collections import Counter
from dataclasses import dataclass
from math import log2
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from log_analysis import plot_distributions_plotly, plot_distributions_seaborn
from matplotlib.axes import Axes
from pcomp.emd.comparators.permutation_test import PermutationTestComparisonResult
from pm4py import read_xes

RESULTS_PATH = Path("results")


WEIGHTED_TIME_RESULTS_PATH = Path("road_traffic_synthetic_results_WEIGHTED_TIME")


ORDER_CRITERION = [
    "technique",
    "log_path",
    "binner",
    "weighted_time_cost",
    "mutation_probability",
    "mutation_std_shift",
    "has_diff",
    "is_no_diff_log",
]

EQUALITY_CRTERION = [
    "technique",
    "mutation_probability",
    "mutation_std_shift",
    "binner",
    "weighted_time_cost",
    "has_diff",
    "is_no_diff_log",
]


EvalStyle = Literal["Fixed Severity", "Fixed Probability"]


@st.cache_data
def cached_read_csv(path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


@dataclass
class SeededResults:
    seed: int
    root_path: Path

    @classmethod
    def from_root_path(cls, root_path: Path) -> "SeededResults":
        print(root_path)
        return SeededResults(int(root_path.name), root_path)

    def get_summary(self) -> pd.DataFrame:
        return cached_read_csv(self.root_path / "summary.csv")


@st.cache_data
def get_seed_root_paths(weighted: bool) -> list[SeededResults]:
    root_path = RESULTS_PATH if not weighted else WEIGHTED_TIME_RESULTS_PATH
    return [
        SeededResults.from_root_path(path)
        for path in root_path.iterdir()
        if path.is_dir()
    ]


def get_seeds(weighted: bool) -> list[int]:
    return [int(result.root_path.name) for result in get_seed_root_paths(weighted)]


@st.cache_data
def get_seed_summaries(weighted: bool) -> list[pd.DataFrame]:
    return [
        result.get_summary().sort_values(by=ORDER_CRITERION).reset_index(drop=True)
        for result in get_seed_root_paths(weighted)
    ]


@st.cache_data
def get_total_df(summaries: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(summaries)
    total_df = (
        df.groupby(by=EQUALITY_CRTERION)
        .aggregate(
            {
                "pval": list,
                "logs_emd": list,
                "detection": list,
                "correct": list,
                "classification_class": list,
                "duration": list,
            }
        )
        .reset_index()
    )
    total_df["mean_pval"] = total_df["pval"].apply(np.mean)
    total_df["mean_is_detection"] = total_df["mean_pval"] < 0.05
    total_df["majority_vote"] = total_df["detection"].apply(
        lambda detections: Counter(detections).most_common(1)[0][0]
    )
    # I am aware this isnt really percent since it goes from 0-1, but its late and I cant
    # think of another word for it
    total_df["percent_correct"] = total_df["correct"].apply(
        lambda corrects: sum(corrects) / len(corrects)
    )
    total_df["pval_range"] = total_df["pval"].apply(
        lambda pvals: (min(pvals), max(pvals))
    )
    total_df["pval_range_size"] = total_df["pval_range"].apply(
        lambda minmax: minmax[1] - minmax[0]
    )
    total_df["detection_entropy"] = total_df["detection"].apply(entropy)

    return total_df


def get_sensitivity_matrix(df: pd.DataFrame, column: str = "detection") -> pd.DataFrame:
    """Create a dataframe showing the `column` value for each mutation setting.
    Makes the mutation probability the df index, and the severities are the column names.

    Made for displaying with `st.dataframe`.

    Args:
        df (pd.DataFrame): The dataframe.
        column (str, optional): The column for which to show the sensitivity. Defaults to
            "detection".

    Returns:
        pd.DataFrame: The sensitivity matrix.
    """

    value_pairs_to_detection = list(
        df[["mutation_probability", "mutation_std_shift", column]].itertuples(
            index=False, name=None
        )
    )

    probabilities = df["mutation_probability"].unique()
    rows = [
        {k: v for p, k, v in value_pairs_to_detection if p == prob}
        for prob in probabilities
    ]

    # Make probabilities the row index, std shift the column index (column names)
    res = pd.DataFrame(rows).set_index(probabilities).sort_index()
    res: pd.DataFrame = res[sorted(list(res.columns))]  # type: ignore
    return res


def entropy(items: list[bool]) -> float:
    population = Counter(items)
    pop_len = len(items)
    return -sum(
        (count / pop_len) * log2(count / pop_len) for _, count in population.items()
    )


def plot_pval_progression(
    df: pd.DataFrame,
    key_to_track: Literal["mutation_std_shift", "mutation_probability"],
    pval_column: str = "pval",
    rename_pval_column: str = "P-Value",
) -> None:
    renamed_columns = {
        "mutation_std_shift": "Standard Deviations Increase",
        "mutation_probability": "% Affected Events",
        pval_column: rename_pval_column,
        # "pval": "P-Value",
    }
    plot_df = df.rename(columns=renamed_columns)
    st.line_chart(plot_df, x=renamed_columns[key_to_track], y=rename_pval_column)


def plot_progression(
    df: pd.DataFrame,
    selectbox_key: str,
    pval_column: str = "pval",
    rename_pval_column: str = "P-Value",
) -> None:
    eval_styles: list[EvalStyle] = list(get_args(EvalStyle))
    match st.selectbox(
        "Evaluation Style", eval_styles, index=0, key=f"ev_style_{selectbox_key}"
    ):
        case "Fixed Severity":
            severities = sorted(df["mutation_std_shift"].unique())
            severity = st.select_slider(
                "Standard Deviations Shift",
                options=severities,
                value=1.0,
                key=f"std_slider_{selectbox_key}",
            )
            plot_pval_progression(
                df[df["mutation_std_shift"] == severity],  # type: ignore
                "mutation_probability",
                pval_column,
                rename_pval_column,
            )
        case "Fixed Probability":
            probabilities = sorted(df["mutation_probability"].unique())
            probability = st.select_slider(
                "Mutation Probability",
                options=probabilities,
                value=1.0,
                key=f"prob_slider_{selectbox_key}",
            )
            plot_pval_progression(
                df[df["mutation_probability"] == probability],  # type: ignore
                "mutation_std_shift",
                pval_column,
                rename_pval_column,
            )


def _float_to_str_truncate(x: float):
    if x.is_integer():
        return str(int(x))
    else:
        return str(x)


@st.cache_data
def plot_log_service_time_distributions(
    seed: int,
    probability: float,
    severity: float,
    mode: Literal["seaborn", "plotly"],
) -> go.Figure | Axes:

    sev = _float_to_str_truncate(severity)
    prob = _float_to_str_truncate(probability)

    log_1_path = f"road_traffic_synthetic/{seed}/PartialOrderCreator/LogSplitter_frac_0.5_seed_{seed}/log_1.xes.gz"
    log_2_path = f"road_traffic_synthetic/{seed}/PartialOrderCreator/LogSplitter_frac_0.5_seed_{seed}/ServiceTimeStdShifter_Send Fine_std{sev}_p{prob}_seed_{seed}/log_2.xes.gz"

    log_1 = read_xes(log_1_path, "rustxes")
    log_2 = read_xes(log_2_path, "rustxes")
    if mode == "plotly":
        return plot_distributions_plotly(
            log_1[log_1["concept:name"] == "Send Fine"],  # type: ignore
            log_2[log_2["concept:name"] == "Send Fine"],  # type: ignore
        )
    elif mode == "seaborn":
        return plot_distributions_seaborn(
            log_1[log_1["concept:name"] == "Send Fine"],  # type: ignore
            log_2[log_2["concept:name"] == "Send Fine"],  # type: ignore
        )


def plot_test_result(
    seed: int, probability: float, severity: float, weighted_time: bool
):
    result_path = (
        (RESULTS_PATH if not weighted_time else WEIGHTED_TIME_RESULTS_PATH)
        / str(seed)
        / ("weighted_time" if weighted_time else "normal_time")
        / f"std{_float_to_str_truncate(severity)}_p{_float_to_str_truncate(probability)}"
        / "result.pkl"
    )
    with open(result_path, "rb") as f:
        result_obj: PermutationTestComparisonResult = pkl.load(f)
        st.pyplot(result_obj.plot())


st.set_page_config(layout="wide")
weighted_results = st.checkbox("Weighted Time Function")
seed_specific_tab, total_tab, log_tab = st.tabs(
    ["Seed-Specific Analysis", "Aggregated Analysis", "Log Exploration"]
)

DF_HEIGHT = 527

with total_tab:
    total_df = get_total_df(get_seed_summaries(weighted_results))
    st.header("Classification by Mean P-Value")
    st.dataframe(
        get_sensitivity_matrix(total_df, "mean_is_detection"), height=DF_HEIGHT
    )
    st.header("Classification by Majority Vote")
    st.dataframe(get_sensitivity_matrix(total_df, "majority_vote"), height=DF_HEIGHT)
    st.header("Percent Correct Classifications")
    st.dataframe(get_sensitivity_matrix(total_df, "percent_correct"), height=DF_HEIGHT)
    st.header("P-Value Range")
    st.dataframe(get_sensitivity_matrix(total_df, "pval_range"), height=DF_HEIGHT + 1)
    st.dataframe(
        get_sensitivity_matrix(total_df, "pval_range_size"), height=DF_HEIGHT + 1
    )
    st.header("Detection Entropy")
    st.dataframe(
        get_sensitivity_matrix(total_df, "detection_entropy"), height=DF_HEIGHT
    )

    plot_progression(total_df, "aggregated_analysis", "mean_pval", "Mean P-Value")
with seed_specific_tab:
    seeds = get_seeds(weighted_results)
    seed = st.number_input("Seed", 1, 5)  # TODO: Un-hardcode
    # seed = st.number_input("Seed", min(seeds), max(seeds))
    if seed not in seeds:
        # Could happen if a seed was skipped, e.g., we have seeds 1,2,4 and select 3
        st.warning("No results found for this seed. Please try another")
    else:
        df = get_seed_summaries(weighted_results)[seed - 1]
        st.header(f"Seed {seed}")
        st.dataframe(get_sensitivity_matrix(df), height=DF_HEIGHT)

        # P-Value Progression
        plot_progression(df, "seed_specific", "pval", "P-Value")

        prob = st.select_slider(
            "Probability",
            sorted(df["mutation_probability"].unique().tolist()),
            value=0.0,
        )
        sev = st.select_slider(
            "Severity", sorted(df["mutation_std_shift"].unique().tolist()), value=0.0
        )
        st.header(
            f"Result for Probability {_float_to_str_truncate(prob)}, Severity {_float_to_str_truncate(sev)}"
        )
        plot_test_result(seed, prob, sev, weighted_results)
with log_tab:
    total_df = get_total_df(get_seed_summaries(weighted_results))
    severities = total_df["mutation_std_shift"].unique()
    probabilities = total_df["mutation_probability"].unique()
    probability = st.select_slider(
        "Mutation Probability",
        options=probabilities,
        value=1.0,
        key="log_analysis_probability",
    )
    severity = st.select_slider(
        "Standard Deviations Shift",
        options=severities,
        value=1.0,
        key="log_analysis_severity",
    )

    SEED = 1
    fig: go.Figure = plot_log_service_time_distributions(  # type: ignore
        SEED, probability, severity, "plotly"
    )
    st.plotly_chart(fig)

    ax: Axes = plot_log_service_time_distributions(  # type: ignore
        SEED, probability, severity, "seaborn"
    )
    ax.set_xlim(0, 8_000)
    st.pyplot(ax.get_figure())
