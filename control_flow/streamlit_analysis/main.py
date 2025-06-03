from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import streamlit as st

RESULTS_BASE_DIR = Path("..", "control_flow_results")


SUMMARY_DIR = RESULTS_BASE_DIR / "summary.csv"
ONLY_NOISE_0 = True

if not SUMMARY_DIR.exists():
    RESULTS_BASE_DIR = Path("control_flow_results")
    SUMMARY_DIR = RESULTS_BASE_DIR / "summary.csv"


ostovar_to_change_pattern = {
    "ConditionalMove": "cm",
    "ConditionalRemoval": "cre",
    "ConditionalToSequence": "cf",
    "Frequency": "fr",
    "Loop": "lp",
    "ParallelMove": "pm",
    "ParallelRemoval": "pre",
    "ParallelToSequence": "pl",
    "SerialMove": "sm",
    "SerialRemoval": "sre",
    "Skip": "cb",
    "Substitute": "rp",
    "Swap": "sw",
    # Could use .get() but I want to explicitly catch mistakes
    "IOR": "IOR",
    "IRO": "IRO",
    "OIR": "OIR",
    "ORI": "ORI",
    "RIO": "RIO",
    "ROI": "ROI",
}

LogSource = Literal["ostovar", "ceravolo", "bose"]


def log_name_to_change_pattern(source: LogSource, log_name: str) -> str:
    match source:
        case "ostovar":
            # E.g., SerialMove_noise0
            # E.g., Atomic_Size3_SerialMove_noise0.xes.gz
            return ostovar_to_change_pattern[log_name.split("_")[2]]
        case "ceravolo":
            # E.g., sudden_trace_noise0_1000_sw
            return log_name.split("_")[-1]
        case "bose":
            return "bose"


# @st.cache_data
def get_results_df() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_DIR)
    # df = df[~df["log_name"].str.contains("recurring")]

    df["change_pattern"] = df[["log_source", "log_name"]].apply(
        lambda row: log_name_to_change_pattern(row["log_source"], row["log_name"]),
        axis=1,
    )

    if ONLY_NOISE_0:
        df = df[df["noise_level"] == 0]

    return df


# @st.cache_data()
def get_splitted_results_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = get_results_df()
    return (
        cast(pd.DataFrame, df[df["technique"] == "Bootstrap Test"]),
        cast(pd.DataFrame, df[df["technique"] == "Permutation Test"]),
    )


def get_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["classification_class"].value_counts().to_dict()

    try:
        accuracy = (counts.get("TP", 0) + counts.get("TN", 0)) / sum(counts.values())
    except ZeroDivisionError:
        accuracy = np.NaN

    try:
        precision = counts.get("TP", 0) / (counts.get("TP", 0) + counts.get("FP", 0))
    except ZeroDivisionError:
        precision = np.NaN

    try:
        recall = counts.get("TP", 0) / (counts.get("TP", 0) + counts.get("FN", 0))
    except ZeroDivisionError:
        recall = np.NaN

    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = np.NaN

    try:
        fpr = counts.get("FP", 0) / (counts.get("FP", 0) + counts.get("TN", 0))
    except ZeroDivisionError:
        fpr = np.NaN

    df = pd.DataFrame(
        [
            {
                "TP": counts.get("TP", 0),
                "TN": counts.get("TN", 0),
                "FP": counts.get("FP", 0),
                "FN": counts.get("FN", 0),
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "FPR": fpr,
                "F1-Score": f1,
            }
        ]
    )

    return df


def get_performance_table_per_source(df: pd.DataFrame) -> dict[LogSource, pd.DataFrame]:
    return {
        cast(LogSource, source): get_performance_table(group_df)
        for source, group_df in df.groupby(by="log_source")
    }


def get_change_pattern_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for change_pattern, sub_df in df.groupby(by="change_pattern", sort=False):
        performance_table = get_performance_table(sub_df)
        columns_before = performance_table.columns
        performance_table["Change Pattern"] = change_pattern.upper()
        performance_table = performance_table[["Change Pattern"] + list(columns_before)]
        tables.append(performance_table)
    return pd.concat(tables)


def streamlit_main():
    st.header("Control Flow Analysis")
    st.dataframe(get_results_df(), hide_index=True)

    bootstrap_df, permutation_df = get_splitted_results_df()

    st.markdown("## Bootstrap Test")
    st.dataframe(get_performance_table(bootstrap_df), hide_index=True)

    with st.expander("Per-Source Analysis"):
        for source, performance_table in get_performance_table_per_source(
            bootstrap_df
        ).items():
            st.markdown(f"### {source.title()}")
            st.dataframe(performance_table, hide_index=True)
    with st.expander("Per-Change-Pattern-Analysis"):
        change_pattern_performance_table = get_change_pattern_performance_table(
            bootstrap_df
        )
        st.dataframe(change_pattern_performance_table, hide_index=True)
        st.markdown(
            f"#### Mean F1-Score {change_pattern_performance_table['F1-Score'].mean()}"
        )

    with st.expander("Incorrect Classifications"):
        incorrect_df = bootstrap_df[~bootstrap_df["correct"]]
        st.dataframe(
            incorrect_df[
                ["log_source", "log_name", "has_diff", "pval", "classification_class"]
            ],
            hide_index=True,
        )

    st.markdown("## Permutation Test")
    st.dataframe(get_performance_table(permutation_df), hide_index=True)

    with st.expander("Per-Source Analysis"):
        for source, performance_table in get_performance_table_per_source(
            permutation_df
        ).items():
            st.markdown(f"### {source.title()}")
            st.dataframe(performance_table, hide_index=True)
    with st.expander("Per-Change-Pattern-Analysis"):
        change_pattern_performance_table = get_change_pattern_performance_table(
            permutation_df
        )
        st.dataframe(change_pattern_performance_table, hide_index=True)
        st.markdown(
            f"#### Mean F1-Score {change_pattern_performance_table['F1-Score'].mean()}"
        )
    with st.expander("Incorrect Classifications"):
        incorrect_df = permutation_df[~permutation_df["correct"]]
        st.dataframe(
            incorrect_df[
                ["log_source", "log_name", "has_diff", "pval", "classification_class"]
            ],
            hide_index=True,
        )


if __name__ == "__main__":
    streamlit_main()
