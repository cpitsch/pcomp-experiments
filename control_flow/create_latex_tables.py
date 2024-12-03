from pathlib import Path
from typing import Any, cast

import pandas as pd
from run import LogInstance, get_all_log_instances
from streamlit_analysis.main import (
    get_change_pattern_performance_table,
    get_performance_table,
    log_name_to_change_pattern,
    ostovar_to_change_pattern,
)

RESULTS_BASE_DIR = Path("control_flow_results")


SUMMARY_PATH = RESULTS_BASE_DIR / "summary.csv"
LATEX_FILES_BASE_DIR = Path("latex_tables")

FLOAT_FORMATTER = "{:.3g}".format

short_to_long = {v: k for k, v in ostovar_to_change_pattern.items()}
short_to_long["bose"] = "Bose Log"
short_to_long |= {
    "bose": "Bose Event Log",
    "cd": "Synchronize",
    "cp": "Copy",
    "re": "Add/Remove",
}


def get_results_df() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_PATH)
    df["change_pattern"] = df[["log_source", "log_name"]].apply(
        lambda row: log_name_to_change_pattern(row["log_source"], row["log_name"]),
        axis=1,
    )

    return df


def get_splitted_results_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = get_results_df()
    return (
        cast(pd.DataFrame, df[df["technique"] == "Bootstrap Test"]),
        cast(pd.DataFrame, df[df["technique"] == "Permutation Test"]),
    )


def log_instance_to_df_row(log_instance: LogInstance) -> dict[str, Any]:
    return {
        "Change Pattern": log_name_to_change_pattern(
            log_instance.source, log_instance.path.name.split(".")[0]  # type: ignore
        ),
        "Positive": log_instance.has_drift,
    }


def create_change_pattern_counts_table():
    all_log_instances = get_all_log_instances()
    df = pd.DataFrame(
        [log_instance_to_df_row(log_instance) for log_instance in all_log_instances]
    )
    df["Change Pattern"] = df["Change Pattern"].apply(
        lambda x: short_to_long[x] + f" ({x})"
    )

    df_counts = (
        df.groupby(["Change Pattern", "Positive"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    df_counts = df_counts.rename(
        columns={False: "Negative Instances", True: "Positive Instances"}
    )
    df_counts.columns.name = None
    df_counts = df_counts.sort_values(by="Change Pattern")
    df_counts["Total"] = (
        df_counts["Positive Instances"] + df_counts["Negative Instances"]
    )
    df_counts = df_counts[
        ["Change Pattern", "Positive Instances", "Negative Instances", "Total"]
    ]

    totals = df_counts[["Positive Instances", "Negative Instances", "Total"]].sum()
    total_row = pd.DataFrame(
        [
            [
                "Total",
                totals["Positive Instances"],
                totals["Negative Instances"],
                totals["Total"],
            ]
        ],
        columns=df_counts.columns,
    )
    df_with_total = pd.concat([df_counts, total_row], ignore_index=True)

    latex = (
        df_with_total.to_latex(column_format="lccc", index=False)
        .replace(r"\toprule", r"\hline")
        .replace(r"\midrule", r"\hline")
        .replace(r"\bottomrule", r"\hline")
    )
    with open(LATEX_FILES_BASE_DIR / "change_pattern_logs.tex", "w") as f:
        f.write(latex)


def create_general_metrics_table():
    bootstrap_df, permutation_test_df = get_splitted_results_df()

    performance_bootstrap = get_performance_table(bootstrap_df)
    performance_permutation = get_performance_table(permutation_test_df)

    performance_bootstrap["Approach"] = "Bootstrap Test"
    performance_permutation["Approach"] = "Permutation Test"

    final_df = pd.concat([performance_bootstrap, performance_permutation])
    final_df = final_df[
        [
            "Approach",
            "TP",
            "TN",
            "FP",
            "FN",
            "Accuracy",
            "Precision",
            "Recall",
            "FPR",
            "F1-Score",
        ]
    ]

    latex = (
        final_df.to_latex(
            column_format="lccccccccc",
            index=False,
            float_format=FLOAT_FORMATTER,
            na_rep="",
        )
        .replace(r"\toprule", r"\hline")
        .replace(r"\midrule", r"\hline")
        .replace(r"\bottomrule", r"\hline")
    )
    with open(LATEX_FILES_BASE_DIR / "control_flow_results.tex", "w") as f:
        f.write(latex)


def get_per_change_pattern_table(df: pd.DataFrame):
    change_pattern_performance_table = get_change_pattern_performance_table(df)

    def descriptive_change_pattern(short_pattern: str):
        short_pattern = short_pattern.lower()
        if short_pattern == "bose":
            return short_to_long[short_pattern]
        else:
            return short_to_long[short_pattern] + f" ({short_pattern})"

    change_pattern_performance_table["Change Pattern"] = (
        change_pattern_performance_table["Change Pattern"].apply(
            descriptive_change_pattern
        )
    )
    return change_pattern_performance_table


def create_per_change_pattern_table(df: pd.DataFrame, filename: str):
    change_pattern_performance_table = get_per_change_pattern_table(df)
    latex = (
        change_pattern_performance_table.sort_values(by="Change Pattern")
        .to_latex(
            column_format="lccccccccc",
            index=False,
            float_format=FLOAT_FORMATTER,
            na_rep="",
        )
        .replace(r"\toprule", r"\hline")
        .replace(r"\midrule", r"\hline")
        .replace(r"\bottomrule", r"\hline")
    )
    with open(LATEX_FILES_BASE_DIR / filename, "w") as f:
        f.write(latex)


if __name__ == "__main__":
    if not LATEX_FILES_BASE_DIR.exists():
        LATEX_FILES_BASE_DIR.mkdir()

    create_change_pattern_counts_table()
    create_general_metrics_table()
    bootstrap_df, permutation_df = get_splitted_results_df()
    create_per_change_pattern_table(
        permutation_df, "permutation_test_per_change_pattern.tex"
    )
    create_per_change_pattern_table(
        bootstrap_df, "bootstrap_test_per_change_pattern.tex"
    )
