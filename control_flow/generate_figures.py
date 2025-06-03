from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from streamlit_analysis.main import (
    get_change_pattern_performance_table,
    get_splitted_results_df,
)

sns.set_theme(rc={"figure.figsize": (12, 4)})

OUTPUT_ROOT = Path("figures")
if not OUTPUT_ROOT.exists():
    OUTPUT_ROOT.mkdir(parents=True)


def get_summary_df() -> pd.DataFrame:
    return pd.read_csv("control_flow_results/summary.csv")


def get_noiseless_summary_df() -> pd.DataFrame:
    df = get_summary_df()
    return df[df["noise_level"] == 0]


def get_combined_performance_table() -> pd.DataFrame:
    bootstrap_df, permutation_df = get_splitted_results_df()

    permutation_performance = get_change_pattern_performance_table(permutation_df)
    bootstrap_performance = get_change_pattern_performance_table(bootstrap_df)

    permutation_performance["Technique"] = "Permutation Test"
    bootstrap_performance["Technique"] = "Bootstrap Test"

    return pd.concat([permutation_performance, bootstrap_performance])


def sort_change_patterns(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by="Change Pattern", key=lambda words: words.map(lambda word: [len(word), word])
    )


def create_combined_f1_score_barplot():
    combined = sort_change_patterns(get_combined_performance_table())

    fig, ax = plt.subplots()
    ax = sns.barplot(
        combined.fillna(0), x="Change Pattern", y="F1-Score", hue="Technique", ax=ax
    )
    sns.move_legend(
        ax,
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        title=None,
        frameon=False,
    )

    fig.savefig(OUTPUT_ROOT / "f1-score-general.pdf", bbox_inches="tight")
    plt.close(fig)


def create_combined_harm_mean_power_t1er_barplot():
    combined = sort_change_patterns(get_combined_performance_table())

    # H_MEAN_COLUMN_NAME = r"$H(\mathit{Power}, \mathit{T1-ER})$"
    H_MEAN_COLUMN_NAME = "Harmonic Mean of 1-T1-ER and Power"
    combined["one-minus-fpr"] = 1 - combined["FPR"]
    combined[H_MEAN_COLUMN_NAME] = (
        2 * combined["Recall"] * combined["one-minus-fpr"]
    ) / (combined["Recall"] + combined["one-minus-fpr"])

    # "Harmonic Mean of 1-T1-ER and Power"
    fig, ax = plt.subplots()
    ax = sns.barplot(
        combined.fillna(0),
        x="Change Pattern",
        y=H_MEAN_COLUMN_NAME,
        hue="Technique",
        ax=ax,
    )
    sns.move_legend(
        ax,
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        title=None,
        frameon=False,
    )

    fig.savefig(OUTPUT_ROOT / "t1er_power_fscore.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    create_combined_f1_score_barplot()
    create_combined_harm_mean_power_t1er_barplot()
