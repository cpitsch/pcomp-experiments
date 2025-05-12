from collections import Counter
from math import log2
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from pm4py import read_xes  # type: ignore

FIGURES_BASE_DIR = Path("figures")


RESULTS_BASE_DIR = Path("results")
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

FLOAT_FORMATTER = "{:.3g}".format


def _num_to_str_truncate(x: float | int) -> str:
    """Convert an int/float to string. Follow the rules rust does
    since rust created the directory names. I.e., 1.0 becomes "1", not "1.0"T.

    Args:
        x (float | int): The number to convert.

    Returns:
        str: The converted as a string, first converted to int if it is a whole number.
    """
    if isinstance(x, int):
        return str(x)
    elif x.is_integer():
        return str(int(x))
    else:
        return str(x)


def import_log(log_path: Path) -> pd.DataFrame:
    """Import an event log using rustxes.

    Args:
        log_path (Path): The path to the .xes or .xes.gz file.

    Returns:
        pd.DataFrame: The imported event log
    """
    return cast(pd.DataFrame, read_xes(log_path.as_posix(), variant="rustxes"))


def get_log_paths(seed: int, probability: float, severity: float) -> tuple[Path, Path]:
    """Get the paths to the two compared event logs for a certain parameter setting

    Args:
        seed (int): The seed.
        probability (float): The probability of injecting a mutation.
        severity (float): The severity of the (potentially) injected mutation.

    Returns:
        tuple[Path, Path]: The paths to the event logs. The first path corresponds to
            the event log half with no mutations.
    """
    prob = _num_to_str_truncate(probability)
    sev = _num_to_str_truncate(severity)

    base_path = Path(
        "road_traffic_synthetic",
        str(seed),
        "PartialOrderCreator",
        f"LogSplitter_frac_0.5_seed_{seed}",
    )
    return (
        base_path / "log_1.xes.gz",
        base_path
        / f"ServiceTimeStdShifter_Send Fine_std{sev}_p{prob}_seed_{seed}"
        / "log_2.xes.gz",
    )


def get_logs(
    seed: int, probability: float, severity: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Import the event logs for a given parameter setting.

    Args:
        seed (int): The seed used for mutations.
        probability (float): The probability of injecting a mutation.
        severity (float): The severity of the (potentially) injected mutation.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The event logs. The first event log is the
            half with no injected mutations.
    """
    log_path_1, log_path_2 = get_log_paths(seed, probability, severity)
    log_1 = import_log(log_path_1)
    log_2 = import_log(log_path_2)
    return log_1, log_2


def plot_time_distributions(
    seed: int, probability: float, severity: float, activity: str | None = None
) -> tuple[Figure, Axes]:
    """Plot the distributions of total service time per case for the two event log halves
    for a given parameter setting. The original event log distribution is shown in blue,
    the mutated half in orange.

    Args:
        seed (int): The seed used for mutations.
        probability (float): The probability of injecting a mutation.
        severity (float): The severity of the (potentially) injected mutation.
        activity (str | None, optional): The activity to filter for. Defaults to None
            (all activities considered).

    Returns:
        tuple[Figure, Axes]: The figure.
    """
    log_1, log_2 = get_logs(seed, probability, severity)

    log_1["Log Type"] = "Original"
    log_2["Log Type"] = "Mutated"
    combined = pd.concat([log_1, log_2])

    if activity is not None:
        combined: pd.DataFrame = combined[combined["concept:name"] == activity]  # type: ignore

    combined["@pcomp:duration"] = (
        combined["time:timestamp"] - combined["start_timestamp"]
    )
    combined["Service Time [h]"] = combined["@pcomp:duration"].dt.total_seconds() / 3600

    # plt.rcParams["axes.labelsize"] = 14  # Axis labels
    fig, ax = plt.subplots()
    sns.histplot(combined, x="Service Time [h]", hue="Log Type", element="step", ax=ax)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.get_xaxis().get_label().set_fontsize(14)
    ax.get_yaxis().get_label().set_fontsize(14)
    return fig, ax


def generate_service_time_distribution_plots():
    """Create and save the service time distribution plots for a few hard-coded probability/severity
    pairs (and seed=1).
    """
    BASE_DIR = FIGURES_BASE_DIR / "service_time_distributions"
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    prob_sev_pairs = [
        (0, 0),
        (0.5, 0.5),
        (1, 0.5),
        (0.5, 1),
        (0.7, 1),
        (1, 1),
        (0.4, 0.3),
    ]

    SEED = 1

    for prob, sev in prob_sev_pairs:
        fig, ax = plot_time_distributions(SEED, prob, sev, "Send Fine")

        # Values from visual inspection
        ax.set_xlim(0, 8000)
        ax.set_ylim(0, 1825)

        fig.savefig(
            BASE_DIR / f"prob_{prob}_sev_{sev}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def get_summary_df(seed: int) -> pd.DataFrame:
    """Load the experiment summary dataframe for a given seed.
    Additionally, the column "scaled_emd" is created, which multiplies the measured EMD
    between the event logs by 10.

    Args:
        seed (int): The seed (used for log mutationTs) to consider.

    Returns:
        pd.DataFrame: The dataframe.
    """
    df = pd.read_csv(RESULTS_BASE_DIR / str(seed) / "summary.csv")
    df["scaled_emd"] = df["logs_emd"] * 10
    return df


def get_summary_dfs() -> dict[int, pd.DataFrame]:
    """Load the experiment summary dataframe for each seed.

    Returns:
        dict[int, pd.DataFrame]: A dictionary mapping seeds to the respective results
            dataframe.
    """
    return {
        int(seed.name): get_summary_df(int(seed.name))
        for seed in RESULTS_BASE_DIR.iterdir()
    }


def entropy(items: list[bool]) -> float:
    """Compute the Shannon-Entropy of a list of booleans.

    Args:
        items (list[bool]): The booleans.

    Returns:
        float: The shannon-entropy
    """
    population = Counter(items)
    pop_len = len(items)
    return -sum(
        (count / pop_len) * log2(count / pop_len) for _, count in population.items()
    )


def aggregate_summary_dfs(summaries: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Combine all summary dataframes into one by concatenation. Then, results for
    identical parameters (but different seeds) are collected and aggregated into lists
    of values.
    Additionally, some auxilliary, aggregate, columns are created:

        - `mean_pval`: The mean pvalue over all seeds
        - `pval_stdev`: The pvalue standard deviation over all seeds
        - `mean_is_detection`: Is `mean_pval` less than the significance level (0.05)?
        - `majority_vote`: Majority vote decision over all seeds: is there a difference.
        - `mean_logs_emd`: The mean emd measured between the event logs, over all seeds
        - `10_mean_logs_emd`: `mean_logs_emd` multiplied by 10
        - `percent_correct`: Ratio of correct classifications over all seeds
        - `pval_range`: The range of (min_pval, max_pval) over all seeds
        - `pval_range_size`: If `pval_range` is (min_pval, max_pval), then this is max_pval - min_pval
        - `detection_entropy`: The shannon entropy of the classifications (detection: true, false)

    Args:
        summaries (dict[int, pd.DataFrame]): The summaries dictionary. See `get_summary_dfs`.

    Returns:
        pd.DataFrame: The combined dataframe.
    """
    df = pd.concat(summaries.values())
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
    total_df["pval_stdev"] = total_df["pval"].apply(np.std)
    total_df["mean_is_detection"] = total_df["mean_pval"] < 0.05
    total_df["majority_vote"] = total_df["detection"].apply(
        lambda detections: Counter(detections).most_common(1)[0][0]
    )
    total_df["mean_logs_emd"] = total_df["logs_emd"].apply(np.mean)
    total_df["10_mean_logs_emd"] = 10 * total_df["mean_logs_emd"]
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


def generate_heatmap(
    df: pd.DataFrame,
    value: str,
    reverse_colormap: bool = False,
    legend: bool = True,
    auto_color_range: bool = False,
) -> tuple[Figure, Axes]:
    """Create a heatmap from the summary dataframe, using mutation probability and severity
    as axes.

    Args:
        df (pd.DataFrame): The summary dataframe.
        value (str): The value for which to create the heatmap.
        reverse_colormap (bool, optional): Reverse the colormap. This is useful, e.g., when
            using red/green but low is "good". Defaults to False.
        legend (bool, optional): Draw the legend? Defaults to True.
        auto_color_range (bool, optional): Automatically discover the size of the range to
            use for the colormap. Otherwise, use [0,1]. Defaults to False.

    Returns:
        tuple[Figure, Axes]: The heatmap.
    """
    fig, ax = plt.subplots(figsize=(10, 7.5))

    probability_column = "Mutation Probability"
    severity_column = "Standard Deviations Shift"
    plot_df = df.rename(
        columns={
            "mutation_probability": probability_column,
            "mutation_std_shift": severity_column,
        },
    ).pivot(index=probability_column, columns=severity_column, values=value)
    colors = ["g", "r"]
    if reverse_colormap:
        colors = list(reversed(colors))

    v_min_max: dict[str, int | None] = {"vmin": 0, "vmax": 1}
    if auto_color_range:
        v_min_max = {"vmin": None, "vmax": None}

    sns.heatmap(
        plot_df,
        annot=True,
        ax=ax,
        fmt=".2f",
        cmap=LinearSegmentedColormap.from_list("rg", colors, N=256),
        cbar=legend,
        linewidths=0.3,
        **v_min_max,  # type: ignore
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    return fig, ax


def generate_individual_seed_heatmap(seed: int, df: pd.DataFrame, value: str):
    """Create and save a heatmap for a given seed and value using the summary dataframe.

    If the value column is "logs_emd" or "scaled_emd", autoscaling is turned on since the
    range is unknown

    Args:
        seed (int): The seed the summary dataframe corresponds to. Only used to create
            the filename
        df (pd.DataFrame): The summary dataframe.
        value (str): The column name to create the dataframe for.
    """
    auto_range = value in ["logs_emd", "scaled_emd"]
    fig, _ = generate_heatmap(df, value, auto_color_range=auto_range)
    fig.savefig(FIGURES_BASE_DIR / "seeds" / f"{seed}_{value}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_individual_seed_heatmaps(value: str):
    """For a column name, create the heatmap for each seed.

    Args:
        value (str): The column name.
    """
    (FIGURES_BASE_DIR / "seeds").mkdir(parents=True, exist_ok=True)

    summaries = get_summary_dfs()
    for seed, df in summaries.items():
        generate_individual_seed_heatmap(seed, df, value)


def generate_aggregate_heatmaps():
    """Create and save heatmaps for aggregate values from the aggregated summary df.
    Heatmaps created for 'mean_pval', 'percent_correct', 'pval_stdev', 'mean_logs_emd',
    '10_mean_logs_emd'.
    """
    BASE_DIR = FIGURES_BASE_DIR / "aggregate"
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    total_df = aggregate_summary_dfs(get_summary_dfs())

    for column in [
        "mean_pval",
        "percent_correct",
        "pval_stdev",
        "mean_logs_emd",
        "10_mean_logs_emd",
    ]:
        for create_legend in [True, False]:
            legend_str = "" if create_legend else "_no_legend"
            auto_range = False  # column == "pval_stdev"
            auto_range = column in ["pval_stdev", "mean_logs_emd", "10_mean_logs_emd"]
            fig = generate_heatmap(
                total_df,
                column,
                reverse_colormap=column == "percent_correct",
                legend=create_legend,
                auto_color_range=auto_range,
            )[0]
            fig.savefig(BASE_DIR / f"{column}{legend_str}.pdf", bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    if not FIGURES_BASE_DIR.exists():
        FIGURES_BASE_DIR.mkdir()

    print("Generating Service Time Distribution Plots")
    generate_service_time_distribution_plots()
    print("Generating P-Value Heatmaps for each seed")
    generate_individual_seed_heatmaps("pval")
    print("Generating EMD Heatmaps for each seed")
    generate_individual_seed_heatmaps("logs_emd")
    generate_individual_seed_heatmaps("scaled_emd")
    print("Generating Detection Heatmaps for each seed")
    generate_individual_seed_heatmaps("detection")
    print("Generating aggregate heatmaps")
    generate_aggregate_heatmaps()
