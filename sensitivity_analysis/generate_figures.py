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
from pm4py import read_xes

FIGURES_BASE_DIR = Path("figures")


RESULTS_BASE_DIR = Path("road_traffic_synthetic_results")
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


def _num_to_str_truncate(x: float | int):

    if isinstance(x, int):
        return str(x)
    elif x.is_integer():
        return str(int(x))
    else:
        return str(x)


def import_log(log_path: Path) -> pd.DataFrame:
    return cast(pd.DataFrame, read_xes(log_path.as_posix(), variant="rustxes"))


def get_log_paths(seed: int, probability: float, severity: float) -> tuple[Path, Path]:
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
    log_path_1, log_path_2 = get_log_paths(seed, probability, severity)
    log_1 = import_log(log_path_1)
    log_2 = import_log(log_path_2)
    return log_1, log_2


def plot_time_distributions(
    seed: int, probability: float, severity: float, activity: str | None = None
) -> tuple[Figure, Axes]:
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
    df = pd.read_csv(RESULTS_BASE_DIR / str(seed) / "summary.csv")
    df["scaled_emd"] = df["logs_emd"] * 10
    return df


def get_summary_dfs():
    return {
        int(seed.name): get_summary_df(int(seed.name))
        for seed in RESULTS_BASE_DIR.iterdir()
    }


def entropy(items: list[bool]) -> float:
    population = Counter(items)
    pop_len = len(items)
    return -sum(
        (count / pop_len) * log2(count / pop_len) for _, count in population.items()
    )


def aggregate_summary_dfs(summaries: dict[int, pd.DataFrame]) -> pd.DataFrame:
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

    v_min_max = {"vmin": 0, "vmax": 1}
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
    auto_range = value in ["logs_emd", "scaled_emd"]
    fig, _ = generate_heatmap(df, value, auto_color_range=auto_range)
    fig.savefig(FIGURES_BASE_DIR / "seeds" / f"{seed}_{value}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_individual_seed_heatmaps(value: str):
    (FIGURES_BASE_DIR / "seeds").mkdir(parents=True, exist_ok=True)

    summaries = get_summary_dfs()
    for seed, df in summaries.items():
        generate_individual_seed_heatmap(seed, df, value)


def generate_aggregate_heatmaps():
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
