from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_BASE_PATH = Path("results")
FIGURES_BASE_PATH = Path("figures_after")


def generate_fpr_plot(result_dir: Path, fpr_column_name: str):
    """Create and save a plot of the False Positive Rate (FPR) (Or: Type-I Error Rate) as a function
    of the significance level, $\alpha$

    Args:
        result_dir (Path): The path to the results directory.
        fpr_column_name (str): The name to use for the False Positive Rate in the figure.
    """
    df = pd.read_csv(result_dir / "summary.csv")

    SIG_LVL_COLUMN = r"Significance Level $\alpha$"
    FPR_COLUMN = fpr_column_name

    plot_df = pd.DataFrame(
        [
            {
                SIG_LVL_COLUMN: alpha,
                FPR_COLUMN: df[df["pval"] < alpha].shape[0] / df.shape[0],
            }
            for alpha in set(df["pval"].unique()).union([0.0, 1.0])
        ]
    )

    fig, ax = plt.subplots()
    sns.lineplot(data=plot_df, x=SIG_LVL_COLUMN, y=FPR_COLUMN, ax=ax)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend(ax.get_lines(), ["Observed", "Expected"])

    filename = (
        "fpr.pdf"
        if fpr_column_name == "False Positive Rate"
        else f"fpr_{fpr_column_name.lower().replace(' ', '_')}.pdf"
    )

    figure_path = FIGURES_BASE_PATH / result_dir.name / filename
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(figure_path, bbox_inches="tight")


def generate_combined_fpr_plot(
    permutation_test_dir: Path,
    bootstrap_test_dir: Path,
    fpr_column_name: str,
    filename: str,
):
    """Create and save a plot of the False Positive Rate (FPR) (Or: Type-I Error Rate) as a function
    of the significance level, $\alpha$ for both techniques.

    Args:
        permutation_test_dir (Path): The path to the results directory of the permutation test approach.
        bootstrap_test_dir (Path): The path to the results directory of the bootstrap test approach.
        fpr_column_name (str): The name to use for the False Positive Rate in the figure.
        filename (str): The filename to which to save the figure
    """
    permutation_df = pd.read_csv(permutation_test_dir / "summary.csv")
    bootstrap_df = pd.read_csv(bootstrap_test_dir / "summary.csv")
    permutation_df["Technique"] = "Permutation Test"
    bootstrap_df["Technique"] = "Bootstrap Test"
    df = pd.concat([permutation_df, bootstrap_df])

    SIG_LVL_COLUMN = r"Significance Level $\alpha$"
    FPR_COLUMN = fpr_column_name

    plot_df = pd.DataFrame(
        [
            {
                SIG_LVL_COLUMN: alpha,
                FPR_COLUMN: technique_df[technique_df["pval"] < alpha].shape[0]
                / technique_df.shape[0],
                "Technique": technique,
            }
            for technique, technique_df in df.groupby(by="Technique")
            for alpha in set(technique_df["pval"].unique()).union([0.0, 1.0])
        ]
        # Make permutation test be first for consistent colors
    ).sort_values(by="Technique", ascending=False)

    fig, ax = plt.subplots()
    sns.lineplot(data=plot_df, x=SIG_LVL_COLUMN, y=FPR_COLUMN, ax=ax, hue="Technique")
    sns.move_legend(ax, "lower right")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.get_xaxis().get_label().set_fontsize(12)
    ax.get_yaxis().get_label().set_fontsize(12)

    plt.setp(ax.get_legend().get_texts(), fontsize=12)
    plt.setp(ax.get_legend().get_title(), fontsize=12)

    # ax.legend(ax.get_lines(), ["Observed", "Expected"])

    figure_path = FIGURES_BASE_PATH / filename
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(figure_path, bbox_inches="tight")


# def generate_total_fpr_plot(
#     fpr_column_name: str,
#     filename: str,
# ):
#     """Create and save a plot of the False Positive Rate (FPR) (Or: Type-I Error Rate)
#     as a function of the significance level, $\alpha$ for both techniques and both perspectives.
#
#     Creates two plots next to each other, one for control flow, and one for timed-control-flow.
#     The hue is the chosen technique
#
#     Args:
#         fpr_column_name (str): The name to use for the False Positive Rate in the figure.
#         filename (str): The filename to which to save the figure
#     """
#     cf_permutation_df = pd.read_csv(
#         RESULTS_BASE_PATH / "permutation_cf" / "summary.csv"
#     )
#     time_permutation_df = pd.read_csv(
#         RESULTS_BASE_PATH / "permutation_time" / "summary.csv"
#     )
#     cf_bootstrap_df = pd.read_csv(RESULTS_BASE_PATH / "bootstrap_cf" / "summary.csv")
#     time_bootstrap_df = pd.read_csv(
#         RESULTS_BASE_PATH / "bootstrap_time" / "summary.csv"
#     )
#     cf_permutation_df["Technique"] = "Permutation Test"
#     cf_permutation_df["Perspective"] = "Control Flow"
#     time_permutation_df["Technique"] = "Permutation Test"
#     time_permutation_df["Perspective"] = "Timed Control Flow"
#     cf_bootstrap_df["Technique"] = "Bootstrap Test"
#     cf_bootstrap_df["Perspective"] = "Control Flow"
#     time_bootstrap_df["Technique"] = "Bootstrap Test"
#     time_bootstrap_df["Perspective"] = "Timed Control Flow"
#     df = pd.concat(
#         [cf_permutation_df, cf_bootstrap_df, time_permutation_df, time_bootstrap_df]
#     )
#
#     SIG_LVL_COLUMN = r"Significance Level $\alpha$"
#     FPR_COLUMN = fpr_column_name
#     plot_df = pd.DataFrame(
#         [
#             {
#                 SIG_LVL_COLUMN: alpha,
#                 FPR_COLUMN: technique_df[technique_df["pval"] < alpha].shape[0]
#                 / technique_df.shape[0],
#                 "Technique": technique,
#                 "Perspective": perspective,
#             }
#             for (technique, perspective), technique_df in df.groupby(
#                 by=["Technique", "Perspective"]
#             )
#             for alpha in set(technique_df["pval"].unique()).union([0.0, 1.0])
#         ]
#     )
#
#     g = sns.FacetGrid(plot_df, col="Perspective", hue="Technique")
#     g.map(sns.lineplot, SIG_LVL_COLUMN, FPR_COLUMN)
#     fig, ax = plt.subplots()
#     sns.lineplot(data=plot_df, x=SIG_LVL_COLUMN, y=FPR_COLUMN, ax=ax, hue="Technique")
#     sns.move_legend(ax, "lower right")
#     ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
#
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#
#     # ax.legend(ax.get_lines(), ["Observed", "Expected"])
#
#     figure_path = FIGURES_BASE_PATH / filename
#     figure_path.parent.mkdir(parents=True, exist_ok=True)
#
#     fig.savefig(figure_path, bbox_inches="tight")


if __name__ == "__main__":
    FIGURES_BASE_PATH.mkdir(exist_ok=True)

    for result_dir in RESULTS_BASE_PATH.iterdir():
        assert result_dir.is_dir()
        print("Creating figures for", result_dir.name)
        # Create the figure once with the label False Positive Rate, and once with Type I Error Rate
        generate_fpr_plot(result_dir, "False Positive Rate")
        generate_fpr_plot(result_dir, "Type I Error Rate")

    generate_combined_fpr_plot(
        RESULTS_BASE_PATH / "permutation_cf",
        RESULTS_BASE_PATH / "bootstrap_cf",
        "Type I Error Rate",
        "fpr_t1er_cf_combined.pdf",
    )
    generate_combined_fpr_plot(
        RESULTS_BASE_PATH / "permutation_cf",
        RESULTS_BASE_PATH / "bootstrap_cf",
        "Type I Error Rate",
        "fpr_t1er_cf_combined.svg",
    )
    generate_combined_fpr_plot(
        RESULTS_BASE_PATH / "permutation_time",
        RESULTS_BASE_PATH / "bootstrap_time",
        "Type I Error Rate",
        "fpr_t1er_time_combined.pdf",
    )
    generate_combined_fpr_plot(
        RESULTS_BASE_PATH / "permutation_time",
        RESULTS_BASE_PATH / "bootstrap_time",
        "Type I Error Rate",
        "fpr_t1er_time_combined.svg",
    )
