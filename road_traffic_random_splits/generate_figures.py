from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_BASE_PATH = Path("random_split_results")
FIGURES_BASE_PATH = Path("figures")

CSV_PATH = RESULTS_BASE_PATH / "summary.csv"


def generate_fpr_plot():
    df = pd.read_csv(CSV_PATH)

    SIG_LVL_COLUMN = r"Significance Level $\alpha$"
    FPR_COLUMN = "False Positive Rate"

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

    fig.savefig(FIGURES_BASE_PATH / "fpr.pdf", bbox_inches="tight")


if __name__ == "__main__":
    FIGURES_BASE_PATH.mkdir(exist_ok=True)
    generate_fpr_plot()
