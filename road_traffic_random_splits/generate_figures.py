from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_BASE_PATH = Path("results")
FIGURES_BASE_PATH = Path("figures")


def generate_fpr_plot(result_dir: Path, fpr_column_name: str):
    df = pd.read_csv(result_dir / "summary.csv")

    SIG_LVL_COLUMN = r"Significance Level $\alpha$"
    # FPR_COLUMN = "False Positive Rate"
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


if __name__ == "__main__":
    FIGURES_BASE_PATH.mkdir(exist_ok=True)

    for result_dir in RESULTS_BASE_PATH.iterdir():
        assert result_dir.is_dir()
        print("Creating figures for", result_dir.name)
        generate_fpr_plot(result_dir, "False Positive Rate")
        generate_fpr_plot(result_dir, "Type I Error Rate")
