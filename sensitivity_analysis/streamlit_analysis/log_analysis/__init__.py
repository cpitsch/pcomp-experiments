import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.axes import Axes
from pcomp.emd.emd import add_duration_column_to_log


def get_service_times(log: pd.DataFrame, activity: pd.DataFrame) -> list[float]:
    log = add_duration_column_to_log(log, duration_key="@pcomp:duration")
    return log[log["concept:name"] == activity]["@pcomp:duration"].tolist()


def plot_distributions_plotly(log_1: pd.DataFrame, log_2: pd.DataFrame) -> go.Figure:
    log_1["Log Type"] = "Before Change"
    log_2["Log Type"] = "After Change"

    plot_log = pd.concat(
        [add_duration_column_to_log(log_1), add_duration_column_to_log(log_2)]
    )
    plot_log["Duration [h]"] = plot_log["@pcomp:duration"] / 3600
    fig = px.histogram(plot_log, x="Duration [h]", color="Log Type")
    return fig


def plot_distributions_seaborn(log_1: pd.DataFrame, log_2: pd.DataFrame) -> Axes:
    log_1["Log Type"] = "Before Change"
    log_2["Log Type"] = "After Change"

    plot_log = pd.concat(
        [add_duration_column_to_log(log_1), add_duration_column_to_log(log_2)]
    )
    plot_log["Duration [h]"] = plot_log["@pcomp:duration"] / 3600
    return sns.histplot(
        plot_log,
        x="Duration [h]",
        hue="Log Type",
        # kde=True,
    )
