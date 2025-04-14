from typing import Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def plot_side_by_side_hists(data: pd.DataFrame, bin_edges: np.ndarray):
    """
    Plot a basic side by side histogram for the labels
    in the data.
    """
    hist_data = {}
    for i, label in enumerate(data.label.unique()):
        lengths = data[data["label"] == label]["_word_counts"]
        hist, _ = np.histogram(lengths, bins=bin_edges)
        hist_data[label] = hist

    bar_width = 10

    for i, label in enumerate(data.label.unique()):
        plt.bar(
            bin_edges[:-1] + i * bar_width,
            hist_data[label],
            width=bar_width,
            label=f'{"negative"*(int(label)==1) + "positive"*(int(label)==0)}',
        )
    _ = plt.legend()
    _ = plt.xlabel("Word count")
    _ = plt.title("Distribution of word counts per class.")
    plt.show()


def _preprocess_run_data(
    run_data: pd.DataFrame,
    separation_criterion_column: str,
    result_grouping_label_column: str,
    line_plot_columns: list[str],
    bar_plot_columns: list[str],
    *,
    out_group_idx_col="group_id",
    out_var_name_col="metric",
    out_value_name="value",
):
    """
    Preprocess the run data into long format:
    Instead of {"model": str, "metric1": float, "metric2": float}, we now have:
    {"model": str, "metric": str, "value": float} such that metric is the var_name param
    i.e. it holds the name of the column, whose value is contained in the "value"
    (value_name param) column. The other values from the original row are repeated
    for all values of the long-formatted columns.

    Args:
        run_data (pd.Dataframe): Dataframe containing metrics and experiment info.
        separation_criterion_column (str): The column, e.g. "model_name" or "experiment_name", by
            which the data is to be ordered in the AAABBBCCC format.
        result_grouping_label_column (str): The column,  e.g. "model_name" or "experiment_name", by
            which a single the group of bar plots in the set of groups in A, B, etc. is distinguished
            from the other groups - i.e. if the separation criterion is model_name, the result grouping
            name can be experiment name.
        line_plot_columns (list[str]): Columns which are to be plotted as lines. In the output of this
            function they will be repeated for all the values of the long-formatted bar_plot_columns.
        bar_plot_columns (list[str]): Columns which are to be long-formatted - i.e. whose name will be
            inside the new "metrics" column and whose value will be in the new "value" column.
        out_group_idx_col (str): How to name the id column to be created, which holds the
            sequential id assigned within the groups of different values of separation_criterion_column.
        out_var_name_col (str): How to name the column holding the names of the long-formatted
            columns (bar_plot_columns).
        out_value_name (str): How to name the column holding the values of the long-formatted
            columns (bar_plot_columns).

    Returns:
        pd.Dataframe in the long format, wich contains the separation_criterion_column
        all the line_plot_columns and also the new "experiment_id", "
    """
    run_data = run_data.copy()
    run_data[out_group_idx_col] = run_data.groupby(
        separation_criterion_column
    ).cumcount()

    run_data = run_data.melt(
        id_vars=[
            separation_criterion_column,
            out_group_idx_col,
            result_grouping_label_column,
        ]
        + line_plot_columns,
        value_vars=bar_plot_columns,
        var_name=out_var_name_col,
        value_name=out_value_name,
    )

    return run_data.sort_values(
        by=[
            separation_criterion_column,
            result_grouping_label_column,
            out_group_idx_col,
            out_var_name_col,
        ]
    ).reset_index(drop=True)


def plot_side_by_side(
    run_data: pd.DataFrame,
    separation_criterion_column: str,
    result_grouping_label_column: str,
    line_plot_columns: list[str],
    bar_plot_columns: list[str],
    *,
    groups_bar_color_palette: dict[dict[str, str]],
    line_plot_colors: list[str],
    output_file: str = None,
    order: Literal["aabb", "abab"] = "aabb",
):
    """
    Given the mlflow runs dataframe from search_runs, build a
    bar plot of the metrics with some of the metrics displayed as
    a line on top of the bars such that the plots can be fully
    separated by a certain criteria- e.g. by model name, experiment name, etc.,
    through vertical lines. Assuming the combinations of the values of
    separation_criterion_column and result_grouping_label_column are unique!
    Args:
        run_data (pd.Dataframe): Dataframe containing metrics and experiment info.
        separation_criterion_column (str): The column, e.g. "model_name" or "experiment_name", by
            which the data is to be ordered in the AAABBBCCC format.
        result_grouping_label_column (str): The column,  e.g. "model_name" or "experiment_name", by
            which a single the group of bar plots in the set of groups in A, B, etc. is distinguished
            from the other groups - i.e. if the separation criterion is model_name, the result grouping
            name can be experiment name.
        line_plot_columns (list[str]): Columns which are to be plotted as lines. In the output of this
            function they will be repeated for all the values of the long-formatted bar_plot_columns.
        bar_plot_columns (list[str]): Columns which are to be long-formatted - i.e. whose name will be
            inside the new "metrics" column and whose value will be in the new "value" column.
        groups_bar_color_palette (dict[dict[str, str]]): Dict holding the color palette per separation
            criterion group per bar plot column, e.g.:
            {
                'Model A': {
                    'recall': '#1f3b73',  # dark blue
                    'precision': '#3f6ebf',  # medium blue
                    'fnr': '#8ab6f2'  # light blue
                },
                'Model B': {
                    'recall': '#1f734a',  # dark green
                    'precision': '#3fbf88',  # medium green
                    'fnr': '#8af2c3'  # light green
                }
            }
        line_plot_colors (list[str]): The color to use for each of the line_plot_columns, e.g.:
            ["red", "blue"] must be of the same length as line_plot_columns.
        output_file (str): Where to save the figure to. If None, fig is not persisted.
        order (str): Determines how the results are going to be grouped - aabb means the datapoints
            related to one value in the separation_criterion_column are plotted before all the
            datapoints related to any other value - hence aabb. If abab - then the order will be:
            a group, related to one value in the separation column and one value in the result_grouping_label_column
            will be followed by a group related to another value in the separation column and
            the same value in the result_grouping_label_column.
    """
    group_idx_col = "__group_id__"
    out_var_name_col = "__metric__"
    out_value_name_col = "__value__"
    palette_key_column = separation_criterion_column

    if order == "abab":
        separation_criterion_column, result_grouping_label_column = (
            result_grouping_label_column,
            separation_criterion_column,
        )

    df_long = _preprocess_run_data(
        run_data,
        separation_criterion_column,
        result_grouping_label_column,
        line_plot_columns,
        bar_plot_columns,
        out_group_idx_col=group_idx_col,
        out_var_name_col=out_var_name_col,
        out_value_name=out_value_name_col,
    )

    df_long["color"] = df_long.apply(
        lambda row: groups_bar_color_palette[row[palette_key_column]][
            row[out_var_name_col]
        ],
        axis=1,
    )

    # Plot setup
    fig, ax = plt.subplots(figsize=(20, 7))
    x = np.arange(len(df_long))

    # Plot bars
    ax.bar(x, df_long[out_value_name_col], color=df_long["color"], edgecolor="black")

    # get the center of each group of bars
    group_centers = []
    x_ticks = []
    x_labels = []

    bar_groups = df_long.groupby(
        [separation_criterion_column, result_grouping_label_column, group_idx_col]
    )
    accumulated_centers = []
    for i, (_, group) in enumerate(bar_groups):
        bar_indices = group.index
        center = bar_indices.values.mean()
        label = group[
            (
                result_grouping_label_column
                if order == "aabb"
                else separation_criterion_column
            )
        ].iloc[0]
        group_centers.append(center)
        accumulated_centers.append(center)
        n_result_groups = run_data[result_grouping_label_column].nunique()
        if order == "aabb":
            x_ticks.append(center)
            x_labels.append(label)

        elif (i + 1) % n_result_groups == 0:
            x_ticks.append(np.array(accumulated_centers).mean())
            accumulated_centers = []
            x_labels.append(label)

    if order == "aabb":
        # line_plot_columns are plotted as lines line
        for column, color in zip(line_plot_columns, line_plot_colors):
            ax.plot(x, df_long[column], color=color, label=column)

        # we want to plot the name of the group above all values for each group only in the aabb

        overall_max_val = float(
            df_long[[out_value_name_col] + line_plot_columns]
            .to_numpy(dtype=float)
            .max()
        )
        y_text = overall_max_val + 0.1 * overall_max_val
        for separation_group_name, group in df_long.groupby(
            separation_criterion_column
        ):
            start = group.index.min()
            end = group.index.max()
            center = (start + end) / 2

            # Label above bars
            ax.text(
                center,
                y_text,
                separation_group_name,
                ha="center",
                va="bottom",
                fontsize=10,
                color="white",
                fontweight="bold",
                bbox=dict(
                    facecolor="black", edgecolor="none", boxstyle="round,pad=0.4"
                ),
            )
    else:
        # line_plot_columns are plotted as dots
        for column, color in zip(line_plot_columns, line_plot_colors):
            # Assuming the combinations of the values of
            # separation_criterion_column and result_grouping_label_column are unique!
            ax.scatter(
                group_centers,
                np.concat(bar_groups[column].unique().values),
                color=color,
                marker="*",
                label=column,
            )

    ax.set_xlim(-0.8, len(df_long) - 0.2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=11)

    # set legend up: the colors of each of the bar_plot_columns are added to the legend
    legend_elements = [
        Line2D([0], [0], color=color, label=metric.replace("metrics.", ""))
        for metric, color in zip(line_plot_columns, line_plot_colors)
    ]

    all_colors = []
    for group_name, group_color_pallette in groups_bar_color_palette.items():
        for metric, color in group_color_pallette.items():
            all_colors.append((f"{group_name}:\n{metric}", color))

    legend_elements += [
        Patch(facecolor=color, edgecolor="black", label=metric.replace("metrics.", ""))
        for metric, color in all_colors
    ]

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
    )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Classification metric value")
    ax.set_title("Side-by-side model classification performance comparison.")

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    plt.show()
