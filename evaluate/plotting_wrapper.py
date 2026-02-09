"""This module contains a class for creating plots."""
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class PlottingWrapper:
    """Utility wrapper for evaluate functions."""

    def __init__(self, personas: list[str]):
        self.color_map = self.create_persona_colormap(personas)
        self.judgment_colors = ["red", "grey", "green"]
        self.judgment_labels = ["No Persona win", "Equal", "Persona win"]

    @staticmethod
    def create_persona_colormap(
        columns: list[str]
    ) -> dict[str, Any]:
        """Create a static color map to use across plots.

        Args:
            columns (list[str]): List of columns to create the color map for.

        Returns:
            dict[str, Any]: Dictionary containing color values.
        """
        cmap = plt.get_cmap("Dark2")
        colors = {col: cmap(i % 8) for i, col in enumerate(columns)}
        return colors

    def _plot_bar(
        self,
        ax: Axes,
        plot_dict: dict[str, Any],
        bar_label: str,
        title: str | None,
        y_label: str = "Accuracy",
        legend: bool = False
    ):
        bars = []
        for column, value in plot_dict.items():
            label = " ".join(
                [c.capitalize() for c in column.replace("_answer_option", "").split("_")]
            )
            if label == "No":
                label = "No Persona"
            plot_bar = ax.bar(column, value, label=label, color=self.color_map[column])
            bars.append([plot_bar, value])

        # Add text labels
        for plot_bar, val in bars:
            for rect in plot_bar:
                y_pos = rect.get_height()
                if val < 0:
                    y_pos = 0
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    y_pos,
                    bar_label.format(val=val),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xlabel("Persona Type", fontsize=12)
        if title is not None:
            ax.set_title(title, fontsize=12, loc="left")
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.spines[["right", "top"]].set_visible(False)
        if legend:
            ax.legend(title="Persona types")

    def plot_abs_accuracy(
        self,
        ax: Axes,
        plot_dict: dict[str, Any],
        bar_label_template: str = "{val:.3f}",
        title: str | None = None,
        y_label: str = "Accuracy",
        legend: bool = True
    ):
        """Plots a bar plot with absolute accuracies.

        Args:
            ax (Axes): Axes object on which the plot will be created.
            plot_dict (dict[str, Any]): Dictionary with values to plot. The keys are used as labels.
            bar_label_template (str, optional): Template to use for the bar label. Defaults to "{val:.3f}".
            title (str, optional): Title of the plot. Defaults to "Accuracy".
            y_label (str, optional): y-axis label. Defaults to "Accuracy".
            legend (bool, optional): If True, a legend is added to the plot. Defaults to True.
        """
        self._plot_bar(
            ax,
            plot_dict,
            bar_label=bar_label_template,
            title=title,
            y_label=y_label,
            legend=legend
        )

    def plot_rel_accuracy(
        self,
        ax: Axes,
        plot_dict: dict[str, Any],
        bar_label_template: str = "{val:+.3f}",
        title: str = "Title",
        y_label: str = "$\Delta$ Accuracy",
        legend: bool = False
    ):
        """Plots a bar plot with relative accuracy differences.

        Args:
            ax (Axes): Axes object on which the plot will be created.
            plot_dict (dict[str, Any]): Dictionary with values to plot. The keys are used as labels.
            bar_label_template (_type_, optional): Template to use for the bar label. Defaults to "{val:+.3f}".
            title (str, optional): Title of the plot. Defaults to "Title".
            y_label (str, optional): y-axis label. Defaults to "Accuracy Gain".
            legend (bool, optional): If True, a legend is added to the plot. Defaults to False.
        """
        self._plot_bar(
            ax,
            plot_dict,
            bar_label=bar_label_template,
            title=title,
            y_label=y_label,
            legend=legend
        )

    def plot_stacked_barchart(
        self,
        ax: Axes,
        plot_dict: dict[str, list[int]],
        cols: list[str],
        y_label: str = "Win Rate",
        title: str | None = None,
    ) -> None:
        """Plot a stacked bar chart.

        Args:
            ax (Axes): Axis object on which the plot will be created.
            plot_dict (dict[str, list[int]]): Dictionary containing plot values.
            cols (list[str]): The columns to plot.
            y_label (str, optional): y-axis label. Defaults to "Percentage".
            title (str, optional): Axis title. Defaults to None.
        """
        assert_cols = [col for col in cols if col in plot_dict]
        for col_idx, col in enumerate(assert_cols):
            bottom = 0
            for val_idx, val in enumerate(plot_dict[col]):
                x_label = " ".join(
                    [c.capitalize() for c in col.replace("_judgment", "").split("_")]
                )
                if x_label == "no":
                    x_label = "empty"
                ax.bar(
                    x_label, val, bottom=bottom,
                    label=self.judgment_labels[val_idx] if col_idx == 0 else "",
                    color=self.judgment_colors[val_idx])

                # add bar label
                if val > 0.05:
                    ax.text(
                        col_idx, bottom + val / 2, f"{val * 100:.1f}%",
                        ha="center", va="center", color="white", fontsize=9)
                bottom += val

        ax.set_ylabel(y_label, fontsize=12)
        ax.set_ylim(0, 1)
        if title is not None:
            ax.set_title(title, fontsize=12, loc="left")
        ax.legend()
        ax.spines[["right", "top"]].set_visible(False)

        for label in ax.get_xticklabels()[1::2]:
            label.set_y(label.get_position()[1] - 0.05)
