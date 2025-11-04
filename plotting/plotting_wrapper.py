from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class PlottingWrapper:
    """Utility wrapper for plotting functions."""

    def __init__(self, personas: list[str]):
        self.color_map = self.create_persona_colormap(personas)

    @staticmethod
    def create_persona_colormap(
        columns: list[str]
    ) -> dict[str, Any]:
        """Create a static color map to use across plots.

        Args:
            columns (list[str]): List of columns to create the color map for.

        Returns:
            dict[str, Any]: Dictionary containig color values.
        """
        cmap = plt.get_cmap("tab10")
        colors = {col: cmap(i % 10) for i, col in enumerate(columns)}
        return colors

    def _plot_bar(
        self,
        ax: Axes,
        plot_dict: dict[str, Any],
        bar_label: str,
        title: str = "Title",
        y_label: str = "Accuracy",
        legend: bool = False
    ):
        bars = []
        for column, value in plot_dict.items():
            label = " ".join(column.replace("_answer_option", "").split("_"))
            if label == "no":
                label = "empty"
            plot_bar = ax.bar(column, value, label=label, color=self.color_map[column])
            bars.append([plot_bar, value])

        # Add text labels
        for plot_bar, val in bars:
            for rect in plot_bar:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height(),
                    bar_label.format(val=val),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_ylabel(y_label)
        ax.set_xlabel("")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_xticklabels([])
        if legend:
            ax.legend(title="Persona types")

    def plot_abs_accuracy(
        self,
        ax: Axes,
        plot_dict: dict[str, Any],
        bar_label_template: str = "{val:.3f}",
        title: str = "Title",
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
        y_label: str = "Accuracy Gain",
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
