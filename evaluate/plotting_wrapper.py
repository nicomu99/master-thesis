"""This module contains a class for creating plots."""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from inference.utils import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class PlottingWrapper:
    """Utility wrapper for evaluate functions."""

    def __init__(self):
        self.personas = [
            "no", "helpful", "base", "static_short", "static_medium", "static_long", "dynamic_short",
            "dynamic_medium", "dynamic_long", "beginner_teacher", "intermediate_teacher", "expert_teacher"
        ]   # hardcoded to preserve ordering

        self.color_map = self.create_persona_colormap(self.personas)
        self.judgment_colors = ["red", "grey", "green"]
        self.judgment_labels = ["No Persona win", "Equal", "Persona win"]

    @staticmethod
    def create_persona_colormap(
        columns: list[str]
    ) -> dict:
        """Create a static color map to use across plots.

        Args:
            columns (list[str]): List of columns to create the color map for.

        Returns:
            dict: Dictionary containing color values.
        """
        cmap = plt.get_cmap("Dark2")
        colors = {col: cmap(i % 8) for i, col in enumerate(columns)}
        return colors

    def _plot_bar(
        self,
        ax: Axes,
        plot_values: pd.Series,
        bar_label: str,
        title: str | None,
        y_label: str = "Accuracy",
        legend: bool = False,
    ):
        bars = []
        for persona in self.personas:
            if persona not in plot_values.index:
                continue
            value = plot_values[persona]
            label = " ".join(
                [c.capitalize() for c in persona.split("_")]
            )
            if label == "No":
                label = "No Persona"
            plot_bar = ax.bar(persona, value, label=label, color=self.color_map[persona])
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
            ax.set_title(title, fontsize=12, loc="left", y=1.05)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.spines[["right", "top"]].set_visible(False)
        if legend:
            ax.legend(title="Persona types")

    def plot_abs_accuracy(
        self,
        ax: Axes,
        plot_series: pd.Series,
        bar_label_template: str = "{val:.3f}",
        title: str | None = None,
        y_label: str = "Accuracy",
        legend: bool = True
    ):
        """Plots a bar plot with absolute accuracies.

        Args:
            ax (Axes): Axes object on which the plot will be created.
            plot_series (pd.Series): Series of binary outcomes indexed by persona.
            bar_label_template (str, optional): Template to use for the bar label. Defaults to "{val:.3f}".
            title (str, optional): Title of the plot. Defaults to "Accuracy".
            y_label (str, optional): y-axis label. Defaults to "Accuracy".
            legend (bool, optional): If True, a legend is added to the plot. Defaults to True.
        """
        plot_series = plot_series.groupby(level=0).mean()
        self._plot_bar(
            ax,
            plot_series,
            bar_label=bar_label_template,
            title=title,
            y_label=y_label,
            legend=legend
        )

    def plot_rel_accuracy(
        self,
        ax: Axes,
        plot_series: pd.Series,
        baseline_label: str = "no",
        bar_label_template: str = "{val:+.2f}",
        y_label: str = "$\Delta$ Accuracy",
        legend: bool = False,
    ):
        """Plots a bar plot with relative accuracy differences.

        Args:
            ax (Axes): Axes object on which the plot will be created.
            plot_series (pd.Series): Series of binary outcomes indexed by persona.
            baseline_label (str): Persona to use as baseline. Defaults to "no".
            bar_label_template (_type_, optional): Template to use for the bar label. Defaults to "{val:+.3f}".
            y_label (str, optional): y-axis label. Defaults to "Accuracy Gain".
            legend (bool, optional): If True, a legend is added to the plot. Defaults to False.
        """
        values = plot_series.groupby(level=0).mean()
        baseline = values.get(baseline_label)
        if baseline is None or pd.isna(baseline):
            log.warning("Baseline persona %s missing in plot data", baseline_label)
            baseline = 0.0
        plot_series = values - baseline
        plot_series = plot_series.drop(baseline_label)

        title = f"Gain vs. {baseline_label.capitalize()} Persona"
        self._plot_bar(
            ax,
            plot_series,
            bar_label=bar_label_template,
            title=title,
            y_label=y_label,
            legend=legend,
        )

    def create_accuracy_plots(
        self,
        dataset_name: str,
        category_col: str | None = None,
        baseline_col_1: str = "no",
        baseline_col_2: str = "helpful",
    ):
        """Create per-model accuracy figures from a long-format CSV.

        The CSV is expected at `data/evaluation/{dataset_name}.csv` with columns
        "persona", "model", "category", and "score". For each model, this
        creates one figure with one row per category and three columns: absolute
        accuracy and relative accuracy vs two baselines. It also creates an
        additional per-model figure aggregated across all categories.

        Args:
            dataset_name (str): Base filename (without extension) to load.
            category_col (str, optional): Column name for categories. Defaults to None.
            baseline_col_1 (str, optional): Persona baseline for the first relative plot.
                Defaults to "no".
            baseline_col_2 (str, optional): Persona baseline for the second relative plot.
                Defaults to "helpful".
        """
        df = pd.read_csv(f"data/evaluation/{dataset_name}.csv", index_col=0)
        if dataset_name == "alpaca":
            df["score"] = df["score"] >= 1.5

        for model, model_df in df.groupby("model"):
            overall_series = model_df.set_index("persona")["score"]
            fig_overall, axes_overall = plt.subplots(
                nrows=1, ncols=3, figsize=(15, 4)
            )
            self.plot_abs_accuracy(
                axes_overall[0], overall_series,
                title=f"Absolute Accuracy",
            )
            self.plot_rel_accuracy(axes_overall[1], overall_series, baseline_col_1)
            self.plot_rel_accuracy(axes_overall[2], overall_series, baseline_col_2)
            fig_overall.suptitle(
                f"Cumulative Accuracy and Relative Gains for {dataset_name}",
            )
            fig_overall.tight_layout()

            save_path = Path(f"plots/{dataset_name}/")
            save_path.mkdir(parents=True, exist_ok=True)
            fig_overall.savefig(save_path / f"{model}_overall", dpi=300)
            plt.close(fig_overall)
            if category_col is None:
                continue

            num_categories = len(model_df[category_col].unique())
            fig, axes = plt.subplots(
                nrows=num_categories, ncols=3,
                figsize=(15, num_categories * 4)
            )

            for idx, (category, cat_df) in enumerate(model_df.groupby(category_col)):
                series = cat_df.set_index("persona")["score"]

                axis = axes[idx]
                legend = idx == 0
                self.plot_abs_accuracy(axis[0], series, title=f"Absolute Accuracy for Task {category}", legend=legend)
                self.plot_rel_accuracy(axis[1], series, baseline_col_1)
                self.plot_rel_accuracy(axis[2], series, baseline_col_2)

            fig.suptitle(f"Accuracy and Relative Gains for {dataset_name}", y=0.99)
            fig.tight_layout()
            fig.savefig(save_path / f"{model}_per_category", dpi=300)
            plt.close(fig)

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
