"""This module contains preprocessing functions for creating plots."""
from typing import Any

import pandas as pd


def compute_accuracy_df(
    df: pd.DataFrame,
    columns: list[str],
    ground_truth_col: str = "answer",
    category_col: str = "category"
) -> pd.DataFrame:
    """Computes accuracies against ground truth.

    Args:
        df (pd.DataFrame): DataFrame instance.
        columns (list[str]): Columns on which to compute the accuracy on.
        ground_truth_col (str, optional): The column containing the ground truth. Defaults to "answer".
        category_col (str, optional): The column on which to group instances on. Defaults to "category".

    Returns:
        pd.DataFrame: A DataFrame containing computed accuracies.
    """
    accuracy_rows = []
    for category, group in df.groupby(category_col):
        row = {category_col: category}
        for col in columns:
            row[col] = (group[ground_truth_col] == group[col]).mean()
        accuracy_rows.append(row)
    return pd.DataFrame(accuracy_rows)


def compute_accuracy_difference(
    df: pd.DataFrame,
    reference_column: str,
    drop_cols: list[str],
    category_col: str = "category"
) -> pd.DataFrame:
    """Computes accuracy difference against a reference column.

    Args:
        df (pd.DataFrame): DataFrame instance containing accuracy values.
        reference_column (str): The name of the baseline used as a reference.
        drop_cols (list[str]): List of columns to drop in the resulting DataFrame.
        category_col (str, optional): The column on which to group instances on. Defaults to "category".

    Returns:
        pd.DataFrame: A DataFrame containing computed accuracy differences.
    """
    gains_vs_reference = df.copy()
    for col in df.columns:
        if col in (category_col, reference_column):
            continue
        gains_vs_reference[col] = df[col] - df[reference_column]
    drop_cols.extend([reference_column])
    gains_vs_reference.drop(columns=drop_cols)
    return gains_vs_reference


def get_plot_dict(
    accuracy_df: pd.DataFrame,
    category_name: str,
    category_col: str = "category",
    ignore_cols: list[str] | None = None
) -> dict[str, Any]:
    """Returns a dictionary with persona types as keys and accuracies as values.

    Args:
        accuracy_df (pd.DataFrame): DataFrame containing accuracies per persona type.
        category_name (str): String identifier of the category for which to return the dictionary for.
        category_col (str, optional): Column identifier where the category is stored. Defaults to "category".
        ignore_cols (list[str] | None, optional): Columns that should not be in the resulting dictionary.
            Defaults to None.

    Returns:
        dict[str, Any]: Dictionary with persona types as keys and accuracies as values
    """
    if ignore_cols is None:
        ignore_cols = [category_col]
    else:
        ignore_cols.append(category_col)
    row = accuracy_df.loc[accuracy_df[category_col] == category_name].iloc[0]

    plot_dict = {}
    for category in row.index:
        if category in ignore_cols:
            continue
        plot_dict[category] = row[category]
    return plot_dict


def get_plot_dict_stacked(
    df: pd.DataFrame,
    columns: list[str],
    category_col: str = "category",
) -> dict[str, dict[str, list[int]]]:
    """Returns a dictionary for creating stacked bar charts.

    The dictionary has format:

    {
        category_1: {
            column_a: [percentage_ref, percentage_tied, percentage_persona],
        },
        ...
    }

    Args:
        df (pd.DataFrame): DataFrame with plot data.
        columns (list[str]): The columns for which to calculate the percentages for.
        category_col (str, optional): The dataframe column containing categorical identifiers. Defaults to "category".

    Returns:
        dict[str, dict[str, list[int]]]: A dictionary with one entry per category. Each entry has another dictionary
            containing persona identifiers and lists with percentage values of win rates.
    """
    labels_order = ["reference", "both", "persona"]
    accuracy_dict = {}
    for category, group in df.groupby(category_col):
        category_values = {}
        for col in columns:
            if col not in group.columns:
                continue

            valid = group[group[col] != "no response"][col]
            if valid.empty:
                shares = [0.0, 0.0, 0.0]
            else:
                shares = (
                    valid.value_counts(normalize=True)
                    .rename(index={0: "reference", 1: "both", 2: "persona"})
                    .reindex(labels_order, fill_value=0.0)
                    .tolist()
                )
            category_values[col] = shares
        accuracy_dict[category] = category_values
    return accuracy_dict
