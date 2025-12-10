from typing import Any

import pandas as pd
import statsmodels.formula.api as smf


def prepare_data(
        df: pd.DataFrame,
        value_vars: list[str],
        var_name: str = "persona",
        value_name: str = "correct",
        id_vars: list[str] | None = None,
        format_mode: str = "none"
) -> pd.DataFrame:
    """Refactors the data to correctly fit the requirements of a linear mixed model.

    Args:
        df (pd.DataFrame): The dataframe to update.
        value_vars (list[str]): Column identifiers, which will be turned into a single column.
        var_name (str): The name of the newly created column. Defaults to "persona".
        value_name (str): The name of the value column. Defaults to "correct"
        id_vars (list[str]): Column identifiers of the identification column. Defaults to ["static_id"]
        format_mode (str): String identifier of possible formatting. If this is "len",
            the variable column will be turned into a numeric type. Defaults to "none".

    Returns:
        pd.DataFrame: An updated DataFrame.
    """
    if id_vars is None:
        id_vars = ["static_id"]
    df = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
    if format_mode == "len":
        df = len_formatting(df, var_name, value_vars)
    return df


def len_formatting(
        df: pd.DataFrame,
        column: str,
        vals: list[str],
        lengths: list[int] | None = None
) -> pd.DataFrame:
    """Updates the values in `column`. The values in `cols` are turned into the respective values in `lengths`.

    Args:
        df (pd.DataFrame): DataFrame on which the update will happen.
        column (str): Column identifier of the updated column.
        vals (list[str]): Old values in the column.
        lengths (list[int]): Newly inserted values. Defaults to [1, 3, 10].

    Returns:
        pd.DataFrame: A dataframe with updated column.
    """
    if lengths is None:
        lengths = [1, 3, 10]
    for idx, e_col in zip(lengths, vals):
        df.loc[df[column] == e_col, column] = idx
    df[column] = df[column].astype(float)
    return df


def test(
        df: pd.DataFrame,
        group_col: str = "static_id"
) -> Any:
    lin_model = smf.mixedlm("correct ~ persona", data=df, groups=df[group_col])
    return lin_model.fit()


def run_test(
        df: pd.DataFrame,
        value_vars: list[str],
        format_mode: str = "none"
) -> Any:
    df = prepare_data(df, value_vars, format_mode=format_mode)
    return test(df)
