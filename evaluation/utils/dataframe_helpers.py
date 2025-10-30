from typing import List

import pandas as pd


def columns_full(
    dataframe: pd.DataFrame,
    column_names: List[str],
) -> bool:
    """Check whether all values in the given columns are non-null.

    Args:
        dataframe (pd.DataFrame): The pandas DataFrame to check.
        column_names (list[str]): Columns to check for non-null values.

    Returns:
        bool: True if all specified columns have no missing values, False otherwise.
    """
    if not set(column_names).issubset(dataframe.columns):
        return False

    df = dataframe[column_names]
    return bool(df.notnull().all().all())
