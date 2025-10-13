from typing import Optional, Any, List

import pandas as pd


def columns_not_full(
    dataframe: pd.DataFrame,
    column_names: List[str],
    row_mask: Optional[slice | pd.Series] = None
) -> bool:
    """Check whether any of the given columns contain empty values.

    The check can be further refined using a row_mask.

    Args:
        dataframe (pd.DataFrame): pandas DataFrame.
        column_names (List[str]): Column used to check for empty values.
        row_mask (slice | pd.Series[bool] | None): A row mask.

    Returns:
        bool: True if any of the specified columns contain missing values, False otherwise.
    """
    if not set(column_names).issubset(dataframe.columns):
        return True

    df = dataframe.loc[row_mask, column_names] if row_mask is not None else dataframe[column_names]
    return bool(df.isnull().any().any())
