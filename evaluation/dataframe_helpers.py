from typing import Optional, Any, List

import pandas as pd


def add_empty_column(
        dataframe: pd.DataFrame,
        column_name: str,
        default_value: Optional[Any] = None
):
    """Helper function for adding columns to pandas dataframes

    Adds a new column column_name to the dataframe, if it does not exist yet.

    Args:
        dataframe (pd.DataFrame): pandas DataFrame.
        column_name: The name of the newly created column.
        default_value: The value inserted into the column

    Returns:
        pd.DataFrame: Dataframe with newly created column.
    """
    if column_name not in dataframe.columns:
        dataframe[column_name] = default_value


def construct_row_mask(
    dataframe: pd.DataFrame,
    mask_column: str | None,
    mask: str | None,
) -> slice | pd.Series:
    """_summary_

    Args:
        dataframe (pd.DataFrame): _description_
        mask_column (str | None): _description_
        mask (str | None): _description_

    Returns:
        slice | pd.Series[bool]: _description_
    """
    if not mask_column or not mask:
        return slice(None)      # Selects all rows
    return dataframe[mask_column] == mask

def get_with_row_mask(
    dataframe: pd.DataFrame,
    mask_column: str | None,
    mask: str | None,
):
    row_mask = construct_row_mask(dataframe, mask_column, mask)
    return dataframe[row_mask]


def insert_if_empty(
    dataframe: pd.DataFrame,
    insert_column: str,
    insert_value: Any,
    row_mask: Optional[slice | pd.Series] = None
):
    """Inserts a static value to rows in a certain column.

    The insertion can be further refined using ``mask`` and ``mask_column``. If both are set, only rows containing
    a value ``mask`` in column ``mask_column`` will be updated.

    Args:
        dataframe (pd.DataFrame): pandas DataFrame.
        insert_column (str): String identifier of the column where the new value shall be inserted.
        insert_value (Any): The value to write into the column.
        row_mask (slice | pd.Series[bool]): A row mask.
    """
    if row_mask is not None:
        if dataframe.loc[row_mask, insert_column].isnull().any():
            # Only insert if some rows have an empty base persona
            dataframe.loc[row_mask, insert_column] = insert_value
    else:
        if dataframe.loc[:, insert_column].isnull().any():
            # Only insert if some rows have an empty base persona
            dataframe.loc[:, insert_column] = insert_value


def column_not_full(
    dataframe: pd.DataFrame,
    column_name: str,
    row_mask: Optional[slice | pd.Series] = None
) -> bool:
    """Checks whether a column contains empty values.

    The check can be further refined using ``mask`` and ``mask_column``. If both are set, only rows containing
    a value ``mask`` in column ``mask_column`` will be checked.

    Args:
        dataframe (pd.DataFrame): pandas DataFrame.
        column_name (str): Column used to check for empty values.
        row_mask (slice | pd.Series[bool]): A row mask.

    Returns:
        bool: Returns true if any row in ``column_name`` is empty. Else false.
    """
    if column_name not in dataframe.columns:
        return True

    if row_mask is not None:
        return bool(dataframe.loc[row_mask, column_name].isnull().any())
    return bool(dataframe.loc[column_name].isnull().any())

def columns_not_full(
    dataframe: pd.DataFrame,
    column_names: List[str],
    row_mask: Optional[slice | pd.Series] = None
) -> bool:
    """Check whether any of the given columns contain empty values.

    The check can be further refined using a row_mask.

    Args:
        dataframe (pd.DataFrame): pandas DataFrame.
        column_name (List[str]): Column used to check for empty values.
        row_mask (slice | pd.Series[bool] | None): A row mask.

    Returns:
        bool: True if any of the specified columns contain missing values, False otherwise.
    """
    if not set(column_names).issubset(dataframe.columns):
        return True

    df = dataframe.loc[row_mask, column_names] if row_mask is not None else dataframe[column_names]
    return bool(df.isnull().any().any())


def get_unique_value(
    dataframe: pd.DataFrame,
    column_name: str,
    row_mask: slice | pd.Series
) -> str:
    """Returns the first unique value in column ``column_name``.

    The check can be further refined using ``mask`` and ``mask_column``. If both are set, only rows containing
    a value ``mask`` in column ``mask_column`` will be checked.

    Args:
        dataframe (pd.DataFrame): pandas DataFrame.
        column_name (str): Column identifier.
        row_mask (slice | pd.Series[bool]): A row mask.

    Returns:
        Any: The first unique value in column ``column_name``.
    """
    return dataframe.loc[row_mask, column_name].unique()[0]
