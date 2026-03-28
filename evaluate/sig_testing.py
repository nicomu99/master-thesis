"""Module for significance testing."""
from typing import Any, Literal

import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.proportion import binom_test

_DYNAMIC_LEN_COLS = ["base", "dynamic_short", "dynamic_medium", "dynamic_long"]
_STATIC_LEN_COLS = ["base", "static_short", "static_medium", "static_long"]


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
    value_vars = [col for col in value_vars if col in df.columns]
    if id_vars is None:
        id_vars = ["static_id"]
    df = pd.melt(df, id_vars=id_vars, var_name=var_name, value_name=value_name, value_vars=value_vars)
    if format_mode == "len":
        df = len_formatting(df, var_name, value_vars)
    return df


def len_formatting(
    df: pd.DataFrame,
    mode: Literal["static", "dynamic"]
) -> pd.DataFrame:
    """Updates the values in `column`. The values in `cols` are turned into the respective values in `lengths`.

    Args:
        df (pd.DataFrame): DataFrame on which the update will happen.
        mode: Literal["static", "dynamic"]

    Returns:
        pd.DataFrame: A dataframe with updated column.
    """
    if mode == "static":
        vals = _STATIC_LEN_COLS
    elif mode == "dynamic":
        vals = _STATIC_LEN_COLS
    else:
        raise ValueError(f"Unknown mode {mode}")
    df = df.loc[df["persona"] in vals]
    lengths = [1, 3, 5, 10]
    for idx, e_col in zip(lengths, vals):
        df.loc[df["persona"] == e_col, "persona"] = idx
    df["length"] = df["persona"].astype(float)
    return df


def lin_test(
        df: pd.DataFrame,
) -> Any:
    """Fit a linear mixed effects model with the reference as treatment variable and grouped by the question id.

    Args:
        df (pd.DataFrame): DataFrame in long format containing.

    Returns:
        The fitted model.
    """
    lin_model = smf.mixedlm(
        "correct ~ C(persona, Treatment(reference='no'))",
        data=df, groups=df["static_id"])

    return lin_model.fit()


def test_binary_baseline(
    df: pd.DataFrame
) -> Any:
    """Fit a linear mixed effects model with the reference as treatment variable and grouped by the question id.

    Args:
        df (pd.DataFrame): DataFrame in long format containing.

    Returns:
        The fitted model.
    """
    model = smf.logit("score ~ C(persona, Treatment(reference='no')) + C(model)", data=df)
    return model.fit()


def test_binary_length(
    df: pd.DataFrame,
    mode: Literal["static", "dynamic"]
):
    df = len_formatting(df, mode)
    model = smf.logit(
        "score ~ length + C(model)",
        data=df
    )
    return model.fit()


def test_ordinal_baseline(
    df: pd.DataFrame
):
    model = OrderedModel.from_formula(
        "score ~ C(persona)",
        data=df,
        groups=df["model"],
        distr="logit"
    )

    return model.fit(method="bfgs")


def test_ordinal_length(
    df: pd.DataFrame,
    mode: Literal["static", "dynamic"]
):
    df = len_formatting(df, mode)
    model = OrderedModel.from_formula(
        "score ~ length + C(model)",
        data=df,
        distr="logit"
    )
    return model.fit(method="bfgs")


def test_numeric_baseline(
    df: pd.DataFrame
):
    model = smf.mixedlm("score ~ C(persona, Treatment(reference='no'))", data=df, groups=df["model"])
    # model = smf.logit("score ~ C(persona) + C(model)", data=df)
    return model.fit()


def test_numeric_length(
    df: pd.DataFrame,
    mode: Literal["static", "dynamic"]
):
    df = len_formatting(df, mode)
    model = smf.mixedlm(
        "score ~ length",
        data=df,
        groups=df["model"]
    )
    return model.fit()


def run_test_categorical(
        df: pd.DataFrame,
        pred_cols: list[str],
        category_id: str,
        category_col: str = "category"
):
    """Runs a full test pipeline on the categorical persona strings.

    Args:
        df (pd.DataFrame): Dataframe with correctness indicators.
        pred_cols (list[str]): The columns to be used as predictors.
        category_id (str): Question category identifier.
        category_col (str): Dataframe column to be used.

    Returns:
        pd.Series: p-Values of the predictors.
    """
    df = df.loc[df[category_col] == category_id]
    df = prepare_data(df, pred_cols)
    results = lin_test(df)
    results = results.pvalues

    # remove intercept
    del results["Intercept"]
    del results["Group Var"]

    # clean results index
    results.index = (
        results.index
        .str.extract(r"\[(T\.[^\]]+)\]")[0]
        .str.replace(r'T\.', "", regex=True)
        .str.replace("_answer_option", "", regex=False)
    )
    return results


def _length_test(
        df: pd.DataFrame,
        category_id: str,
        length_cols: list[str],
        category_col: str = "category",
        dataset_preprocess: str = "none"
):
    df = df.loc[df[category_col] == category_id]
    df = prepare_data(df, length_cols, format_mode="len")
    if dataset_preprocess == "flores":
        df = df[df["correct"] != 1].copy()
        df["correct"] = (df["correct"] == 2).astype(int)

    results = test_binary_baseline(df)
    results = results.pvalues

    # remove intercept
    del results["Intercept"]
    del results["Group Var"]
    return results


def run_test_dynamic(
        df: pd.DataFrame,
        category_id: str,
        category_col: str = "category",
        dataset_preprocess: str = "none"
):
    """Runs a full test pipeline on the length of dynamic persona strings.

    Args:
        df (pd.DataFrame): Dataframe with correctness indicators.
        category_id (str): Question category identifier.
        category_col (str): Dataframe column to be used.
        dataset_preprocess (str): Preprocessing indicator. Use "flores" for the FLores dataset.

    Returns:
        pd.Series: p-Values of the predictors.
    """
    length_cols = ["base", "dynamic_short", "dynamic_long"]
    return _length_test(df, category_id, length_cols, category_col, dataset_preprocess)


def run_test_static(
        df: pd.DataFrame,
        category_id: str,
        category_col: str = "category",
        dataset_preprocess: str = "none"
):
    """Runs a full test pipeline on the length of static persona strings.

    Args:
        df (pd.DataFrame): Dataframe with correctness indicators.
        category_id (str): Question category identifier.
        category_col (str): Dataframe column to be used.
        dataset_preprocess (str): Preprocessing indicator. Use "flores" for the FLores dataset.

    Returns:
        pd.Series: p-Values of the predictors.
    """
    length_cols = ["base", "static_short", "static_long"]
    return _length_test(df, category_id, length_cols, category_col, dataset_preprocess)


def test_flores(
        df: pd.DataFrame,
        pred_cols: list[str],
        category_id: str,
        category_col: str = "iso_639_3"
):
    """Runs a test comparing the reference vs. other predictor columns on the flores dataset.

    Args:
        df (pd.DataFrame): Dataframe containing predictor variables.
        pred_cols (list[str]): Columns to use as predictor columns.
        category_id (str): String identifier of the category.
        category_col (str): String identifier of the category column.

    Returns:
        pd.Series: p-Values of the predictors.
    """
    df = df.loc[df[category_col] == category_id]
    df = prepare_data(df, pred_cols)  # , id_vars=[category_col])
    df = df[df["correct"] != 1].copy()
    df["correct"] = (df["correct"] == 2).astype(int)

    results = {"personas": [], "p-Value": []}
    for persona, group in df.groupby("persona"):
        wins = group["correct"].sum()
        n = len(group)

        p_val = binom_test(count=wins, nobs=n, alternative="larger")
        results["personas"].append(persona)
        results["p-Value"].append(p_val)

    return pd.Series(results["p-Value"], index=results["personas"])


def test_static_vs_dynamic(
    df: pd.DataFrame,
    category_id: str,
    category_col: str = "category",
    dataset_preprocess: str = "none"
):
    """Runs a test comparing the static vs. dynamic predictors.

    Args:
        df (pd.DataFrame): Dataframe containing predictor variables.
        category_id (str): String identifier of the category.
        category_col (str): String identifier of the category column.
        dataset_preprocess (str): Preprocessing indicator. Use "flores" for the FLores dataset.

    Returns:
        pd.Series: p-Values of the predictors.
    """
    df = df.loc[df[category_col] == category_id].copy()
    dep_cols = ["base", "dynamic_short", "dynamic_long",
                "static_short", "static_long"]
    df = prepare_data(df, dep_cols)
    if dataset_preprocess == "flores":
        df = df[df["correct"] != 1].copy()
        df["correct"] = (df["correct"] == 2).astype(int)

    # if "dynamic" in persona columns -> add identifier
    df["is_dynamic"] = df["persona"].str.contains("dynamic")

    df["correct"] = df["correct"].astype(int)
    df["is_dynamic"] = df["is_dynamic"].astype(int)

    model = smf.mixedlm("correct ~ is_dynamic", df, groups=df["static_id"])
    results = model.fit().pvalues

    del results["Group Var"]
    del results["Intercept"]
    return results


def test_vs_baseline(
    df: pd.DataFrame,
    baseline_col: str,
    persona_cols: list[str]
):
    results = []

    for col in persona_cols:
        stat, p = stats.wilcoxon(df[baseline_col], df[col])
        results.append({
            "persona": col,
            "statistic": stat,
            "p_value": p
        })

    return results
