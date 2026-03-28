"""Module for significance testing."""
from typing import Any, Literal

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel

_DYNAMIC_LEN_COLS = ["base", "dynamic_short", "dynamic_medium", "dynamic_long"]
_STATIC_LEN_COLS = ["base", "static_short", "static_medium", "static_long"]
_TEACHER_PERSONAS = ["beginner_teacher", "intermediate_teacher", "expert_teacher"]
_STATIC_PERSONAS = ["static_short", "static_medium", "static_long"]
_DYNAMIC_PERSONAS = ["dynamic_short", "dynamic_medium", "dynamic_long"]
_BASELINE_REFERENCE = "no"


def _baseline_reference(df: pd.DataFrame) -> str:
    """Return the baseline reference persona for the given DataFrame.

    Prefers 'no'; falls back to 'base' if 'no' is absent (e.g. flores).
    """
    if _BASELINE_REFERENCE in df["persona"].values:
        return _BASELINE_REFERENCE
    return "base"


def len_formatting(
    df: pd.DataFrame,
    mode: Literal["static", "dynamic"]
) -> pd.DataFrame:
    """Filters to the length variants of `mode` and adds a numeric `length` column.

    Args:
        df (pd.DataFrame): DataFrame with a "persona" column.
        mode: "static" or "dynamic"

    Returns:
        pd.DataFrame: Filtered dataframe with a numeric `length` column added.
    """
    if mode == "static":
        vals = _STATIC_LEN_COLS
    elif mode == "dynamic":
        vals = _DYNAMIC_LEN_COLS
    else:
        raise ValueError(f"Unknown mode {mode}")
    df = df.loc[df["persona"].isin(vals)].copy()
    lengths = [1, 3, 5, 10]
    for length, col in zip(lengths, vals):
        df.loc[df["persona"] == col, "persona"] = length
    df["length"] = df["persona"].astype(float)
    return df


def test_binary_baseline(
    df: pd.DataFrame
) -> Any:
    """Logistic regression: all personas vs baseline, controlling for model.

    Uses 'no' as reference when present, otherwise 'base' (e.g. for flores, where
    the score already encodes comparison against the baseline via LLM-as-a-judge).

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns score, persona, model.

    Returns:
        The fitted model.
    """
    ref = _baseline_reference(df)
    model = smf.logit(f"score ~ C(persona, Treatment(reference='{ref}')) + C(model)", data=df)
    return model.fit()


def test_ordinal_baseline(
    df: pd.DataFrame
) -> Any:
    """Ordered logit: all personas vs baseline, controlling for model.

    Uses 'no' as reference when present, otherwise 'base' (e.g. for flores, where
    the score already encodes comparison against the baseline via LLM-as-a-judge).

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns score, persona, model.

    Returns:
        The fitted model.
    """
    ref = _baseline_reference(df)
    model = OrderedModel.from_formula(
        f"score ~ C(persona, Treatment(reference='{ref}')) + C(model)",
        data=df,
        distr="logit"
    )
    return model.fit(method="bfgs")


def test_numeric_baseline(
    df: pd.DataFrame
) -> Any:
    """Linear mixed model: all personas vs baseline, with model as random grouping.

    Uses 'no' as reference when present, otherwise 'base' (e.g. for flores, where
    the score already encodes comparison against the baseline via LLM-as-a-judge).

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns score, persona, model.

    Returns:
        The fitted model.
    """
    ref = _baseline_reference(df)
    model = smf.mixedlm(
        f"score ~ C(persona, Treatment(reference='{ref}'))", data=df, groups=df["model"]
    )
    return model.fit()


def test_binary_length(
    df: pd.DataFrame,
    mode: Literal["static", "dynamic"]
) -> Any:
    """Logistic regression: effect of length on binary score, controlling for model.

    Args:
        df (pd.DataFrame): Long-format DataFrame.
        mode: "static" or "dynamic"

    Returns:
        The fitted model.
    """
    df = len_formatting(df, mode)
    model = smf.logit("score ~ length + C(model)", data=df)
    return model.fit()


def test_ordinal_length(
    df: pd.DataFrame,
    mode: Literal["static", "dynamic"]
) -> Any:
    """Ordered logit: effect of length on ordinal score, controlling for model.

    Args:
        df (pd.DataFrame): Long-format DataFrame.
        mode: "static" or "dynamic"

    Returns:
        The fitted model.
    """
    df = len_formatting(df, mode)
    model = OrderedModel.from_formula(
        "score ~ length + C(model)",
        data=df,
        distr="logit"
    )
    return model.fit(method="bfgs")


def test_numeric_length(
    df: pd.DataFrame,
    mode: Literal["static", "dynamic"]
) -> Any:
    """Linear mixed model: effect of length on numeric score, with model as random grouping.

    Args:
        df (pd.DataFrame): Long-format DataFrame.
        mode: "static" or "dynamic"

    Returns:
        The fitted model.
    """
    df = len_formatting(df, mode)
    model = smf.mixedlm("score ~ length", data=df, groups=df["model"])
    return model.fit()


def test_binary_teacher(
    df: pd.DataFrame
) -> Any:
    """Logistic regression: teacher persona audience level effect on binary score.

    Uses beginner_teacher as reference. Controls for model.

    Args:
        df (pd.DataFrame): Long-format DataFrame.

    Returns:
        The fitted model.
    """
    df = df.loc[df["persona"].isin(_TEACHER_PERSONAS)].copy()
    model = smf.logit(
        "score ~ C(persona, Treatment(reference='beginner_teacher')) + C(model)",
        data=df
    )
    return model.fit()


def test_ordinal_teacher(
    df: pd.DataFrame
) -> Any:
    """Ordered logit: teacher persona audience level effect on ordinal score.

    Uses beginner_teacher as reference. Controls for model.

    Args:
        df (pd.DataFrame): Long-format DataFrame.

    Returns:
        The fitted model.
    """
    df = df.loc[df["persona"].isin(_TEACHER_PERSONAS)].copy()
    model = OrderedModel.from_formula(
        "score ~ C(persona, Treatment(reference='beginner_teacher')) + C(model)",
        data=df,
        distr="logit"
    )
    return model.fit(method="bfgs")


def test_numeric_teacher(
    df: pd.DataFrame
) -> Any:
    """Linear mixed model: teacher persona audience level effect on numeric score.

    Uses beginner_teacher as reference. Random intercepts per model.

    Args:
        df (pd.DataFrame): Long-format DataFrame.

    Returns:
        The fitted model.
    """
    df = df.loc[df["persona"].isin(_TEACHER_PERSONAS)].copy()
    model = smf.mixedlm(
        "score ~ C(persona, Treatment(reference='beginner_teacher'))",
        data=df,
        groups=df["model"]
    )
    return model.fit()


def _prepare_static_vs_dynamic(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to static/dynamic length variants and add binary is_dynamic column."""
    personas = _STATIC_PERSONAS + _DYNAMIC_PERSONAS
    df = df.loc[df["persona"].isin(personas)].copy()
    df["is_dynamic"] = df["persona"].str.startswith("dynamic").astype(int)
    return df


def test_binary_static_vs_dynamic(
    df: pd.DataFrame
) -> Any:
    """Logistic regression: static vs. dynamic effect on binary score, controlling for model.

    Args:
        df (pd.DataFrame): Long-format DataFrame.

    Returns:
        The fitted model.
    """
    df = _prepare_static_vs_dynamic(df)
    model = smf.logit("score ~ is_dynamic + C(model)", data=df)
    return model.fit()


def test_ordinal_static_vs_dynamic(
    df: pd.DataFrame
) -> Any:
    """Ordered logit: static vs. dynamic effect on ordinal score, controlling for model.

    Args:
        df (pd.DataFrame): Long-format DataFrame.

    Returns:
        The fitted model.
    """
    df = _prepare_static_vs_dynamic(df)
    model = OrderedModel.from_formula(
        "score ~ is_dynamic + C(model)",
        data=df,
        distr="logit"
    )
    return model.fit(method="bfgs")


def test_numeric_static_vs_dynamic(
    df: pd.DataFrame
) -> Any:
    """Linear mixed model: static vs. dynamic effect on numeric score, with model as random grouping.

    Args:
        df (pd.DataFrame): Long-format DataFrame.

    Returns:
        The fitted model.
    """
    df = _prepare_static_vs_dynamic(df)
    model = smf.mixedlm("score ~ is_dynamic", data=df, groups=df["model"])
    return model.fit()
