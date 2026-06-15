"""Module for significance testing."""
from typing import Any, Literal

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel

_DYNAMIC_LEN_COLS = [("base", 1), ("dynamic_short", 2), ("dynamic_medium", 3), ("dynamic_long", 4)]
_STATIC_LEN_COLS = [("base", 1), ("static_short", 2), ("static_medium", 3), ("static_long", 4)]
_TEACHER_PERSONAS = ["beginner_teacher", "intermediate_teacher", "expert_teacher"]
_STATIC_PERSONAS = ["static_short", "static_medium", "static_long"]
_DYNAMIC_PERSONAS = ["dynamic_short", "dynamic_medium", "dynamic_long"]
_BASELINE_REFERENCE = "helpful"


def _baseline_reference(df: pd.DataFrame) -> str:
    """Return the baseline reference persona for the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a "persona" column.

    Returns:
        str: "no" if this is one of the available personas, else "base".
    """
    if _BASELINE_REFERENCE in df["persona"].values:
        return _BASELINE_REFERENCE
    return "base"


def prepare_length(
    df: pd.DataFrame,
    mode: Literal["static", "dynamic", "combined"]
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
    elif mode == "combined":
        vals = set(_STATIC_LEN_COLS).union(_DYNAMIC_LEN_COLS)
    else:
        raise ValueError(f"Unknown mode {mode}")
    personas = [persona[0] for persona in vals]
    df = df.loc[df["persona"].isin(personas)].copy()
    for persona, length in vals:
        df.loc[df["persona"] == persona, "length"] = int(length)
    return df


def prepare_teacher(df: pd.DataFrame) -> pd.DataFrame:
    """Filters to the teacher personas and adds a numeric `level` column.

    Args:
        df (pd.DataFrame): DataFrame with a "persona" column.

    Returns:
        pd.DataFrame: Filtered dataframe with a numeric `level` column added.
    """
    df = df.loc[df["persona"].isin(_TEACHER_PERSONAS)].copy()
    for level, persona in enumerate(_TEACHER_PERSONAS):
        df.loc[df["persona"] == persona, "level"] = int(level)
    return df


def prepare_static_vs_dynamic(df: pd.DataFrame) -> pd.DataFrame:
    """Filters to the static and dynamic personas and adds a numeric `is_dynamic` column.

    Args:
        df (pd.DataFrame): DataFrame with a "persona" column.

    Returns:
        pd.DataFrame: Filtered dataframe with a numeric `is_dynamic` column added.
    """
    personas = _STATIC_PERSONAS + _DYNAMIC_PERSONAS + ["base"]
    df = df.loc[df["persona"].isin(personas)].copy()
    df["is_dynamic"] = df["persona"].str.startswith("dynamic").astype(int)
    return df


def test_binary_baseline(
    df: pd.DataFrame
) -> Any:
    """Logistic regression; testing significance of all personas vs baseline.

    For binary outcomes. Uses "no" as reference when present, otherwise "base".

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".

    Returns:
        The fitted model.
    """
    ref = _baseline_reference(df)
    model = smf.logit(f"score ~ C(persona, Treatment(reference='{ref}')) + C(model)", data=df)
    return model.fit(method="bfgs", maxiter=300)


def test_ordinal_baseline(
    df: pd.DataFrame
) -> Any:
    """Ordered logistic regression; testing significance of all personas vs baseline.

    For ordinal outcomes. Uses "no" as reference when present, otherwise "base".

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".

    Returns:
        The fitted model.
    """
    df["score"] = pd.Categorical(df["score"], categories=[0, 1, 2], ordered=True)
    model = OrderedModel.from_formula(
        "score ~ C(persona) + C(model)",
        data=df,
        distr="logit"
    )
    return model.fit(method="bfgs")


def test_numeric_baseline(
    df: pd.DataFrame
) -> Any:
    """Linear mixed model; testing significance of all personas vs baseline.

    For numeric outcomes. Uses "no" as reference when present, otherwise "base". Model
    acts as a random grouping effect.

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".

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
    mode: Literal["static", "dynamic", "combined"]
) -> Any:
    """Logistic regression; testing significance of the persona length.

    For binary outcomes.

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".
        mode (Literal["static", "dynamic"]): Choose whether to test static or dynamic personas.

    Returns:
        The fitted model.
    """
    df = prepare_length(df, mode)
    model = smf.logit("score ~ length + C(model)", data=df)
    return model.fit(method="bfgs", maxiter=300)


def test_binary_length_by_model(
    df: pd.DataFrame,
    mode: Literal["static", "dynamic", "combined"]
) -> Any:
    """Logistic regression with interaction between length and model.

    For numeric outcomes. Model acts as a random grouping effect.

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".
        mode (Literal["static", "dynamic"]): Choose whether to test static or dynamic personas.

    Returns:
        The fitted model.
    """
    df = prepare_length(df, mode)
    model = smf.logit("score ~ length * C(model)", data=df)
    return model.fit(method="bfgs", maxiter=300)


def test_ordinal_length(
    df: pd.DataFrame,
    mode: Literal["static", "dynamic", "combined"]
) -> Any:
    """Ordered logistic regression; testing significance of the persona length.

    For ordinal outcomes.

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".
        mode (Literal["static", "dynamic"]): Choose whether to test static or dynamic personas.

    Returns:
        The fitted model.
    """
    df = prepare_length(df, mode)
    df["score"] = pd.Categorical(df["score"], categories=[0, 1, 2], ordered=True)
    model = OrderedModel.from_formula(
        "score ~ length + C(model)",
        data=df,
        distr="logit"
    )
    return model.fit(method="bfgs")


def test_ordinal_length_by_model(
    df: pd.DataFrame,
    mode: Literal["static", "dynamic", "combined"]
) -> Any:
    """Ordered logistic regression with interaction between length and model.

    For ordinal outcomes.

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".
        mode (Literal["static", "dynamic"]): Choose whether to test static or dynamic personas.

    Returns:
        The fitted model.
    """
    df = prepare_length(df, mode)
    df["score"] = pd.Categorical(df["score"], categories=[0, 1, 2], ordered=True)

    model = OrderedModel.from_formula(
        "score ~ length * C(model)",
        data=df,
        distr="logit"
    )

    return model.fit(method="bfgs")


def test_numeric_length(
    df: pd.DataFrame,
    mode: Literal["static", "dynamic", "combined"]
) -> Any:
    """Linear mixed model; testing significance of the persona length.

    For numeric outcomes. Model acts as a random grouping effect.

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".
        mode (Literal["static", "dynamic"]): Choose whether to test static or dynamic personas.

    Returns:
        The fitted model.
    """
    df = prepare_length(df, mode)
    model = smf.mixedlm("score ~ length", data=df, groups=df["model"])
    return model.fit()


def test_numeric_length_by_model(
    df: pd.DataFrame,
    mode: Literal["static", "dynamic", "combined"]
) -> Any:
    """Linear mixed model; testing whether the length effect varies by model.

    Model acts as a random grouping effect with both random intercepts and
    random slopes for length.

    This allows each model to have:
    - its own baseline score
    - its own length effect
    """
    df = prepare_length(df, mode)

    model = smf.mixedlm(
        "score ~ length",
        data=df,
        groups=df["model"],
        re_formula="~length"
    )

    return model.fit(reml=False, method="lbfgs")


def test_binary_teacher(
    df: pd.DataFrame
) -> Any:
    """Logistic regression; testing significance of the audience level effect.

    For binary outcomes.

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".

    Returns:
        The fitted model.
    """
    df = prepare_teacher(df)
    model = smf.logit("score ~ level + C(model)", data=df)
    return model.fit(method="bfgs", maxiter=300)


def test_binary_teacher_by_model(
    df: pd.DataFrame
) -> Any:
    """Logistic regression with interaction between level and model.

    For binary outcomes.

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".

    Returns:
        The fitted model.
    """
    df = prepare_teacher(df)
    model = smf.logit("score ~ level * C(model)", data=df)
    return model.fit(method="bfgs", maxiter=300)


def test_ordinal_teacher(
    df: pd.DataFrame
) -> Any:
    """Ordered logistic regression; testing significance of the audience level effect.

    For ordinal outcomes.

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".

    Returns:
        The fitted model.
    """
    df = prepare_teacher(df)
    df["score"] = pd.Categorical(df["score"], categories=[0, 1, 2], ordered=True)
    model = OrderedModel.from_formula(
        "score ~ level + C(model)",
        data=df,
        distr="logit"
    )
    return model.fit(method="bfgs")


def test_ordinal_teacher_by_model(
    df: pd.DataFrame
) -> Any:
    """Ordered logistic regression with interaction between level and model.

    For ordinal outcomes.

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".

    Returns:
        The fitted model.
    """
    df = prepare_teacher(df)
    df["score"] = pd.Categorical(df["score"], categories=[0, 1, 2], ordered=True)
    model = OrderedModel.from_formula(
        "score ~ level * C(model)",
        data=df,
        distr="logit"
    )
    return model.fit(method="bfgs")


def test_numeric_teacher(
    df: pd.DataFrame
) -> Any:
    """Linear mixed model; testing significance of the audience level effect.

    For numeric outcomes. Model acts as a random grouping effect.

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".

    Returns:
        The fitted model.
    """
    df = prepare_teacher(df)
    model = smf.mixedlm("score ~ level", data=df, groups=df["model"])
    return model.fit()


def test_binary_static_vs_dynamic(
    df: pd.DataFrame
) -> Any:
    """Logistic regression; testing significance of static vs. dynamic effect.

    For binary outcomes.

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".

    Returns:
        The fitted model.
    """
    df = prepare_static_vs_dynamic(df)
    model = smf.logit("score ~ is_dynamic + C(model)", data=df)
    return model.fit(method="bfgs", maxiter=300)


def test_binary_static_vs_dynamic_by_model(
    df: pd.DataFrame
) -> Any:
    """Logistic regression with interaction between is_dynamic and model.

    For binary outcomes.

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".

    Returns:
        The fitted model.
    """
    df = prepare_static_vs_dynamic(df)
    model = smf.logit("score ~ is_dynamic * C(model)", data=df)
    return model.fit(method="bfgs", maxiter=300)


def test_ordinal_static_vs_dynamic(
    df: pd.DataFrame
) -> Any:
    """Ordered logistic regression; testing significance of static vs. dynamic effect.

    For ordinal outcomes.

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".

    Returns:
        The fitted model.
    """
    df = prepare_static_vs_dynamic(df)
    df["score"] = pd.Categorical(df["score"], categories=[0, 1, 2], ordered=True)
    model = OrderedModel.from_formula(
        "score ~ is_dynamic + C(model)",
        data=df,
        distr="logit"
    )
    return model.fit(method="bfgs")


def test_ordinal_static_vs_dynamic_by_model(
    df: pd.DataFrame
) -> Any:
    """Ordered logistic regression with interaction between is_dynamic and model.

    For ordinal outcomes.

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".

    Returns:
        The fitted model.
    """
    df = prepare_static_vs_dynamic(df)
    df["score"] = pd.Categorical(df["score"], categories=[0, 1, 2], ordered=True)
    model = OrderedModel.from_formula(
        "score ~ is_dynamic * C(model)",
        data=df,
        distr="logit"
    )
    return model.fit(method="bfgs")


def test_numeric_static_vs_dynamic(
    df: pd.DataFrame
) -> Any:
    """Linear mixed model; testing significance of static vs. dynamic effect.

    For numeric outcomes. Model acts as a random grouping effect.

    Args:
        df (pd.DataFrame): Long-format DataFrame with columns "score", "persona", "model".

    Returns:
        The fitted model.
    """
    df = prepare_static_vs_dynamic(df)
    model = smf.mixedlm("score ~ is_dynamic", data=df, groups=df["model"])
    return model.fit()
