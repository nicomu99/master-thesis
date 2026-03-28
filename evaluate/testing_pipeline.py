"""Pipeline to run statistical tests on evaluation datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import warnings

import pandas as pd

from evaluate.sig_testing import (
    test_binary_baseline,
    test_ordinal_baseline,
    test_numeric_baseline,
    test_binary_length,
    test_ordinal_length,
    test_numeric_length,
    test_binary_teacher,
    test_ordinal_teacher,
    test_numeric_teacher,
    test_binary_static_vs_dynamic,
    test_ordinal_static_vs_dynamic,
    test_numeric_static_vs_dynamic,
)


_SUBSET_COLUMN_BY_DATASET = {
    "mmlu-pro": "category",
    "MATH": "subject",
    "flores": "iso_639_3",
}


def _subset_column_for_dataset(dataset: str) -> str | None:
    return _SUBSET_COLUMN_BY_DATASET.get(dataset)


def _baseline_test_for_dataset(dataset: str) -> Callable:
    if dataset == "flores":
        return test_ordinal_baseline
    if dataset == "alpaca":
        return test_numeric_baseline
    return test_binary_baseline


def _length_test_for_dataset(dataset: str, mode: str) -> Callable:
    if dataset == "flores":
        return lambda df: test_ordinal_length(df, mode)
    if dataset == "alpaca":
        return lambda df: test_numeric_length(df, mode)
    return lambda df: test_binary_length(df, mode)


def _teacher_test_for_dataset(dataset: str) -> Callable:
    if dataset == "flores":
        return test_ordinal_teacher
    if dataset == "alpaca":
        return test_numeric_teacher
    return test_binary_teacher


def _static_vs_dynamic_test_for_dataset(dataset: str) -> Callable:
    if dataset == "flores":
        return test_ordinal_static_vs_dynamic
    if dataset == "alpaca":
        return test_numeric_static_vs_dynamic
    return test_binary_static_vs_dynamic


def _collect_results(
    df: pd.DataFrame,
    dataset: str,
    subset_column: str | None,
    subset_value: str | None,
    test_fn: Callable | None = None,
) -> pd.DataFrame:
    if test_fn is None:
        test_fn = _baseline_test_for_dataset(dataset)
    try:
        results = test_fn(df)
    except Exception as exc:  # noqa: BLE001 - want a resilient pipeline
        warnings.warn(
            f"test failed for {dataset} "
            f"{subset_column}={subset_value}: {exc}"
        )
        return pd.DataFrame(
            {
                "dataset": [dataset],
                "subset_column": [subset_column],
                "subset_value": [subset_value],
                "term": [None],
                "coef": [pd.NA],
                "stderr": [pd.NA],
                "pvalue": [pd.NA],
            }
        )

    params = results.params
    if hasattr(params, "index"):
        terms = list(params.index)
        coef = list(params.values)
    else:
        terms = _get_param_names(results, len(params))
        coef = list(params)

    stderr = _get_optional_vector(results, "bse", len(coef))
    pvalues = _get_optional_vector(results, "pvalues", len(coef))

    out = pd.DataFrame(
        {
            "term": terms,
            "coef": coef,
            "stderr": stderr,
            "pvalue": pvalues,
        }
    )
    out = out[out["term"] != "Group Var"].copy()
    out.insert(0, "subset_value", subset_value)
    out.insert(0, "subset_column", subset_column)
    out.insert(0, "dataset", dataset)
    return out


def _get_param_names(results, count: int) -> list[str]:
    if hasattr(results, "param_names"):
        names = list(results.param_names)
        return names[:count]
    model = getattr(results, "model", None)
    if model is not None:
        names: list[str] = []
        if hasattr(model, "exog_names"):
            names.extend(list(model.exog_names))
        if hasattr(model, "vcp_names"):
            names.extend(list(model.vcp_names))
        elif hasattr(model, "vc_names"):
            names.extend(list(model.vc_names))
        if names:
            return names[:count]
    return [f"param_{idx}" for idx in range(count)]


def _get_optional_vector(results, attr: str, count: int) -> list:
    values = getattr(results, attr, None)
    if values is None:
        return [pd.NA] * count
    if hasattr(values, "values"):
        values = values.values
    values = list(values)
    if len(values) < count:
        values.extend([pd.NA] * (count - len(values)))
    return values[:count]


def _iter_csv(data_dir: Path):
    """Yield (csv_path, df, dataset) for valid evaluation CSVs."""
    for csv_path in sorted(data_dir.iterdir()):
        if csv_path.suffix != ".csv":
            continue
        df = pd.read_csv(csv_path, index_col=0)
        required = {"score", "persona", "model"}
        if not required.issubset(df.columns):
            print(csv_path)
            warnings.warn(f"Skipping {csv_path.name}: missing {required - set(df.columns)}")
            continue
        yield csv_path, df, csv_path.stem


def run_baseline_tests(
    data_dir: str | Path = "data/evaluation",
) -> pd.DataFrame:
    """Test whether any persona has a significant effect against the 'no' baseline.

    Runs on the full dataset and, for datasets with a subset column
    (mmlu-pro, MATH, flores), also per subset.

    Returns:
        pd.DataFrame: Coefficients, standard errors, and p-values per term.
    """
    data_dir = Path(data_dir)
    results = []

    for csv_path, df, dataset in _iter_csv(data_dir):
        test_fn = _baseline_test_for_dataset(dataset)
        results.append(_collect_results(df, dataset, None, None, test_fn))

        subset_col = _SUBSET_COLUMN_BY_DATASET.get(dataset)
        if subset_col and subset_col in df.columns:
            for value in df[subset_col].dropna().unique():
                subset = df.loc[df[subset_col] == value]
                results.append(_collect_results(subset, dataset, subset_col, str(value), test_fn))

    if not results:
        return pd.DataFrame(
            columns=["dataset", "subset_column", "subset_value", "term", "coef", "stderr", "pvalue"]
        )
    return pd.concat(results, ignore_index=True)


def run_length_tests(
    data_dir: str | Path = "data/evaluation",
) -> pd.DataFrame:
    """Test whether persona length has a statistically significant effect on score.

    Runs separate tests for static and dynamic length variants on each dataset.

    Returns:
        pd.DataFrame: Results with an additional `mode` column ("static" or "dynamic").
    """
    data_dir = Path(data_dir)
    results = []

    for csv_path, df, dataset in _iter_csv(data_dir):
        for mode in ("static", "dynamic"):
            test_fn = _length_test_for_dataset(dataset, mode)
            out = _collect_results(df, dataset, None, None, test_fn)
            out.insert(3, "mode", mode)
            results.append(out)

    if not results:
        return pd.DataFrame(
            columns=["dataset", "subset_column", "subset_value", "mode", "term", "coef", "stderr", "pvalue"]
        )
    return pd.concat(results, ignore_index=True)


def run_teacher_tests(
    data_dir: str | Path = "data/evaluation",
) -> pd.DataFrame:
    """Test whether teacher persona audience level has a statistically significant effect.

    Uses beginner_teacher as the reference category.

    Returns:
        pd.DataFrame: Coefficients, standard errors, and p-values per term.
    """
    data_dir = Path(data_dir)
    results = []

    for csv_path, df, dataset in _iter_csv(data_dir):
        test_fn = _teacher_test_for_dataset(dataset)
        results.append(_collect_results(df, dataset, None, None, test_fn))

    if not results:
        return pd.DataFrame(
            columns=["dataset", "subset_column", "subset_value", "term", "coef", "stderr", "pvalue"]
        )
    return pd.concat(results, ignore_index=True)


def run_static_vs_dynamic_tests(
    data_dir: str | Path = "data/evaluation",
) -> pd.DataFrame:
    """Test whether dynamic personas perform significantly differently from static personas.

    Excludes 'base' persona. Uses a binary is_dynamic predictor (0=static, 1=dynamic).

    Returns:
        pd.DataFrame: Coefficients, standard errors, and p-values per term.
    """
    data_dir = Path(data_dir)
    results = []

    for csv_path, df, dataset in _iter_csv(data_dir):
        test_fn = _static_vs_dynamic_test_for_dataset(dataset)
        results.append(_collect_results(df, dataset, None, None, test_fn))

    if not results:
        return pd.DataFrame(
            columns=["dataset", "subset_column", "subset_value", "term", "coef", "stderr", "pvalue"]
        )
    return pd.concat(results, ignore_index=True)


def run_all_tests(
    data_dir: str | Path = "data/evaluation",
    output_dir: str | Path = "data/evaluation/tests",
) -> None:
    """Run all four test suites and write one CSV per suite to output_dir.

    Files written:
        results_baseline.csv
        results_length.csv
        results_teacher.csv
        results_static_vs_dynamic.csv

    Args:
        data_dir: Directory containing the evaluation CSVs.
        output_dir: Directory where result CSVs are written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suites = [
        ("baseline", run_baseline_tests),
        ("length", run_length_tests),
        ("teacher", run_teacher_tests),
        ("static_vs_dynamic", run_static_vs_dynamic_tests),
    ]

    for name, fn in suites:
        print(f"\n=== {name} ===")
        df = fn(data_dir)
        path = output_dir / f"results_{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} rows → {path}")


if __name__ == "__main__":
    run_all_tests()
