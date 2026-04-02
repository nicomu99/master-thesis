"""Pipeline to run statistical tests on evaluation datasets."""
from typing import Callable, Literal

import re
import argparse
from pathlib import Path

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

from inference.utils import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


_CATEGORY_COLUMN_BY_DATASET = {
    "mmlu-pro": "category",
    "MATH": "subject",
    "flores": "iso_639_3",
}


def _baseline_test_for_dataset(dataset: str) -> Callable:
    if dataset == "flores":
        return test_ordinal_baseline
    if dataset == "alpaca":
        return test_numeric_baseline
    return test_binary_baseline


def _length_test_for_dataset(dataset: str, mode: Literal["static", "dynamic", "combined"]) -> Callable:
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
    test_fn: Callable,
    category_name: str | None,
) -> pd.DataFrame:
    try:
        results = test_fn(df)
    except Exception as e:
        log.error("Test failed for dataset %s %s: %s", dataset, category_name, e)
        return pd.DataFrame(
            {
                "dataset": [dataset],
                "category": [category_name],
                "term": [None],
                "coef": [pd.NA],
                "stderr": [pd.NA],
                "pvalue": [pd.NA],
            }
        )

    terms = list(results.params.index)
    coef = list(results.params.values)
    stderr = _get_results(results, "bse", len(coef))
    pvalues = _get_results(results, "pvalues", len(coef))

    out = pd.DataFrame(
        {
            "term": [_clean_term(t) for t in terms],
            "coef": coef,
            "stderr": stderr,
            "pvalue": pvalues,
        }
    )
    out = out[out["term"] != "Group Var"]
    out.insert(0, "category", category_name)
    out.insert(0, "dataset", dataset)
    return out


def _get_results(results, attr: str, count: int) -> list:
    """Safely retrieves some attribute from statsmodels return values."""
    values = getattr(results, attr, None)
    if values is None:
        return [pd.NA] * count
    if hasattr(values, "values"):
        values = values.values
    values = list(values)
    if len(values) < count:
        values.extend([pd.NA] * (count - len(values)))
    return values[:count]


def _clean_term(term: str) -> str:
    """Extract the level name from a statsmodels contrast term.

    E.g. "C(persona, Treatment(reference='no'))[T.dynamic_medium]" becomes "dynamic_medium"
    """
    m = re.search(r'\[T\.(.+)]', term)
    return m.group(1) if m else term


def _iter_csv(data_dir: Path):
    """Yields dataset name and dataframe for valid evaluation CSVs."""
    for csv_path in data_dir.iterdir():
        if csv_path.suffix != ".csv":
            continue
        df = pd.read_csv(csv_path, index_col=0)
        yield csv_path.stem, df


def run_baseline_tests(
    data_dir: str | Path = "data/evaluation",
) -> pd.DataFrame:
    """Test whether any persona has a significant effect against the "no" baseline.

    Runs on the full dataset and, for datasets with a category column also per category.

    Returns:
        pd.DataFrame: DataFrame with coefficients, standard errors, and p-values per term.
    """
    data_dir = Path(data_dir)
    results = []

    for dataset, df in _iter_csv(data_dir):
        test_fn = _baseline_test_for_dataset(dataset)
        results.append(_collect_results(df, dataset, test_fn, None))

        category_col = _CATEGORY_COLUMN_BY_DATASET.get(dataset)
        if not category_col or category_col not in df.columns:
            continue
        for category, cat_df in df.groupby(category_col):
            results.append(_collect_results(cat_df, dataset, test_fn, category))

    if not results:
        return pd.DataFrame(
            columns=["dataset", "category", "term", "coef", "stderr", "pvalue"]
        )
    return pd.concat(results, ignore_index=True)


def run_length_tests(
    data_dir: str | Path = "data/evaluation",
) -> pd.DataFrame:
    """Test whether persona length has a statistically significant effect on score.

    Runs separate tests for static and dynamic length variants on each dataset and also
    per category, if the dataset has any.

    Returns:
        pd.DataFrame: DataFrame with coefficients, standard errors, and p-values per term and
            `mode` column ("static" or "dynamic").
    """
    data_dir = Path(data_dir)
    results = []

    for dataset, df in _iter_csv(data_dir):
        for mode in ("static", "dynamic", "combined"):
            mode: Literal["static", "dynamic", "combined"] = mode   # to silence warning
            test_fn = _length_test_for_dataset(dataset, mode)
            out = _collect_results(df, dataset, test_fn, None)
            out.insert(3, "mode", mode)
            results.append(out)

            category_col = _CATEGORY_COLUMN_BY_DATASET.get(dataset)
            if not category_col or category_col not in df.columns:
                continue
            for category, cat_df in df.groupby(category_col):
                out = _collect_results(cat_df, dataset, test_fn, category)
                out.insert(3, "mode", mode)
                results.append(out)

    if not results:
        return pd.DataFrame(
            columns=["dataset", "category", "mode", "term", "coef", "stderr", "pvalue"]
        )
    return pd.concat(results, ignore_index=True)


def run_teacher_tests(
    data_dir: str | Path = "data/evaluation",
) -> pd.DataFrame:
    """Test whether teacher persona audience level has a statistically significant effect.

    The expertise is handled as a numeric value. Runs tests for the whole dataset
    and also per category, if the dataset has any.

    Returns:
        pd.DataFrame: DataFrame with coefficients, standard errors, and p-values per term.
    """
    data_dir = Path(data_dir)
    results = []

    for dataset, df in _iter_csv(data_dir):
        test_fn = _teacher_test_for_dataset(dataset)
        results.append(_collect_results(df, dataset, test_fn, None))

        category_col = _CATEGORY_COLUMN_BY_DATASET.get(dataset)
        if not category_col or category_col not in df.columns:
            continue
        for category, cat_df in df.groupby(category_col):
            results.append(_collect_results(cat_df, dataset, test_fn, category))

    if not results:
        return pd.DataFrame(
            columns=["dataset", "category", "term", "coef", "stderr", "pvalue"]
        )
    return pd.concat(results, ignore_index=True)


def run_static_vs_dynamic_tests(
    data_dir: str | Path = "data/evaluation",
) -> pd.DataFrame:
    """Test whether dynamic personas perform significantly differently from static personas.

    Uses a binary is_dynamic predictor (0=static, 1=dynamic). Runs tests for the whole dataset
    and also per category, if the dataset has any.

    Returns:
        pd.DataFrame: DataFrame with coefficients, standard errors, and p-values per term.
    """
    data_dir = Path(data_dir)
    results = []

    for dataset, df in _iter_csv(data_dir):
        test_fn = _static_vs_dynamic_test_for_dataset(dataset)
        results.append(_collect_results(df, dataset, test_fn, None))

        category_col = _CATEGORY_COLUMN_BY_DATASET.get(dataset)
        if not category_col or category_col not in df.columns:
            continue
        for category, cat_df in df.groupby(category_col):
            results.append(_collect_results(cat_df, dataset, test_fn, category))

    if not results:
        return pd.DataFrame(
            columns=["dataset", "category", "term", "coef", "stderr", "pvalue"]
        )
    return pd.concat(results, ignore_index=True)


def run_all_tests(
    data_dir: str | Path = "data/evaluation",
    output_dir: str | Path = "data/evaluation/tests",
    suite: str = "all",
) -> None:
    """Run test suites.

    By default, all four test suites are executed. Optionally, a single test can be chosen
    to be run.

    Args:
        data_dir (str | Path): Directory containing the evaluation CSVs.
        output_dir (str | Path): Directory where result CSVs are written.
        suite (str): Which suite to run.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suites = {
        "baseline": run_baseline_tests,
        "length": run_length_tests,
        "teacher": run_teacher_tests,
        "static_vs_dynamic": run_static_vs_dynamic_tests,
    }

    selected_suites = suites.items() if suite == "all" else [(suite, suites[suite])]

    for name, fn in selected_suites:
        print(f"\n=== {name} ===")
        df = fn(data_dir)
        path = output_dir / f"results_{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} rows to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation test suites.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="data/evaluation",
        help="Directory containing the evaluation CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/evaluation/tests",
        help="Directory where result CSVs are written.",
    )
    parser.add_argument(
        "--suite",
        choices=["all", "baseline", "length", "teacher", "static_vs_dynamic"],
        default="all",
        help="Which test suite to run. Defaults to %(default)s.",
    )
    args = parser.parse_args()

    run_all_tests(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        suite=args.suite
    )
