"""Pipeline to run lin_test_numerical on evaluation datasets and subcategories."""
from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd

from evaluate import test_binary_baseline, test_ordinal_baseline, test_numeric_baseline


_SUBSET_COLUMN_BY_DATASET = {
    "mmlu-pro": "category",
    "MATH": "subject",
    "flores": "iso_639_3",
}


def _subset_column_for_dataset(dataset: str) -> str | None:
    return _SUBSET_COLUMN_BY_DATASET.get(dataset)


def _test_for_dataset(dataset: str):
    if dataset == "flores":
        return test_ordinal_baseline
    elif dataset == "alpaca":
        return test_numeric_baseline
    return test_binary_baseline


def _collect_results(
    df: pd.DataFrame,
    dataset: str,
    subset_column: str | None,
    subset_value: str | None,
) -> pd.DataFrame:
    try:
        test_fn = _test_for_dataset(dataset)
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


def run_evaluation_lin_tests(
    data_dir: str | Path = "data/evaluation",
) -> pd.DataFrame:
    """Run lin_test_numerical on all datasets and subcategories in data_dir."""
    data_dir = Path(data_dir)
    results = []

    for csv_path in data_dir.iterdir():
        df = pd.read_csv(csv_path, index_col=0)

        required = {"score", "persona", "model"}
        if not required.issubset(df.columns):
            warnings.warn(f"Skipping {csv_path.name}: missing {required - set(df.columns)}")
            continue

        dataset = csv_path.stem
        print(dataset)
        results.append(_collect_results(df, dataset, None, None))

        subset_col = _SUBSET_COLUMN_BY_DATASET.get(dataset)
        if subset_col and subset_col in df.columns:
            for value in df[subset_col].dropna().unique():
                subset = df.loc[df[subset_col] == value]
                results.append(_collect_results(subset, dataset, subset_col, str(value)))

    if not results:
        return pd.DataFrame(
            columns=["dataset", "subset_column", "subset_value", "term", "coef", "stderr", "pvalue"]
        )
    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    results_df = run_evaluation_lin_tests()
    results_df.to_csv("data/evaluation/lin_test_results.csv", index=False)
