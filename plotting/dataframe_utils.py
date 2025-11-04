import pandas as pd


def compute_accuracy_df(
    df: pd.DataFrame,
    columns: list[str],
    ground_truth_col: str = "answer"
) -> pd.DataFrame:
    accuracy_rows = []

    for category, group in df.groupby("category"):
        row = {"category": category}
        for col in columns:
            row[col] = (group[result_col] == group[col]).mean()
        accuracy_rows.append(row)
    return pd.DataFrame(accuracy_rows)