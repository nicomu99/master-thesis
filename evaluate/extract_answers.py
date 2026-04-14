"""Helper functions for extracting the answer from LLM generations."""
import re
import pandas as pd


def extract_answers(
    dataframe: pd.DataFrame,
    columns: list[str],
    extraction_fn: str
) -> pd.DataFrame:
    """Extracts answers from LLM completions.

    Args:
        dataframe (pd.DataFrame): Dataframe to extract answers from.
        columns (list[str]): Columns to extract answers from.
        extraction_fn (str): String identifier of the extraction function to use.
            Must be 'mmlu', 'flores' or 'MATH'.

    Returns:
        pd.DataFrame: A dataframe with extracted answers.
    """
    _extraction_fns = {
        "mmlu-pro": extract_answer_mmlu,
        "MATH": extract_answer_math,
        "flores": extract_answer_flores
    }
    if extraction_fn not in _extraction_fns:
        raise ValueError(
            "Extraction function unknown. Must be 'mmlu', 'MATH' or 'flores'."
        )
    _extraction_fn = _extraction_fns[extraction_fn]

    dataframe = dataframe.assign(
        **{
            f"{column.replace("_answer", "").replace("_judgment", "")}": dataframe.apply(
                lambda row: _extraction_fn(row, column), axis=1)
            for column in columns
        }
    )
    return dataframe


def extract_answer_mmlu(
    sample: pd.Series,
    column: str
) -> str | list[str]:
    """Extracts answers for the MMLU-pro dataset.

    Args:
        sample (pd.Series): Sample row.
        column (str): Column identifier.

    Returns:
        str | list[str]: The extracted number or a list of numbers.
    """
    completion = sample[column]
    if not completion:
        return "Invalid response format"

    answer_range = len(sample["answers"])
    option_range = f"A-{chr(65 + answer_range - 1)}"
    pattern = rf"^(?:The correct answer is:\s*)?\(?([{option_range}])\)?\b"
    match = re.search(pattern, completion.strip())

    if not match:
        return "Invalid response format"

    return match.group(1)


def extract_answer_math(
    sample: pd.Series,
    column: str
) -> str:
    """Extracts answers for the MATH dataset.

    Args:
        sample (pd.Series): Sample row.
        column (str): Column identifier.

    Returns:
        str: The extracted answer.
    """
    completion = sample[column]
    if not completion:
        return ""
    lines = [line.strip() for line in completion.splitlines() if line.strip()]
    if not lines:
        return ""
    if lines[-1] == "\\]":
        lines = lines[:-1]
    if lines[-1] == "}":
        lines = lines[:-1]

    answer = lines[-1]
    answer = "".join(answer.split())

    # Get rid of leading point
    if answer.endswith("."):
        answer = answer[:-1]
    if answer.startswith("\\[") and answer.endswith("\\]"):
        answer = answer[2:-2]

    wrappers = ("boxed", "text", "mathrm", "mathbf")
    wrapper_re = re.compile(rf"^\\(?:{'|'.join(wrappers)})\{{(.*)\}}$")
    while True:
        m = wrapper_re.match(answer)
        if not m:
            break
        answer = m.group(1)

    # Remove wrapping $$...$$ or $...$
    if answer.startswith("$$") and answer.endswith("$$"):
        answer = answer[2:-2]
    elif answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1]
    # Remove \( and \)
    if answer.startswith("\\(") and answer.endswith("\\)"):
        answer = answer[2:-2]

    return answer


def extract_answer_flores(
    sample: pd.Series,
    column: str,
) -> int:
    """Extracts answers for the flores dataset.

    Args:
        sample (pd.Series): Sample row.
        column (str): Column identifier.

    Returns:
        int: The extracted answer.
    """
    completion = sample[column]
    sample_id = sample["static_id"]
    sample_number = int(sample_id.split("_")[-1])
    if completion is None:
        return -1

    match_single = re.match(r"The better translation is:\s*([12])\b(.*)", completion, re.IGNORECASE | re.DOTALL)
    if match_single:
        if (
            sample_number % 2 == 0 and int(match_single.group(1)) == 2 or
            sample_number % 2 == 1 and int(match_single.group(1)) == 1
        ):
            return 0
        return 2

    match_equal = re.match(r"Both translations are equal:\s*(.*)", completion, re.IGNORECASE | re.DOTALL)
    if match_equal:
        return 1
    return 0
