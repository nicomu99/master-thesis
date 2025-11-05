import re


def extract_answer_mmlu(
    completion: str,
    answer_range: int
) -> str | list[str]:
    """Extracts answers for the MMLU-pro dataset.

    Args:
        completion (str): LLM completion.
        answer_range (int): Number of answer options in the dataframe.

    Returns:
        str | list[str]: The extracted number or a list of numbers.
    """
    option_range = f"A-{chr(65 + answer_range - 1)}"
    pattern = rf"The correct answer is:\s*([{option_range}])\b"
    match = re.search(pattern, completion)

    if not match:
        return "Invalid response format"

    return match.group(1)


def extract_answer_math(completion: str) -> str:
    """Extracts answers for the MATH dataset.

    Args:
        completion (str): LLM completion.

    Returns:
        str: The extracted answer.
    """
    lines = [line.strip() for line in completion.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


def extract_answer_flores(
    completion: str,
    sample_id: str
) -> str:
    """Extracts answers for the flores dataset.

    Args:
        completion (str): LLM completion.
        sample_id (str): Sample identifier.

    Returns:
        str: The extracted answer.
    """
    sample_number = int(sample_id.split("_")[-1])
    match_single = re.match(r"The better translation is:\s*(1|2)\b(.*)", completion, re.IGNORECASE | re.DOTALL)
    if match_single:
        if (
            sample_number % 2 == 0 and int(match_single.group(1)) == 2 or
            sample_number % 2 == 1 and int(match_single.group(1)) == 1
        ):
            return "reference"
        return "persona"

    match_equal = re.match(r"Both translations are equal:\s*(.*)", completion, re.IGNORECASE | re.DOTALL)
    if match_equal:
        return "both"
    return "no response"
