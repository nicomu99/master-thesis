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

    double_parentheses = re.search(rf"\(([{option_range}])\)", completion)
    if double_parentheses:
        return double_parentheses.group(1)

    single_parantheses = re.search(rf"([{option_range}])\)", completion)
    if single_parantheses:
        return single_parantheses.group(1)

    all_matches = re.findall(rf"\(?([{option_range}])\)?", completion)
    if all_matches:
        if len(all_matches) == 1:
            return all_matches[0]

        first_letter = re.match(rf"^([{option_range}])\s", completion)
        if first_letter:
            return first_letter.group(1)

        letter_dot = re.match(rf"^([{option_range}])\.", completion)
        if letter_dot:
            return letter_dot.group(1)

        answer_letter_pattern = re.compile(rf"^Answer:\s*([{option_range}])(?:\s+(?!and\b).+)?$", re.MULTILINE)
        answer_letter = re.findall(answer_letter_pattern, completion)
        if len(answer_letter) == 1:
            return answer_letter[0]

        correct_option_pattern = re.compile(rf"^Correct option:\s*([{option_range}])(?:\s+(?!and\b).+)?$", re.MULTILINE)
        correct_option = re.findall(correct_option_pattern, completion)
        if len(correct_option) == 1:
            return correct_option[0]

        return all_matches

    return "Not found"


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
