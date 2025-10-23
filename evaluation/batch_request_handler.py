from typing import Any, Dict, Optional, List, Tuple, TextIO, cast

import json
import string
from pathlib import Path
from collections import defaultdict

import pandas as pd

from .utils import TaskInfo, QuestionType
from .utils import TEMP_PATH, STATIC_ID_COLUMN, QUESTION_COLUMN, GROUND_TRUTH_COLUMN
from .utils import TRANSLATION_JUDGE_TEMPLATE
from .persona_registry import PersonaConfig


class BatchRequestHandler:
    """Helper class used to read and write json files with request and response data."""

    @staticmethod
    def _write_prompt_to_file(
        f: TextIO,
        custom_id: str,
        prompt: str,
        model: str,
        instruction: Optional[str] = None
    ):
        """Function that writes a prompt request in JSON format to a file.

        Args:
            f (TextIO): File output buffer. The request will be written to this file.
            custom_id (str): An identifier, which can be used to map client outputs to the input samples.
            prompt (str): Request prompt.
            model (str): Model identifier of the LLM API.
            instruction (str | None): System prompt instruction. If this is None, the default system prompt
                will be used. Defaults to None.
        """
        body = {"model": model, "input": prompt}
        if instruction:
            body["instructions"] = instruction

        api_request_dict = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body}

        f.write(json.dumps(api_request_dict) + "\n")

    @staticmethod
    def _create_question_prompt(
        task_data: Dict[str, Any],
        question_type: QuestionType,
    ) -> str:
        """Returns a question prompt template with filled out placeholders.

        For a given question type, fetches the correct prompt template and fills all placeholders.

        Args:
            task_data (Dict[str, str]): A dictionary containing the relevant information to fill in placeholders.
            question_type (QuestionType): String identifier of the correct question template. Must be "open_question",
                "mc_question" or "summarization".

        Returns:
            str: Returns the filled template string.
        """
        template = question_type.template

        prompt_kwargs = {"question": task_data[QUESTION_COLUMN]}
        if question_type == QuestionType.MC:
            letters = string.ascii_uppercase
            choices = "\n".join(
                f"({letters[i]}) {answer}" for i, answer in enumerate(task_data[GROUND_TRUTH_COLUMN])
            )
            prompt_kwargs["choices"] = choices

        return template.format(**prompt_kwargs)

    @staticmethod
    def create_persona_batch_request_file(
        task_config: TaskInfo,
        dataframe: pd.DataFrame,
        persona_configs: List[PersonaConfig],
        model: str
    ) -> Tuple[Path, int]:
        """Creates a request file for persona generation.

        Args:
            task_config (TaskConfig): Task configuration attributes.
            dataframe (pd.DataFrame): Dataframe containing samples of the task.
            persona_configs (List[PersonaConfig]): List of persona configurations.
            model (str): Model string identifier.

        Returns:
            Tuple[Path, int]: A string identifier of the created task request file and number of requests created.

        Raises:
            OSError: If the request file cannot be written to disk.
        """
        request_count = 0
        request_file = TEMP_PATH / f"{task_config.task_id}_persona_request.jsonl"
        with request_file.open("w", encoding="utf-8") as f:
            for row_dict in dataframe.to_dict(orient="records"):
                for persona_config in persona_configs:
                    template = persona_config.template
                    persona_name = persona_config.name

                    if persona_name in row_dict and row_dict[persona_name] is not None:
                        continue

                    prompt = template.format(
                        task_type=task_config.field,
                        question=row_dict[QUESTION_COLUMN])

                    custom_id = f"{row_dict[STATIC_ID_COLUMN]}_{persona_name}"
                    BatchRequestHandler._write_prompt_to_file(f, custom_id, prompt, model)
                    request_count += 1
        return request_file, request_count

    @staticmethod
    def create_answer_batch_request_file(
        task_config: TaskInfo,
        dataframe: pd.DataFrame,
        question_type: QuestionType,
        persona_configs: List[PersonaConfig],
        model: str
    ) -> Tuple[Path, int]:
        """Creates a request file for task question answering.

        Args:
            task_config (TaskConfig): Task configuration attributes.
            dataframe (pd.DataFrame): Dataframe containing samples of the task.
            question_type (str): The question type of this task.
            persona_configs (List[PersonaConfig]): List of persona configurations.
            model (str): Model string identifier.

        Returns:
            Tuple[Path, int]: A string identifier of the created task request file and number of requests created.

        Raises:
            OSError: If the request file cannot be written to disk.
        """
        request_count = 0
        request_file = TEMP_PATH / f"{task_config.task_id}_request.jsonl"
        with request_file.open("w", encoding="utf-8") as f:
            for row_dict in dataframe.to_dict(orient="records"):
                row_dict = cast(Dict[str, Any], row_dict)
                prompt = BatchRequestHandler._create_question_prompt(
                    row_dict, question_type)

                for persona_config in persona_configs:
                    persona_name = persona_config.name
                    answer_column_df = persona_config.answer_column

                    if answer_column_df in row_dict and row_dict[answer_column_df] is not None:
                        continue

                    custom_id = f"{row_dict[STATIC_ID_COLUMN]}_{answer_column_df}"
                    BatchRequestHandler._write_prompt_to_file(f, custom_id, prompt, model, row_dict[persona_name])
                    request_count += 1
        return request_file, request_count

    @staticmethod
    def create_judgment_batch_request_file(
        task_config: TaskInfo,
        dataframe: pd.DataFrame,
        persona_configs: List[PersonaConfig],
        model: str
    ) -> Tuple[Path, int]:
        """Creates a request file for judgment evaluation.

        Args:
            task_config (TaskConfig): Task configuration.
            dataframe (pd.DataFrame): Dataframe containing samples of the task.
            persona_configs (List[PersonaConfig]): List of persona configurations.
            model (str): Model string identifier.

        Returns:
            Tuple[Path, int]: A string identifier of the created task request file and number of requests created.

        Raises:
            ValueError: If the dataframe is missing required columns or contains invalid data.
            OSError: If the request file cannot be written to disk.
        """
        if GROUND_TRUTH_COLUMN not in dataframe.columns:
            raise ValueError(f"Missing answer column for task {task_config.task_id}")

        request_count = 0
        request_file = TEMP_PATH / f"{task_config.task_id}_judgment_request.jsonl"
        with request_file.open("w", encoding="utf-8") as f:
            for row_dict in dataframe.to_dict(orient="records"):

                prompt_template = TRANSLATION_JUDGE_TEMPLATE
                for persona_config in persona_configs:
                    answer_column = persona_config.answer_column
                    judgment_column = persona_config.judgment_column

                    if judgment_column in row_dict and row_dict[judgment_column] is not None:
                        continue

                    prompt = prompt_template.format(
                        reference=row_dict[QUESTION_COLUMN],
                        translation_1=row_dict[answer_column],
                        translation_2=row_dict[GROUND_TRUTH_COLUMN],
                    )
                    custom_id = f"{row_dict[STATIC_ID_COLUMN]}_{judgment_column}"
                    BatchRequestHandler._write_prompt_to_file(f, custom_id, prompt, model)
                    request_count += 1
        return request_file, request_count

    @staticmethod
    def read_response_file(file_name: Path) -> pd.DataFrame:
        """Helper function that reads the contents of a batch response json file.

        The contents are returned as a pandas dataframe.

        Args:
            file_name (Path): Path to the json file to read.

        Returns:
            pd.DataFrame: Dataframe containing one row per sample and one column for each persona type.
        """
        response_data = defaultdict(lambda: defaultdict(str))

        with file_name.open("r", encoding="utf-8") as f:
            for line in f:
                response_line = json.loads(line)

                # The custom id is constructed from the manually defined static_id and the persona type
                response_id = response_line["custom_id"].split("_")
                sample_id = "_".join(response_id[:2])
                column_id = "_".join(response_id[2:])

                response = response_line["response"]
                if "body" in response:
                    response = response["body"]

                completion = "".join(
                    c["text"]
                    for o in response["output"]
                    for c in o.get("content", [])
                    if c.get("type") == "output_text"
                )

                response_data[sample_id][column_id] = completion

        structured_response = [
            {STATIC_ID_COLUMN: static_id, **responses}
            for static_id, responses in response_data.items()
        ]

        return pd.DataFrame(structured_response)

    @staticmethod
    def read_error_file(file_name: Path) -> List[str]:
        """Reads the contents of an error file and returns a list with each unique error message.

        Args:
            file_name (Path): File path to the error file.

        Returns:
            List[str]: A list of error messages
        """

        error_messages = set()
        with file_name.open("r", encoding="utf-8") as f:
            for line in f:
                response_line = json.loads(line)
                response = response_line["response"]["body"]["error"]["message"]
                error_messages.add(response)

        return list(error_messages)
