from typing import Any, Dict, Optional, List

import json
import string
from pathlib import Path
from collections import defaultdict

import pandas as pd

from .utils import TaskConfig, QuestionType, BatchType
from .utils import QUESTION_COLUMN, ANSWER_COLUMN
from .prompt_templates import OPEN_QUESTION_TEMPLATE, MC_QUESTION_TEMPLATE, SUMMARIZATION_TEMPLATE


class BatchRequestHandler:
    """Helper class used to create json files with request data.
    
    Attributes:
        temp_path (Path): A temporary directory path. Request files will be saved into this directory.
    """
    def __init__(self):
        self.model = "gpt-5-nano"

        self.temp_path = Path("temp")
        self.temp_path.mkdir(parents=True, exist_ok=True)

    def _write_prompt_to_file(
        self,
        f: Any,
        custom_id: str,
        prompt: str,
        instruction: Optional[str] = None
    ):
        """Function that writes a prompt reqeust in JSON format to a file.

        Args:
            f (Any): File output buffer. The request will be written to this file.
            custom_id (str): An identifier, which can be used to map client outputs to the input samples.
            prompt (str): Request prompt.
            instruction (str | None): A system prompt overwrite string. If this is empty, the default system prompt
                will be used. Defaults to None.
        """
        body = {"model": self.model, "input": prompt}
        if instruction:
            body["instructions"] = instruction

        api_request_dict = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body
        }

        f.write(json.dumps(api_request_dict) + "\n")

    @staticmethod
    def create_question_prompt(
        task_data: Dict[str, str],
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

        Raises:
            ValueError: Wrong question type used.
        """

        prompt_kwargs = {"question": task_data[QUESTION_COLUMN]}
        if question_type == QuestionType.MC:
            template = MC_QUESTION_TEMPLATE
            letters = string.ascii_uppercase
            choices = "\n".join(
                f"({letters[i]}) {answer}" for i, answer in enumerate(task_data[ANSWER_COLUMN])
            )
            prompt_kwargs["choices"] = choices
        elif question_type == QuestionType.OPEN:
            template = OPEN_QUESTION_TEMPLATE
        elif question_type == QuestionType.SUMMARIZATION:
            template = SUMMARIZATION_TEMPLATE

        else:
            raise ValueError(
                f"Question type {question_type} not recognized. "
                "Should be 'mc', 'open' or 'summarization'"
            )
        return template.format(**prompt_kwargs)

    def create_task_request_file(
        self,
        task_config: TaskConfig,
        dataframe: pd.DataFrame,
        question_type: QuestionType,
        persona_types: List[str]
    ) -> str:
        """Creates a request file for task question answering.

        Args:
            task_config (TaskConfig): Task configuration attributes.
            dataframe (pd.DataFrame): Dataframe containing samples of the task.
            question_type (str): The question type of this task. Can be "mc_question", 
                "open_question" or "summarization".
            persona_types (List[str]): Persona types for which a request should be sent to the API.

        Returns:
            str: A string identifier of the created task request.
        """

        task_request_file = f"{self.temp_path}/{task_config.task_id}_request.jsonl"

        with open(task_request_file, "w", encoding="utf-8") as f:
            for row in dataframe.itertuples(index=False):
                # noinspection PyCallingNonCallable
                row_dict = row._asdict()  # type: ignore

                prompt = self.create_question_prompt(
                    row_dict,
                    question_type
                )
                for persona_type in persona_types:
                    custom_id = f"{row_dict["static_id"]}_{persona_type}"
                    self._write_prompt_to_file(f, custom_id, prompt, row_dict[persona_type])
        return task_request_file

    def create_persona_request_file(
        self,
        task_config: TaskConfig,
        dataframe: pd.DataFrame,
        persona_templates: Dict[str, str]
    ) -> str:
        """Creates a request file for persona generation.

        Args:
            task_config (TaskConfig): Task configuration attributes.
            dataframe (pd.DataFrame): Dataframe containing samples of the task.
            persona_templates (Dict[str, str]): Persona types and templates for which a request should be sent to the
                API.

        Returns:
            str: A string identifier of the created request file.
        """

        persona_request_file = f"{self.temp_path}/{task_config.task_id}_persona_request.jsonl"

        with open(persona_request_file, "w", encoding="utf-8") as f:
            for row in dataframe.itertuples(index=False):
                # noinspection PyCallingNonCallable
                row_dict = row._asdict()  # type: ignore

                for persona_type, prompt_template in persona_templates.items():
                    prompt = prompt_template.format(
                        task_type=task_config.field,
                        persona_string=task_config.static_persona,
                        question=row_dict[QUESTION_COLUMN]
                    )
                    custom_id = f"{row_dict["static_id"]}_{persona_type}"
                    self._write_prompt_to_file(f, custom_id, prompt)

        return persona_request_file

    @staticmethod
    def read_response_file(
        file_name: str,
        batch_type: BatchType = BatchType.PERSONAS
    ) -> pd.DataFrame:
        response_data = defaultdict(lambda: defaultdict(str))

        with open(file_name, "rb") as f:
            for line in f:
                response_line = json.loads(line)

                # The custom id is constructred from the manually defined static_id and the persona type
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

                if batch_type == BatchType.ANSWERS:
                    column_id = column_id.replace("persona", "answer")

                response_data[sample_id][column_id] = completion

        structured_response = [
            {"static_id": static_id, **responses}
            for static_id, responses in response_data.items()
        ]

        return pd.DataFrame(structured_response)

    @staticmethod
    def read_error_file(file_name: str) -> List[str]:

        error_messages = set()
        with open(file_name, "rb") as f:
            for line in f:
                response_line = json.loads(line)
                response = response_line["response"]["body"]["error"]["message"]
                error_messages.add(response)

        return list(error_messages)
