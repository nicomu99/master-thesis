from typing import Dict, Optional, Iterable, List

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .dataset_handler import DatasetHandler
from .dataset_config import DatasetConfig
from .task_config import TaskConfig
from .llm_client import LLMClient
from .dataframe_helpers import (
    add_empty_column,
    insert_if_empty,
    is_not_full_column,
    construct_row_mask,
    get_unique_value
)
from .prompt_templates import (
    STATIC_SHORT_TEMPLATE,
    STATIC_LONG_TEMPLATE,
    DYNAMIC_SHORT_TEMPLATE,
    DYNAMIC_LONG_TEMPLATE,
    OPEN_QUESTION_TEMPLATE,
    MC_QUESTION_TEMPLATE,
    SUMMARIZATION_TEMPLATE
)

from .log_conf import get_logger

log = get_logger(__name__)

# TODO: Download open ai client responses
# TODO: Send requests: Check if personas have already been generated
# For now, we save each task id and batch id from the api

# Will have two types of requests: New personas and new Q/A answers


class Evaluator:
    """Main evaluation class
    """
    def __init__(
        self,
        llm_client: LLMClient,
        include_datasets: Optional[Iterable[str]] = None,
        exclude_datasets: Optional[Iterable[str]] = None,
    ):
        self.temp_path = Path("temp")
        self.dataset_handler = DatasetHandler()
        self.llm_client = llm_client

        self.task_configs: Dict[str, List[TaskConfig]] = self.dataset_handler.load(
            include_datasets,
            exclude_datasets
        )

        self.persona_types = [
            "base_persona", "static_short_persona", "static_long_persona",
            "dynamic_short_persona", "dynamic_long_persona"
        ]
        self.persona_prompt_templates = {
            "static_short_persona":     STATIC_SHORT_TEMPLATE,
            "static_long_persona":      STATIC_LONG_TEMPLATE,
            "dynamic_short_persona":    DYNAMIC_SHORT_TEMPLATE,
            "dynamic_long_persona":     DYNAMIC_LONG_TEMPLATE
        }

    def generate_static_personas(
            self,
            dataframe: pd.DataFrame,
            task_config: TaskConfig,
            dataset_config: DatasetConfig,
    ):
        """Creates static personas on task level.

        The function lets an LLM client generate personas that are shared by all samples of the same task in three
        different, increasing length formats. The shortest length is defined manually. The other two formats are
        generated iteratively, using the previous persona as a prefix.

        Args:
            task_config: A task configuration with information about the task.
            dataframe: A DataFrame holding samples for which the personas should be generated.
            dataset_config: A DataConfig object holding information about ``dataset``.
        """
        task_name = task_config.name
        task_column = dataset_config.task_column
        persona = task_config.static_persona

        # Insert base persona first
        base_persona_column = "base_persona"
        row_mask = construct_row_mask(dataframe, task_column, task_name)
        insert_if_empty(dataframe, base_persona_column, persona, row_mask)

        for static_persona_type in ["static_short_persona", "static_long_persona"]:
            if is_not_full_column(dataframe, static_persona_type, row_mask):
                log.debug("Creating persona %s", static_persona_type)
                prompt_template = self.persona_prompt_templates[static_persona_type]
                client_kwargs: Dict[str, str] = {
                    "task_type": task_config.field,
                    "persona_string": persona
                }
                persona = self.llm_client.get_api_response(
                    prompt_template,
                    static_persona_type,
                    **client_kwargs
                )

                dataframe.loc[row_mask, static_persona_type] = persona
            else:

                # Keep value in case if short persona gets skipped
                log.debug("Persona type %s already exists", static_persona_type)
                persona = get_unique_value(dataframe, static_persona_type, row_mask)

    def generate_personas(self):
        """Helper script for generating personas.

        Iterates all task configurations and creates personas for them.
        """
        # Assume data is loaded already
        log.info("Starting persona generation")

        # First, for each dataset category, create the static personas
        for dataset_id, dataset_config, dataframe in self.dataset_handler.iter_datasets(
            "Processing",
            log, kind="items"
        ):
            log.debug("Generating personas for dataset %s", dataset_id)

            # Add new column, if it does not exist yet
            for persona_type in self.persona_types:
                add_empty_column(dataframe, persona_type)

            for task_config in self.task_configs[dataset_id]:
                self.generate_static_personas(dataframe, task_config, dataset_config)
                self.dataset_handler.write_dataframe(dataset_id)
                # self.generate_dynamic_personas(dataframe, task_config, dataset_config)
                # Create the task specific personas

        log.debug("Finished generating static personas")
        # Then, create personas specific to the sample
        # if data_config.task_column != "":
        #     # Some datasets do not have subtasks, in which case we do not have to filter
        #     task_data = task_data.filter(lambda x: x[data_config] == task.name)
        # Save everything to disk

    @staticmethod
    def create_question_prompt(
        question_type: str,
        task_data: Dict[str, str],
        question_key: str,
        answer_key: Optional[str] = None,
    ) -> str:
        """Returns a filled question prompt template.

        For a given question type, fetches the correct prompt template and fills all placeholders.

        Args:
            question_type (str): String identifier of the correct question template. Must be 'open_question',
                'mc_question' or 'summarization'.
            task_data (Dict[str, str]): A dictionary containing the relevant information to fill into placeholders.
            question_key (str): A string identifier corresponding to the key of the question in ``task_data``.
            answer_key (Optional[str], optional): A string identifier corresponding to the key of the answers in
                ``task_data``. Defaults to None.

        Returns:
            str: Returns the filled template string.

        Raises:
            ValueError: Wrong question type used.
        """
        if question_type == "mc_question":
            template = MC_QUESTION_TEMPLATE

            assert isinstance(answer_key, str)
            prompt_kwargs = {
                f"choice_{i + 1}": task_data[answer_key][i] for i in range(4)
            }
            prompt_kwargs["question"] = task_data[question_key]
        elif question_type == "open_question":
            template = OPEN_QUESTION_TEMPLATE
            prompt_kwargs = {"question": task_data[question_key]}
        elif question_type == "summarization":
            template = SUMMARIZATION_TEMPLATE
            prompt_kwargs = {"text": task_data[question_key]}
        else:
            raise ValueError(
                f"Question type {question_type} not recognized. "
                "Should be 'mc_question', 'open_question' or 'summarization'"
            )

        return template.format(**prompt_kwargs)

    def create_task_request_file(
        self,
        task_config: TaskConfig,
        dataframe: pd.DataFrame,
        dataset_config: DatasetConfig,
    ) -> str:
        """_summary_

        Args:
            task_config (TaskConfig): _description_
            dataframe (pd.DataFrame): _description_
            dataset_config (DatasetConfig): _description_

        Returns:
            str: _description_
        """
        task_request_file = f"{self.temp_path}/{task_config.task_id}_request.jsonl"

        with open(task_request_file, "w", encoding="utf-8") as f:
            for row in tqdm(dataframe.itertuples(index=False), total=len(dataframe)):
                # noinspection PyCallingNonCallable
                row_dict = row._asdict()  # type: ignore

                prompt = self.create_question_prompt(
                    dataset_config.question_type,
                    row_dict,
                    dataset_config.question_field,
                    dataset_config.answer_field
                )
                for persona_type in ["base_persona", "static_short_persona", "static_long_persona"]:
                    api_request_dict = {
                        "custom_id": f"{row_dict["static_id"]}_{persona_type}",
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": {
                            "model": "gpt-5-nano",
                            "instructions": row_dict[persona_type],
                            "input": prompt,
                        }
                    }

                    f.write(json.dumps(api_request_dict) + "\n")
        return task_request_file

    def send_task_requests(self):
        """

        Returns:

        """
        log.info("Sending task requests")
        self.temp_path.mkdir(parents=True, exist_ok=True)

        for dataset_id, dataset_config, dataframe in self.dataset_handler.iter_datasets(
            "Processing",
            log, kind="items"
        ):
            for task_config in self.task_configs[dataset_id]:
                if task_config.task_id in self.llm_client.batches_info_store:
                    log.debug("Skipping %s, batch already sent", task_config.task_id)
                    continue

                row_mask = construct_row_mask(dataframe, dataset_config.task_column, task_config.name)
                task_df = dataframe[row_mask]
                task_file = self.create_task_request_file(task_config, task_df, dataset_config)

                self.llm_client.send_batch(task_file, task_config.task_id)
        log.debug("Finished sending task requests")
