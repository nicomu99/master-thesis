from typing import Dict, Optional, Iterable, List, Any

from collections import defaultdict
from pathlib import Path

import pandas as pd
from openai import OpenAI

from dataset_handler import DatasetHandler
from dataset_config import DatasetConfig
from task_config import TaskConfig
from dataframe_helpers import (
    add_empty_column,
    insert_if_empty,
    is_not_full_column,
    construct_row_mask,
    get_unique_value
)
from prompt_templates import (
    STATIC_SHORT_TEMPLATE,
    STATIC_LONG_TEMPLATE,
    DYNAMIC_SHORT_TEMPLATE,
    DYNAMIC_LONG_TEMPLATE
)

from log_conf import get_logger

log = get_logger(__name__)


class Evaluator:
    """Main evaluation class
    """
    def __init__(self):
        self.config_path = Path("config.json")
        self.dataset_handler = DatasetHandler()

        self.task_configs: Dict[str, List[TaskConfig]] = defaultdict(list)

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

        self.client = OpenAI()

    def main(
        self,
        include_datasets: Optional[Iterable[str]] = None,
        exclude_datasets: Optional[Iterable[str]] = None,
    ) -> None:
        """_summary_

        Args:
            include_datasets (Optional[Iterable[str]], optional): _description_. Defaults to None.
            exclude_datasets (Optional[Iterable[str]], optional): _description_. Defaults to None.
        """
        self.task_configs = self.dataset_handler.load(include_datasets, exclude_datasets)
        self.generate_personas()

    def get_openai_api_response(
            self,
            template: str,
            prompt_cache_key: str,
            **kwargs: Any
    ) -> str:
        """Helper function for getting output from the OpenAI API.

        The function fills the template with the given keyword arguments and sends the prompt to the
        OpenAI API.

        Args:
            template: A string template with placeholders.
            prompt_cache_key: A key to be used for caching API calls.
            **kwargs: Keyword arguments to be inserted into ``template``. Must match the placeholders.

        Returns:
            str: The generated text returned by the API.
        """
        prompt = template.format(**kwargs)
        response = self.client.responses.create(
            model="gpt-5-nano",
            input=prompt,
            prompt_cache_key=prompt_cache_key
        )

        return response.output_text

    def create_static_personas(
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
            dataset: A DataFrame holding samples for which the personas should be generated.
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
                    "task_type": task_config.name,
                    "persona_string": persona
                }
                persona = self.get_openai_api_response(
                    prompt_template,
                    static_persona_type,
                    **client_kwargs
                )
            else:
                dataframe.loc[row_mask, static_persona_type] = persona

                # Keep value in case if short persona gets skipped
                log.debug("Personas already exist %s", static_persona_type)
                persona = get_unique_value(dataframe, static_persona_type, row_mask)

    def _write_dataframe(
            self,
            dataset_id: str,
            dataset: pd.DataFrame
    ):
        """Writes the dataframe to the data folder.

        Args:
            dataset_id: A string identifier that will be used as the name of the file.
            dataset: A dataframe object that should be written to a file.
        """

    def generate_personas(self):
        """Helper script for generating personas.

        Iterates all task configurations and creates personas for them.
        """
        # Assume data is loaded already
        log.info("Generating static personas")

        # First, for each dataset category, create the static personas
        for dataset_id, dataset_config, dataframe in self.dataset_handler.iter_datasets("Processing", kind="items"):
            log.debug("Generating personas for dataset %s", dataset_id)

            # Add new column, if it does not exist yet
            for persona_type in self.persona_types:
                add_empty_column(dataframe, persona_type)

            for task_config in self.task_configs[dataset_id]:
                self.create_static_personas(dataframe, task_config, dataset_config)
                self.dataset_handler.write_dataframe(dataset_id)
                # Create the task specific personas

        log.info("Finished generating static personas")
        # Then, create personas specific to the sample
        # if data_config.task_column != "":
        #     # Some datasets do not have subtasks, in which case we do not have to filter
        #     task_data = task_data.filter(lambda x: x[data_config] == task.name)
        # Save everything to disk


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.main()
