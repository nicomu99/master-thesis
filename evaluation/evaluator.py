from typing import Dict, Optional, Iterable, List

import pandas as pd

from .dataset_handler import DatasetHandler
from .batch_request_creator import BatchRequestCreator
from .task_config import TaskConfig
from .llm_client import LLMClient
from .persona_registry import PersonaRegistry
from .dataframe_helpers import (
    columns_not_full,
    insert_if_empty,
    get_with_row_mask
)

from .log_conf import get_logger

log = get_logger(__name__)

# TODO: Allow several batches per task
# For now, we save each task id and batch id from the api

# Will have two types of requests: New personas and new Q/A answers


class Evaluator:
    """Main evaluation class
    """
    def __init__(
        self,
        include_datasets: Optional[Iterable[str]] = None,
        exclude_datasets: Optional[Iterable[str]] = None,
    ):
        self.dataset_handler = DatasetHandler()
        self.llm_client = LLMClient()
        self.batch_request_creator = BatchRequestCreator()

        self.task_configs: Dict[str, List[TaskConfig]] = self.dataset_handler.load(
            include_datasets,
            exclude_datasets
        )

        self.persona_registry = PersonaRegistry()

    def generate_static_personas(
            self,
            task_config: TaskConfig,
    ) -> Dict[str, str]:
        """Creates static personas on task level.

        The function lets an LLM client generate personas that are shared by all samples of the same task in three
        different, increasing length formats. The shortest length is defined manually. The other two formats are
        generated iteratively, using the previous persona as a prefix.

        Args:
            task_config: A task configuration with information about the task.
        """
        persona = task_config.static_persona

        # Insert base persona first
        personas = {"base_persona": persona}
        for persona_type, prompt_template in self.persona_registry.get_static_templates().items():
            log.debug("Creating persona %s", persona_type)

            client_kwargs: Dict[str, str] = {"task_type": task_config.field, "persona_string": persona }
            persona = self.llm_client.get_api_response(
                prompt_template,
                persona_type,
                **client_kwargs
            )

            personas[persona_type] = persona
        return personas

    def generate_personas(self):
        """Helper script for generating personas.

        Iterates all task configurations and creates personas for them.
        """
        # Assume data is loaded already
        log.info("Starting persona generation")

        # First, for each dataset category, create the static personas
        for dataset_id, dataset_config in self.dataset_handler.iter_datasets(
            "Processing",
            log, kind="configs"
        ):
            log.debug("Generating personas for dataset %s", dataset_id)

            # Add new column, if it does not exist yet
            self.dataset_handler.insert_columns(dataset_id, self.persona_registry.get_names())

            for task_config in self.task_configs[dataset_id]:
                task_df = self.dataset_handler.get_task_dataframe(dataset_id, task_config.name)

                if columns_not_full(task_df, self.persona_registry.get_static_names()):
                    static_personas = self.generate_static_personas(task_config)
                    for name, value in static_personas.items():
                        insert_if_empty(task_df, name, value)
                    self.dataset_handler.merge_and_write(dataset_id, task_df)

                persona_request_file = self.batch_request_creator.create_persona_request_file(
                    task_config,
                    task_df,
                    dataset_config,
                    self.persona_registry.get_dynamic_templates()
                )
                self.llm_client.send_batch(f"{task_config.task_id}_personas", persona_request_file)

        log.debug("Finished generating static personas")

    def _check_persona_existence(
        self,
        dataframe: pd.DataFrame,
        row_mask: Optional[slice | pd.Series] = None
    ):
        return columns_not_full(dataframe, self.persona_registry.get_static_names(), row_mask)

    def send_task_requests(self):
        """

        Returns:

        """
        log.info("Sending task requests")

        for dataset_id, dataset_config, dataframe in self.dataset_handler.iter_datasets(
            "Processing",
            log, kind="items"
        ):
            for task_config in self.task_configs[dataset_id]:
                if not self.llm_client.should_send_batch(task_config.task_id):
                    log.debug("Skipping %s, batch already sent", task_config.task_id)
                    continue

                log.debug("Sending %s", task_config.task_id)

                task_df = get_with_row_mask(dataframe, dataset_config.task_column, task_config.name)
                if not self._check_persona_existence(task_df):
                    log.warning(
                    "Peronas for task %s not created yet. Please run persona creation first.",
                        task_config.task_id
                    )
                    continue

                task_file = self.batch_request_creator.create_task_request_file(
                    task_config,
                    task_df,
                    dataset_config,
                    self.persona_registry.get_names()
                )

                self.llm_client.send_batch(task_file, task_config.task_id)
        log.debug("Finished sending task requests")

    def check_batch_statuses(self):
        """Prints the batch statuses."""
        self.llm_client.check_batch_statuses()

    def fetch_batch_responses(self):
        """Fetches batch responses and saves them to disk."""
        self.llm_client.fetch_batch_responses()

    def get_batch_infos(self):
        """Retrieves the batch info store"""
        return self.llm_client.batches_info_store

    def update_batch_info(
        self,
        task_id: str,
        new_status: str
    ):
        """Retrieves batch infos.

        Args:
            task_id (str): String identifier of the task specific task.
            new_status (str): Updated status.
        """
        self.llm_client.update_batch_info(task_id, new_status)
