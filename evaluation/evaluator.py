from typing import Dict, Optional, Iterable, List, Set

import json
from collections import defaultdict

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .dataset_handler import DatasetHandler
from .llm_client import LLMClient
from .persona_registry import PersonaRegistry
from .utils import columns_not_full, load_dataclass_dict, save_dataclass_dict
from .utils import TaskConfig, BatchType

from .utils import get_logger

log = get_logger(__name__)

# TODO: Update batch info update (now in TaskConfig)
# TODO: For batches: Create second storage data struct for batches that have been downloaded already.


class Evaluator:
    """Main evaluation class
    """
    def __init__(
        self,
        include_datasets: Optional[Iterable[str]] = None,
        exclude_datasets: Optional[Iterable[str]] = None,
    ):
        self.dataset_handler = DatasetHandler(include_datasets, exclude_datasets)

        self.llm_client = LLMClient()
        self.persona_registry = PersonaRegistry()

        self.config_path = "config_task.json"
        self.task_configs: Dict[str, TaskConfig] = {}
        self._load()

    def _load(self):
        self.task_configs = load_dataclass_dict(
            self.config_path,
            TaskConfig,
            "task_id"
        )

    def _save(self):
        save_dataclass_dict(
            self.config_path,
            self.task_configs,
            "task_id"
        )

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

            client_kwargs: Dict[str, str] = {"task_type": task_config.field, "persona_string": persona}
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
        for task_id, task_config in self.task_iterator(desc="Processing"):
            log.info("Generating personas for %s", task_id)
            if not task_config.generate_personas:
                log.debug("Skipping persona generation")
                continue

            # Add new column, if it does not exist yet
            dataset_id = task_config.dataset_id
            self.dataset_handler.insert_columns(dataset_id, self.persona_registry.get_names())

            task_df = self.dataset_handler.get_task_dataframe(dataset_id, task_config.category_name)
            if columns_not_full(task_df, self.persona_registry.get_static_names()):
                log.debug("Creating static personas")

                static_personas = self.generate_static_personas(task_config)
                task_df = task_df.assign(**static_personas)
                self.dataset_handler.merge_and_write(dataset_id, task_df)

            if columns_not_full(task_df, self.persona_registry.get_dynamic_names()):
                log.debug("Creating dynamic personas")

                self.llm_client.send_persona_batch(
                    task_config, task_df,
                    self.persona_registry.get_dynamic_templates())

            task_config.generate_personas = False
        self._save()


    def send_task_requests(self):
        """

        Returns:

        """
        log.info("Sending task requests.")
        # So we have batches_info_store that has batch_id, batch_info items.
        # batch_info can both be persona batches or task batches.
        # Before sending a task batch, we should check whether we have already sent a request for it
        # Solutions:
        # add flag to TaskConfig to check whether it has already been sent
        # simply iterate batch info

        persona_names = self.persona_registry.get_names()
        for task_id, task_config in self.task_iterator(desc="Processing"):
            log.info("Sending answer request %s", task_id)
            if not task_config.generate_answers:
                log.debug("Skipping %s, batch already sent", task_id)
                continue

            dataset_id = task_config.dataset_id
            question_type = self.dataset_handler.get_config(dataset_id).question_type
            task_df = self.dataset_handler.get_task_dataframe(dataset_id, task_config.category_name)
            if columns_not_full(task_df, persona_names):
                log.warning(
                    "Personas for task %s not created yet. Please run persona creation first.",
                    task_id
                )
                continue

            self.llm_client.send_task_batch(
                task_config, task_df, question_type, persona_names)

            task_config.generate_answers = False
        self._save()

    def task_iterator(
        self,
        desc: str,
    ):
        task_iterator = tqdm(self.task_configs.items(), desc=desc)
        with logging_redirect_tqdm(loggers=[log]):
            for task_id, task_config in task_iterator:
                task_iterator.set_description(f"{desc} {task_config.task_id}")
                yield task_id, task_config

    def check_batch_statuses(self):
        """Prints the batch statuses."""
        self.llm_client.check_batch_statuses()

    @staticmethod
    def get_response_data(file_name: str) -> List[Dict]:
        response_data = defaultdict(lambda: defaultdict(str))

        with open(file_name, "rb") as f:
            for line in f:
                response_line = json.loads(line)

                response_id = response_line["custom_id"].split("_")
                sample_id = "_".join(response_id[:2])
                persona = "_".join(response_id[2:])

                response = response_line["response"]
                if "body" in response:
                    response = response["body"]

                completion = "".join(
                    c["text"]
                    for o in response["output"]
                    for c in o.get("content", [])
                    if c.get("type") == "output_text"
                )

                response_data[sample_id][persona] = completion

        structured_response = [
            {"static_id": static_id, **responses}
            for static_id, responses in response_data.items()
        ]

        return structured_response

    @staticmethod
    def get_error_data(file_name: str) -> Set[str]:

        error_messages = set()
        with open(file_name, "rb") as f:
            for line in f:
                response_line = json.loads(line)
                response = response_line["response"]["body"]["error"]["message"]
                error_messages.add(response)

        return error_messages

    def fetch_batch_responses(self):
        """Fetches batch responses and saves them to disk."""
        retrieved_batches = self.llm_client.fetch_batch_responses()
        for batch_info in retrieved_batches:
            task_id = batch_info.task_id
            dataset_id = self.task_configs[task_id].dataset_id

            output_file = batch_info.local_output_file
            if output_file:
                response_dict = self.get_response_data(output_file)
                self.dataset_handler.merge_and_write(dataset_id, response_dict)

            error_file = batch_info.local_error_file
            if error_file:
                error_messages = self.get_error_data(error_file)
                batch_type = batch_info.batch_type
                if batch_type == BatchType.PERSONAS:
                    self.task_configs[task_id].generate_personas = True
                for message in error_messages:
                    log.error("Task %s %s failed", task_id, batch_info.batch_type)
                    log.error(message)
        self._save()

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
