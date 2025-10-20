from typing import Dict, Optional, Iterable, List

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .dataset_handler import DatasetHandler
from .llm_client import LLMClient
from .persona_registry import PersonaRegistry
from .batch_request_handler import BatchRequestHandler
from .utils import columns_not_full, load_dataclass_dict, save_dataclass_dict
from .utils import TaskConfig, BatchInfo

from .utils import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


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
            "task_id")

    def _save(self):
        save_dataclass_dict(
            self.config_path,
            self.task_configs,
            "task_id")

    def get_unfinished_tasks_personas(self) -> List[str]:
        """Returns a list with all tasks that have missing personas.

        Returns:
            List[str]: A list of string identifiers of tasks that have missing personas.
        """
        return [k for k, v in self.task_configs.items() if v.is_personas_pending()]

    def get_unfinished_tasks_answers(self) -> List[str]:
        """Returns a list with all tasks that have missing answers.

        Returns:
            List[str]: A list of string identifiers of tasks that have missing answers.
        """
        return [k for k, v in self.task_configs.items() if v.is_answers_pending()]

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
        persona = self.persona_registry.get_base_persona_string()
        # Insert base persona first
        personas = {}
        for persona_name, prompt_template in self.persona_registry.get_static_templates().items():
            client_kwargs: Dict[str, str] = {
                "task_type": task_config.field, "persona_string": persona}
            persona = self.llm_client.get_api_response(
                prompt_template, persona_name, **client_kwargs)

            personas[persona_name] = persona
        return personas

    def generate_personas(
        self,
        task_id: str
    ):
        """Helper script for generating personas.

        Iterates all task configurations and creates personas for them.
        """
        log.info("Generating personas for %s", task_id)
        task_config = self.task_configs[task_id]
        dataset_id = task_config.dataset_id

        task_df = self.dataset_handler.get_task_dataframe(
            dataset_id, task_config.category_name)
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

        task_config.increment_status()
        self._save()

    def send_task_requests(
        self,
        task_id: str
    ):
        log.info("Sending answer request %s", task_id)
        task_config = self.task_configs[task_id]

        persona_names, persona_configs = self.persona_registry.get_names_and_configs()
        dataset_id = task_config.dataset_id
        dataset_config = self.dataset_handler.get_config(dataset_id)
        question_type = dataset_config.question_type

        task_df = self.dataset_handler.get_task_dataframe(dataset_id, task_config.category_name)

        if not task_config.is_answers_pending():
            log.warning(
                "Skipping %s, make sure personas were generated and no batch is currently being processed.",
                task_id)
            return

        if columns_not_full(task_df, persona_names):
            log.warning(
                "Personas for task %s not created yet. Please run persona creation first.",
                task_id)
            return

        self.llm_client.send_task_batch(
            task_config, task_df, question_type, persona_configs)

        task_config.increment_status()
        self._save()

    def task_iterator(
        self,
        desc: str,
    ):
        task_iterator = tqdm(self.task_configs.items(), desc=desc)
        with logging_redirect_tqdm():
            for task_id, task_config in task_iterator:
                task_iterator.set_description(f"{desc} {task_config.task_id}")
                yield task_id, task_config

    def check_batch_statuses(self) -> Dict[str, BatchInfo]:
        """Prints the batch statuses."""
        active_batches = self.llm_client.check_batch_statuses()

        failed_task_ids = [v.task_id for v in active_batches.values() if v.is_error()]
        for failed_id in failed_task_ids:
            self.task_configs[failed_id].decrement_status()
        return active_batches

    def fetch_batch_responses(self):
        """Fetches batch responses and saves them to disk."""
        retrieved_batches = self.llm_client.fetch_batch_responses()

        for batch_info in retrieved_batches:
            task_id = batch_info.task_id
            task_config = self.task_configs[task_id]
            dataset_id = task_config.dataset_id

            batch_type = batch_info.batch_type
            batch_output_file = batch_info.local_output_file
            batch_error_file = batch_info.local_error_file

            if batch_output_file:
                response_df = BatchRequestHandler.read_response_file(batch_output_file, batch_type)
                self.dataset_handler.merge_and_write(dataset_id, response_df)

            if batch_error_file:
                error_messages = BatchRequestHandler.read_error_file(batch_error_file)
                for message in error_messages:
                    log.error("Task %s %s failed", task_id, batch_type)
                    log.error(message)

            task_config.update_status(batch_info.is_retrieved())
        self._save()
