from typing import Dict, Optional, Iterable, List

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .dataset_handler import DatasetHandler
from .llm_client import LLMClient
from .persona_registry import PersonaRegistry
from .batch_request_handler import BatchRequestHandler
from .utils import columns_not_full, load_dataclass_dict, save_dataclass_dict
from .utils import TaskConfig, BatchType, TaskStatus

from .utils import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

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

    def get_unfinished_tasks_personas(self) -> List[str]:
        """Returns a list with all tasks that have missing personas.

        Returns:
            List[str]: A list of string identifiers of tasks that have missing personas.
        """
        return [k for k, v in self.task_configs.items() if v.status == TaskStatus.PERSONAS_PENDING]

    def get_unfinished_tasks_answers(self) -> List[str]:
        """Returns a list with all tasks that have missing answers.

        Returns:
            List[str]: A list of string identifiers of tasks that have missing answers.
        """
        return [k for k, v in self.task_configs.items() if v.status == TaskStatus.ANSWERS_PENDING]

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

    def generate_personas(
        self,
        task_id: str
    ):
        """Helper script for generating personas.

        Iterates all task configurations and creates personas for them.
        """

        # First, for each dataset category, create the static personas
        log.info("Generating personas for %s", task_id)
        task_config = self.task_configs[task_id]
        if not self.llm_client.queue_is_empty():
            log.info("Batch queue currently not empty. Please wait for other batches to finish first.")
            return

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

        task_config.status = TaskStatus.PERSONAS_REQUESTED
        self._save()


    def send_task_requests(
        self,
        task_id: str
    ):
        """

        Returns:

        """
        # So we have batches_info_store that has batch_id, batch_info items.
        # batch_info can both be persona batches or task batches.
        # Before sending a task batch, we should check whether we have already sent a request for it
        # Solutions:
        # add flag to TaskConfig to check whether it has already been sent
        # simply iterate batch info
        log.info("Sending answer request %s", task_id)

        task_config = self.task_configs[task_id]
        persona_names = self.persona_registry.get_names()
        if not task_config.status == TaskStatus.ANSWERS_PENDING:
            log.debug("Skipping %s, batch already sent", task_id)
            return

        dataset_id = task_config.dataset_id
        question_type = self.dataset_handler.get_config(dataset_id).question_type

        task_df = self.dataset_handler.get_task_dataframe(dataset_id, task_config.category_name)
        if columns_not_full(task_df, persona_names):
            log.warning(
                "Personas for task %s not created yet. Please run persona creation first.",
                task_id
            )
            return

        self.llm_client.send_task_batch(
            task_config, task_df, question_type, persona_names)

        task_config.status = TaskStatus.ANSWERS_REQUESTED
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

    def check_batch_statuses(self):
        """Prints the batch statuses."""
        self.llm_client.check_batch_statuses()

    def fetch_batch_responses(self):
        """Fetches batch responses and saves them to disk."""
        retrieved_batches = self.llm_client.fetch_batch_responses()

        for batch_info in retrieved_batches:
            task_id = batch_info.task_id
            task_config = self.task_configs[task_id]
            dataset_id = task_config.dataset_id

            output_file = batch_info.local_output_file
            if output_file:
                response_dict = BatchRequestHandler.read_response_file(output_file, batch_info.batch_type)

                self.dataset_handler.merge_and_write(dataset_id, response_dict)
                if batch_info.batch_type == BatchType.PERSONAS:
                    task_config.status = TaskStatus.ANSWERS_PENDING
                elif batch_info.batch_type == BatchType.ANSWERS:
                    task_config.status = TaskStatus.FINISHED

            error_file = batch_info.local_error_file
            if error_file:
                error_messages = BatchRequestHandler.read_error_file(error_file)

                batch_type = batch_info.batch_type
                if batch_type == BatchType.PERSONAS:
                    task_config.status = TaskStatus.PERSONAS_PENDING
                elif batch_type == BatchType.ANSWERS:
                    task_config.status = TaskStatus.ANSWERS_PENDING

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
