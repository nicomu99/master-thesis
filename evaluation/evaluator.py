from typing import Dict, Optional, Iterable, List

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .dataset_handler import DatasetHandler
from .llm_client import LLMClient
from .persona_registry import PersonaRegistry
from .batch_request_handler import BatchRequestHandler
from .utils import columns_not_full, load_dataclass_dict, save_dataclass_dict, load_task_config
from .utils import TaskInfo, BatchInfo, TEMP_PATH

from .utils import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Evaluator:
    """Main evaluation class."""

    def __init__(
        self,
        include_datasets: Optional[Iterable[str]] = None,
        exclude_datasets: Optional[Iterable[str]] = None,
    ):
        self.dataset_handler = DatasetHandler(include_datasets, exclude_datasets)

        self.llm_client = LLMClient()
        self.persona_registry = PersonaRegistry()

        self.config_path = "config_task.json"

        self.task_info_file = TEMP_PATH / "task_info.json"
        self.task_infos: Dict[str, TaskInfo] = {}
        self._load()

    def _load(self):
        task_configs = load_task_config(self.config_path)
        dataset_configs = self.dataset_handler.dataset_configs

        if self.task_info_file.exists():
            task_infos = load_dataclass_dict(
                self.task_info_file, TaskInfo,
                key_field=TaskInfo.get_key_field())

            self.task_infos = {
                task_id: task_infos.get(task_id, info)
                for task_id, info in task_configs.items()
                if info.dataset_id in dataset_configs}
        else:
            self.task_infos = task_configs

    def _save(self):
        save_dataclass_dict(
            self.task_info_file, self.task_infos,
            key_field=TaskInfo.get_key_field())

    def get_personas_pending_tasks(self) -> List[str]:
        """Returns a list with all tasks that have missing personas.

        Returns:
            List[str]: A list of string identifiers of tasks that have missing personas.
        """
        return [k for k, v in self.task_infos.items() if v.is_personas_pending()]

    def get_answer_pending_tasks(self) -> List[str]:
        """Returns a list with all tasks that have missing answers.

        Returns:
            List[str]: A list of string identifiers of tasks that have missing answers.
        """
        return [k for k, v in self.task_infos.items() if v.is_answers_pending()]

    def get_judgment_pending_tasks(self) -> List[str]:
        """Returns a list with all tasks that have missing judgment evaluation.

        Returns:
            List[str]: A list of string identifiers of tasks that have missing judgment evaluation.
        """
        return [k for k, v in self.task_infos.items() if v.is_judgment_pending()]

    def generate_static_personas(
            self,
            task_info: TaskInfo,
    ) -> Dict[str, str]:
        """Creates static personas on task level.

        The function lets an LLM client generate personas that are shared by all samples of the same task in three
        different, increasing length formats. The shortest length is defined manually. The other two formats are
        generated iteratively, using the previous persona as a prefix.

        Args:
            task_info (TaskInfo): A task configuration with information about the task.
        """
        persona = self.persona_registry.get_base_persona_string()
        personas = {}
        for persona_name, prompt_template in self.persona_registry.get_static_templates().items():
            client_kwargs = {
                "task_type": task_info.field, "persona_string": persona}
            persona = self.llm_client.get_api_response(
                prompt_template, persona_name, **client_kwargs)

            personas[persona_name] = persona
        return personas

    def generate_all_static_personas(self) -> None:
        """Creates static personas on task level for all tasks."""
        for _, task_info in self.task_iterator(desc="Processing"):
            dataset_id = task_info.dataset_id
            category_name = task_info.category_name
            task_df = self.dataset_handler.get_task_dataframe(dataset_id, category_name)

            if columns_not_full(task_df, self.persona_registry.get_empty_names()):
                empty_personas = self.persona_registry.get_empty_templates()
                task_df = task_df.assign(**empty_personas)
                self.dataset_handler.merge_and_write(dataset_id, task_df)

            if columns_not_full(task_df, self.persona_registry.get_static_names()):
                print("Creating static personas...")

                static_personas = self.generate_static_personas(task_info)
                task_df = task_df.assign(**static_personas)
                self.dataset_handler.merge_and_write(dataset_id, task_df)

            if not columns_not_full(task_df, self.persona_registry.get_names()):
                task_info.skip_personas()
        self._save()

    def generate_personas(
        self,
        task_id: str
    ) -> int:
        """Generates static personas and sends requests for dynamic personas.

        Args:
            task_id (str): String identifier of the task.

        Returns:
            int: Number of requests sent to the LLM API.
        """
        task_info = self.task_infos[task_id]
        dataset_id = task_info.dataset_id
        category_name = task_info.category_name
        task_df = self.dataset_handler.get_task_dataframe(dataset_id, category_name)

        if columns_not_full(task_df, self.persona_registry.get_empty_names()):
            empty_personas = self.persona_registry.get_empty_templates()
            task_df = task_df.assign(**empty_personas)
            self.dataset_handler.merge_and_write(dataset_id, task_df)

        if columns_not_full(task_df, self.persona_registry.get_static_names()):
            print("Creating static personas...")

            static_personas = self.generate_static_personas(task_info)
            task_df = task_df.assign(**static_personas)
            self.dataset_handler.merge_and_write(dataset_id, task_df)

        request_count = 0
        if columns_not_full(task_df, self.persona_registry.get_dynamic_names()):
            print("Creating dynamic personas...")

            request_count = self.llm_client.send_persona_batch(
                task_info, task_df, self.persona_registry.get_dynamic_configs())

        if request_count == 0:
            task_info.skip_personas()
        else:
            task_info.increment_status()
        self._save()
        return request_count

    def send_answer_requests(
        self,
        task_id: str
    ) -> int:
        """Sends answer completion requests to the LLM API.

        Args:
            task_id (str): String identifier of the task.

        Returns:
            int: Number of requests sent to the LLM API.
        """
        task_info = self.task_infos[task_id]
        dataset_id = task_info.dataset_id
        dataset_config = self.dataset_handler.get_config(dataset_id)
        question_type = dataset_config.question_type
        persona_names, persona_configs = self.persona_registry.get_names_and_configs()
        task_df = self.dataset_handler.get_task_dataframe(dataset_id, task_info.category_name)

        if not task_info.is_answers_pending():
            log.warning(
                "Skipping %s, make sure personas were generated and no batch is currently being processed.",
                task_id)
            return 0

        if columns_not_full(task_df, persona_names):
            log.warning(
                "Personas for task %s not created yet. Please run persona creation first.",
                task_id)
            return 0

        request_count = self.llm_client.send_answer_batch(
            task_info, task_df, question_type, persona_configs)

        task_info.increment_status()
        self._save()
        return request_count

    def send_judgment_requests(
        self,
        task_id: str
    ) -> int:
        """Sends judgment evaluation requests to the LLM API.

        Args:
            task_id (str): String identifier of the task.

        Returns:
            int: Number of requests sent to the LLM API.
        """
        task_info = self.task_infos[task_id]
        dataset_id = task_info.dataset_id
        persona_configs = self.persona_registry.get_configs()

        task_df = self.dataset_handler.get_task_dataframe(dataset_id, task_info.category_name)

        if not task_info.is_judgment_pending():
            log.warning(
                "Skipping %s, make sure answers were generated and no batch is currently being processed.",
                task_id)
            return 0

        request_count = self.llm_client.send_judgment_batch(
            task_info, task_df, persona_configs)

        task_info.increment_status()
        self._save()
        return request_count

    def check_batch_statuses(self) -> Dict[str, BatchInfo]:
        """Prints the batch statuses.

        Returns:
            Dict[str, BatchInfo]: Dictionary with batch information of active batches.
        """
        active_batches = self.llm_client.check_batch_statuses()
        for bid, batch_info in active_batches.items():
            if batch_info.is_error():
                self.task_infos[bid].decrement_status()
        return active_batches

    def fetch_batch_responses(self):
        """Fetches batch responses and saves them to disk."""
        retrieved_batches = self.llm_client.download_batch_files()

        for batch_info in retrieved_batches:
            task_info = self.task_infos[batch_info.task_id]
            dataset_id = task_info.dataset_id

            if batch_info.has_output():
                output_file = batch_info.get_output_file()
                response_df = BatchRequestHandler.read_response_file(output_file.local_file_path)
                self.dataset_handler.merge_and_write(dataset_id, response_df)

            if batch_info.has_error():
                error_file = batch_info.get_error_file()
                error_messages = BatchRequestHandler.read_error_file(error_file.local_file_path)
                for message in error_messages:
                    log.error("Task %s failed: %s", task_info.task_id, message)

            task_info.update_status(batch_info.is_retrieved())
        self._save()

    def task_iterator(
        self,
        desc: str,
    ):
        task_iterator = tqdm(self.task_infos.items(), desc=desc)
        with logging_redirect_tqdm(loggers=[log]):
            for task_id, task_info in task_iterator:
                task_iterator.set_description(f"{desc} {task_id}")
                yield task_id, task_info
