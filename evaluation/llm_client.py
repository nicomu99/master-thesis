from typing import Any, Dict, List

from pathlib import Path

import pandas as pd
from openai import OpenAI, APIConnectionError

from .batch_request_handler import BatchRequestHandler
from .persona_registry import PersonaConfig
from .utils import TaskConfig, BatchInfo, BatchType, QuestionType, BatchStatus
from .utils import load_dataclass_dict, save_dataclass_dict
from .utils import TEMP_PATH
from .utils import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class LLMClient:
    """Primary class for communicating with the api."""

    def __init__(self):
        self.model = "gpt-5-nano"
        self.client = OpenAI()

        self.batches_info_file = TEMP_PATH / "batches_info_file.json"
        self.batches_info_store: Dict[str, BatchInfo] = {}
        self._load()

    def _load(self) -> None:
        self.batches_info_store: Dict[str, BatchInfo] = load_dataclass_dict(
            self.batches_info_file,
            BatchInfo,
            "batch_id"
        )

    def _save(self):
        save_dataclass_dict(self.batches_info_file, self.batches_info_store, "batch_id")

    def get_api_response(
        self,
        template: str,
        prompt_cache_key: str | None = None,
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
        if prompt_cache_key:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                prompt_cache_key=prompt_cache_key
            )
        else:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
            )

        return response.output_text

    def send_persona_batch(
        self,
        task_config: TaskConfig,
        task_df: pd.DataFrame,
        persona_templates: Dict[str, str]
    ) -> None:
        """Sends a persona batch request to the LLM API.

        Args:
            task_config (TaskConfig): Configuration parameters of the task.
            task_df (pd.DataFrame): Dataframe containing task samples.
            persona_templates (Dict[str, str]): Template to use for persona generation.
        """
        batch_file_name = BatchRequestHandler.create_persona_request_file(
            task_config, task_df, persona_templates, self.model)

        self.send_batch(
            task_config.task_id, batch_file_name, BatchType.PERSONAS)

    def send_task_batch(
        self,
        task_config: TaskConfig,
        task_df: pd.DataFrame,
        question_type: QuestionType,
        persona_configs: List[PersonaConfig]
    ) -> None:
        """Sends a task question answering request to the LLM API.

        Args:
            task_config (TaskConfig): Configuration parameters of the task.
            task_df (pd.DataFrame): Dataframe containing task samples.
            question_type (QuestionType): The type of questions of the task samples.
            persona_types (List[PersonaConfig]): The persona configurations to use for generating answers.
        """
        batch_file_name = BatchRequestHandler.create_task_request_file(
            task_config, task_df, question_type, persona_configs, self.model)

        self.send_batch(
            task_config.task_id, batch_file_name, BatchType.ANSWERS)

    def send_batch(
        self,
        task_id: str,
        batch_file: str,
        batch_type: BatchType
    ):
        """Sends batch files to the llm api.

        Args:
            task_id (str): String identifier of the task the batch file belongs to.
            batch_file_name (str): File name of the file containing the request objects.
            batch_type (str): Enumeration indicating whether the batch contains persona generation requests
                or task answering prompts.
        """

        log.info("Sending batch for task %s", task_id)

        with open(batch_file, "rb") as f:
            batch_input_file = self.client.files.create(
                file=f,
                purpose="batch"
            )

            batch_input_file_id = batch_input_file.id
            batch_job = self.client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/responses",
                completion_window="24h",
            )

            batch_info = BatchInfo(batch_job.id, task_id, batch_type)
            self.batches_info_store[batch_job.id] = batch_info

            self._save()

    def check_batch_statuses(self) -> List[str]:
        """Fetches and prints statuses of batch requests.

        Returns:
            List[str]: A list of batch identifiers of batches that did not finish correctly.
        """
        active_batches = {
            k: v
            for k, v in self.batches_info_store.items()
            if not v.has_finished()
        }
        log.info("Checking batch statuses; %s active batches found.", len(active_batches))

        failed_task_ids = []
        for batch_id, batch_info in active_batches.items():
            try:
                remote_batch = self.client.batches.retrieve(batch_id)

                task_id = batch_info.task_id
                batch_info.update_status(remote_batch.status)

                log.info("Task %s %-8s batch status is: %s", task_id, batch_info.batch_type, batch_info.status)
                if batch_info.is_failed():
                    errors = getattr(remote_batch, "errors", None)
                    if not errors or not getattr(errors, "data", None):
                        log.error("Batch failed with no error details.")
                        continue

                    for error in errors.data:
                        log.error("Error %s, %s", error.code, error.message)

                    failed_task_ids.append(batch_info.task_id)
                elif batch_info.is_in_progress():
                    request_counts = getattr(remote_batch, "request_counts", None)
                    if not request_counts:
                        continue

                    log.info(
                        "Progress: %s out of %s finished; %s requests failed.",
                        request_counts.completed, request_counts.total,
                        request_counts.failed)

                elif batch_info.is_completed():
                    batch_info.output_file_id = remote_batch.output_file_id if remote_batch.output_file_id else None
                    batch_info.error_file_id = remote_batch.error_file_id if remote_batch.error_file_id else None

            except APIConnectionError:
                log.error("Connection error.")
        self._save()
        return failed_task_ids

    def _save_batch_response(
        self,
        remote_file_id: str,
        local_file_path: Path
    ):
        batch_response_stream = self.client.files.content(remote_file_id)

        with open(local_file_path, "wb") as f:
            f.write(batch_response_stream.read())

    def fetch_batch_responses(self) -> List[BatchInfo]:
        """Fetches responses for batch requests and saves them to files.

        Returns:
            List[BatchInfo]: A list with batch information of batches that finished.
        """
        self.check_batch_statuses()
        completed_batches = [
            batch_info
            for batch_info in self.batches_info_store.values()
            if batch_info.is_completed()
        ]
        log.info("Fetching batch responses, %s completed batches found.", len(completed_batches))

        for batch_info in completed_batches:
            if batch_info.output_file_id:
                local_output_file = batch_info.create_output_file()
                self._save_batch_response(batch_info.output_file_id, local_output_file)
                batch_info.set_status(BatchStatus.RETRIEVED)

            if batch_info.error_file_id:
                local_error_file = batch_info.create_error_file()
                self._save_batch_response(batch_info.error_file_id, local_error_file)
                batch_info.set_status(BatchStatus.ERROR)

        self._save()
        return completed_batches
