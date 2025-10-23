from typing import Any, Dict, List

from pathlib import Path

import pandas as pd
from openai import OpenAI, APIConnectionError

from .batch_request_handler import BatchRequestHandler
from .persona_registry import PersonaConfig
from .utils import TaskInfo, BatchInfo, BatchType, QuestionType, BatchStatus
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
            BatchInfo.get_key_field()
        )

    def _save(self):
        save_dataclass_dict(self.batches_info_file, self.batches_info_store, BatchInfo.get_key_field())

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
                model=self.model, input=prompt,
                prompt_cache_key=prompt_cache_key)
        else:
            response = self.client.responses.create(
                model=self.model, input=prompt)
        return response.output_text

    def send_persona_batch(
        self,
        task_info: TaskInfo,
        task_df: pd.DataFrame,
        persona_configs: List[PersonaConfig]
    ) -> int:
        """Sends a persona batch request to the LLM API.

        Args:
            task_info (TaskConfig): Configuration parameters of the task.
            task_df (pd.DataFrame): Dataframe containing task samples.
            persona_configs (List[PersonaConfig]): List of persona configurations.

        Returns:
            int: Number of sent requests.
        """
        request_count = 0
        try:
            file_name, request_count = BatchRequestHandler.create_persona_batch_request_file(
                task_info, task_df, persona_configs, self.model)

            self.send_batch(
                task_info.task_id, file_name, BatchType.PERSONAS)
        except OSError as e:
            log.error(
                "Unexpected error for task %s: %s",
                task_info.task_id, e, exc_info=True)
        return request_count

    def send_answer_batch(
        self,
        task_info: TaskInfo,
        task_df: pd.DataFrame,
        question_type: QuestionType,
        persona_configs: List[PersonaConfig]
    ) -> int:
        """Sends a task question answering request to the LLM API.

        Args:
            task_info (TaskConfig): Configuration parameters of the task.
            task_df (pd.DataFrame): Dataframe containing task samples.
            question_type (QuestionType): The type of questions of the task samples.
            persona_configs (List[PersonaConfig]): The persona configurations to use for generating answers.

        Returns:
            int: Number of sent requests.
        """
        request_count = 0
        try:
            file_name, request_count = BatchRequestHandler.create_answer_batch_request_file(
                task_info, task_df, question_type, persona_configs, self.model)

            self.send_batch(
                task_info.task_id, file_name, BatchType.ANSWERS)
        except OSError as e:
            log.error(
                "Unexpected error for task %s: %s",
                task_info.task_id, e, exc_info=True)
        return request_count

    def send_judgment_batch(
        self,
        task_info: TaskInfo,
        task_df: pd.DataFrame,
        persona_configs: List[PersonaConfig]
    ) -> int:
        """Sends a LLM-as-a-judge request to the LLM API.

        Args:
            task_info (TaskConfig): Task metadata.
            task_df (pd.DataFrame): Dataframe containing samples associated with this task.
            persona_configs (List[PersonaConfig]): Persona configuration list.

        Returns:
            int: Number of sent requests.
        """
        request_count = 0
        try:
            file_name, request_count = BatchRequestHandler.create_judgment_batch_request_file(
                task_info, task_df, persona_configs, self.model)

            self.send_batch(
                task_info.task_id, file_name, BatchType.JUDGMENT)
        except ValueError as e:
            log.error(
                "Failed to create and send judgment batch for task %s: %s",
                task_info.task_id, e, exc_info=True)
        except OSError as e:
            log.error(
                "Unexpected error for task %s: %s",
                task_info.task_id, e, exc_info=True)
        return request_count

    def send_batch(
        self,
        task_id: str,
        batch_file: Path,
        batch_type: BatchType
    ):
        """Sends batch files to the llm api.

        Args:
            task_id (str): String identifier of the task the batch file belongs to.
            batch_file (Path): File name of the file containing the request objects.
            batch_type (str): Enumeration indicating whether the batch contains persona generation requests
                or task answering prompts.
        """

        log.info("Sending batch for task %s", task_id)

        with batch_file.open("rb") as f:
            batch_input_file = self.client.files.create(
                file=f, purpose="batch")

            batch_input_file_id = batch_input_file.id
            batch_job = self.client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/responses",
                completion_window="24h")

            batch_info = BatchInfo(batch_job.id, task_id, batch_type)
            self.batches_info_store[batch_job.id] = batch_info

            self._save()

    def check_batch_statuses(self) -> Dict[str, BatchInfo]:
        """Fetches and returns statuses of batch requests.

        Returns:
            Dict[str, BatchInfo]: A dictionary of batch information.
        """
        active_batches = {
            k: v for k, v in self.batches_info_store.items()
            if not v.has_finished()}

        for batch_id, batch_info in active_batches.items():
            try:
                remote_batch = self.client.batches.retrieve(batch_id)
                batch_info.update_status(remote_batch.status)

                if batch_info.is_failed():
                    errors = getattr(remote_batch, "errors", None)
                    if not errors or not getattr(errors, "data", None):
                        log.warning("Batch failed with no error details.")
                        continue
                    for error in errors.data:
                        log.warning(
                            "Batch for task %s failed: %s %s",
                            batch_info.task_id, error.code, error.message)
                # TODO: Extract common logic
                elif batch_info.is_in_progress():
                    request_counts = getattr(remote_batch, "request_counts", None)
                    if not request_counts:
                        continue
                    batch_info.progress_message = (
                        f"Progress: {request_counts.completed} out of {request_counts.total} finished; "
                        f"{request_counts.failed} requests failed.")

                elif batch_info.is_completed():
                    if remote_batch.output_file_id:
                        batch_info.init_output_file(remote_batch.output_file_id)
                    if remote_batch.error_file_id:
                        batch_info.init_error_file(remote_batch.error_file_id)

                    request_counts = getattr(remote_batch, "request_counts", None)
                    if not request_counts:
                        continue
                    batch_info.progress_message = (
                        f"Progress: {request_counts.completed} out of {request_counts.total} finished; "
                        f"{request_counts.failed} requests failed.")

            except APIConnectionError as e:
                log.error("Error: %s", e, exc_info=True)
        self._save()
        return active_batches

    def _download_batch_file(
        self,
        remote_file_id: str,
        local_file_path: Path
    ):
        batch_response_stream = self.client.files.content(remote_file_id)

        with local_file_path.open("wb") as f:
            f.write(batch_response_stream.read())

    def download_batch_files(self) -> List[BatchInfo]:
        """Fetches responses for batch requests and saves them to files.

        Returns:
            List[BatchInfo]: A list with batch information of batches that finished.
        """
        self.check_batch_statuses()
        completed_batches = [
            batch_info for batch_info in self.batches_info_store.values()
            if batch_info.is_completed()]
        log.info("Fetching batch responses, %s completed batches found.", len(completed_batches))

        for batch_info in completed_batches:
            if batch_info.has_output():
                output_file = batch_info.get_output_file()
                self._download_batch_file(output_file.remote_file_id, output_file.local_file_path)
                batch_info.set_status(BatchStatus.RETRIEVED)

            if batch_info.has_error():
                error_file = batch_info.get_error_file()
                self._download_batch_file(error_file.remote_file_id, error_file.local_file_path)
                batch_info.set_status(BatchStatus.ERROR)

        self._save()
        return completed_batches
