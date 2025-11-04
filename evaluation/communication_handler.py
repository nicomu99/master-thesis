from typing import Any, Literal
from collections.abc import Iterable, Callable

from pathlib import Path

import pandas as pd
from openai import APIConnectionError

from .clients import LLMClient, OpenAIClient, GenAIClient
from .persona_registry import PersonaConfig
from .utils import TaskInfo, BatchInfo, QuestionType, BatchType
from .batch_request_handler import BatchRequestHandler
from .utils import load_dataclass_dict, save_dataclass_dict
from .utils import TEMP_PATH
from .utils import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class CommunicationHandler:
    """Primary class for communicating with the api."""

    def __init__(self):
        self.clients = {
            "openai": OpenAIClient(),
            "genai": GenAIClient()
        }

        self.batches_info_file = TEMP_PATH / "batches_info_file.json"
        self.batches_info_store = {}
        self._load()

    def _load(self) -> None:
        self.batches_info_store = load_dataclass_dict(
            self.batches_info_file, BatchInfo, BatchInfo.get_key_field())

    def _save(self):
        save_dataclass_dict(self.batches_info_file, self.batches_info_store, BatchInfo.get_key_field())

    def _get_client(self, name: str) -> LLMClient:
        if name not in self.clients:
            raise ValueError(f"Unknown client name {name}. Must be in {self.clients.keys()}")
        return self.clients[name]

    def get_api_response(
        self,
        template: str,
        llm_name: Literal["openai", "genai"] = "openai",
        **kwargs: Any
    ) -> str:
        """Helper function for getting output from an LLM API.

        Args:
            template: A string template with placeholders.
            llm_name (Literal["openai", "genai"]): String identifier of the LLM client. Defaults to "openai".
            **kwargs: Keyword arguments to be inserted into ``template``. Must match the placeholders.

        Returns:
            str: The generated text returned by the API.
        """
        try:
            client = self._get_client(llm_name)
            return client.get_api_response(template, **kwargs)
        except (ConnectionError, ValueError) as e:
            log.error("Error: %s", e, exc_info=True)
            return "No response"

    def send_persona_batch(
        self,
        task_info: TaskInfo,
        task_df: pd.DataFrame,
        persona_configs: list[PersonaConfig],
        llm_name: Literal["openai", "genai"] = "openai"
    ) -> int | bool:
        """Sends a persona batch request to the LLM API.

        Args:
            task_info (TaskConfig): Configuration parameters of the task.
            task_df (pd.DataFrame): Dataframe containing task samples.
            persona_configs (list[PersonaConfig]): List of persona configurations.
            llm_name (Literal["openai", "genai"]): String identifier of the LLM client. Defaults to "openai".

        Returns:
            int: Number of sent requests.
        """
        return self._send_batch(
            task_info,
            task_df,
            BatchType.PERSONAS,
            BatchRequestHandler.create_persona_batch_request_file,
            persona_configs,
            llm_name=llm_name)

    def send_answer_batch(
        self,
        task_info: TaskInfo,
        task_df: pd.DataFrame,
        question_type: QuestionType,
        persona_configs: list[PersonaConfig],
        llm_name: Literal["openai", "genai"] = "openai"
    ) -> int | bool:
        """Sends a task question answering request to the LLM API.

        Args:
            task_info (TaskConfig): Configuration parameters of the task.
            task_df (pd.DataFrame): Dataframe containing task samples.
            question_type (QuestionType): The type of questions of the task samples.
            persona_configs (list[PersonaConfig]): The persona configurations to use for generating answers.
            llm_name (Literal["openai", "genai"]): String identifier of the LLM client. Defaults to "openai".

        Returns:
            int: Number of sent requests.
        """
        return self._send_batch(
            task_info,
            task_df,
            BatchType.ANSWERS,
            BatchRequestHandler.create_answer_batch_request_file,
            question_type,
            persona_configs,
            llm_name=llm_name
        )

    def send_judgment_batch(
        self,
        task_info: TaskInfo,
        task_df: pd.DataFrame,
        persona_configs: list[PersonaConfig],
        llm_name: Literal["openai", "genai"] = "openai"
    ) -> int | bool:
        """Sends a LLM-as-a-judge request to the LLM API.

        Args:
            task_info (TaskConfig): Task metadata.
            task_df (pd.DataFrame): Dataframe containing samples associated with this task.
            persona_configs (list[PersonaConfig]): Persona configuration list.
            llm_name (Literal["openai", "genai"]): String identifier of the LLM client. Defaults to "openai".

        Returns:
            int: Number of sent requests.
        """
        return self._send_batch(
            task_info,
            task_df,
            BatchType.JUDGMENT,
            BatchRequestHandler.create_judgment_batch_request_file,
            persona_configs,
            llm_name=llm_name
        )

    def check_batch_statuses(self) -> dict[str, BatchInfo]:
        """Fetches and returns statuses of batch requests.

        Returns:
            dict[str, BatchInfo]: A dictionary of batch information.
        """
        active_batches = {
            k: v for k, v in self.batches_info_store.items()
            if not v.has_finished()}
        log.info("Checking batch statuses, %s active batches found.", len(active_batches))

        for batch_info in active_batches.values():
            batch_id = batch_info.batch_id
            try:
                client = self._get_client(batch_info.client)
                status_output = client.get_batch_progress(batch_id)
                batch_info.update_batch(status_output)
            except (APIConnectionError, ConnectionError, ValueError) as e:
                log.error("Error: %s", e, exc_info=True)
        self._save()
        return active_batches

    def download_batch_files(
        self,
        task_ids: Iterable[str]
    ) -> list[BatchInfo]:
        """Fetches responses for batch requests and saves them to files.

        Returns:
            list[BatchInfo]: A list with batch information of batches that finished.
        """
        self.check_batch_statuses()
        completed_batches = [
            batch_info for batch_info in self.batches_info_store.values()
            if batch_info.is_completed()]
        log.info("Fetching batch responses, %s completed batches found.", len(completed_batches))

        retrieved_batches = []
        for batch_info in completed_batches:
            tid = batch_info.task_id
            if batch_info.task_id not in task_ids:
                log.warning("Skipping batch for inactive task %s", tid)
                continue

            try:
                client = self._get_client(batch_info.client)
                if batch_info.has_output():
                    output_file = batch_info.get_output_file()
                    BatchRequestHandler.read_response_stream(
                        output_file, client.fetch_response)

                if batch_info.has_error():
                    error_file = batch_info.get_error_file()
                    BatchRequestHandler.read_response_stream(
                        error_file, client.fetch_response)
                    error_messages = BatchRequestHandler.read_error_file(error_file.local_file_path)
                    for message in error_messages:
                        log.error("Task %s failed: %s", tid, message)

                retrieved_batches.append(batch_info)
            except (ValueError, APIConnectionError) as e:
                log.error("Error: %s", e, exc_info=True)

        self._save()
        return retrieved_batches

    def read_batch_file(
        self,
        batch_info: BatchInfo
    ) -> bool | pd.DataFrame:
        """Reads the response object of a batch.

        Args:
            batch_info (BatchInfo): Batch information object.

        Returns:
            bool | pd.DataFrame: Returns false if the response could not be loaded, else a dataframe containing
                the response contents.
        """
        try:
            if batch_info.has_output():
                client = self._get_client(batch_info.client)
                output_file = batch_info.get_output_file()
                return BatchRequestHandler.read_response_file(output_file.local_file_path, client.read_response_line)
        except (KeyError, TypeError) as e:
            log.error("Error: %s", e, exc_info=True)
        return False

    def _send_batch(
        self,
        task_info: TaskInfo,
        task_df: pd.DataFrame,
        batch_type: BatchType,
        create_fn: Callable[..., tuple[Path, int]],
        *create_args,
        llm_name: Literal["openai", "genai"] = "openai",
    ) -> int | bool:
        """Generic helper to send any type of batch to the LLM API."""
        try:
            client = self._get_client(llm_name)
            file_name, request_count = create_fn(task_info, task_df, *create_args, client)

            if request_count < 1:
                return 0

            batch_id = client.send_batch(file_name)
            batch_info = BatchInfo(
                batch_id, task_info.task_id, batch_type, request_count=request_count, client=llm_name)
            self.batches_info_store[batch_info.batch_id] = batch_info
            self._save()
            return batch_info.request_count
        except (ValueError, OSError, ConnectionError, KeyError) as e:
            log.error(
                "Failed to create and send %s batch for task %s: %s",
                batch_type.name.lower(), task_info.task_id, e, exc_info=True,
            )
        return False
