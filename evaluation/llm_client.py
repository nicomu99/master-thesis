from typing import Any, Dict, List

from pathlib import Path

import pandas as pd
from openai import OpenAI

from .batch_request_handler import BatchRequestHandler
from .utils import TaskConfig, BatchInfo, BatchType, QuestionType
from .utils import load_dataclass_dict, save_dataclass_dict
from .utils import get_logger

log = get_logger(__name__)


class LLMClient:
    """Primary class for communicating with the api."""

    def __init__(self):
        self.model = "gpt-5-nano"
        self.client = OpenAI()
        self.batch_request_creator = BatchRequestHandler()

        self.temp_path = Path("temp")
        self.batches_info_file = self.temp_path / "batches_info_file.json"
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
        batch_file_name = self.batch_request_creator.create_persona_request_file(
            task_config, task_df, persona_templates)

        self.send_batch(
            task_config.task_id, batch_file_name, BatchType.PERSONAS)

    def send_task_batch(
        self,
        task_config: TaskConfig,
        task_df: pd.DataFrame,
        question_type: QuestionType,
        persona_types: List[str]
    ) -> None:
        batch_file_name = self.batch_request_creator.create_task_request_file(
            task_config, task_df, question_type, persona_types)

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

    def check_batch_statuses(self):
        """Fetches and prints statuses of batch requests."""

        log.info("Checking batch statuses; %s batches found.", len(self.batches_info_store))
        if len(self.batches_info_store) == 0:
            return

        for batch_id, batch_info in self.batches_info_store.items():
            batch = self.client.batches.retrieve(batch_id)
            if batch_info.status in ["retrieved", "error"]:
                continue

            task_id = batch_info.task_id
            log.info("Task %s %-8s status is: %s", task_id, batch_info.batch_type, batch.status)
            if batch.status == "failed":
                assert batch.errors is not None
                assert batch.errors.data is not None
                for error in batch.errors.data:
                    log.error("Error %s: %s", error.code, error.message)

                batch_info.status = "failed"
            elif batch.status == "in_progress":
                request_counts = batch.request_counts
                if request_counts is not None:
                    log.info(
                        "Progress: %s out of %s finished; %s requests failed.",
                        request_counts.completed,
                        request_counts.total,
                        request_counts.failed
                    )

                    batch_info.status = "in_progress"
            elif batch.status == "completed":
                batch_info.status = "completed"
                batch_info.output_file_id = batch.output_file_id
                if batch.error_file_id:
                    batch_info.error_file_id = batch.error_file_id

        self._save()

    def _save_batch_response(
        self,
        remote_file_id: str,
        local_file_path: Path
    ):
        batch_response_stream = self.client.files.content(remote_file_id)

        with open(local_file_path, "wb") as f:
            f.write(batch_response_stream.read())

    def fetch_batch_responses(self)  -> List[BatchInfo]:
        """Fetches responses for batch requests and saves them to files."""

        log.info("Fetching batch responses")
        if len(self.batches_info_store) < 1:
            log.debug("No batches found")
            return []

        retrieved_batches = []
        for batch_id, batch_info in self.batches_info_store.items():
            if not batch_info.status == "completed":
                continue

            task_id = batch_info.task_id
            for remote_file_id_attr, file_type, local_file_attr in [
                ("output_file_id",  "output",   "local_output_file"),
                ("error_file_id",   "error",    "local_error_file")
            ]:
                remote_file_id = getattr(batch_info, remote_file_id_attr)
                if not remote_file_id:
                    continue

                local_batch_file = self.temp_path / f"{task_id}_{batch_id}_{file_type}.jsonl"
                self._save_batch_response(remote_file_id, local_batch_file)
                setattr(batch_info, local_file_attr, str(local_batch_file))
                batch_info.status = "error" if file_type == "error" else "retrieved"
                retrieved_batches.append(batch_info)

        self._save()

        return retrieved_batches

    def update_batch_info(
        self,
        task_id: str,
        new_status: str = "sent"
    ):
        self.batches_info_store[task_id].status = new_status
        print(self.batches_info_store[task_id].status, task_id)
        self._save()
