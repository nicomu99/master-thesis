from typing import Any, Dict

import json
import pickle
from pathlib import Path

from openai import OpenAI

from batch_info import BatchInfo
from log_conf import get_logger

log = get_logger(__name__)


class LLMClient:
    def __init__(self):
        self.client = OpenAI()

        self.temp_path = Path("temp")
        self.batches_info_file = self.temp_path / "batches_info_file.pickle"

        self.batches_info_store: Dict[str, BatchInfo] = {}
        self._load_batches_info()

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
                model="gpt-5-nano",
                input=prompt,
                prompt_cache_key=prompt_cache_key
            )
        else:
            response = self.client.responses.create(
                model="gpt-5-nano",
                input=prompt,
            )

        return response.output_text

    def send_batch(
        self,
        batch_file_name: str,
        task_id: str
    ):
        log.info("Sending batch for task %s", task_id)

        with open(batch_file_name, "rb") as f:
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

            batch_info = BatchInfo(batch_job.id, "sent", None)
            self.batches_info_store[task_id] = batch_info

            self._save_batch_info()

    def check_batch_statuses(self):
        log.info("Checking batch statuses; %s batches found.", len(self.batches_info_store))
        if len(self.batches_info_store) == 0:
            return

        for task_id, batch_info in self.batches_info_store.items():
            batch = self.client.batches.retrieve(batch_info.batch_id)

            log.info("Task %s status is: %s", task_id, batch.status)
            if batch.status == "failed":
                assert batch.errors is not None
                assert batch.errors.data is not None
                for error in batch.errors.data:
                    log.info("Error %s: %s", error.code, error.message)

                self.batches_info_store[task_id].status = "failed"
            elif batch.status == "in_progress":
                request_counts = batch.request_counts
                if request_counts is not None:
                    log.info(
                        "Progress: %s out of %s finished; %s requests failed.",
                        request_counts.completed,
                        request_counts.total,
                        request_counts.failed
                    )

                    self.batches_info_store[task_id].status = "in_progress"
            elif batch.status == "completed":
                self.batches_info_store[task_id].status = "completed"
                self.batches_info_store[task_id].output_file_id = batch.output_file_id

        self._save_batch_info()

    def fetch_batch_responses(self):
        log.info("Fetching batch responses")
        if len(self.batches_info_store) < 1:
            log.debug("No batches found")
            return

        for task_id, batch_info in self.batches_info_store.items():
            if not batch_info.status == "completed":
                log.debug("Task %s not finished yet", task_id)
                continue

            if not batch_info.output_file_id:
                log.info("No output file found for task %s with batch id %s", task_id, batch_info.batch_id)
                continue
            batch_response = self.client.files.content(batch_info.output_file_id)

            batch_response_file = self.temp_path / f"{task_id}_{batch_info.batch_id}.jsonl"
            with open(batch_response_file, "w", encoding="utf-8") as f:
                for response_line in batch_response.text:
                    f.write(json.dumps(response_line) + "\n")

            # Retrieve batch
            self.batches_info_store[task_id].status = "retrieved"
        self._save_batch_info()

    def _load_batches_info(self) -> None:
        if not self.batches_info_file.exists():
            return

        with open(self.batches_info_file, "rb") as f:
            self.batches_info_store = pickle.load(f)

    def _save_batch_info(self):
        with open(self.batches_info_file, mode="wb") as f:
            pickle.dump(self.batches_info_store, f)
