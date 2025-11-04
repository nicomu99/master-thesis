import io
import json
from pathlib import Path

from openai import OpenAI

from evaluation.utils import logging
from .llm_client import LLMClient


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class OpenAIClient(LLMClient):
    """Implementation of :class:`LLMClient` for interacting with the OpenAI API.

    Attributes:
        model (str): Default model name used for all requests (e.g., ``"gpt-5-nano"``).
        client (OpenAI): Instance of the official OpenAI client used to perform API operations.
    """

    def __init__(self):
        self.model = "gpt-5-nano"
        self.client = OpenAI()

    def get_api_response(self, template: str, **kwargs) -> str:
        prompt = template.format(**kwargs)
        response = self.client.responses.create(
            model=self.model, input=prompt)
        return response.output_text

    def write_prompt(
        self,
        file: io.TextIOBase,
        custom_id: str,
        prompt: str,
        instruction: str | None = None
    ):
        body = {"model": self.model, "input": prompt}
        if instruction:
            body["instructions"] = instruction

        api_request_dict = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body}

        file.write(json.dumps(api_request_dict) + "\n")

    def send_batch(self, batch_file: Path) -> str:
        with batch_file.open("rb") as f:
            batch_input_file = self.client.files.create(
                file=f, purpose="batch")

        batch_job = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/responses",
            completion_window="24h")

        return batch_job.id

    def read_response_line(self, line: str) -> tuple[str, str]:
        response_line = json.loads(line)

        response = response_line["response"]
        if "body" in response:
            response = response["body"]

        completion = "".join(
            c["text"]
            for o in response["output"]
            for c in o.get("content", [])
            if c.get("type") == "output_text"
        )

        return response_line["custom_id"], completion

    def get_batch_progress(self, batch_id: str) -> tuple[str, int, int, str | None, str | None]:
        remote_batch = self.client.batches.retrieve(batch_id)
        status = remote_batch.status
        output_file_id = remote_batch.output_file_id
        error_file_id = remote_batch.error_file_id

        completed, failed = 0, 0
        if remote_batch.request_counts is not None:
            progress = remote_batch.request_counts
            completed = progress.completed
            failed = progress.failed

        if remote_batch.errors is None:
            return status, completed, failed, output_file_id, error_file_id

        errors = remote_batch.errors
        if errors.data is None:
            log.warning("Batch for task %s failed with no error message.", batch_id)
            return status, completed, failed, output_file_id, error_file_id

        errors = errors.data
        for error in errors:
            log.warning(
                "Batch for task failed: %s %s",
                error.code, error.message)
        return status, completed, failed, output_file_id, error_file_id

    def fetch_response(self, remote_file_id: str) -> bytes:
        response = self.client.files.content(remote_file_id)
        return response.read()
