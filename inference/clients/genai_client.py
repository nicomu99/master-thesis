import io
import json
from pathlib import Path

from google import genai
from google.genai import types

from inference.utils import logging

from .llm_client import LLMClient


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class GenAIClient(LLMClient):
    """Implementation of :class:`LLMClient` for interacting with the Google Gemini API.

    Attributes:
        model (str): Default model name used for all requests (e.g., ``"gemini-2.5-flash"``).
        client (OpenAI): Instance of the official OpenAI client used to perform API operations.
    """

    def __init__(self):
        self.model = "gemini-2.5-flash"
        self.client = genai.Client()

    def get_api_response(self, template: str, **kwargs) -> str:
        prompt = template.format(**kwargs)
        response = self.client.models.generate_content(
            model=self.model, contents=prompt)
        if response.text is None:
            raise ConnectionError("No response retrieved.")
        return response.text

    def write_prompt(self, file: io.TextIOBase, custom_id: str, prompt: str, instruction: str | None = None) -> None:
        request: dict[str, object] = {"contents": [{"parts": [{"text": prompt}]}]}
        if instruction:
            request["system_instruction"] = instruction

        api_request_dict = {
            "key": custom_id,
            "request": request
        }

        file.write(json.dumps(api_request_dict) + "\n")

    def read_response_line(self, line: str) -> tuple[str, str]:
        response_line = json.loads(line)
        completion = ""
        try:
            response = response_line["response"]["candidates"][0]["content"]
            if "parts" in response:
                completion = "".join(o["text"] for o in response["parts"])
        except KeyError:
            completion = "No response"
        return response_line["key"], completion

    def send_batch(self, batch_file: Path) -> str:
        batch_input_file = self.client.files.upload(
            file=str(batch_file),
            config=types.UploadFileConfig(
                mime_type="application/json",
                display_name=batch_file.name
            )
        )
        if batch_input_file.name is None:
            raise ConnectionError(f"Unexpected error occurred during upload of {str(batch_file)}")

        batch_job = self.client.batches.create(
            model=self.model,
            src=batch_input_file.name)
        if batch_job.name is None:
            raise ConnectionError(f"Unexpected error occurred during upload of {str(batch_file)}")
        return batch_job.name

    @staticmethod
    def _status_transition(remote_status: str) -> str:
        status_transitions = {
            "JOB_STATE_PENDING": "validating",
            "JOB_STATE_RUNNING": "in_progress",
            "JOB_STATE_FAILED": "failed",
            "JOB_STATE_CANCELLED": "cancelled",
            "JOB_STATE_SUCCEEDED": "completed",
            "JOB_STATE_EXPIRED": "expired",
        }
        return status_transitions[remote_status]

    def get_batch_progress(self, batch_id: str) -> tuple[str, int, int, str | None, str | None]:
        batch_job = self.client.batches.get(name=batch_id)
        if batch_job is None or batch_job.state is None:
            raise ConnectionError(f"Unexpected error occurred during upload of {batch_id}")

        new_status = self._status_transition(batch_job.state.name)
        completed, failed = 0, 0
        remote_file_id = None
        if batch_job.dest and batch_job.dest.file_name:
            remote_file_id = batch_job.dest.file_name

        if new_status == "failed":
            log.warning(
                "Batch for task failed: %s",
                batch_job.error)

        return new_status, completed, failed, remote_file_id, None

    def fetch_response(self, remote_file_id: str) -> bytes:
        return self.client.files.download(file=remote_file_id)
