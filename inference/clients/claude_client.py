from typing import Iterable, cast

import io
import json
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import MessageParam

from inference.utils import logging, BatchInfo
from .llm_client import LLMClient


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class ClaudeClient(LLMClient):
    """Implementation of :class:`LLMClient` for interacting with the OpenAI API.

    Attributes:
        model (str): Default model name used for all requests (e.g., ``"gpt-5-nano"``).
        client (OpenAI): Instance of the official OpenAI client used to perform API operations.
    """

    def __init__(self):
        self.model = "claude-haiku-3"
        self.client = Anthropic()

    def _get_api_response(self, template: str, **kwargs) -> str:
        prompt = template.format(**kwargs)
        message = [{"role": "user", "content": prompt}]
        message = cast(Iterable[MessageParam], cast(object, message))
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=message)
        return response.content[0].text

    def write_prompt(
        self,
        file: io.TextIOBase,
        custom_id: str,
        prompt: str,
        instruction: str | None = None
    ):
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024}

        api_request_dict = {
            "custom_id": custom_id,
            "params": body}

        file.write(json.dumps(api_request_dict) + "\n")

    def send_batch(self, batch_file: Path) -> str:
        data = []
        with batch_file.open("r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

        batch_job = self.client.messages.batches.create(requests=data)
        return batch_job.id

    def read_response_line(self, line: str) -> tuple[str, str]:
        response_line = json.loads(line)
        custom_id = response_line["custom_id"]
        result = response_line.get("result")
        if not result or result.get("type") != "succeeded":
            return custom_id, "No response"

        message = result.get("message") or {}
        content = message.get("content", [])
        completion = "".join(
            block.get("text", "")
            for block in content
            if block.get("type") in ("text", "output_text")
        )
        return custom_id, completion

    def get_batch_progress(self, batch_id: str) -> tuple[str, int, int]:
        remote_batch = self.client.messages.batches.retrieve(batch_id)
        status = remote_batch.processing_status

        completed, failed = 0, 0
        if remote_batch.request_counts is not None:
            progress = remote_batch.request_counts
            completed = progress.succeeded
            failed = progress.canceled + progress.errored + progress.expired

        return status, completed, failed

    def get_batch_files(self, batch_id: str) -> tuple[str | None, str | None]:
        _ = batch_id
        return None, None

    def requires_output_file(self) -> bool:
        return False

    def fetch_response(self, remote_file_id: str) -> bytes:
        raise NotImplementedError("This function is not implemented.")

    def write_response_to(self, batch_info: BatchInfo, file: io.TextIOBase) -> None:
        batch_id = batch_info.batch_id
        for result in self.client.messages.batches.results(batch_id):
            file.write(result.model_dump_json())
            file.write("\n")
