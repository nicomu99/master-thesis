from abc import ABC, abstractmethod
import io
import time

from pathlib import Path

from inference.utils import BatchInfo


class LLMClient(ABC):
    """Abstract base class for interacting with different LLM (Large Language Model) APIs.

    This class serves as a unified interface for communicating with different LLM APIs. The concrete function
    implementations depend on the API. Each subclass implements all abstract methods.
    Each subclass must implement all abstract methods to ensure consistent behavior across APIs.
    """
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0

    def get_api_response(self, template: str, **kwargs) -> str:
        """Return a single, synchronous output from an LLM API.

        The function fills the template with the given keyword arguments and sends the prompt to the
        LLM API.

        Args:
            template (str): A string template with placeholders.
            **kwargs: Keyword arguments to be inserted into ``template``. Must match the placeholders.

        Returns:
            str: The generated text returned by the API.
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._get_api_response(template, **kwargs)
            except (ConnectionError, TimeoutError) as e:
                last_error = e
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_seconds * (2 ** attempt))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to retrieve API response.")

    @abstractmethod
    def _get_api_response(self, template: str, **kwargs) -> str:
        """Implementation hook for :meth:`get_api_response`."""

    @abstractmethod
    def write_prompt(self, file: io.TextIOBase, custom_id: str, prompt: str, instruction: str | None = None) -> None:
        """Writes a prompt request in JSONL format to a file.

        Args:
            file (io.TextIOBase): File output buffer. The request will be written to this file.
            custom_id (str): An identifier, which can be used to map client outputs to the input samples.
            prompt (str): Request prompt.
            instruction (str | None): System prompt instruction. If this is None, the default system prompt
                will be used. Defaults to None.
        """

    @abstractmethod
    def read_response_line(self, line: str) -> tuple[str, str]:
        """Reads the contents of a JSONL file.

        Args:
            line (str): JSONL line.

        Returns:
            tuple[str, str]: A tuple containing the sample ID and completion.

        Raises:
            KeyError: If the response line does not match the correct format.
        """

    @abstractmethod
    def send_batch(self, batch_file: Path) -> str:
        """Send a batch request.

        Args:
            batch_file (Path): File name of the file containing the request objects.

        Returns:
            str: Batch identifier of the remote API.
        """

    @abstractmethod
    def get_batch_progress(self, batch_id: str) -> tuple[str, int, int]:
        """Sends a synchronous request to the LLM API to retrieve the current progress of the batch.

        Args:
            batch_id (str): Remote identifier of the batch.

        Returns:
            str: Remote batch status.
            int: Completed requests.
            int: Failed requests.
        """

    @abstractmethod
    def get_batch_files(self, batch_id: str) -> tuple[str | None, str | None]:
        """Retrieve remote output/error file identifiers for a batch, if available.

        Args:
            batch_id (str): Remote identifier of the batch.

        Returns:
            tuple[str | None, str | None]: Output file id and error file id (if present).
        """

    def requires_output_file(self) -> bool:
        """Indicates whether this client needs a remote output file to write responses."""
        return True

    @abstractmethod
    def fetch_response(self, remote_file_id: str) -> bytes:
        """Get the response stream from the remote API.

        Args:
            remote_file_id (str): Remote file identifier.

        Returns:
            bytes: Response stream.
        """

    @abstractmethod
    def write_response_to(self, batch_info: BatchInfo, file: io.TextIOBase) -> None:
        """Write batch results to a file in JSONL format.

        Args:
            batch_info (BatchInfo): Metadata required to retrieve the batch results.
            file (io.TextIOBase): Open file-like object in text mode.

        Raises:
            RuntimeError: If the batch results cannot be retrieved or written.
            KeyError: If the response format does not match the expected structure.
        """
