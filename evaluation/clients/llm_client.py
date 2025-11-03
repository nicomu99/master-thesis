from abc import ABC, abstractmethod
from typing import TextIO, Optional, Any, Tuple

from pathlib import Path


class LLMClient(ABC):
    """Abstract base class for interacting with different LLM (Large Language Model) APIs.

    This class serves as a unified interface for communicating with different LLM APIs. The concrete function
    implementations depend on the API. Each subclass implements all abstract methods.
    Each subclass must implement all abstract methods to ensure consistent behavior across APIs.
    """

    @abstractmethod
    def get_api_response(self, template: str, **kwargs: Any) -> str:
        """Return a single, synchronous output from an LLM API.

        The function fills the template with the given keyword arguments and sends the prompt to the
        LLM API.

        Args:
            template: A string template with placeholders.
            **kwargs: Keyword arguments to be inserted into ``template``. Must match the placeholders.

        Returns:
            str: The generated text returned by the API.
        """

    @abstractmethod
    def write_prompt(self, file: TextIO, custom_id: str, prompt: str, instruction: Optional[str] = None) -> None:
        """Writes a prompt request in JSONL format to a file.

        Args:
            file (TextIO): File output buffer. The request will be written to this file.
            custom_id (str): An identifier, which can be used to map client outputs to the input samples.
            prompt (str): Request prompt.
            instruction (str | None): System prompt instruction. If this is None, the default system prompt
                will be used. Defaults to None.
        """

    @abstractmethod
    def read_response_line(self, line: str) -> Tuple[str, str | None]:
        """Reads the contents of a JSONL file.

        Args:
            line (str): JSONL line.

        Returns:
            Tuple[str, str]: A tuple containing the sample ID and completion.

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
    def get_batch_progress(self, batch_id: str) -> Tuple[str, int, int, str | None, str | None]:
        """Sends a synchronous request to the LLM API to retrieve the current progress of the batch.

        Args:
            batch_id (str): Remote identifier of the batch.

        Returns:
            str: Remote batch status.
            int: Completed requests.
            int: Failed requests.
            str | None: Output file id.
            str | None: Error file id.
        """

    @abstractmethod
    def fetch_response(self, remote_file_id: str) -> bytes:
        """Get the response stream from the remote API.

        Args:
            remote_file_id (str): Remote file identifier.

        Returns:
            bytes: Response stream.
        """
