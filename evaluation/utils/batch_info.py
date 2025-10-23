from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from openai.types import Batch

from .enums import BatchType, BatchStatus
from .constants import TEMP_PATH
from .log_conf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


@dataclass
class BatchFile:
    """Data class holding metadata linking remote file ids to local file paths.

    Attributes:
        remote_file_id (str): Remote output file identifier.
        local_file_path (Path): Local file path, where the remote file will be stored.
    """
    remote_file_id: str
    local_file_path: Path


@dataclass
class BatchInfo:
    """Data class holding metadata and state information for a data batch.

    Attributes:
        batch_id (str): Unique batch identifier.
        task_id (str): Identifier of the associated task.
        batch_type (BatchType): Type of batch.
        status (BatchStatus): Possibly outdated process status of the batch.
        progress_message (Optional[str]): Remote status message.
        output_file (Optional[BatchFile]): Output metadata of the generated responses. Defaults to None.
        output_file (Optional[BatchFile]): Metadata for possible errors during processing of the batch.
            Defaults to None.
    """

    batch_id: str
    task_id: str
    batch_type: BatchType
    status: BatchStatus = BatchStatus.SENT
    progress_message: Optional[str] = None
    output_file: Optional[BatchFile] = None
    error_file: Optional[BatchFile] = None

    _STR_TRANSITIONS = {
        "validating": BatchStatus.IN_PROGRESS,
        "in_progress": BatchStatus.IN_PROGRESS,
        "finalizing": BatchStatus.IN_PROGRESS,
        "expired": BatchStatus.ERROR,
        "failed": BatchStatus.FAILED,
        "completed": BatchStatus.COMPLETED
    }

    def is_completed(self) -> bool:
        """Check whether the batch status is completed.

        Returns:
            bool: True if the status equals BatchStatus.COMPLETED, otherwise False.
        """
        return self.status == BatchStatus.COMPLETED

    def is_failed(self) -> bool:
        """Check whether the batch status is failed.

        Returns:
            bool: True if the status equals BatchStatus.FAILED, otherwise False.
        """
        return self.status == BatchStatus.FAILED

    def is_error(self) -> bool:
        """Check whether the batch status is error or failed.

        Returns:
            bool: True if the status equals BatchStatus.ERROR or BatchStatus.FAILED, otherwise False.
        """
        return self.status in (BatchStatus.ERROR, BatchStatus.FAILED)

    def is_in_progress(self) -> bool:
        """Check whether the batch status is in progress.

        Returns:
            bool: True if the status equals BatchStatus.IN_PROGRESS, otherwise False.
        """
        return self.status == BatchStatus.IN_PROGRESS

    def has_finished(self) -> bool:
        """Check whether the batch status is retrieved or error.

        Returns:
            bool: True if the status equals BatchStatus.RETRIEVED or BatchStatus.ERROR, otherwise False.
        """
        return self.status in (BatchStatus.RETRIEVED, BatchStatus.ERROR)

    def is_retrieved(self) -> bool:
        """Check whether the batch status is retrieved.

        Returns:
            bool: True if the status equals BatchStatus.RETRIEVED, otherwise False.
        """
        return self.status == BatchStatus.RETRIEVED

    def set_status(self, status: BatchStatus):
        """Set the current batch status.

        Args:
            status (BatchStatus): The new status to assign to the batch.
        """
        self.status = status

    def init_output_file(self, remote_id: str):
        """Initialize the output file metadata for the current batch.

        This method creates a BatchFile object linking the remote file
        identifier with the corresponding local file path where the output
        will be stored.

        Args:
            remote_id (str): The unique identifier of the remote output file.
        """
        local_file_path = TEMP_PATH / f"{self.task_id}_{self.batch_id}_output.jsonl"
        self.output_file = BatchFile(remote_id, local_file_path)

    def get_output_file(self) -> BatchFile:
        """Return the initialized output file metadata.

        Retrieves the BatchFile instance. If the output file has not been
        initialized yet, a RuntimeError is raised.

        Returns:
            BatchFile: The metadata object describing the remote and local output file.

        Raises:
            RuntimeError: If the output file has not been initialized.
        """
        if not self.output_file:
            raise RuntimeError("Output file not created yet.")
        return self.output_file

    def has_output(self) -> bool:
        """Check whether an output file has been initialized.

        Returns:
            bool: True if the output file metadata exists, False otherwise.
        """
        return self.output_file is not None

    def init_error_file(self, remote_id: str):
        """Initialize the error file metadata for the current batch.

        This method creates a :class:`BatchFile` object linking the remote file
        identifier with the corresponding local file path where the output
        will be stored.

        Args:
            remote_id (str): The unique identifier of the remote error file.
        """
        local_file_path = TEMP_PATH / f"{self.task_id}_{self.batch_id}_error.jsonl"
        self.error_file = BatchFile(remote_id, local_file_path)

    def get_error_file(self) -> BatchFile:
        """Return the initialized error file metadata.

        Retrieves the BatchFile instance. If the error file has not been
        initialized yet, a RuntimeError is raised.

        Returns:
            BatchFile: The metadata object describing the remote and local output file.

        Raises:
            RuntimeError: If the output file has not been initialized.
        """
        if not self.error_file:
            raise RuntimeError("Output file not created yet.")
        return self.error_file

    def has_error(self):
        """Check whether an error file has been initialized.

        Returns:
            bool: True if the error file metadata exists, False otherwise.
        """
        return self.error_file is not None

    def update_status(self, new_status: str):
        """Update the current status using a string-based transition mapping.

        Args:
            new_status (str): The string key representing the new status.

        Notes:
            If ``new_status`` is not defined in ``_STR_TRANSITIONS``, a warning is logged
            and the existing status remains unchanged.
        """
        try:
            self.status = self._STR_TRANSITIONS[new_status]
        except KeyError:
            log.warning(
                "No status transition defined for status %s. Will keep old status.",
                new_status)

    def update_progress_message(self, remote_batch: Batch):
        """Updates the progress message, if the remote batch holds one.

        Args:
            remote_batch (Batch): Remote batch object.
        """
        request_counts = getattr(remote_batch, "request_counts", None)
        if not request_counts:
            return

        self.progress_message = (
            f"Progress: {request_counts.completed} out of {request_counts.total} finished; "
            f"{request_counts.failed} requests failed.")

    @staticmethod
    def get_key_field() -> str:
        """Returns the key field of this data class.

        Returns:
            str: Key field identifier.
        """
        return "batch_id"
