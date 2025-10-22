from typing import Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

from .enums import BatchType, BatchStatus
from .constants import TEMP_PATH
from .log_conf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


@dataclass
class BatchInfo:
    """Data class holding metadata and state information for a data batch.

    Attributes:
        batch_id (str): Unique batch identifier.
        task_id (str): Identifier of the associated task.
        batch_type (BatchType): Type of batch.
        progress_message (Optional[str]): Remote status message.
        remote_messages (Set[str]): Remote error messages.
        status (BatchStatus): Possibly outdated process status of the batch.
        output_file_id (Optional[str]): Remote output file identifier.
        local_output_file (Optional[Path]): Local file path of the response.
        error_file_id (Optional[str]): Remote error file identifier.
        local_error_file (Path): Local file path of the error response.
    """

    batch_id: str
    task_id: str
    batch_type: BatchType
    progress_message: Optional[str] = None
    remote_messages: Set[str] = field(default_factory=set)
    status: BatchStatus = BatchStatus.SENT
    output_file_id: Optional[str] = None
    local_output_file: Optional[Path] = None
    error_file_id: Optional[str] = None
    local_error_file: Optional[Path] = None

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

    def create_output_file(self) -> Path:
        """Create a new output file path object.

        Returns:
            Path: Output file path.
        """
        self.local_output_file = TEMP_PATH / f"{self.task_id}_{self.batch_id}_output.jsonl"
        return self.local_output_file

    def create_error_file(self) -> Path:
        """Create a new error file path object.

        Returns:
            Path: Error file path.
        """
        self.local_error_file = TEMP_PATH / f"{self.task_id}_{self.batch_id}_error.jsonl"
        return self.local_error_file

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
