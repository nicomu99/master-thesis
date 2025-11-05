from dataclasses import dataclass

from .batch_file import BatchFile
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
        status (BatchStatus): Possibly outdated process status of the batch.
        progress_message (str | None): Remote status message.
        output_file (BatchFile | None): Output metadata of the generated responses. Defaults to None.
        output_file (BatchFile | None): Metadata for possible errors during processing of the batch.
            Defaults to None.
    """

    batch_id: str
    task_id: str
    batch_type: BatchType
    status: BatchStatus = BatchStatus.SENT
    request_count: int = 0
    client: str = "openai"
    progress_message: str | None = None
    output_file: BatchFile | None = None
    error_file: BatchFile | None = None

    _STR_TRANSITIONS = {
        "validating": BatchStatus.VALIDATING,
        "in_progress": BatchStatus.IN_PROGRESS,
        "finalizing": BatchStatus.IN_PROGRESS,
        "expired": BatchStatus.ERROR,
        "failed": BatchStatus.FAILED,
        "cancelled": BatchStatus.RETRIEVED,
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

    def is_validating(self) -> bool:
        """Check whether the batch status is validating.

        Returns:
            bool: True if the status equals BatchStatus.VALIDATING, otherwise False.
        """
        return self.status == BatchStatus.VALIDATING

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
        file_id = self.batch_id.replace("/", "_")
        local_file_path = TEMP_PATH / f"{self.task_id}/output/{file_id}.jsonl"
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
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
        local_file_path = TEMP_PATH / f"{self.task_id}/output/{self.batch_id}_error.jsonl"
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
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

    def update_batch(
        self,
        batch_update_data: tuple[str, int, int, str | None, str | None]
    ):
        """Updates the batch info

        Args:
            batch_update_data (tuple[str, int, int, str  |  None, str  |  None]): Tuple consisting of new status,
                number of completed and failed requests, output file id and error file id.
        """
        new_status, completed, failed, output_file_id, error_file_id = batch_update_data
        self.update_status(new_status)
        self.update_progress_message(completed, failed)

        if output_file_id:
            self.init_output_file(output_file_id)
        if error_file_id:
            self.init_error_file(error_file_id)

    def finish_batch(self):
        """Set the batch to the final status.

        If there were errors during processing, the final status is BatchStatus.ERROR, else BatchStatus.RETRIEVED.
        """
        if self.has_error():
            self.status = BatchStatus.ERROR
        else:
            self.status = BatchStatus.RETRIEVED

    def update_progress_message(self, completed: int, failed: int):
        """Updates the progress message.

        Args:
            completed (int): Number of successful completions.
            failed (int): Number of failed completions.
        """
        if completed > 0 or failed > 0:
            self.progress_message = (
                f"Progress: {completed} out of {self.request_count} finished; "
                f"{failed} requests failed.")
        elif self.request_count > 0:
            self.progress_message = (
                f"Processing {self.request_count} requests.")
        else:
            self.progress_message = (
                "No progress information.")

    @staticmethod
    def get_key_field() -> str:
        """Returns the key field of this data class.

        Returns:
            str: Key field identifier.
        """
        return "batch_id"
