from typing import Optional
from dataclasses import dataclass
from .enums import TaskStatus
from .log_conf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


@dataclass
class TaskInfo:
    task_id: str
    dataset_id: str
    field: str
    need_judge: bool = False
    status: TaskStatus = TaskStatus.PERSONAS_PENDING
    category_name: Optional[str] = None
    """Data class containing task-specific configuration parameters.

    Attributes:
        task_id (str): Identifier of the task of the format "<dataset_id>_<name>".
            If name is empty, "_all" is appended to the identifier instead.
        dataset_id (str): Identifier of the dataset where samples of the task are stored.
        field (str): High-level field of study of this task.
        status (TaskStatus): The current status of the task.
        category_name (None | str): Category identifier of the task. If the dataset of this task contains
            several areas of expertise, this value is used to identify rows of the same expertise.
            If this field is not defined, the field will be used as a fallback.
    """

    _STATUS_INCREMENT = {
        TaskStatus.PERSONAS_PENDING: TaskStatus.PERSONAS_REQUESTED,
        TaskStatus.PERSONAS_REQUESTED: TaskStatus.ANSWERS_PENDING,
        TaskStatus.ANSWERS_PENDING: TaskStatus.ANSWERS_REQUESTED,
        TaskStatus.ANSWERS_REQUESTED: TaskStatus.FINISHED,
        TaskStatus.JUDGE_PENDING: TaskStatus.JUDGE_REQUESTED,
        TaskStatus.JUDGE_REQUESTED: TaskStatus.FINISHED
    }

    _STATUS_DECREMENT = {
        TaskStatus.PERSONAS_REQUESTED: TaskStatus.PERSONAS_PENDING,
        TaskStatus.ANSWERS_REQUESTED: TaskStatus.ANSWERS_PENDING,
        TaskStatus.JUDGE_REQUESTED: TaskStatus.JUDGE_PENDING
    }

    def __post_init__(self):
        if self.category_name is None:
            self.category_name = self.field

    def is_personas_pending(self) -> bool:
        """Returns true if the task status is TaskStatus.PERSONAS_PENDING.

        Returns:
            bool: True if the task status is TaskStatus.PERSONAS_PENDING, else false.
        """
        return self.status == TaskStatus.PERSONAS_PENDING

    def is_answers_pending(self) -> bool:
        """Returns true if the task status is TaskStatus.ANSWERS_PENDING.

        Returns:
            bool: True if the task status is TaskStatus.ANSWERS_PENDING, else false.
        """
        return self.status == TaskStatus.ANSWERS_PENDING

    def is_judge_pending(self) -> bool:
        """Returns true if the task status is TaskStatus.JUDGE_PENDING.

        Returns:
            bool: True if the task status is TaskStatus.JUDGE_PENDING, else false.
        """
        return self.status == TaskStatus.JUDGE_PENDING

    def increment_status(self):
        """Increments the status of the task."""
        try:
            if self.status == TaskStatus.ANSWERS_REQUESTED and self.need_judge:
                self.status = TaskStatus.JUDGE_PENDING
            else:
                self.status = self._STATUS_INCREMENT[self.status]
        except KeyError:
            log.warning(
                "No status transition defined for task status %s. Will keep old status.",
                self.status)

    def decrement_status(self):
        """Decrements the status of the task."""
        try:
            self.status = self._STATUS_DECREMENT[self.status]
        except KeyError:
            log.warning(
                "No status transition defined for task status %s. Will keep old status.",
                self.status)

    def update_status(self, increment_status: bool):
        """Increments the status if increment status is true, else decrements it.

        Args:
            increment_status (bool): Boolean used to decide whether to increment or decrement the status.
        """
        if increment_status:
            self.increment_status()
        else:
            self.decrement_status()

    @staticmethod
    def get_key_field() -> str:
        """Returns the key field of this data class.

        Returns:
            str: Key field identifier.
        """
        return "task_id"
