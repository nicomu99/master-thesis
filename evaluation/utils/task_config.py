from typing import Optional
from dataclasses import dataclass
from .enums import TaskStatus
from .log_conf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


@dataclass
class TaskConfig:
    """Data class containing task-specific configuration parameters.

    Attributes:
        task_id (str): Identifier of the task of the format "<dataset_id>_<name>".
            If name is empty, "_all" is appended to the identifier instead.
        dataset_id (str): Identifier of the dataset where samples of the task are stored.
        field (str): High-level field of study of this task.
        status (TaskStatus): The current status of the task.
        category_name (None | str): Category identifier of the task. If the dataset of this task contains
            several areas of expertise, this value is used to identify rows of the same expertise.
            If this field is not defined, the field field will be used as a fallback.
    """
    task_id: str
    dataset_id: str
    field: str
    status: TaskStatus = TaskStatus.PERSONAS_PENDING
    category_name: Optional[str] = None

    _STATUS_INCREMENT = {
        TaskStatus.PERSONAS_PENDING: TaskStatus.PERSONAS_REQUESTED,
        TaskStatus.PERSONAS_REQUESTED: TaskStatus.ANSWERS_PENDING,
        TaskStatus.ANSWERS_PENDING: TaskStatus.ANSWERS_REQUESTED,
        TaskStatus.ANSWERS_REQUESTED: TaskStatus.FINISHED
    }

    _STATUS_DECREMENT = {
        TaskStatus.PERSONAS_REQUESTED: TaskStatus.PERSONAS_PENDING,
        TaskStatus.ANSWERS_REQUESTED: TaskStatus.ANSWERS_PENDING
    }

    def __post_init__(self):
        if self.category_name is None:
            self.category_name = self.field

    def is_personas_pending(self):
        return self.status == TaskStatus.PERSONAS_PENDING

    def is_answers_pending(self):
        return self.status == TaskStatus.ANSWERS_PENDING

    def increment_status(self):
        try:
            self.status = self._STATUS_INCREMENT[self.status]
        except KeyError:
            log.warning(
                "No status transition defined for task status %s. Will keep old status.",
                self.status)

    def decrement_status(self):
        try:
            self.status = self._STATUS_DECREMENT[(self.status)]
        except KeyError:
            log.warning(
                "No status transition defined for task status %s. Will keep old status.",
                self.status)

    def update_status(
        self,
        increment_status: bool
    ):
        if increment_status:
            self.increment_status()
        else:
            self.decrement_status()
