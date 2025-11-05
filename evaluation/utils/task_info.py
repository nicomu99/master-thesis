from dataclasses import dataclass
from .enums import TaskStatus
from .log_conf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


@dataclass
class TaskInfo:
    """Data class containing task-specific metadata.

    Attributes:
        task_id (str): Task identifier.
        dataset_id (str): Identifier of the associated dataset.
        field (str): High-level field of study of this task.
        need_judgment (bool): Whether the task requires LLM-as-a-judge evaluation.
        status (TaskStatus): The current status of the task. Defaults to PERSONAS_PENDING.
        category_name (None | str): Category identifier of the task. If the dataset of this task contains
            several areas of expertise, this value is used to identify rows of the same expertise.
            Defaults to field.
    """
    task_id: str
    dataset_id: str
    field: str
    need_judgment: bool = False
    status: TaskStatus = TaskStatus.PERSONAS_PENDING
    category_name: str | None = None

    _STATUS_INCREMENT = {
        TaskStatus.PERSONAS_PENDING: TaskStatus.PERSONAS_REQUESTED,
        TaskStatus.PERSONAS_REQUESTED: TaskStatus.ANSWERS_PENDING,
        TaskStatus.ANSWERS_PENDING: TaskStatus.ANSWERS_REQUESTED,
        TaskStatus.ANSWERS_REQUESTED: TaskStatus.FINISHED,
        TaskStatus.JUDGMENT_PENDING: TaskStatus.JUDGMENT_REQUESTED,
        TaskStatus.JUDGMENT_REQUESTED: TaskStatus.FINISHED
    }

    _STATUS_DECREMENT = {
        TaskStatus.PERSONAS_REQUESTED: TaskStatus.PERSONAS_PENDING,
        TaskStatus.ANSWERS_REQUESTED: TaskStatus.ANSWERS_PENDING,
        TaskStatus.JUDGMENT_REQUESTED: TaskStatus.JUDGMENT_PENDING
    }

    def __post_init__(self):
        if self.category_name is None:
            self.category_name = self.field

    def reset(self):
        """Reset the task status to TaskStatus.PERSONAS_PENDING."""
        self.status = TaskStatus.PERSONAS_PENDING

    def reset_to_answers(self):
        """Reset the task status to TaskStatus.ANSWERS_PENDING."""
        self.status = TaskStatus.ANSWERS_PENDING

    def reset_to_judgments(self):
        """Reset the task status to TaskStatus.JUDGMENT_PENDING."""
        self.status = TaskStatus.JUDGMENT_PENDING

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

    def is_judgment_pending(self) -> bool:
        """Returns true if the task status is TaskStatus.JUDGE_PENDING.

        Returns:
            bool: True if the task status is TaskStatus.JUDGE_PENDING, else false.
        """
        return self.status == TaskStatus.JUDGMENT_PENDING

    def is_finished(self) -> bool:
        """Returns true if the task status is TaskStatus.FINISHED.

        Returns:
            bool: True if the task status is TaskStatus.FINISHED, else false.
        """
        return self.status == TaskStatus.FINISHED

    def increment_status(self):
        """Increments the status of the task."""
        try:
            if self.status == TaskStatus.ANSWERS_REQUESTED and self.need_judgment:
                self.status = TaskStatus.JUDGMENT_PENDING
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

    def update_status(
        self,
        skip: bool = False,
        increment: bool = True,
    ):
        """Updates the status information.

        Args:
            skip (bool, optional): If true, the next phase is skipped. Defaults to False.
            increment (bool, optional): If true, the status is incremented. Defaults to True.
        """
        if increment:
            self.increment_status()
            if skip:
                self.increment_status()
        else:
            self.decrement_status()

    def skip_status(self):
        """Skips the requested phase."""
        self.increment_status()
        self.increment_status()

    @staticmethod
    def get_key_field() -> str:
        """Returns the key field of this data class.

        Returns:
            str: Key field identifier.
        """
        return "task_id"
