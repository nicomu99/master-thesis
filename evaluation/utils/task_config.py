from typing import Optional
from dataclasses import dataclass
from .enums import TaskStatus


@dataclass
class TaskConfig:
    """Data class containing task-specific configuration parameters.
    
    Attributes:
        task_id (str): Identifier of the task of the format "<dataset_id>_<name>". 
            If name is empty, "_all" is appended to the identifier instead.
        dataset_id (str): Identifier of the dataset where samples of the task are stored.
        field (str): High-level field of study of this task.
        static_persona (str): A single-sentence, base persona string in the format "You are <occupation>". 
        status (TaskStatus): The current status of the task.
        category_name (None | str): Category identifier of the task. If the dataset of this task contains
            several areas of expertise, this value is used to identify rows of the same expertise.
            Else, name will be none.
    """
    task_id: str
    dataset_id: str
    field: str
    static_persona: str
    status: TaskStatus = TaskStatus.PERSONAS_PENDING
    category_name: Optional[str] = None
