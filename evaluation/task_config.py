from typing import Optional

from dataclasses import dataclass


@dataclass
class TaskConfig:
    """Data class containing task-specific configuration parameters.
    
    Attributes:
        task_id (str): Identifier of the task of the format "<dataset_id>_<name>". 
            If name is empty, "_all" is appended to the identifier instead.
        dataset_id (str): Identifier of the dataset where samples of the task are stored.
        name (None | str): Category identifier of the task. If the dataset of this task contains
            several areas of expertise, this value is used to identify rows of the same expertise.
            Else, name will be none.
        field (str): High-level field of study of this task.
        static_persona (str): A single-sentence, base persona string in the format "You are <occupation>". 
    """
    task_id: str
    dataset_id: str
    name: Optional[str]
    field: str
    static_persona: str

    def __post_init__(self):
        if self.name == "":
            self.name = None
            self.task_id = f"{self.dataset_id}_all"
