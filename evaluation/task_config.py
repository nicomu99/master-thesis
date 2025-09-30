from typing import Optional

from dataclasses import dataclass


@dataclass
class TaskConfig:
    task_id: str
    dataset_id: str
    name: Optional[str]       # If task name is "general", the whole dataset will be used
    field: str      # In this case, the field will be an empty string
    static_persona: str

    def __post_init__(self):
        if self.name == "":
            self.name = None
