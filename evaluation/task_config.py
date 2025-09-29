from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    dataset_id: str
    name: str       # If task name is "general", the whole dataset will be used
    field: str      # In this case, the field will be an empty string
    static_persona: str
