from typing import Optional

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    dataset_id: str
    huggingface_id: str
    split: str
    question_type: str
    question_field: str
    answer_field: str
    task_column: Optional[str]
    load_name: Optional[str] = None

    def __post_init__(self):
        if self.task_column == "":
            self.task_column = None
