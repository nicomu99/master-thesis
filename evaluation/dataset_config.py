from typing import Optional

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    huggingface_id: str
    split: str
    question_field: str
    task_column: str
    load_name: Optional[str] = None
