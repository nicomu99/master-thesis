from typing import Optional

from pydantic import BaseModel

class DatasetConfig(BaseModel):
    dataset_id: str
    huggingface_id: str
    split: str
    question_field: str
    task_column: str
    load_name: Optional[str] = None