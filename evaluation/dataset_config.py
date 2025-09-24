from typing import Dict, Optional, List

from pydantic import BaseModel

class DatasetConfig(BaseModel):
    id: str
    huggingface_id: str
    split: str
    question_field: str
    load_name: Optional[str] = None
    personas: Optional[List[Dict[str, str]]] = None