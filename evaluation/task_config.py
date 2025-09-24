from pydantic import BaseModel

class TaskConfig(BaseModel):
    dataset_id: str
    name: str
    field: str
    static_persona: str
