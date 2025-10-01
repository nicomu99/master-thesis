from typing import Optional

from dataclasses import dataclass


@dataclass
class BatchInfo:
    batch_id: str
    status: str
    output_file_id: Optional[str]
