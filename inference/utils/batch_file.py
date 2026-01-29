from dataclasses import dataclass
from pathlib import Path


@dataclass
class BatchFile:
    """Data class holding metadata linking remote file ids to local file paths.

    Attributes:
        remote_file_id (str): Remote output file identifier.
        local_file_path (Path): Local file path, where the remote file will be stored.
    """
    remote_file_id: str
    local_file_path: Path
