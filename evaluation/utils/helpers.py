from typing import Dict, Type, TypeVar, Any, get_origin

import json
from enum import Enum
from pathlib import Path
from dataclasses import fields, is_dataclass, asdict

from .task_info import TaskInfo

T = TypeVar("T")


def encode_dataclass(obj: object):
    """Converts enum objects to their values.

    This is a helper class for serializing enum members of dataclasses. It simply converts each enum member field to
    its string value.

    Args:
        obj (object): The object which should be converted.

    Raises:
        TypeError: If obj is not an enum, an error is thrown.

    Returns:
        str: The string value of the enum object.
    """
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def decode_dataclass(data: dict, cls: Any) -> Any:
    """Helper function that recursively decodes dataclass data.

    Args:
        data (dict): A dictionary holding the data of a dataclass.
        cls (Any): The dataclass template to use to decode the data dictionary.

    Returns:
        Any: A dataclass object.
    """
    init_kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue

        val = data.get(f.name)
        if isinstance(val, dict) and is_dataclass(f.type):
            init_kwargs[f.name] = decode_dataclass(val, f.type)
        elif isinstance(val, str) and isinstance(f.type, type) and issubclass(f.type, Enum):
            init_kwargs[f.name] = f.type(val)
        elif isinstance(val, list) and get_origin(f.type) is set:
            init_kwargs[f.name] = set(map(str, val))
        else:
            init_kwargs[f.name] = val
    return cls(**init_kwargs)


def load_dataclass_dict(
    file_path: str | Path,
    cls: Type[T],
    key_field: str
) -> Dict[str, T]:
    """Loads a json file with dataclass objects into a dictionary.

    Args:
        file_path (str | Path): The path to the file to be loaded.
        cls (Type[T]): The data class template to use.
        key_field (str): The identifier field of the dataclass object.

    Returns:
        Dict[str, T]: A dictionary with dataclass identifiers as keys and the dataclass object as values.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {}

    with file_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    return {k: decode_dataclass({key_field: k, **v}, cls) for k, v in raw_data.items()}


def load_task_config(path: str | Path) -> Dict[str, TaskInfo]:
    """Loads and decodes all task configurations from a JSON file.

    The JSON file is expected to contain a nested mapping of dataset IDs to
    task configurations. Each configuration entry is automatically converted
    into a ``TaskInfo`` dataclass instance using ``decode_dataclass``

    Args:
        path (str | Path): Path to the JSON configuration file.

    Returns:
        Dict[str, TaskInfo]: A mapping from task IDs to their corresponding
        ``TaskInfo`` instances, each annotated with its ``dataset_id`` and
        ``task_id``.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    task_infos: Dict[str, TaskInfo] = {}

    for dataset_id, tasks in data.items():
        for task_id, cfg in tasks.items():
            # Fill in required fields before decoding
            cfg = {
                **cfg,
                "task_id": task_id,
                "dataset_id": dataset_id,
            }
            # Decode using the helper
            task_info = decode_dataclass(cfg, TaskInfo)
            task_infos[task_id] = task_info

    return task_infos


def save_dataclass_dict(
    file_path: str | Path,
    data: dict[str, T],
    key_field: str | None,
) -> None:
    """Save a dictionary of dataclasses to JSON.

    Args:
        file_path (str | Path): The file path to use to save the data.
        data (dict[str, T]): A dictionary containing dataclasses as values.
        key_field (str): The field of the dataclass used as an identifier.
    """
    save_dict = {
        k: {
            field: value for field, value in asdict(v).items()
            if field != key_field
        }
        for k, v in data.items()
        if is_dataclass(v) and not isinstance(v, type)
    }

    with Path(file_path).open("w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=4, ensure_ascii=False, default=encode_dataclass)
