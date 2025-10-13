from typing import Dict, Type, TypeVar

import json
from enum import Enum
from pathlib import Path
from dataclasses import fields, is_dataclass, asdict

T = TypeVar("T")


def encode_dataclass(obj):
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Type {type(obj)} not serializable")


def decode_dataclass(data: dict, cls):
    init_kwargs = {}
    for f in fields(cls):
        val = data.get(f.name)
        if isinstance(val, dict) and is_dataclass(f.type):
            init_kwargs[f.name] = decode_dataclass(val, f.type)
        elif isinstance(val, str) and isinstance(f.type, type) and issubclass(f.type, Enum):
            init_kwargs[f.name] = f.type(val)
        else:
            init_kwargs[f.name] = val
    return cls(**init_kwargs)


def load_dataclass_dict(
    file_path: str | Path,
    cls: Type[T],
    key_field: str
) -> Dict[str, T]:
    file_path = Path(file_path)
    if not file_path.exists():
        return {}

    with file_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    return {k: decode_dataclass({key_field: k, **v}, cls) for k, v in raw_data.items()}


def save_dataclass_dict(
    file_path: str | Path,
    data: dict[str, T],
    key_field: str,
) -> None:
    """Save a dictionary of dataclasses to JSON."""
    save_dict = {
        k: {
            field: value for field, value in asdict(v).items() # pyright: ignore[reportArgumentType]
            if field != key_field
        }
        for k, v in data.items()
        if is_dataclass(v)
    }

    with Path(file_path).open("w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=4, ensure_ascii=False, default=encode_dataclass)
