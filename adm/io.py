"""JSON export for data consumed by OJS cells (CLAUDE.md §6.3)."""

import json
from pathlib import Path
from typing import Any

import numpy as np


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot write {type(obj).__name__} to JSON")


def write_json(obj: Any, path: str | Path) -> Path:
    """Write `obj` to `path` as compact JSON, creating parent folders.

    NumPy arrays become nested lists (row-major). NaN and infinity raise ValueError,
    because JSON has no representation for them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, default=_to_builtin, allow_nan=False, separators=(",", ":"))
    path.write_text(text + "\n", encoding="utf-8")
    return path
