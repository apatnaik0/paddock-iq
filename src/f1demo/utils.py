from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def round_or_none(value: float, ndigits: int = 3) -> float | None:
    if value is None or np.isnan(value):
        return None
    return round(float(value), ndigits)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
