from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    cache: Path
    outputs: Path
    plots: Path
    tables: Path
    models: Path
    site: Path
    templates: Path


ROOT = Path(__file__).resolve().parents[2]
PATHS = Paths(
    root=ROOT,
    data=ROOT / "data",
    cache=ROOT / "data" / "cache",
    outputs=ROOT / "outputs",
    plots=ROOT / "outputs" / "plots",
    tables=ROOT / "outputs" / "tables",
    models=ROOT / "outputs" / "models",
    site=ROOT / "site",
    templates=ROOT / "templates",
)

SESSION_ORDER = ["FP1", "FP2", "FP3", "Qualifying", "Race"]
