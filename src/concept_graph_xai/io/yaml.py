"""YAML loaders/dumpers for the nested-dict ConceptGraph format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a nested-dict ConceptGraph definition from a YAML file."""

    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML at {path!s} must define a mapping at the top level")
    return loaded


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Dump a nested-dict ConceptGraph definition to YAML."""

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
