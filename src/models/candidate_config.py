"""Candidate config resolution from config.yaml.

A candidate is a recipe: algorithm, preprocessor settings, optional
upstream model choices. Defined in ``models.{type}.candidates`` in
config.yaml. This module reads that config without going through the
project's typed Config object so the candidate block remains a free-form
dict the trainer copies verbatim into the experiment's ``config.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")


def _load_raw(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    p = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    return yaml.safe_load(p.read_text())


def list_candidates(
    model_type: str, config_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    raw = _load_raw(config_path)
    cands = (raw.get("models") or {}).get(model_type, {}).get("candidates") or []
    return [c["name"] for c in cands]


def find_candidate(
    model_type: str, candidate: str, config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    raw = _load_raw(config_path)
    cands = (raw.get("models") or {}).get(model_type, {}).get("candidates") or []
    for c in cands:
        if c.get("name") == candidate:
            return c
    raise KeyError(
        f"No candidate {candidate!r} under models.{model_type}.candidates "
        f"(available: {[c.get('name') for c in cands]})"
    )
