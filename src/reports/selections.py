"""Load and query a selections YAML produced by the Streamlit selector.

Shape (see docs/superpowers/specs/2026-07-24-selections-workflow-design.md):

    name: rocky_bilbao_2026
    username: phenrickson
    criteria: {...}          # informational
    games:
      205637: {selection: lock,  status: yes}
      177736: {selection: maybe, status: no}

Label roles:
    selection (lock|maybe|no) -> which section a game appears in
    status    (yes|no)        -> the "bringing it" flag -> The Menu
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_SELECTIONS = {"lock", "maybe", "other", "no"}
_STATUSES = {"yes", "no"}


@dataclass
class Selections:
    name: str
    username: str
    criteria: dict = field(default_factory=dict)
    # game_id -> {"selection": ..., "status": ...}
    games: dict = field(default_factory=dict)

    def _ids_where(self, **kv) -> list[int]:
        out = [
            gid
            for gid, lab in self.games.items()
            if all(lab.get(k) == v for k, v in kv.items())
        ]
        return sorted(out)

    def locks(self) -> list[int]:
        return self._ids_where(selection="lock")

    def maybes(self) -> list[int]:
        return self._ids_where(selection="maybe")

    def others(self) -> list[int]:
        """Games explicitly tagged 'other' (not lock/maybe, but still shown)."""
        return self._ids_where(selection="other")

    def menu(self) -> list[int]:
        """The final list: everything marked status: yes."""
        return self._ids_where(status="yes")


def load_selections(path: str | Path) -> Selections:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    games_raw = raw.get("games") or {}
    games: dict[int, dict] = {}
    def _norm(v, default):
        # YAML turns unquoted yes/no into booleans; normalize back to strings.
        if v is True:
            return "yes"
        if v is False:
            return "no"
        return default if v is None else str(v)

    for gid, lab in games_raw.items():
        lab = lab or {}
        sel = _norm(lab.get("selection"), "no")
        stat = _norm(lab.get("status"), "no")
        if sel not in _SELECTIONS:
            raise ValueError(f"game {gid}: bad selection {sel!r} (expected {_SELECTIONS})")
        if stat not in _STATUSES:
            raise ValueError(f"game {gid}: bad status {stat!r} (expected {_STATUSES})")
        games[int(gid)] = {"selection": sel, "status": stat}
    return Selections(
        name=raw.get("name", ""),
        username=raw.get("username", ""),
        criteria=raw.get("criteria") or {},
        games=games,
    )


def dump_selections(sel: Selections, path: str | Path) -> None:
    """Write a Selections back to YAML (used by the Streamlit exporter)."""
    payload = {
        "name": sel.name,
        "username": sel.username,
        "criteria": sel.criteria,
        "games": {int(g): dict(lab) for g, lab in sorted(sel.games.items())},
    }
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))
