"""Build the JSON payload for the custom faceted-filter collection table.

Reuses ``format_collection_table`` for the exact display cells, then attaches a
per-column filter ``kind`` the client JS uses to render controls.

The separate **Best** and **Recommended** string columns produced by
``format_collection_table`` are merged here into a single **Player Counts**
column of kind ``badges``: its cell is ``{"best": [...], "rec": [...]}`` (both
numeric-sorted lists of ints). The client renders each recommended count as a
badge, filled when it is also a best count, and filters against these sets.
"""

from __future__ import annotations

import polars as pl

from src.reports.tables import format_collection_table

# label -> filter kind. Columns absent here default to "none".
#   discrete       — one chip per distinct value
#   range          — numeric min/max; row matches on overlap, unknowns kept
#   range-contains — cell is a range string ("2–4"); chips 1..8/8+, matches when
#                    the chip value falls within [min,max]
#   badges         — cell is {"best":[...],"rec":[...]}; rendered as colored
#                    badges, filtered by a Best|Recommended toggle + count chips
_COLUMN_KINDS = {
    "Status": "discrete",
    "Your rating": "range",
    "Players": "range-contains",
    "Recommended": "badges",
    "Playtime": "range",
    "Complexity": "range",
}


def _parse_counts(raw) -> list:
    """"3, 4" / "" -> [3, 4] / []. Non-numeric tokens (e.g. "8+") dropped."""
    if raw is None:
        return []
    out = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
    return sorted(out)


def build_explorer_payload(collection: pl.DataFrame, games: pl.DataFrame) -> dict:
    """Return ``{"columns": [{"label","kind"}], "rows": [[cell, ...], ...]}``.

    Cell values match ``format_collection_table`` except **Best**/**Recommended**
    are replaced by one **Player Counts** cell holding
    ``{"best": [...], "rec": [...]}``.
    """
    table = format_collection_table(collection, games)
    src_labels = list(table.columns)

    # Locate the two source columns; merge them into one where "Best" sat.
    best_i = src_labels.index("Best") if "Best" in src_labels else None
    rec_i = src_labels.index("Recommended") if "Recommended" in src_labels else None

    out_labels = []
    for l in src_labels:
        if l == "Best":
            out_labels.append("Recommended")
        elif l == "Recommended":
            continue
        else:
            out_labels.append(l)
    columns = [{"label": l, "kind": _COLUMN_KINDS.get(l, "none")} for l in out_labels]

    if table.empty:
        return {"columns": columns, "rows": []}

    filled = table.astype(object).where(table.notna(), "")
    src_rows = filled.values.tolist()
    rows = []
    for r in src_rows:
        best = _parse_counts(r[best_i]) if best_i is not None else []
        rec = _parse_counts(r[rec_i]) if rec_i is not None else []
        # Union so every best count is also shown as a badge (best ⊆ rec in the
        # source data, but be defensive).
        rec_union = sorted(set(rec) | set(best))
        merged = {"best": best, "rec": rec_union}
        out = []
        for i, v in enumerate(r):
            if i == rec_i:
                continue
            out.append(merged if i == best_i else v)
        rows.append(out)
    return {"columns": columns, "rows": rows}
