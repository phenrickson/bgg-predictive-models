"""Standalone Streamlit app: curate a proposed list of games from a collection.

Filter a user's collection by criteria, assign per-game **selection**
(lock/maybe/no) and **status** (yes/no), and export to
``reports/selections/<name>.yaml`` for the menu report to consume.

This is a self-contained tool (not part of the main dashboard). Launch with:

    just select
    # or: uv run streamlit run src/selections/app.py

Step 1 of the selections workflow
(docs/superpowers/specs/2026-07-24-selections-workflow-design.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode  # noqa: E402

from src.reports.collection_data import (  # noqa: E402
    _fetch_collection_snapshot,
    _fetch_games_metadata,
)
from src.reports.selections import Selections, dump_selections, load_selections  # noqa: E402

st.set_page_config(page_title="Game Selections", layout="wide")

SELECTIONS_DIR = PROJECT_ROOT / "reports" / "selections"
SELECTION_OPTS = ["no", "maybe", "lock"]
STATUS_OPTS = ["no", "yes"]


@st.cache_data(show_spinner="Loading collection…")
def _load(username: str):
    coll = _fetch_collection_snapshot(username)
    games = _fetch_games_metadata()
    return coll, games


def _status_str(row) -> str:
    for flag, label in [
        ("owned", "Own"),
        ("preordered", "Preordered"),
        ("wishlist", "Wishlist"),
        ("want", "Want"),
        ("want_to_buy", "Want"),
        ("previously_owned", "Prev. Owned"),
        ("prev_owned", "Prev. Owned"),
    ]:
        if bool(row.get(flag, False)):
            return label
    return "—"


def _rng(lo, hi):
    lo = None if lo in (None, 0) else int(lo)
    hi = None if hi in (None, 0) else int(hi)
    if lo is None and hi is None:
        return ""
    if lo == hi or hi is None:
        return str(lo or hi)
    return f"{lo}–{hi}"


def _rec_contains(raw, n) -> bool:
    if raw is None:
        return False
    return str(n) in [t.strip() for t in str(raw).split(",")]


st.title("🎲 Game Selections")
st.caption(
    "Filter a collection to the games meeting your criteria, tag each as "
    "**lock / maybe**, set **status**, then export the list for the menu report."
)

# ---- user + selection name ----
c1, c2 = st.columns([1, 1])
with c1:
    username = st.text_input("BGG username", value="phenrickson")
with c2:
    sel_name = st.text_input("Selection name", value="rocky_bilbao_2026")

if not username:
    st.stop()

coll, games = _load(username)
if coll.height == 0:
    st.warning(f"No collection found for {username!r}.")
    st.stop()

# Owned games joined to metadata (players/rec/complexity/playtime).
owned = coll.filter(pl.col("owned")) if "owned" in coll.columns else coll
meta = {int(r["game_id"]): r for r in games.iter_rows(named=True)}

# ---- filters ----
st.subheader("Criteria")
f1, f2, f3, f4 = st.columns(4)
with f1:
    rec_player = st.selectbox("Recommended at N players", ["any", 1, 2, 3, 4, 5, 6, 7, 8], index=0)
with f2:
    cx = st.slider("Complexity", 1.0, 5.0, (1.0, 5.0), 0.1)
with f3:
    pt_max = st.number_input("Max playtime (min, 0 = any)", min_value=0, value=0, step=15)
with f4:
    supports = st.selectbox("Supports N players", ["any", 1, 2, 3, 4, 5, 6, 7, 8], index=0)

# ---- build the criteria-matching universe ----
rows = []
for r in owned.iter_rows(named=True):
    gid = int(r["game_id"])
    m = meta.get(gid, {})
    weight = m.get("average_weight")
    mn, mx = m.get("min_players"), m.get("max_players")
    mnt, mxt = m.get("min_playtime"), m.get("max_playtime")

    if rec_player != "any" and not _rec_contains(m.get("recommended_player_counts"), rec_player):
        continue
    if supports != "any":
        if mn is None or mx is None or not (int(mn) <= int(supports) <= int(mx)):
            continue
    if weight is not None and weight and not (cx[0] <= float(weight) <= cx[1]):
        continue
    if pt_max and mxt and int(mxt) > pt_max:
        continue

    rows.append(
        {
            "game_id": gid,
            "Game": (m.get("name") or r.get("game_name") or ""),
            "Year": int(m["year_published"]) if m.get("year_published") else None,
            "Status": _status_str(r),
            "Players": _rng(mn, mx),
            "Recommended": m.get("recommended_player_counts") or "",
            "Best": m.get("best_player_counts") or "",
            "Playtime": _rng(mnt, mxt),
            "Complexity": round(float(weight), 2) if weight else None,
            "selection": "no",
            "status": "no",
        }
    )

if not rows:
    st.info("No games match the criteria. Loosen the filters above.")
    st.stop()

df = pd.DataFrame(rows)

# ---- pre-populate labels from an existing selections file ----
sel_path = SELECTIONS_DIR / f"{sel_name}.yaml"
if sel_path.exists() and st.session_state.get("_loaded_sel") != str(sel_path):
    existing = load_selections(sel_path)
    df["selection"] = df["game_id"].map(
        lambda g: existing.games.get(g, {}).get("selection", "no")
    )
    df["status"] = df["game_id"].map(
        lambda g: existing.games.get(g, {}).get("status", "no")
    )
    st.session_state["_loaded_sel"] = str(sel_path)

st.subheader(f"Possible games ({len(df)})")
st.caption("Edit the **selection** and **status** columns. Locks are auto-set to status = yes on export.")

# ---- editable grid ----
gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_default_column(editable=False, resizable=True, sortable=True, filter=True)
gb.configure_column("game_id", hide=True)
gb.configure_column("Best", hide=True)
gb.configure_column(
    "selection",
    editable=True,
    cellEditor="agSelectCellEditor",
    cellEditorParams={"values": SELECTION_OPTS},
    width=120,
)
gb.configure_column(
    "status",
    editable=True,
    cellEditor="agSelectCellEditor",
    cellEditorParams={"values": STATUS_OPTS},
    width=110,
)
grid = AgGrid(
    df,
    gridOptions=gb.build(),
    update_mode=GridUpdateMode.VALUE_CHANGED,
    fit_columns_on_grid_load=False,
    allow_unsafe_jscode=True,
    height=520,
    key=f"grid_{sel_name}",
)

edited = pd.DataFrame(grid["data"])

# ---- export ----
st.subheader("Export")
n_lock = (edited["selection"] == "lock").sum()
n_maybe = (edited["selection"] == "maybe").sum()
st.write(f"**{n_lock}** locks · **{n_maybe}** maybes · **{len(edited)}** possible games")

if st.button("💾 Export selections", type="primary"):
    games_map = {}
    for _, row in edited.iterrows():
        selv = str(row["selection"])
        statv = str(row["status"])
        if selv == "lock":  # locks are always on the menu
            statv = "yes"
        games_map[int(row["game_id"])] = {"selection": selv, "status": statv}
    criteria = {
        "recommended_at": rec_player,
        "supports": supports,
        "complexity": list(cx),
        "max_playtime": pt_max or None,
    }
    s = Selections(name=sel_name, username=username, criteria=criteria, games=games_map)
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    dump_selections(s, sel_path)
    st.success(f"Wrote {sel_path.relative_to(PROJECT_ROOT)} — {len(games_map)} games.")
    st.caption(f"Render the report:  `just render-menu {sel_name}`")
