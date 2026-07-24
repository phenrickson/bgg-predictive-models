"""Standalone Streamlit app: curate a proposed list of games from a collection.

Filter a collection by criteria, then tag each game's **selection**
(lock / maybe / other / no) and **status** (yes / no) and export to
``reports/selections/<name>.yaml`` for the menu report to consume.

Editing happens inside a form: nothing reruns until you hit **Apply changes**,
so tagging a batch of games is smooth. Selections persist in session state,
keyed by game_id, so changing the filters never loses your labels. Games left
at ``no`` are excluded from the export entirely.

Launch:  just select   (or  uv run streamlit run src/selections/app.py)

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

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode  # noqa: E402

from src.reports.collection_data import (  # noqa: E402
    _fetch_collection_snapshot,
    _fetch_games_metadata,
)
from src.reports.game_cards import complexity_label  # noqa: E402
from src.reports.selections import Selections, dump_selections, load_selections  # noqa: E402

SELECTIONS_DIR = PROJECT_ROOT / "reports" / "selections"
SELECTION_OPTS = ["no", "other", "maybe", "lock"]
STATUS_OPTS = ["no", "yes"]

st.set_page_config(page_title="Game Selections", layout="wide")

# --- warm identity, matched to the Rocky Bilbao report ---
st.markdown(
    """
    <style>
      :root { --sol:#e2571f; --rioja:#8c2f2a; --olive:#6b7238; --ink:#2a1e14; }
      h1, h2, h3 { font-family: Georgia, "Iowan Old Style", serif !important; }
      h1 span.accent { color: var(--sol); }
      .stButton>button[kind="primary"] { background: var(--sol); border-color: var(--sol); }
      div[data-baseweb="slider"] [role="slider"] { background: var(--sol) !important; }
      .chip { display:inline-block; padding:.15rem .55rem; border-radius:1rem;
              font-size:.8rem; font-weight:600; margin-right:.4rem; }
      .chip-lock { background:#8c2f2a; color:#fff; }
      .chip-maybe { background:#6b7238; color:#fff; }
      .chip-other { background:#3b6b7a; color:#fff; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading collection…")
def _load(username: str):
    return _fetch_collection_snapshot(username), _fetch_games_metadata()


def _status_str(row) -> str:
    for flag, label in [
        ("owned", "Own"), ("preordered", "Preordered"), ("wishlist", "Wishlist"),
        ("want", "Want"), ("want_to_buy", "Want"),
        ("previously_owned", "Prev. Owned"), ("prev_owned", "Prev. Owned"),
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


st.markdown("# 🎲 Game <span class='accent'>Selections</span>", unsafe_allow_html=True)
st.caption(
    "Filter a collection, then tag games **lock / maybe / other**. Anything left "
    "at **no** is excluded. Edits apply on the button — the grid won't jump around "
    "while you work."
)

# ---- labels live in session state, keyed by game_id, so filters never lose them ----
if "labels" not in st.session_state:
    st.session_state.labels = {}  # gid -> {"selection","status"}


def _label(gid: int) -> dict:
    return st.session_state.labels.get(int(gid), {"selection": "no", "status": "no"})


# ---- user + selection name ----
c1, c2, c3 = st.columns([1.2, 1.2, 1])
with c1:
    username = st.text_input("BGG username", value="phenrickson")
with c2:
    sel_name = st.text_input("Selection name", value="rocky_bilbao_2026")
with c3:
    st.write("")
    reload_saved = st.button("↺ Reload from file", help="Discard edits, reload the saved file")

if not username:
    st.stop()

coll, games = _load(username)
if coll.height == 0:
    st.warning(f"No collection found for {username!r}.")
    st.stop()

sel_path = SELECTIONS_DIR / f"{sel_name}.yaml"

# Auto-load the saved selections whenever the name changes to one that has a
# file on disk (or when Reload is pressed). Tracked so we don't clobber edits
# on every rerun — only when the target name actually changes.
_should_load = reload_saved or st.session_state.get("_loaded_name") != sel_name
if _should_load and sel_path.exists():
    existing = load_selections(sel_path)
    st.session_state.labels = {int(g): dict(lab) for g, lab in existing.games.items()}
    st.session_state["_loaded_name"] = sel_name
    st.caption(f"Loaded {len(existing.games)} saved labels from `{sel_path.name}`.")
elif _should_load:
    # New name with no file yet — start fresh.
    st.session_state.labels = {}
    st.session_state["_loaded_name"] = sel_name

owned = coll.filter(pl.col("owned")) if "owned" in coll.columns else coll
meta = {int(r["game_id"]): r for r in games.iter_rows(named=True)}

# ---- filters + editor live in a fragment, so changing a filter re-runs ONLY
#      this region (table updates in place) instead of repainting the page ----
@st.fragment
def selector_panel():
    st.subheader("Criteria")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        rec_player = st.selectbox("Recommended at N players", ["any", 1, 2, 3, 4, 5, 6, 7, 8])
    with f2:
        cx = st.slider("Complexity", 1.0, 5.0, (1.0, 5.0), 0.1)
    with f3:
        pt_max = st.number_input("Max playtime (min, 0 = any)", min_value=0, value=0, step=15)
    with f4:
        supports = st.selectbox("Supports N players", ["any", 1, 2, 3, 4, 5, 6, 7, 8])
    # stash criteria so the export (outside the fragment) can record it
    st.session_state.criteria = {
        "recommended_at": rec_player, "supports": supports,
        "complexity": list(cx), "max_playtime": pt_max or None,
    }

    rows = []
    for r in owned.iter_rows(named=True):
        gid = int(r["game_id"])
        m = meta.get(gid, {})
        weight = m.get("average_weight")
        mn, mx = m.get("min_players"), m.get("max_players")
        mnt, mxt = m.get("min_playtime"), m.get("max_playtime")

        if rec_player != "any" and not _rec_contains(m.get("recommended_player_counts"), rec_player):
            continue
        if supports != "any" and (mn is None or mx is None or not (int(mn) <= int(supports) <= int(mx))):
            continue
        if weight and not (cx[0] <= float(weight) <= cx[1]):
            continue
        if pt_max and mxt and int(mxt) > pt_max:
            continue

        lab = _label(gid)
        rows.append(
            {
                "game_id": gid,
                "Game": m.get("name") or r.get("game_name") or "",
                "Year": int(m["year_published"]) if m.get("year_published") else None,
                "Owned": _status_str(r),
                "Players": _rng(mn, mx),
                "Recommended": m.get("recommended_player_counts") or "",
                "Playtime": _rng(mnt, mxt),
                "Complexity": complexity_label(weight),
                "selection": lab["selection"],
                "status": lab["status"],
            }
        )

    st.subheader(f"Possible games ({len(rows)})")
    if not rows:
        st.info("No games match the criteria. Loosen the filters above.")
        return

    df = pd.DataFrame(rows)

    # AgGrid: editing and sorting happen CLIENT-SIDE. update_mode=MODEL_CHANGED
    # returns the edited data on the natural fragment reruns (filter change,
    # export) WITHOUT forcing a rerun on each individual cell edit — so changing
    # a label does not re-sort or repaint the grid. We merge whatever comes back
    # into session_state so edits persist across filter changes.
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        editable=False, sortable=True, resizable=True, filter=True, flex=1
    )
    gb.configure_column("game_id", hide=True)
    gb.configure_column("Game", flex=3, minWidth=220)
    gb.configure_column("Recommended", flex=1, minWidth=110)
    gb.configure_column(
        "selection", header_name="Selection", editable=True, flex=1, minWidth=120,
        cellEditor="agSelectCellEditor", cellEditorParams={"values": SELECTION_OPTS},
    )
    gb.configure_column(
        "status", header_name="On menu?", editable=True, flex=1, minWidth=110,
        cellEditor="agSelectCellEditor", cellEditorParams={"values": STATUS_OPTS},
    )
    # Tint rows by their selection so the table reads at a glance.
    row_style = JsCode(
        """
        function(p){
          var v = p.data.selection;
          if (v==='lock')  return {backgroundColor:'rgba(140,47,42,0.16)'};
          if (v==='maybe') return {backgroundColor:'rgba(107,114,56,0.16)'};
          if (v==='other') return {backgroundColor:'rgba(59,107,122,0.14)'};
          return {};
        }
        """
    )
    grid_opts = gb.build()
    grid_opts["getRowStyle"] = row_style

    # Key on the FILTER values (not labels): a filter change re-seeds the grid
    # with the newly-filtered rows; editing a label leaves the key unchanged so
    # the grid keeps its sort/scroll and doesn't thrash.
    filter_key = f"grid_{rec_player}_{supports}_{cx[0]}_{cx[1]}_{pt_max}"
    grid = AgGrid(
        df,
        gridOptions=grid_opts,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        theme="balham",
        height=560,
        use_container_width=True,
        key=filter_key,
    )
    # Merge the grid's current data back into session_state (persists edits
    # across filter changes; the grid keeps its own sort while you edit).
    for row in grid["data"].to_dict("records") if hasattr(grid["data"], "to_dict") else grid["data"]:
        st.session_state.labels[int(row["game_id"])] = {
            "selection": str(row["selection"]),
            "status": str(row["status"]),
        }

    labels = st.session_state.labels
    nc = lambda s: sum(1 for v in labels.values() if v["selection"] == s)  # noqa: E731
    st.markdown(
        f"<span class='chip chip-lock'>{nc('lock')} locks</span>"
        f"<span class='chip chip-maybe'>{nc('maybe')} maybes</span>"
        f"<span class='chip chip-other'>{nc('other')} others</span>",
        unsafe_allow_html=True,
    )


selector_panel()

# ---- export (page-level; reads labels + criteria from session state) ----
if st.button("💾 Export selections", type="primary"):
    labels = st.session_state.labels
    games_map = {}
    for gid, lab in labels.items():
        selv = lab["selection"]
        if selv == "no":
            continue  # excluded entirely
        statv = "yes" if selv == "lock" else lab["status"]
        games_map[int(gid)] = {"selection": selv, "status": statv}
    if not games_map:
        st.warning("Nothing selected — tag some games lock/maybe/other first.")
    else:
        s = Selections(
            name=sel_name, username=username,
            criteria=st.session_state.get("criteria", {}), games=games_map,
        )
        SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        dump_selections(s, sel_path)
        st.success(f"Wrote {sel_path.relative_to(PROJECT_ROOT)} — {len(games_map)} games.")
        st.caption(f"Render the report:  `just render-menu {sel_name}`")
