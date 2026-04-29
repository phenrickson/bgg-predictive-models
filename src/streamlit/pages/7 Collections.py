"""Streamlit page for exploring collection-model candidates per user.

Three tabs:

- **Overview** — comparison of all candidates trained for the selected
  ``(user, outcome)``. Reads ``registration.json`` from disk; never trains.
- **Candidate Detail** — drill into one candidate (tuning curve, feature
  importance, predictions on val/test).
- **Finalized Model** — load any candidate that has a ``finalized.pkl`` and
  use it to score upcoming/unseen games via the BigQuery feature universe.

The page is a thin layout layer; data access lives in
``src/streamlit/components/collection_loader.py`` and visual primitives are
reused from ``src/collection/viz.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import plotly.express as px
import polars as pl
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collection.viz import plot_feature_importance  # noqa: E402
from src.streamlit.components.collection_loader import (  # noqa: E402
    comparison_frame,
    get_storage,
    list_candidates,
    list_finalized_candidates,
    list_outcomes,
    list_splits_versions,
    list_users,
    load_feature_importance,
    load_finalized_run,
    load_predictions,
    load_runs,
    load_tuning_results,
    long_metrics_frame,
)
from src.streamlit.components.footer import render_footer  # noqa: E402

logger = logging.getLogger(__name__)
st.set_page_config(page_title="Collections | BGG Models Dashboard", layout="wide")
st.title("Collections")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feature_group_label(name: str) -> str:
    from src.collection.viz import feature_group
    return feature_group(name)


@st.cache_data(show_spinner=False)
def _load_universe(min_year: int) -> pl.DataFrame:
    from src.data.loader import BGGDataLoader
    from src.utils.config import load_config

    bq_config = load_config().get_bigquery_config()
    loader = BGGDataLoader(bq_config)
    universe = loader.load_features(use_predicted_complexity=True, use_embeddings=False)
    return universe.filter(pl.col("year_published") >= min_year)


def _score_universe(
    pipeline,
    min_year: int,
    top_n: int,
    task: str,
    threshold: float | None,
) -> pl.DataFrame:
    universe = _load_universe(min_year)
    if universe.height == 0:
        return universe.with_columns(pl.lit(None).cast(pl.Float64).alias("score"))

    X = universe.drop("label") if "label" in universe.columns else universe
    X_pd = X.to_pandas()
    if task == "classification":
        proba = pipeline.predict_proba(X_pd)[:, 1]
        scored = universe.with_columns(pl.Series("score", proba))
        if threshold is not None:
            scored = scored.with_columns(
                (pl.col("score") >= threshold).alias("pred")
            )
    else:
        pred = pipeline.predict(X_pd)
        scored = universe.with_columns(pl.Series("score", pred))

    keep = [
        c for c in ["game_id", "name", "year_published", "users_rated", "score", "pred"]
        if c in scored.columns
    ]
    return scored.select(keep).sort("score", descending=True).head(top_n)

# ---------------------------------------------------------------------------
# Top-level selectors (apply to every tab)
# ---------------------------------------------------------------------------

users = list_users()
if not users:
    st.warning(
        "No collection models found on disk under `models/collections/`. "
        "Train candidates first via `python -m src.collection.train`."
    )
    render_footer()
    st.stop()

col1, col2 = st.columns([1, 1])
with col1:
    username = st.selectbox("User", users, index=0)
storage = get_storage(username)

outcomes = list_outcomes(storage)
if not outcomes:
    st.warning(f"No outcomes trained for user `{username}` yet.")
    render_footer()
    st.stop()
with col2:
    outcome = st.selectbox("Outcome", outcomes, index=0)

splits_versions = sorted(list_splits_versions(storage, outcome), reverse=True)
splits_filter: int | None = None
if splits_versions:
    splits_choice = st.selectbox(
        "Splits version",
        [f"v{v}" for v in splits_versions],
        index=0,
        key="splits_filter",
    )
    splits_filter = int(splits_choice[1:])

filtered_runs = load_runs(storage, outcome, splits_version=splits_filter)
candidates = sorted({r.get("candidate") for r in filtered_runs if r.get("candidate")})
if not candidates:
    st.warning(
        f"No candidate runs found for `{username}` / `{outcome}` against "
        f"splits v{splits_filter}."
        if splits_filter is not None
        else f"No candidate runs found for `{username}` / `{outcome}`. "
        f"Run `python -m src.collection.train --username {username} --outcome {outcome}`."
    )
    render_footer()
    st.stop()

splits_caption = f"splits v{splits_filter}" if splits_filter is not None else "—"
st.caption(
    f"User: **{username}** &nbsp;·&nbsp; Outcome: **{outcome}** "
    f"&nbsp;·&nbsp; {splits_caption} &nbsp;·&nbsp; {len(candidates)} candidate(s)"
)

(
    tab_collection,
    tab_overview,
    tab_detail,
    tab_compare,
    tab_topn,
    tab_finalized,
) = st.tabs(
    [
        "Collection",
        "Overview",
        "Candidate Detail",
        "Compare Predictions",
        "Top N",
        "Finalized Model",
    ]
)

# ---------------------------------------------------------------------------
# Tab 0 — Collection (raw BGG snapshot from BigQuery)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_user_collection(username: str) -> pl.DataFrame | None:
    from src.collection.collection_storage import CollectionStorage

    storage = CollectionStorage(environment="dev")
    return storage.get_latest_collection(username)


with tab_collection:
    st.subheader(f"{username}'s collection")
    try:
        coll = _load_user_collection(username)
    except Exception as e:  # noqa: BLE001
        logger.exception("collection load failed")
        st.error(f"Failed to load collection from BigQuery: {e}")
        coll = None

    if coll is None or coll.height == 0:
        st.info(f"No stored collection rows for `{username}` in BigQuery.")
    else:
        flag_cols = [
            c for c in ["owned", "prev_owned", "previously_owned", "want",
                        "wishlist", "preordered", "for_trade", "want_to_play",
                        "want_to_buy"]
            if c in coll.columns
        ]
        rated = (
            coll.filter(pl.col("user_rating") > 0).height
            if "user_rating" in coll.columns else 0
        )

        cols = st.columns(min(len(flag_cols) + 1, 5) or 1)
        cols[0].metric("Total games", f"{coll.height:,}")
        for i, c in enumerate(flag_cols[:4], start=1):
            n = coll.filter(pl.col(c) == True).height
            cols[i].metric(c, f"{n:,}")
        if rated and len(cols) > 1 + len(flag_cols[:4]):
            pass  # already in the metrics row

        if rated:
            mean_r = coll.filter(pl.col("user_rating") > 0)["user_rating"].mean()
            st.caption(f"Rated games: {rated:,} · mean rating: {mean_r:.2f}")

        f1, f2 = st.columns([1, 1])
        with f1:
            search = st.text_input("Search game name", key="coll_search").strip().lower()
        with f2:
            flag_choice = st.selectbox(
                "Filter by flag", ["(any)"] + flag_cols, key="coll_flag"
            )

        view = coll
        if search:
            name_col = "game_name" if "game_name" in view.columns else "name"
            if name_col in view.columns:
                view = view.filter(
                    pl.col(name_col).str.to_lowercase().str.contains(search)
                )
        if flag_choice != "(any)":
            view = view.filter(pl.col(flag_choice) == True)

        preferred = [
            c for c in ["game_id", "game_name", "name", "user_rating",
                        "owned", "prev_owned", "previously_owned", "want",
                        "wishlist", "preordered", "for_trade",
                        "wishlist_priority", "first_seen_at", "updated_at"]
            if c in view.columns
        ]
        rest = [c for c in view.columns if c not in preferred]
        view = view.select(preferred + rest)

        if "user_rating" in view.columns:
            view = view.sort("user_rating", descending=True, nulls_last=True)

        st.caption(f"{view.height:,} rows")
        st.dataframe(view.head(1000).to_pandas(), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------

with tab_overview:
    runs = filtered_runs
    if not runs:
        st.info("No registrations on disk for this outcome yet.")
    else:
        wide = comparison_frame(runs)
        st.subheader("Comparison")
        st.dataframe(wide.to_pandas(), use_container_width=True)

        long_df = long_metrics_frame(runs)
        if long_df.height:
            metric_options = sorted(long_df["metric"].unique().to_list())
            default_idx = (
                metric_options.index("roc_auc") if "roc_auc" in metric_options else 0
            )
            split_options = sorted(
                long_df["split"].unique().to_list(),
                key=lambda s: {"val": 0, "oof": 1, "test": 2}.get(s, 99),
            )
            mc1, mc2 = st.columns([1, 1])
            with mc1:
                metric_choice = st.selectbox("Metric", metric_options, index=default_idx)
            with mc2:
                default_split_idx = (
                    split_options.index("test") if "test" in split_options else 0
                )
                split_choice = st.selectbox(
                    "Split", split_options, index=default_split_idx
                )

            chart_df = long_df.filter(
                (pl.col("metric") == metric_choice) & (pl.col("split") == split_choice)
            ).sort("value", descending=True)
            if chart_df.height:
                fig = px.bar(
                    chart_df.to_pandas(),
                    x="candidate",
                    y="value",
                    title=f"{metric_choice} ({split_choice})",
                    text="value",
                )
                fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                fig.update_layout(yaxis_title=metric_choice, xaxis_title=None)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No values for metric={metric_choice}, split={split_choice}.")

        # Splits + run metadata
        with st.expander("Run metadata"):
            meta_rows = []
            for r in runs:
                meta_rows.append({
                    "candidate": r.get("candidate"),
                    "version": r.get("version"),
                    "splits_version": r.get("splits_version"),
                    "n_train_used": r.get("n_train_used"),
                    "n_val": r.get("n_val"),
                    "n_test": r.get("n_test"),
                    "threshold": r.get("threshold"),
                    "trained_at": r.get("trained_at"),
                    "git_sha": (r.get("git_sha") or "")[:8],
                    "finalize_through": r.get("finalize_through"),
                    "finalized_at": r.get("finalized_at"),
                })
            st.dataframe(pl.DataFrame(meta_rows).to_pandas(), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2 — Candidate Detail
# ---------------------------------------------------------------------------

with tab_detail:
    candidate = st.selectbox("Candidate", candidates, key="detail_candidate")
    # Use the version of this candidate's run that matches the chosen splits filter
    # (filtered_runs already enforces that). Fall back to storage's latest if missing.
    matched_run = next(
        (r for r in filtered_runs if r.get("candidate") == candidate), None
    )
    if matched_run is not None:
        version = matched_run.get("version")
        registration = matched_run
    else:
        version = storage.latest_candidate_version(outcome, candidate)
        registration = storage.load_candidate_registration(
            outcome, candidate, version=version
        ) or {}

    st.subheader(f"{candidate} (v{version})")

    # Headline metrics
    m_val = registration.get("val_metrics") or {}
    m_test = registration.get("metrics") or {}
    m_oof = (registration.get("oof_metrics") or {}).get("overall") or {}
    if m_val or m_test:
        cols = st.columns(4)
        for i, key in enumerate(["roc_auc", "pr_auc", "f1", "log_loss"]):
            with cols[i]:
                val_v = m_val.get(key)
                test_v = m_test.get(key)
                oof_v = m_oof.get(key)
                val_s = f"{val_v:.4f}" if isinstance(val_v, (int, float)) else "—"
                test_s = f"{test_v:.4f}" if isinstance(test_v, (int, float)) else "—"
                oof_s = f"{oof_v:.4f}" if isinstance(oof_v, (int, float)) else None
                delta_text = f"val: {val_s}" + (f"  ·  oof: {oof_s}" if oof_s else "")
                st.metric(label=f"{key} (test)", value=test_s, delta=delta_text, delta_color="off")

    with st.expander("Best params + spec"):
        st.json({
            "best_params": registration.get("best_params"),
            "candidate_spec": registration.get("candidate_spec"),
            "threshold": registration.get("threshold"),
            "splits_version": registration.get("splits_version"),
        })

    # Tuning curve
    tuning = load_tuning_results(storage, outcome, candidate, version=version)
    if tuning is not None and tuning.height:
        st.subheader("Tuning")
        score_cols = [c for c in tuning.columns if c.startswith("score_") or c == "score"]
        if "score" in tuning.columns:
            tuning_sorted = tuning.sort("score").to_pandas().reset_index(drop=True)
            tuning_sorted["trial"] = tuning_sorted.index
            fig = px.line(tuning_sorted, x="trial", y="score", markers=True,
                          title="Tuning trials (sorted by score)")
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Tuning trace"):
            st.dataframe(tuning.to_pandas(), use_container_width=True)

    # Feature importance
    importance = load_feature_importance(storage, outcome, candidate, version=version)
    if importance is not None and importance.height:
        st.subheader("Feature importance")
        groups_present = sorted({
            _feature_group_label(name) for name in importance["feature"].to_list()
        })
        ic1, ic2 = st.columns([1, 1])
        with ic1:
            group_choice = st.selectbox(
                "Group filter", ["(all)"] + groups_present, key="fi_group"
            )
        with ic2:
            top_n = st.slider("Top N (each direction)", 5, 50, 20, key="fi_top_n")

        try:
            fig = plot_feature_importance(
                importance.to_pandas(),
                group=None if group_choice == "(all)" else group_choice,
                top_pos=top_n,
                top_neg=top_n,
                interactive=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not render feature importance: {e}")

    # OOF metrics (when present) — separate section since they describe a
    # different read on the same model than val/test.
    oof_block = registration.get("oof_metrics") or {}
    if oof_block:
        st.subheader("OOF cross-validation")
        st.caption(
            f"{oof_block.get('n_folds', '?')} folds · "
            f"seed {oof_block.get('seed', '?')} · "
            f"stratified on `{oof_block.get('stratified_on') or 'none'}` · "
            f"threshold {oof_block.get('threshold')}"
        )
        per_fold = oof_block.get("per_fold") or []
        overall = oof_block.get("overall") or {}
        if per_fold:
            rows = list(per_fold)
            if overall:
                rows = rows + [{"fold": "overall", **overall}]
            st.dataframe(pl.DataFrame(rows).to_pandas(), use_container_width=True)

    # Predictions
    st.subheader("Predictions")
    split = st.radio(
        "Split", ["test", "val", "oof"], index=0, horizontal=True, key="pred_split"
    )
    preds = load_predictions(storage, outcome, candidate, split=split, version=version)
    if preds is None or preds.height == 0:
        st.info(f"No {split} predictions saved for this run.")
    else:
        search = st.text_input("Search game name", key="pred_search").strip().lower()
        view = preds
        if search and "name" in view.columns:
            view = view.filter(pl.col("name").str.to_lowercase().str.contains(search))

        if "proba" in view.columns:
            view = view.sort("proba", descending=True)
        elif "prediction" in view.columns:
            view = view.sort("prediction", descending=True)

        # Surface predicted score next to the name so the descending sort
        # is visible without scrolling.
        preferred = [
            c for c in ["fold", "proba", "pred", "prediction", "actual",
                        "game_id", "name", "year_published", "users_rated", "label"]
            if c in view.columns
        ]
        rest = [c for c in view.columns if c not in preferred]
        view = view.select(preferred + rest)

        st.caption(f"{view.height:,} rows")
        st.dataframe(view.head(500).to_pandas(), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3 — Compare Predictions
# ---------------------------------------------------------------------------

with tab_compare:
    compare_split = st.radio(
        "Split",
        ["test", "val", "oof"],
        index=0,
        horizontal=True,
        key="compare_split",
    )

    candidate_choices = st.multiselect(
        "Candidates",
        candidates,
        default=candidates,
        key="compare_candidates",
    )

    if not candidate_choices:
        st.info("Pick at least one candidate.")
    else:
        per_candidate: dict[str, pl.DataFrame] = {}
        missing: list[str] = []
        meta_columns_seen: dict[str, pl.DataFrame] = {}
        for cand in candidate_choices:
            run = next((r for r in filtered_runs if r.get("candidate") == cand), None)
            if run is None:
                missing.append(cand)
                continue
            preds = load_predictions(
                storage, outcome, cand, split=compare_split, version=run.get("version")
            )
            if preds is None or preds.height == 0:
                missing.append(cand)
                continue
            if "proba" not in preds.columns:
                missing.append(f"{cand} (no proba column — regression?)")
                continue
            per_candidate[cand] = preds
            meta_columns_seen[cand] = preds

        if missing:
            st.caption(
                f"Skipped (no `{compare_split}` predictions or missing proba): "
                + ", ".join(missing)
            )

        if not per_candidate:
            st.info(f"No candidates have saved `{compare_split}` predictions.")
        else:
            # Build the wide frame: meta columns + one proba_<cand> per candidate.
            # Use the first candidate's frame as the spine; left-join the rest by game_id.
            spine_cand, spine = next(iter(per_candidate.items()))
            meta_cols = [
                c for c in ["game_id", "name", "year_published", "users_rated", "label"]
                if c in spine.columns
            ]
            spine_view = spine.select(
                meta_cols + ["proba"]
            ).rename({"proba": f"proba_{spine_cand}"})

            wide = spine_view
            for cand, df in per_candidate.items():
                if cand == spine_cand:
                    continue
                if "game_id" not in df.columns:
                    continue
                slim = df.select(["game_id", "proba"]).rename(
                    {"proba": f"proba_{cand}"}
                )
                wide = wide.join(slim, on="game_id", how="left")

            proba_cols = [f"proba_{c}" for c in per_candidate.keys()]
            rank_options = ["mean across candidates"] + proba_cols
            rank_choice = st.selectbox(
                "Rank by", rank_options, index=0, key="compare_rank_by"
            )

            search = st.text_input("Search game name", key="compare_search").strip().lower()
            if search and "name" in wide.columns:
                wide = wide.filter(
                    pl.col("name").str.to_lowercase().str.contains(search)
                )

            if rank_choice == "mean across candidates":
                wide = wide.with_columns(
                    pl.mean_horizontal(*proba_cols).alias("_rank")
                )
                wide = wide.sort("_rank", descending=True, nulls_last=True).drop("_rank")
            else:
                wide = wide.sort(rank_choice, descending=True, nulls_last=True)

            # Pairwise scatter: pick two candidates, plot proba vs proba colored by label.
            if len(per_candidate) >= 2:
                st.subheader("Pairwise scatter")
                cand_list = list(per_candidate.keys())
                sc1, sc2 = st.columns([1, 1])
                with sc1:
                    x_cand = st.selectbox(
                        "X-axis candidate", cand_list, index=0, key="compare_scatter_x"
                    )
                with sc2:
                    y_default = 1 if len(cand_list) > 1 else 0
                    y_cand = st.selectbox(
                        "Y-axis candidate", cand_list, index=y_default, key="compare_scatter_y"
                    )
                if x_cand == y_cand:
                    st.caption("Pick two different candidates to see disagreement.")
                else:
                    x_col = f"proba_{x_cand}"
                    y_col = f"proba_{y_cand}"
                    scatter_df = wide.select(
                        [c for c in ["name", "year_published", "label", x_col, y_col]
                         if c in wide.columns]
                    ).drop_nulls(subset=[x_col, y_col])
                    if scatter_df.height == 0:
                        st.info("No overlapping rows between those two candidates.")
                    else:
                        color_col = "label" if "label" in scatter_df.columns else None
                        hover = [c for c in ["name", "year_published"] if c in scatter_df.columns]
                        fig = px.scatter(
                            scatter_df.to_pandas(),
                            x=x_col,
                            y=y_col,
                            color=color_col,
                            hover_data=hover or None,
                            opacity=0.55,
                            title=f"{x_cand} vs {y_cand} (proba)",
                        )
                        # 45-degree reference line so agreement is the diagonal.
                        fig.add_shape(
                            type="line", x0=0, y0=0, x1=1, y1=1,
                            line={"color": "gray", "dash": "dash", "width": 1},
                        )
                        fig.update_layout(
                            xaxis_title=f"proba — {x_cand}",
                            yaxis_title=f"proba — {y_cand}",
                            xaxis_range=[0, 1],
                            yaxis_range=[0, 1],
                        )
                        st.plotly_chart(fig, use_container_width=True)

            st.caption(f"{wide.height:,} rows · split = `{compare_split}`")
            st.dataframe(wide.head(500).to_pandas(), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4 — Top N picks for a selected candidate
# ---------------------------------------------------------------------------

with tab_topn:
    topn_candidate = st.selectbox("Candidate", candidates, key="topn_candidate")
    matched_run = next(
        (r for r in filtered_runs if r.get("candidate") == topn_candidate), None
    )
    topn_version = matched_run.get("version") if matched_run else None

    topn_split = st.radio(
        "Split",
        ["test", "val", "oof"],
        index=0,
        horizontal=True,
        key="topn_split",
    )

    topn_preds = load_predictions(
        storage, outcome, topn_candidate, split=topn_split, version=topn_version
    )

    if topn_preds is None or topn_preds.height == 0:
        st.info(f"No `{topn_split}` predictions for `{topn_candidate}`.")
    elif "proba" not in topn_preds.columns:
        st.info("Top-N currently surfaces classification proba; this is a regression run.")
    elif "year_published" not in topn_preds.columns or "name" not in topn_preds.columns:
        st.info("Predictions parquet is missing `year_published` or `name`; cannot pivot by year.")
    else:
        all_years = sorted(
            int(y) for y in topn_preds["year_published"].drop_nulls().unique().to_list()
        )
        if not all_years:
            st.info("No year_published values in this prediction set.")
        else:
            data_min_year = all_years[0]
            data_max_year = all_years[-1]
            # OOF spans the full training set; fix the slider bounds at 1990
            # (or the data's min, whichever is later) → present.
            if topn_split == "oof":
                min_year = max(1990, data_min_year)
            else:
                min_year = data_min_year
            max_year = data_max_year
            default_lo = max(2015, min_year)
            default_hi = max_year
            if default_lo > default_hi:
                default_lo = min_year

            c1, c2 = st.columns([1, 2])
            with c1:
                top_n = st.slider("Top N", 5, 100, 25, key="topn_n")
            with c2:
                if min_year == max_year:
                    st.caption(f"Year: **{min_year}** (only year in this prediction set)")
                    lo = hi = min_year
                else:
                    year_range = st.slider(
                        "Year range",
                        min_value=min_year,
                        max_value=max_year,
                        value=(default_lo, default_hi),
                        step=1,
                        key="topn_year_range",
                    )
                    lo, hi = int(year_range[0]), int(year_range[1])

            view = topn_preds.filter(
                (pl.col("year_published") >= lo) & (pl.col("year_published") <= hi)
            )
            # year_published is sometimes Float64 in saved predictions; cast to
            # Int64 so pivoted column names render as "2024" not "2024.0".
            view = view.with_columns(pl.col("year_published").cast(pl.Int64))

            # Per year: take top-N rows by proba.
            view = view.with_columns(
                pl.col("proba")
                .rank(method="ordinal", descending=True)
                .over("year_published")
                .alias("_rank")
            ).filter(pl.col("_rank") <= top_n)

            if view.height == 0:
                st.info("No rows in the selected year range.")
            else:
                # Build name and label tables: rows = rank (1..N), cols = year.
                year_cols = sorted(
                    int(y) for y in view["year_published"].unique().to_list()
                )

                name_pivot = view.pivot(
                    values="name", index="_rank", on="year_published"
                ).sort("_rank").rename({"_rank": "rank"})
                # Make sure column order is ascending year and all selected years appear.
                ordered = ["rank"] + [str(y) for y in year_cols]
                name_pivot = name_pivot.select(
                    [c for c in ordered if c in name_pivot.columns]
                )

                if "label" in view.columns:
                    label_pivot = view.pivot(
                        values="label", index="_rank", on="year_published"
                    ).sort("_rank").rename({"_rank": "rank"})
                    label_pivot = label_pivot.select(
                        [c for c in ordered if c in label_pivot.columns]
                    )
                else:
                    label_pivot = None

                name_pdf = name_pivot.to_pandas().set_index("rank")

                if label_pivot is not None:
                    import pandas as pd

                    label_pdf = label_pivot.to_pandas().set_index("rank")

                    def _style_all(df):
                        out = pd.DataFrame("", index=df.index, columns=df.columns)
                        for col in df.columns:
                            if col not in label_pdf.columns:
                                continue
                            for idx in df.index:
                                if idx not in label_pdf.index:
                                    continue
                                truth = label_pdf.at[idx, col]
                                if truth is True or truth == 1:
                                    out.at[idx, col] = (
                                        "background-color: #1f4e79; color: white"
                                    )
                        return out

                    styler = name_pdf.style.apply(_style_all, axis=None)
                    st.caption(
                        f"`{topn_candidate}` · split = `{topn_split}` · "
                        f"top {top_n} per year · {lo}–{hi} · "
                        "blue = true positive"
                    )
                    st.dataframe(styler, use_container_width=True)
                else:
                    st.caption(
                        f"`{topn_candidate}` · split = `{topn_split}` · "
                        f"top {top_n} per year · {lo}–{hi}"
                    )
                    st.dataframe(name_pdf, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 5 — Finalized Model
# ---------------------------------------------------------------------------

with tab_finalized:
    finalized = list_finalized_candidates(storage, outcome)
    if not finalized:
        st.info(
            f"No finalized model for `{username}` / `{outcome}` yet. Run "
            f"`python -m src.collection.finalize --username {username} "
            f"--outcome {outcome} --candidate <name>`."
        )
    else:
        if len(finalized) == 1:
            cand_name, cand_version = finalized[0]
            st.caption(f"Finalized: **{cand_name}** (v{cand_version})")
        else:
            label_to_pair = {f"{c} (v{v})": (c, v) for c, v in finalized}
            choice = st.selectbox(
                "Finalized candidate", list(label_to_pair.keys()), key="finalized_pick"
            )
            cand_name, cand_version = label_to_pair[choice]

        try:
            pipeline, registration = load_finalized_run(
                storage, outcome, cand_name, cand_version
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to load finalized model: {e}")
            pipeline = None
            registration = {}

        if pipeline is not None:
            finalize_through = registration.get("finalize_through")
            threshold = registration.get("threshold")
            cols = st.columns(3)
            cols[0].metric("Threshold", f"{threshold:.3f}" if threshold else "—")
            cols[1].metric("Finalize through", str(finalize_through or "—"))
            cols[2].metric(
                "Finalized at",
                (registration.get("finalized_at") or "")[:19] or "—",
            )

            with st.expander("Run registration"):
                st.json(registration)

            st.subheader("Score upcoming games")

            year_default = int(finalize_through) + 1 if finalize_through else 2026
            sc1, sc2 = st.columns([1, 1])
            with sc1:
                min_year = st.number_input(
                    "Min year_published",
                    min_value=1900, max_value=2100, value=year_default, step=1,
                )
            with sc2:
                top_n = st.slider("Top N", 10, 200, 50)

            if st.button("Score from BigQuery universe", type="primary"):
                with st.spinner("Loading universe and scoring…"):
                    try:
                        scored = _score_universe(
                            pipeline=pipeline,
                            min_year=int(min_year),
                            top_n=int(top_n),
                            task=registration.get("task", "classification"),
                            threshold=threshold,
                        )
                        st.session_state["_finalized_scored"] = scored
                    except Exception as e:  # noqa: BLE001
                        logger.exception("scoring failed")
                        st.error(f"Scoring failed: {e}")
                        st.session_state.pop("_finalized_scored", None)

            scored = st.session_state.get("_finalized_scored")
            if scored is not None and scored.height:
                search = st.text_input("Search", key="final_search").strip().lower()
                view = scored
                if search and "name" in view.columns:
                    view = view.filter(pl.col("name").str.to_lowercase().str.contains(search))
                st.caption(f"{view.height:,} rows")
                st.dataframe(view.to_pandas(), use_container_width=True)


render_footer()
