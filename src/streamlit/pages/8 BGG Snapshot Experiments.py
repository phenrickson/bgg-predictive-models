"""Browse trained candidates under the snapshot tree.

Pick a (snapshot, model, candidate, version, split) and see:
- The pipeline's coefficients / feature importances
- The tune/test fold predictions vs actuals
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, project_root)

from src.models.snapshot_storage import DEFAULT_BASE_DIR, SnapshotStorage  # noqa: E402


st.set_page_config(page_title="BGG Snapshot Experiments", layout="wide")
st.title("BGG Snapshot Experiments")
st.caption(
    "Inspect models trained under `models/bgg/snapshots/v{N}/`. "
    "Pick a (snapshot, model, candidate, version, split) to see coefficients and predictions."
)


# ---- selectors ----

snapshot_versions = SnapshotStorage.latest_version(base_dir=DEFAULT_BASE_DIR)
if snapshot_versions is None:
    st.warning(
        "No snapshots found at `models/bgg/snapshots/`. Run `just bgg-build` first."
    )
    st.stop()

# Build the list of available snapshot versions
all_versions: list[int] = []
from pathlib import Path  # noqa: E402

base = Path(DEFAULT_BASE_DIR)
for child in base.iterdir():
    if child.is_dir() and child.name.startswith("v"):
        try:
            all_versions.append(int(child.name[1:]))
        except ValueError:
            pass
all_versions = sorted(all_versions, reverse=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    snapshot_version = st.selectbox("Snapshot", all_versions, index=0)

storage = SnapshotStorage(snapshot_version=snapshot_version, base_dir=DEFAULT_BASE_DIR)

# Discover model_types under experiments/
exp_root = storage.snapshot_dir / "experiments"
if not exp_root.exists():
    st.info(f"No experiments under `models/bgg/snapshots/v{snapshot_version}/`. Train something first.")
    st.stop()

model_types = sorted(p.name for p in exp_root.iterdir() if p.is_dir())
with c2:
    model_type = st.selectbox("Model type", model_types)

# Discover candidates under model_type
candidates = sorted(p.name for p in (exp_root / model_type).iterdir() if p.is_dir())
with c3:
    candidate = st.selectbox("Candidate", candidates)

# Discover candidate versions
candidate_versions = sorted(
    storage.list_candidate_versions(model_type, candidate), reverse=True
)
with c4:
    candidate_version = st.selectbox("Version", candidate_versions, index=0)

# Discover splits this candidate has results for
results_root = storage.experiment_dir(model_type, candidate, candidate_version) / "results"
splits = (
    sorted(p.name for p in results_root.iterdir() if p.is_dir())
    if results_root.exists()
    else []
)
if not splits:
    st.warning("This candidate version has no per-split results.")
    st.stop()
with c5:
    split_name = st.selectbox("Split", splits)


# ---- load the result ----

result = storage.load_result(model_type, candidate, candidate_version, split_name)
if result is None:
    st.error("Failed to load result.")
    st.stop()

reg = storage.load_candidate_registration(model_type, candidate, candidate_version) or {}
config = storage.load_candidate_config(model_type, candidate, candidate_version) or {}

# ---- header ----

left, right = st.columns([2, 1])
with left:
    st.subheader(
        f"{model_type} / {candidate} / v{candidate_version} — split: {split_name}"
    )
    st.caption(
        f"Algorithm: `{config.get('algorithm', '?')}`  ·  "
        f"Snapshot v{snapshot_version}  ·  "
        f"Created: {reg.get('created_at', '?')}"
    )
with right:
    if reg.get("upstream_experiments"):
        st.caption(f"Upstream: `{reg['upstream_experiments']}`")


# ---- metrics ----

st.subheader("Metrics")
metrics = result.get("metrics") or {}
cols = st.columns(len(metrics)) if metrics else [st]
for col, (fold, mvals) in zip(cols, metrics.items()):
    with col:
        st.caption(fold.upper())
        # Filter to numeric, scalar values (skip confusion matrices, lists)
        scalar_metrics = {
            k: v for k, v in mvals.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        st.json(scalar_metrics, expanded=False)


# ---- coefficients / feature importance ----

st.subheader("Coefficients / feature importance")

pipeline = result["pipeline"]


def _extract_coefficients(pipe) -> pd.DataFrame | None:
    """Pull coefficients (linear) or feature_importances_ (tree) from a fitted
    sklearn pipeline. Returns None if neither is available."""
    try:
        preprocessor = pipe.named_steps.get("preprocessor")
        model = pipe.named_steps.get("model")
        if preprocessor is None or model is None:
            return None

        feature_names = None
        # Try preprocessor.get_feature_names_out
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            # Fall back to traversing nested steps in reverse
            if hasattr(preprocessor, "named_steps"):
                for _name, step in reversed(list(preprocessor.named_steps.items())):
                    try:
                        feature_names = step.get_feature_names_out()
                        break
                    except Exception:
                        continue

        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            if coef.ndim == 2:
                coef = coef[0]  # binary classification
            if feature_names is None or len(feature_names) != len(coef):
                feature_names = [f"f{i}" for i in range(len(coef))]
            df = pd.DataFrame({
                "feature": feature_names,
                "coefficient": coef,
                "abs_coefficient": np.abs(coef),
            })
            df = df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
            return df
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_)
            if feature_names is None or len(feature_names) != len(imp):
                feature_names = [f"f{i}" for i in range(len(imp))]
            df = pd.DataFrame({
                "feature": feature_names,
                "importance": imp,
            })
            df = df.sort_values("importance", ascending=False).reset_index(drop=True)
            return df
    except Exception as e:
        st.warning(f"Coefficient extraction failed: {e}")
    return None


coef_df = _extract_coefficients(pipeline)
if coef_df is None:
    st.info("No coefficients or feature importances available for this model.")
else:
    top_n = st.slider(
        "Top N features", min_value=10, max_value=200, value=40, step=5,
        key=f"snap_coef_n_{snapshot_version}_{model_type}_{candidate}_{candidate_version}_{split_name}",
    )

    if "coefficient" in coef_df.columns:
        plot_df = coef_df.nlargest(top_n, "abs_coefficient").sort_values("coefficient")
        fig = px.bar(
            plot_df,
            y="feature",
            x="coefficient",
            orientation="h",
            color="coefficient",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title=f"Top {top_n} Coefficients",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            height=max(400, top_n * 20),
            yaxis_title="",
            xaxis_title="Coefficient",
        )
        st.plotly_chart(fig, use_container_width=True)
    elif "importance" in coef_df.columns:
        plot_df = coef_df.nlargest(top_n, "importance").sort_values("importance")
        fig = px.bar(
            plot_df,
            y="feature",
            x="importance",
            orientation="h",
            color="importance",
            color_continuous_scale="Viridis",
            title=f"Top {top_n} Feature Importances",
        )
        fig.update_layout(
            height=max(400, top_n * 20),
            yaxis_title="",
            xaxis_title="Importance",
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander(f"Raw coefficient table ({len(coef_df)} features)"):
        st.dataframe(coef_df, use_container_width=True, height=400)


# ---- predictions vs actuals ----

st.subheader("Predictions vs actuals")

fold_choice = st.radio(
    "Fold",
    options=[f for f in ("tune", "test") if f"{f}_predictions" in result],
    horizontal=True,
)
preds_df: pl.DataFrame = result[f"{fold_choice}_predictions"]
preds_pd = preds_df.to_pandas()

# Sanity: scatter prediction vs actual.
if "prediction" in preds_pd.columns and "actual" in preds_pd.columns:
    pred_col = "predicted_proba" if "predicted_proba" in preds_pd.columns else "prediction"
    fig = px.scatter(
        preds_pd,
        x=pred_col,
        y="actual",
        hover_data={c: True for c in ("game_id", "name") if c in preds_pd.columns},
        title=f"{fold_choice}: {pred_col} vs actual ({preds_pd.shape[0]} games)",
        opacity=0.5,
    )
    # 45° line over the data range
    if pred_col != "predicted_proba":
        lo = float(min(preds_pd[pred_col].min(), preds_pd["actual"].min()))
        hi = float(max(preds_pd[pred_col].max(), preds_pd["actual"].max()))
        fig.add_shape(
            type="line",
            x0=lo, y0=lo, x1=hi, y1=hi,
            line=dict(color="red", width=1, dash="dash"),
        )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Predictions parquet missing `prediction` / `actual` columns.")

with st.expander(f"Raw {fold_choice} predictions ({preds_pd.shape[0]} rows)"):
    st.dataframe(preds_pd, use_container_width=True, height=400)


# ---- training set ----

st.subheader(f"Training set for {model_type}")
split_data = storage.load_split(split_name)
if split_data is None or "train" not in split_data:
    st.info("No train fold found for this split.")
else:
    train_df: pl.DataFrame = split_data["train"]

    # Reproduce what train_one fed to this model: join upstream score
    # columns (if any) and apply the model's prepare_features filtering.
    from types import SimpleNamespace as _SN

    from src.models.outcomes.train import get_model_class as _get_model_class

    upstream = (reg.get("upstream_experiments") or {})
    enriched = train_df
    for u_type, u_cand in upstream.items():
        u_versions = storage.list_candidate_versions(u_type, u_cand)
        if not u_versions:
            continue
        u_score = storage.load_score_predictions(u_type, u_cand, u_versions[-1], split_name)
        if u_score is None:
            continue
        join_cols = [c for c in u_score.columns if c == "game_id" or c not in enriched.columns]
        enriched = enriched.join(u_score.select(join_cols), on="game_id", how="left")

    model_kwargs = {}
    if "min_ratings" in config:
        model_kwargs["min_ratings"] = config["min_ratings"]
    if "min_weights" in config:
        model_kwargs["min_weights"] = config["min_weights"]
    if "mode" in config:
        model_kwargs["mode"] = config["mode"]
    if "include_predictions" in config:
        model_kwargs["include_predictions"] = config["include_predictions"]
    model_obj = _get_model_class(model_type)(**model_kwargs)

    target_col = model_obj.target_column
    enriched_pd = enriched.to_pandas()
    X = enriched_pd.drop(columns=[target_col]) if target_col in enriched_pd.columns else enriched_pd
    y = enriched_pd[target_col] if target_col in enriched_pd.columns else None

    prep_args = _SN(
        use_embeddings=bool(config.get("use_embeddings", False)),
        sub_model_experiments=config.get("sub_model_experiments", {}),
        mode=config.get("mode"),
        include_predictions=config.get("include_predictions", True),
    )
    try:
        X_filt, y_filt = model_obj.prepare_features(X, y, "train", prep_args)
    except Exception as e:
        st.warning(f"prepare_features failed; showing unfiltered split fold. Error: {e}")
        X_filt, y_filt = X, y

    # Recombine into one frame for display
    if y_filt is not None:
        filtered = X_filt.copy()
        filtered[target_col] = y_filt
    else:
        filtered = X_filt

    n_total = len(filtered)
    embed_cols = [c for c in filtered.columns if c.startswith(("emb_", "desc_emb_"))]
    sample = filtered.drop(columns=embed_cols).head(500)

    st.caption(
        f"After upstream join + `{model_type}.prepare_features` · "
        f"{n_total:,} rows total · "
        f"sample of 500 ({sample.shape[1]} cols, embeddings excluded)"
    )
    st.dataframe(sample, use_container_width=True, height=400)
