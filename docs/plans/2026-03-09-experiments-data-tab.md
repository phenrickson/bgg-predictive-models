# Experiments Data Tab Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the Experiments page into 6 focused tabs and add a Data tab for visualizing feature-outcome relationships in training data.

**Architecture:** Extract existing Metrics/Predictions/Coefficients sections from the monolithic Experiments tab into their own top-level tabs. Add two new loader functions to `experiment_loader.py` for raw and preprocessed data. Add a Data tab with raw/preprocessed toggle, feature selection dropdown, and plotly charts.

**Tech Stack:** Streamlit, Polars, Pandas, Plotly, scikit-learn (preprocessor from pipeline.pkl)

---

### Task 1: Add data loading functions to experiment_loader.py

**Files:**
- Modify: `src/streamlit/components/experiment_loader.py`

**Step 1: Add `load_training_data` function**

Add to `experiment_loader.py` after `load_coefficients`:

```python
def load_training_data(exp_path: Path, dataset: str = "train") -> Optional[pl.DataFrame]:
    """Load raw training data parquet for a dataset split."""
    path = exp_path / "data" / f"{dataset}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    return None


def load_preprocessed_data(exp_path: Path, dataset: str = "train") -> Optional["pd.DataFrame"]:
    """Load pipeline and run preprocessor on training data.

    Returns a pandas DataFrame with engineered feature columns.
    """
    import pickle
    import pandas as pd

    pipeline_path = exp_path / "pipeline.pkl"
    data_path = exp_path / "data" / f"{dataset}.parquet"

    if not pipeline_path.exists() or not data_path.exists():
        return None

    try:
        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)

        preprocessor = pipeline.named_steps["preprocessor"]
        raw_df = pl.read_parquet(data_path).to_pandas()
        return preprocessor.transform(raw_df)
    except Exception as e:
        logger.warning(f"Failed to load preprocessed data from {exp_path}: {e}")
        return None
```

**Step 2: Update the imports in `__init__` or verify they're importable**

The file uses direct imports, so just ensure the new functions are importable from the module.

**Step 3: Verify manually**

Run: `uv run python -c "from src.streamlit.components.experiment_loader import load_training_data, load_preprocessed_data; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/streamlit/components/experiment_loader.py
git commit -m "feat: add training data and preprocessed data loaders for experiment explorer"
```

---

### Task 2: Restructure tabs in Experiments page

**Files:**
- Modify: `src/streamlit/pages/2 Experiments.py`

**Step 1: Update imports**

Add `load_training_data` and `load_preprocessed_data` to the import block:

```python
from src.streamlit.components.experiment_loader import (
    discover_experiments,
    load_metrics,
    load_predictions,
    load_coefficients,
    load_training_data,
    load_preprocessed_data,
)
```

**Step 2: Move shared selectors above tabs**

Currently the Outcome/Dataset selectors are inside `tab_eval`. Move them above the tab creation so all experiment-focused tabs share them. The selectors should only show when there are eval experiments.

After the `eval_exps`/`finalized_exps` split (line ~331), replace the tab creation and the filter controls inside `tab_eval` with:

```python
# --- Shared selectors for experiment tabs ---
if eval_exps:
    eval_outcomes = sorted(set(e["outcome"] for e in eval_exps))
    filter_cols = st.columns(2)
    with filter_cols[0]:
        selected_outcome = st.selectbox("Outcome", eval_outcomes, key="eval_outcome")
    with filter_cols[1]:
        selected_dataset = st.selectbox("Dataset", ["test", "tune", "train"], key="eval_dataset")
    outcome_exps = [e for e in eval_exps if e["outcome"] == selected_outcome]

# --- Tabs ---
tab_metrics, tab_predictions, tab_coefficients, tab_data, tab_finalized, tab_metadata = st.tabs(
    ["Metrics", "Predictions", "Coefficients", "Data", "Models", "Metadata"]
)
```

**Step 3: Move Metrics section into `tab_metrics`**

Take the existing Metrics subheader + table + Metrics Over Time chart code (currently inside `tab_eval`) and wrap it in `with tab_metrics:`. Remove the `st.subheader("Metrics")` — replace with `st.header("Metrics")`.

**Step 4: Move Predictions section into `tab_predictions`**

Take the Predictions subheader + experiment selector + scatter plot + dataframe code and wrap in `with tab_predictions:`. Replace subheader with header.

**Step 5: Move Coefficients section into `tab_coefficients`**

Take the Coefficients subheader + coefficient loading + rendering code and wrap in `with tab_coefficients:`. Replace subheader with header.

**Step 6: Leave `tab_data` empty for now with placeholder**

```python
with tab_data:
    st.header("Data")
    st.info("Data explorer coming soon.")
```

**Step 7: Keep Models and Metadata tabs as-is**

Just rename `tab_finalized` and `tab_metadata` to use the new variable names.

**Step 8: Verify**

Run: `uv run streamlit run src/streamlit/Home.py` and check that the Experiments page has 6 tabs and the existing Metrics/Predictions/Coefficients content renders correctly in their new tabs.

**Step 9: Commit**

```bash
git add "src/streamlit/pages/2 Experiments.py"
git commit -m "refactor: split Experiments tab into Metrics, Predictions, Coefficients tabs"
```

---

### Task 3: Implement the Data tab — Raw view

**Files:**
- Modify: `src/streamlit/pages/2 Experiments.py`

**Step 1: Add raw data loading with caching**

Inside the `with tab_data:` block:

```python
with tab_data:
    st.header("Data")

    if not eval_exps or not outcome_exps:
        st.info("No eval experiments found.")
    else:
        # Pick experiment to explore
        data_exp_names = [e["name"] for e in outcome_exps]
        selected_data_exp_name = st.selectbox(
            "Experiment", data_exp_names, key="data_exp_select"
        )
        selected_data_exp = next(
            e for e in outcome_exps if e["name"] == selected_data_exp_name
        )

        view_mode = st.radio("View", ["Raw", "Preprocessed"], horizontal=True, key="data_view_mode")

        @st.cache_data
        def load_raw_cached(path_str, dataset):
            from pathlib import Path
            return load_training_data(Path(path_str), dataset)

        raw_df = load_raw_cached(str(selected_data_exp["path"]), selected_dataset)
```

**Step 2: Classify columns and build feature selector**

```python
        if raw_df is None:
            st.warning(f"No {selected_dataset} data found for this experiment.")
        elif view_mode == "Raw":
            outcome_col = selected_outcome

            # Classify columns
            meta_cols = {"game_id", "name", "image", "thumbnail", "description",
                         "last_updated", "name_right", "year_published_right", "complexity_right"}
            outcome_cols = {"rating", "hurdle", "complexity", "geek_rating", "users_rated",
                            "bayes_average", "average_rating", "log_users_rated"}
            list_cols = [c for c in raw_df.columns if raw_df[c].dtype == pl.List(pl.String)]
            numeric_cols = [c for c in raw_df.columns
                           if c not in meta_cols and c not in outcome_cols and c not in list_cols
                           and raw_df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]

            feature_type = st.radio("Feature type", ["Numeric", "Categorical"], horizontal=True, key="raw_feat_type")

            if feature_type == "Numeric":
                selected_feature = st.selectbox("Feature", sorted(numeric_cols), key="raw_num_feat")
            else:
                selected_feature = st.selectbox("Feature", sorted(list_cols), key="raw_cat_feat")
```

**Step 3: Render numeric scatter plot**

```python
            if feature_type == "Numeric" and selected_feature:
                pdf = raw_df.select([selected_feature, outcome_col]).drop_nulls().to_pandas()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pdf[selected_feature],
                    y=pdf[outcome_col],
                    mode="markers",
                    marker=dict(size=3, opacity=0.3),
                    showlegend=False,
                ))

                if HAS_STATSMODELS and len(pdf) > 10:
                    try:
                        sorted_idx = np.argsort(pdf[selected_feature].values)
                        smoothed = lowess_mod.lowess(
                            pdf[outcome_col].values[sorted_idx],
                            pdf[selected_feature].values[sorted_idx],
                            frac=2 / 3, it=5,
                        )
                        fig.add_trace(go.Scatter(
                            x=smoothed[:, 0], y=smoothed[:, 1],
                            mode="lines", line=dict(color="red", width=2),
                            name="LOESS", showlegend=True,
                        ))
                    except Exception:
                        pass

                label = selected_feature.replace("_", " ").title()
                fig.update_layout(
                    title=f"{label} vs {outcome_col.replace('_', ' ').title()}",
                    xaxis_title=label,
                    yaxis_title=outcome_col.replace("_", " ").title(),
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)
```

**Step 4: Render categorical box plot**

```python
            elif feature_type == "Categorical" and selected_feature:
                top_n_cats = st.slider("Top N categories", 5, 50, 20, 5, key="raw_cat_topn")
                # Explode list column and count frequencies
                exploded = raw_df.select([selected_feature, outcome_col]).explode(selected_feature).drop_nulls()
                # Get top N most frequent values
                top_values = (
                    exploded.group_by(selected_feature)
                    .agg(pl.count().alias("count"))
                    .sort("count", descending=True)
                    .head(top_n_cats)
                    [selected_feature].to_list()
                )
                plot_df = exploded.filter(pl.col(selected_feature).is_in(top_values)).to_pandas()

                # Order by median outcome
                medians = plot_df.groupby(selected_feature)[outcome_col].median().sort_values()
                fig = px.box(
                    plot_df, x=selected_feature, y=outcome_col,
                    category_orders={selected_feature: medians.index.tolist()},
                    title=f"{outcome_col.replace('_', ' ').title()} by {selected_feature.replace('_', ' ').title()}",
                )
                fig.update_layout(height=500, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
```

**Step 5: Verify**

Run `uv run streamlit run src/streamlit/Home.py`, navigate to Experiments > Data tab. Select a numeric feature — should see scatter with LOESS. Select categorical — should see box plot.

**Step 6: Commit**

```bash
git add "src/streamlit/pages/2 Experiments.py"
git commit -m "feat: add Data tab with raw feature vs outcome visualizations"
```

---

### Task 4: Implement the Data tab — Preprocessed view

**Files:**
- Modify: `src/streamlit/pages/2 Experiments.py`

**Step 1: Add preprocessed data loading and feature selection**

In the `with tab_data:` block, after the raw view `elif`, add the preprocessed branch:

```python
        elif view_mode == "Preprocessed":
            @st.cache_data
            def load_preprocessed_cached(path_str, dataset):
                from pathlib import Path
                return load_preprocessed_data(Path(path_str), dataset)

            with st.spinner("Running preprocessor..."):
                processed_df = load_preprocessed_cached(
                    str(selected_data_exp["path"]), selected_dataset
                )

            if processed_df is None:
                st.warning("Could not load preprocessed data (missing pipeline.pkl or data).")
            else:
                # Get outcome values from raw data to pair with processed features
                outcome_col = selected_outcome
                raw_for_outcome = load_raw_cached(str(selected_data_exp["path"]), selected_dataset)
                if raw_for_outcome is not None and outcome_col in raw_for_outcome.columns:
                    outcome_values = raw_for_outcome[outcome_col].to_pandas().values
                else:
                    st.warning(f"Cannot find {outcome_col} in raw data.")
                    outcome_values = None

                if outcome_values is not None:
                    st.caption(f"{processed_df.shape[1]:,} features, {processed_df.shape[0]:,} observations")

                    # Category filter
                    category = st.selectbox(
                        "Feature category",
                        list(FEATURE_CATEGORIES.keys()),
                        key="data_preproc_cat",
                    )
                    prefix = FEATURE_CATEGORIES[category]
                    all_features = list(processed_df.columns)
                    if prefix == "__other__":
                        known_prefixes = [p for p in FEATURE_CATEGORIES.values() if p and p != "__other__"]
                        filtered_features = [f for f in all_features if not any(f.startswith(p) for p in known_prefixes)]
                    elif prefix is not None:
                        filtered_features = [f for f in all_features if f.startswith(prefix)]
                    else:
                        filtered_features = all_features

                    selected_feature = st.selectbox(
                        "Feature", sorted(filtered_features), key="preproc_feat"
                    )

                    if selected_feature:
                        import pandas as pd
                        plot_pdf = pd.DataFrame({
                            selected_feature: processed_df[selected_feature].values,
                            outcome_col: outcome_values,
                        }).dropna()

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=plot_pdf[selected_feature],
                            y=plot_pdf[outcome_col],
                            mode="markers",
                            marker=dict(size=3, opacity=0.3),
                            showlegend=False,
                        ))

                        if HAS_STATSMODELS and len(plot_pdf) > 10:
                            try:
                                sorted_idx = np.argsort(plot_pdf[selected_feature].values)
                                smoothed = lowess_mod.lowess(
                                    plot_pdf[outcome_col].values[sorted_idx],
                                    plot_pdf[selected_feature].values[sorted_idx],
                                    frac=2 / 3, it=5,
                                )
                                fig.add_trace(go.Scatter(
                                    x=smoothed[:, 0], y=smoothed[:, 1],
                                    mode="lines", line=dict(color="red", width=2),
                                    name="LOESS", showlegend=True,
                                ))
                            except Exception:
                                pass

                        label = selected_feature.replace("_", " ").title()
                        fig.update_layout(
                            title=f"{label} (preprocessed) vs {outcome_col.replace('_', ' ').title()}",
                            xaxis_title=label,
                            yaxis_title=outcome_col.replace("_", " ").title(),
                            height=500,
                        )
                        st.plotly_chart(fig, use_container_width=True)
```

**Step 2: Verify**

Run `uv run streamlit run src/streamlit/Home.py`, navigate to Data tab, switch to Preprocessed. Select a category filter (e.g. "Mechanic"), pick a feature, confirm scatter renders.

**Step 3: Commit**

```bash
git add "src/streamlit/pages/2 Experiments.py"
git commit -m "feat: add preprocessed feature view to Data tab"
```
