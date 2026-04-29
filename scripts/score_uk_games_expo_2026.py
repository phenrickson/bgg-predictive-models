#!/usr/bin/env python
"""Score UK Games Expo 2026 preview games with a user's finalized models.

Pulls the BGG geek preview feed for the 2026 UK Games Expo (preview id 86),
loads features from the BigQuery warehouse for those game_ids, and scores
them with every finalized candidate model the user has at their most recent
splits version. Writes a CSV with one column per candidate.

Usage:
    uv run python scripts/score_uk_games_expo_2026.py \
        --username phenrickson \
        --outcome own
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
import requests

from src.collection.collection_artifact_storage import CollectionArtifactStorage
from src.data.loader import BGGDataLoader
from src.utils.config import load_config

PREVIEW_ID = 86
PREVIEW_API = "https://api.geekdo.com/api/geekpreviewitems"
SCORING_SERVICE_URL = "https://bgg-model-scoring-jwkjti5j5a-uc.a.run.app"
BGG_MODEL_NAMES = {
    "hurdle_model_name": "hurdle-v2026",
    "complexity_model_name": "complexity-v2026",
    "rating_model_name": "rating-v2026",
    "users_rated_model_name": "users_rated-v2026",
    "geek_rating_model_name": "geek_rating-v2026",
}
BGG_PREDICTION_COLS = [
    "predicted_hurdle_prob",
    "predicted_complexity",
    "predicted_rating",
    "predicted_users_rated",
    "predicted_geek_rating",
]


def fetch_bgg_predictions(game_ids: List[int]) -> pl.DataFrame:
    """Call /predict_games on the deployed scoring service for the given
    game_ids. Returns a polars DataFrame with game_id + BGG_PREDICTION_COLS."""
    body = {
        **BGG_MODEL_NAMES,
        "game_ids": [int(g) for g in game_ids],
        "upload_to_data_warehouse": False,
    }
    resp = requests.post(
        f"{SCORING_SERVICE_URL}/predict_games", json=body, timeout=300
    )
    resp.raise_for_status()
    payload = resp.json()
    preds = payload.get("predictions") or []
    if not preds:
        return pl.DataFrame({"game_id": []})
    df = pl.from_dicts(preds)
    keep = ["game_id"] + [c for c in BGG_PREDICTION_COLS if c in df.columns]
    return df.select(keep)


def fetch_preview_game_ids(preview_id: int = PREVIEW_ID) -> List[Tuple[int, str]]:
    """Page through the geek preview feed; return [(game_id, name), ...]."""
    out: List[Tuple[int, str]] = []
    seen: set[int] = set()
    page = 1
    while True:
        resp = requests.get(
            PREVIEW_API,
            params={"nosession": 1, "pageid": page, "previewid": preview_id},
            timeout=30,
        )
        resp.raise_for_status()
        items = resp.json()
        if not items:
            break
        added = 0
        for item in items:
            try:
                gid = int(item["objectid"])
            except (KeyError, TypeError, ValueError):
                continue
            if gid in seen:
                continue
            seen.add(gid)
            name = (
                item.get("geekitem", {}).get("item", {}).get("name")
                or item.get("geekitem", {}).get("item", {}).get("href", "")
                or ""
            )
            if isinstance(name, dict):
                name = name.get("primary") or ""
            out.append((gid, str(name)))
            added += 1
        print(f"  page {page}: {added} new ({len(out)} total)")
        page += 1
        time.sleep(0.25)
    return out


def find_finalized_models_for_user(
    storage: CollectionArtifactStorage, outcome: str
) -> Tuple[int, List[Tuple[str, int]]]:
    """Return (latest_splits_version, [(candidate, version), ...]) for all
    candidate runs at that splits version that have a finalized.pkl."""
    latest_splits = storage.latest_canonical_splits_version(outcome)
    if latest_splits is None:
        raise SystemExit(f"No splits found for outcome={outcome!r}")

    results: List[Tuple[str, int]] = []
    for candidate in storage.list_candidates(outcome):
        for v in storage._list_candidate_versions(outcome, candidate):
            reg = storage.load_candidate_registration(outcome, candidate, version=v)
            if reg is None or reg.get("splits_version") != latest_splits:
                continue
            finalized_path = (
                storage.base_dir / outcome / candidate / f"v{v}" / "finalized.pkl"
            )
            if finalized_path.exists():
                results.append((candidate, v))
    return latest_splits, results


def score_with_pipeline(pipeline, df: pl.DataFrame, task: str) -> pl.Series:
    X = df.drop("label") if "label" in df.columns else df
    X_pd = X.to_pandas()
    if task == "classification":
        proba = pipeline.predict_proba(X_pd)[:, 1]
        return pl.Series(proba)
    preds = pipeline.predict(X_pd)
    return pl.Series(preds)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--username", required=True)
    p.add_argument("--outcome", default="own")
    p.add_argument("--environment", default="dev")
    p.add_argument("--local-root", default="models/collections")
    p.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: scratch/uk_games_expo_2026_{user}_{outcome}.csv)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="If set, write only the top-N rows ranked by mean score across models.",
    )
    p.add_argument(
        "--format",
        choices=("csv", "xlsx"),
        default="xlsx",
        help="Output format. xlsx adds a heatmap-style color scale to score columns.",
    )
    p.add_argument(
        "--no-bgg-predictions",
        action="store_true",
        help="Skip the call to the BGG scoring service for universe-level predictions.",
    )
    args = p.parse_args(argv)

    project_config = load_config()
    bq_config = project_config.get_data_warehouse_config()

    storage = CollectionArtifactStorage(
        args.username,
        local_root=args.local_root,
        environment=args.environment,
    )

    splits_v, finalized = find_finalized_models_for_user(storage, args.outcome)
    if not finalized:
        raise SystemExit(
            f"No finalized models found at splits v{splits_v} for "
            f"user={args.username!r} outcome={args.outcome!r}."
        )
    print(f"Using splits v{splits_v}; finalized candidates:")
    for cand, ver in finalized:
        print(f"  - {cand} v{ver}")

    print("\nFetching UK Games Expo 2026 preview feed...")
    preview = fetch_preview_game_ids(PREVIEW_ID)
    print(f"Got {len(preview)} unique games from preview.")
    if not preview:
        raise SystemExit("Preview feed returned no games.")

    game_ids = [gid for gid, _ in preview]

    print("\nLoading features from BigQuery...")
    where = f"game_id IN ({','.join(str(g) for g in game_ids)})"
    universe = BGGDataLoader(bq_config).load_features(
        use_predicted_complexity=True,
        use_embeddings=False,
        where_clause=where,
    )
    print(f"Loaded {universe.height} rows from features (of {len(game_ids)} requested)")

    missing = set(game_ids) - set(universe["game_id"].to_list())
    if missing:
        print(f"  {len(missing)} game_ids not found in warehouse (likely too new/no features)")

    score_cols: Dict[str, pl.Series] = {}
    for cand, ver in finalized:
        print(f"\nScoring with {cand} v{ver}...")
        pipeline = storage.load_finalized_pipeline(args.outcome, cand, version=ver)
        reg = storage.load_candidate_registration(args.outcome, cand, version=ver) or {}
        task = reg.get("task", "classification")
        scores = score_with_pipeline(pipeline, universe, task)
        col_name = f"{cand}_v{ver}"
        score_cols[col_name] = scores
        print(f"  {col_name}: min={scores.min():.4f} max={scores.max():.4f} mean={scores.mean():.4f}")

    scored = universe.select(["game_id", "name", "year_published"]).with_columns(
        [s.alias(name) for name, s in score_cols.items()]
    )
    if scored["year_published"].dtype != pl.Int64:
        scored = scored.with_columns(pl.col("year_published").cast(pl.Int64, strict=False))

    bgg_pred_cols: List[str] = []
    if not args.no_bgg_predictions:
        print("\nCalling BGG scoring service for universe-level predictions...")
        bgg_preds = fetch_bgg_predictions(game_ids)
        print(f"  got predictions for {bgg_preds.height} games")
        if bgg_preds.height > 0:
            scored = scored.join(bgg_preds, on="game_id", how="left")
            bgg_pred_cols = [c for c in BGG_PREDICTION_COLS if c in scored.columns]

    score_col_names = list(score_cols.keys())
    if len(score_col_names) > 1:
        scored = scored.with_columns(
            pl.mean_horizontal([pl.col(c) for c in score_col_names]).alias("mean_score")
        )
        sort_col = "mean_score"
    else:
        sort_col = score_col_names[0]
    scored = scored.sort(sort_col, descending=True)

    final_order = ["game_id", "name", "year_published"]
    final_order += [c for c in score_col_names if c in scored.columns]
    if "mean_score" in scored.columns:
        final_order.append("mean_score")
    final_order += [c for c in BGG_PREDICTION_COLS if c in scored.columns]
    final_order += [c for c in scored.columns if c not in final_order]
    scored = scored.select(final_order)

    if args.top_n is not None:
        scored = scored.head(args.top_n)

    round_cols = score_col_names + (["mean_score"] if "mean_score" in scored.columns else [])
    bgg_round_cols = [c for c in bgg_pred_cols if c != "predicted_users_rated"]
    bgg_int_cols = [c for c in bgg_pred_cols if c == "predicted_users_rated"]
    scored = scored.with_columns(
        [pl.col(c).round(3) for c in round_cols + bgg_round_cols]
        + [pl.col(c).round(0).cast(pl.Int64, strict=False) for c in bgg_int_cols]
    )
    heatmap_score_cols = round_cols + bgg_pred_cols

    ext = ".xlsx" if args.format == "xlsx" else ".csv"
    out_path = (
        Path(args.output)
        if args.output
        else Path("scratch") / f"uk_games_expo_2026_{args.username}_{args.outcome}{ext}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "csv":
        scored.write_csv(out_path)
    else:
        column_formats: Dict[str, str] = {col: "0.000" for col in round_cols + bgg_round_cols}
        for col in bgg_int_cols:
            column_formats[col] = "#,##0"
        if "year_published" in scored.columns:
            column_formats["year_published"] = "0"
        scored.write_excel(
            workbook=str(out_path),
            worksheet=f"{args.username}_{args.outcome}",
            conditional_formats={
                col: {
                    "type": "2_color_scale",
                    "min_color": "#ffffff",
                    "max_color": "#08519c",
                }
                for col in heatmap_score_cols
            },
            column_formats=column_formats,
            autofit=True,
            freeze_panes="A2",
        )
    print(f"\nWrote {scored.height} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
