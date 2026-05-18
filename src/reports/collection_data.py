"""Data loader for the collection Quarto report.

Returns a `CollectionReportData` aggregate that the `.qmd` template
consumes. Splits user-level (outcome-agnostic) data from per-outcome
artifacts so multi-outcome reports can iterate `data.outcomes` without
a loader refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass
class OutcomeArtifacts:
    """All artifacts the report needs for one (user, outcome)."""

    outcome: str
    selected_candidate: str
    selected_version: int

    pipeline: Any
    registration: dict
    threshold: float | None
    feature_importance: pl.DataFrame

    oof_predictions: pl.DataFrame
    val_predictions: pl.DataFrame
    test_predictions: pl.DataFrame

    upcoming_predictions: pl.DataFrame


@dataclass
class CollectionReportData:
    """Full data bundle for a single rendering of the collection report."""

    username: str
    collection: pl.DataFrame
    games: pl.DataFrame
    outcomes: dict[str, OutcomeArtifacts]


from pathlib import Path

DEFAULT_CANDIDATE = "logistic_row_norm"


class MissingArtifactsError(FileNotFoundError):
    """Raised when no finalized model artifacts exist for a (user, outcome).

    Inherits from FileNotFoundError so callers that already handle
    "missing data" via that base class still work, but the named type
    lets the CLI driver render a friendly message instead of a generic
    traceback.
    """


import json as _json
import pickle as _pickle

import fsspec


def _fs_for(uri: str):
    """Return (fs, path-without-protocol) for a local path or gs:// URI."""
    if uri.startswith("gs://"):
        return fsspec.filesystem("gs"), uri.removeprefix("gs://").rstrip("/")
    return fsspec.filesystem("file"), str(uri).rstrip("/")


def _ls_dirs(uri: str) -> list[str]:
    """Return the basenames of immediate subdirectories of `uri`. Empty list
    if `uri` doesn't exist or has no subdirectories."""
    fs, path = _fs_for(uri)
    if not fs.exists(path):
        return []
    return sorted(Path(p).name for p in fs.ls(path) if fs.isdir(p))


def _file_exists(uri: str) -> bool:
    fs, path = _fs_for(uri)
    return fs.exists(path)


def _list_candidate_versions(user_outcome_dir: str, candidate: str) -> list[int]:
    """Return all integer versions for a candidate dir, ascending."""
    cand_dir = f"{user_outcome_dir}/{candidate}"
    versions: list[int] = []
    for name in _ls_dirs(cand_dir):
        if not name.startswith("v"):
            continue
        try:
            versions.append(int(name[1:]))
        except ValueError:
            continue
    return sorted(versions)


def _is_finalized(user_outcome_dir: str, candidate: str, version: int) -> bool:
    """A candidate version is 'finalized' iff it has finalized.pkl."""
    return _file_exists(f"{user_outcome_dir}/{candidate}/v{version}/finalized.pkl")


def _list_finalized_candidates(user_outcome_dir: str) -> list[tuple[str, int]]:
    """Return (candidate, latest_finalized_version) for every candidate in the
    outcome dir that has at least one finalized version."""
    out: list[tuple[str, int]] = []
    for name in _ls_dirs(user_outcome_dir):
        if name.startswith("_") or name.startswith("v"):
            continue
        finalized_versions = [
            v
            for v in _list_candidate_versions(user_outcome_dir, name)
            if _is_finalized(user_outcome_dir, name, v)
        ]
        if finalized_versions:
            out.append((name, max(finalized_versions)))
    return sorted(out)


def select_candidate(
    root: str | Path,
    username: str,
    outcome: str,
    candidate: str | None = None,
) -> tuple[str, int]:
    """Pick a (candidate, version) for a user/outcome.

    `root` is a source string — either a local path or a ``gs://`` URI.
    Both are resolved via fsspec so the same logic works locally and
    against cloud storage.

    Resolution order:
        1. If `candidate` is given and has a finalized version, use its
           latest finalized version.
        2. Otherwise prefer DEFAULT_CANDIDATE if it has a finalized version.
        3. Otherwise pick any finalized candidate (alphabetically first).
        4. Raise MissingArtifactsError if nothing is finalized.
    """
    root_str = str(root).rstrip("/")
    user_dir = f"{root_str}/{username}"
    user_outcome_dir = f"{user_dir}/{outcome}"

    if not _file_exists(user_dir):
        raise MissingArtifactsError(
            f"No artifacts found for user {username!r} at {user_dir}. "
            f"Train a model first (`just sweep` or `just train`), or check "
            f"that the username spelling matches the directory name."
        )
    if not _file_exists(user_outcome_dir):
        available_outcomes = _ls_dirs(user_dir)
        hint = (
            f" Available outcomes: {', '.join(available_outcomes)}."
            if available_outcomes
            else ""
        )
        raise MissingArtifactsError(
            f"No artifacts for outcome {outcome!r} under {username!r} at "
            f"{user_outcome_dir}.{hint}"
        )

    finalized = dict(_list_finalized_candidates(user_outcome_dir))

    if candidate is not None:
        if candidate not in finalized:
            available = sorted(finalized.keys())
            hint = (
                f" Finalized candidates: {', '.join(available)}."
                if available
                else " No candidates are finalized yet — run `just finalize` first."
            )
            raise MissingArtifactsError(
                f"Candidate {candidate!r} is not finalized for "
                f"{username}/{outcome}.{hint}"
            )
        return candidate, finalized[candidate]

    if DEFAULT_CANDIDATE in finalized:
        return DEFAULT_CANDIDATE, finalized[DEFAULT_CANDIDATE]

    if not finalized:
        raise MissingArtifactsError(
            f"No finalized candidate for {username}/{outcome} under {root_str}. "
            f"Run `just finalize` (or `just finalize-all`) before rendering."
        )

    cand = sorted(finalized.keys())[0]
    return cand, finalized[cand]


def _read_bytes(uri: str) -> bytes:
    """Read raw bytes from a local path or gs:// URI."""
    with fsspec.open(uri, "rb") as f:
        return f.read()


def _read_text(uri: str) -> str:
    with fsspec.open(uri, "rt") as f:
        return f.read()


def _read_json(uri: str) -> dict:
    return _json.loads(_read_text(uri))


def _read_pickle(uri: str):
    return _pickle.loads(_read_bytes(uri))


def _read_parquet(uri: str) -> pl.DataFrame:
    """Polars reads gs:// natively; for local paths just pass through."""
    return pl.read_parquet(uri)


from src.collection.viz import extract_finalized_importance


def _outcome_root(source: str, username: str, outcome: str) -> str:
    """Compose the outcome-level URI/path. Trailing slash safe."""
    base = source.rstrip("/")
    return f"{base}/{username}/{outcome}"


def _candidate_root(
    source: str, username: str, outcome: str, candidate: str, version: int
) -> str:
    return f"{_outcome_root(source, username, outcome)}/{candidate}/v{version}"


def _splits_root(source: str, username: str, outcome: str, version: int) -> str:
    return f"{_outcome_root(source, username, outcome)}/_splits/v{version}"


from src.collection.collection_storage import CollectionStorage
from src.utils.config import load_config


def _bq_client():
    """Lazy BigQuery client. Patchable in tests."""
    from google.cloud import bigquery

    return bigquery.Client()


def empty_offline_frame(kind: str) -> "pl.DataFrame":
    """Schema-correct empty frames for offline/test rendering.

    Bare ``pl.DataFrame()`` has zero columns, so the report's
    join/filter cells (``join(..., on="game_id")``,
    ``filter(pl.col("year_published") ...)``) raise ColumnNotFoundError.
    These stubs carry the columns the report actually reads so those
    cells render empty instead of crashing. Mirrors the real fetcher
    output schemas; not exhaustive — only the columns the offline
    render path touches.

    Columns per kind (and why each is needed):

    - ``"games"``: joined into predictions in the predictions-upcoming /
      predictions-older / top-games-training cells and read by the
      table/viz helpers.
        * ``game_id``  — every ``join(..., on="game_id")``.
        * ``name``     — ``data.games.select(["game_id","name","year_published"])``
          in top-games-training-prep; read by ``format_*`` table builders.
        * ``year_published`` — ``filter(pl.col("year_published") >
          finalize_through)`` (predictions-upcoming) and
          ``build_topn_by_year_html`` casts/filters it.
        * ``image`` / ``description`` — ``format_predictions_with_images``.
        * ``users_rated`` — predictions-older does
          ``if "users_rated" in older.columns: older.filter(...)``.
    - ``"upcoming"``: mirrors ``_fetch_upcoming_predictions`` output.
        * ``game_id`` — join key into games.
        * ``predicted_prob`` — ``sort("predicted_prob")`` (upcoming) and
          ``format_predictions_with_images`` reads it as Pr(Yes).
        * ``predicted_label`` / ``score_ts`` / ``model_version`` — round
          out the real query's SELECT list. ``score_ts`` is Datetime to
          match ``pl.from_pandas`` of a BQ TIMESTAMP.
        * ``is_new_7d`` — Boolean; whether the game first appeared in
          this user's predictions within the last 7 days. Drives the
          row highlight in the New & Upcoming table.
    - ``"collection"``: mirrors ``_fetch_collection_snapshot`` output.
        * ``game_id`` — ``build_status_lookup`` keys on it; the
          by-year/by-category plots ``.select("game_id")``.
        * ``owned`` (+ ``preordered``/``wishlist``/``want``/
          ``want_to_buy``/``previously_owned``/``prev_owned``/
          ``user_rating``) — ``build_status_lookup`` /
          ``format_collection_table`` / the collection plots read these
          (status booleans and the filter ``pl.col("owned") == True``).
    """
    if kind == "games":
        return pl.DataFrame(
            schema={
                "game_id": pl.Int64,
                "name": pl.Utf8,
                "year_published": pl.Int64,
                "image": pl.Utf8,
                "description": pl.Utf8,
                "users_rated": pl.Int64,
            }
        )
    if kind == "upcoming":
        return pl.DataFrame(
            schema={
                "game_id": pl.Int64,
                "predicted_prob": pl.Float64,
                "predicted_label": pl.Int64,
                "score_ts": pl.Datetime,
                "model_version": pl.Utf8,
                "is_new_7d": pl.Boolean,
            }
        )
    if kind == "collection":
        return pl.DataFrame(
            schema={
                "game_id": pl.Int64,
                "owned": pl.Boolean,
                "preordered": pl.Boolean,
                "wishlist": pl.Boolean,
                "want": pl.Boolean,
                "want_to_buy": pl.Boolean,
                "previously_owned": pl.Boolean,
                "prev_owned": pl.Boolean,
                "user_rating": pl.Float64,
            }
        )
    raise ValueError(f"Unknown offline frame kind: {kind!r}")


def _fetch_collection_snapshot(username: str) -> pl.DataFrame:
    """Latest BGG collection snapshot for the user."""
    storage = CollectionStorage(environment="dev")
    df = storage.get_latest_collection(username)
    return df if df is not None else pl.DataFrame()


def _fetch_games_metadata() -> pl.DataFrame:
    """Game metadata for joining into predictions tables."""
    from src.data.loader import BGGDataLoader

    bq_config = load_config().get_bigquery_config()
    loader = BGGDataLoader(bq_config)
    return loader.load_features(use_predicted_complexity=True, use_embeddings=False)


def _fetch_upcoming_predictions(username: str, outcome: str) -> pl.DataFrame:
    """Latest deployed-model predictions for the user from
    raw.collection_predictions_landing. Keeps only the most recent
    score per (game_id) — the table is append-only.
    """
    table = load_config().get_collection_landing_table()
    # `is_new_7d` mirrors bgg-dash-viewer's NEW signal: a game is "new"
    # if its first appearance in this user's collection predictions was
    # within the last 7 days. The landing table is append-only, so
    # MIN(score_ts) per game_id is that first-seen timestamp (the local
    # equivalent of the dash's predictions.game_first_prediction).
    sql = f"""
    WITH ranked AS (
        SELECT
            game_id,
            predicted_prob,
            predicted_label,
            score_ts,
            model_version,
            MIN(score_ts) OVER (PARTITION BY game_id) AS first_score_ts,
            ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY score_ts DESC) AS rn
        FROM `{table}`
        WHERE username = {username!r} AND outcome = {outcome!r}
    )
    SELECT
        game_id,
        predicted_prob,
        predicted_label,
        score_ts,
        model_version,
        DATE_DIFF(CURRENT_DATE(), DATE(first_score_ts), DAY) <= 7 AS is_new_7d
    FROM ranked
    WHERE rn = 1
    """
    job = _bq_client().query(sql)
    pdf = job.to_dataframe()
    return pl.from_pandas(pdf)


def _load_outcome(
    source: str,
    username: str,
    outcome: str,
    candidate_override: str | None,
) -> OutcomeArtifacts:
    """Load all per-outcome artifacts. Pure filesystem reads + pipeline
    introspection; no BQ. ``source`` may be a local path or a ``gs://``
    URI; select_candidate handles both via fsspec."""
    candidate, version = select_candidate(
        source, username, outcome, candidate=candidate_override
    )

    cand_root = _candidate_root(source, username, outcome, candidate, version)
    registration = _read_json(f"{cand_root}/registration.json")
    threshold_blob = _read_json(f"{cand_root}/threshold.json")
    threshold = threshold_blob.get("threshold")

    pipeline = _read_pickle(f"{cand_root}/finalized.pkl")

    splits_version = registration.get("splits_version", version)
    splits_train = _read_parquet(
        f"{_splits_root(source, username, outcome, splits_version)}/train.parquet"
    )
    feature_importance_pdf = extract_finalized_importance(
        pipeline, splits_train.head(5).to_pandas()
    )
    if feature_importance_pdf is None:
        feature_importance = pl.DataFrame()
    else:
        feature_importance = pl.from_pandas(feature_importance_pdf)

    oof = _read_parquet(f"{cand_root}/predictions/oof.parquet")
    val = _read_parquet(f"{cand_root}/predictions/val.parquet")
    test = _read_parquet(f"{cand_root}/predictions/test.parquet")

    upcoming = _fetch_upcoming_predictions(username, outcome)

    return OutcomeArtifacts(
        outcome=outcome,
        selected_candidate=candidate,
        selected_version=version,
        pipeline=pipeline,
        registration=registration,
        threshold=threshold,
        feature_importance=feature_importance,
        oof_predictions=oof,
        val_predictions=val,
        test_predictions=test,
        upcoming_predictions=upcoming,
    )


def load(
    username: str,
    outcomes: str | list[str] = "own",
    source: str = "local",
    candidates: dict[str, str] | None = None,
) -> CollectionReportData:
    """Load everything the report template needs for a user."""
    outcome_list: list[str] = (
        [outcomes] if isinstance(outcomes, str) else list(outcomes)
    )
    # `local` means the project's standard models/collections tree.
    # Resolve to an absolute path so the loader works from any cwd
    # (the Quarto kernel runs from the qmd's directory, not the project
    # root, so a relative "models/collections" would miss).
    if source == "local":
        resolved_source = str(
            (Path(__file__).resolve().parents[2] / "models" / "collections")
        )
    else:
        resolved_source = source

    overrides = candidates or {}
    out: dict[str, OutcomeArtifacts] = {}
    for outcome in outcome_list:
        out[outcome] = _load_outcome(
            resolved_source,
            username,
            outcome,
            candidate_override=overrides.get(outcome),
        )

    collection = _fetch_collection_snapshot(username)
    games = _fetch_games_metadata()

    return CollectionReportData(
        username=username,
        collection=collection,
        games=games,
        outcomes=out,
    )
