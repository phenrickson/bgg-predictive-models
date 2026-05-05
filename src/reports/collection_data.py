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


def _list_candidate_versions(user_outcome_dir: Path, candidate: str) -> list[int]:
    """Return all integer versions for a candidate dir, ascending."""
    cand_dir = user_outcome_dir / candidate
    if not cand_dir.exists():
        return []
    versions: list[int] = []
    for child in cand_dir.iterdir():
        if not (child.is_dir() and child.name.startswith("v")):
            continue
        try:
            versions.append(int(child.name[1:]))
        except ValueError:
            continue
    return sorted(versions)


def _is_finalized(user_outcome_dir: Path, candidate: str, version: int) -> bool:
    """A candidate version is 'finalized' iff it has finalized.pkl."""
    return (user_outcome_dir / candidate / f"v{version}" / "finalized.pkl").exists()


def _list_finalized_candidates(user_outcome_dir: Path) -> list[tuple[str, int]]:
    """Return (candidate, latest_finalized_version) for every candidate in the
    outcome dir that has at least one finalized version."""
    if not user_outcome_dir.exists():
        return []
    out: list[tuple[str, int]] = []
    for child in user_outcome_dir.iterdir():
        if not child.is_dir() or child.name.startswith("_") or child.name.startswith("v"):
            continue
        cand = child.name
        finalized_versions = [
            v
            for v in _list_candidate_versions(user_outcome_dir, cand)
            if _is_finalized(user_outcome_dir, cand, v)
        ]
        if finalized_versions:
            out.append((cand, max(finalized_versions)))
    return sorted(out)


def select_candidate(
    root: Path,
    username: str,
    outcome: str,
    candidate: str | None = None,
) -> tuple[str, int]:
    """Pick a (candidate, version) for a user/outcome.

    Resolution order:
        1. If `candidate` is given and has a finalized version, use its
           latest finalized version.
        2. Otherwise prefer DEFAULT_CANDIDATE if it has a finalized version.
        3. Otherwise pick any finalized candidate (alphabetically first).
        4. Raise ValueError if nothing is finalized.
    """
    user_outcome_dir = Path(root) / username / outcome
    finalized = dict(_list_finalized_candidates(user_outcome_dir))

    if candidate is not None:
        if candidate not in finalized:
            raise ValueError(
                f"Candidate {candidate!r} is not finalized for "
                f"{username}/{outcome} under {root}"
            )
        return candidate, finalized[candidate]

    if DEFAULT_CANDIDATE in finalized:
        return DEFAULT_CANDIDATE, finalized[DEFAULT_CANDIDATE]

    if not finalized:
        raise ValueError(
            f"No finalized candidate found for {username}/{outcome} under {root}"
        )

    cand = sorted(finalized.keys())[0]
    return cand, finalized[cand]


import json as _json
import pickle as _pickle

import fsspec


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
    sql = f"""
    WITH ranked AS (
        SELECT
            game_id,
            predicted_prob,
            predicted_label,
            score_ts,
            model_version,
            ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY score_ts DESC) AS rn
        FROM `{table}`
        WHERE username = {username!r} AND outcome = {outcome!r}
    )
    SELECT game_id, predicted_prob, predicted_label, score_ts, model_version
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
    introspection; no BQ."""
    if Path(source).exists() and not source.startswith("gs://"):
        candidate, version = select_candidate(
            Path(source), username, outcome, candidate=candidate_override
        )
    else:
        if candidate_override is None:
            raise NotImplementedError(
                "GCS source requires an explicit candidate override until "
                "GCS-aware candidate selection is implemented."
            )
        candidate = candidate_override
        version = 1

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
