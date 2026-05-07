"""Synthetic data for fast styling iteration on the collection report.

`build_fake_report_data()` returns a `CollectionReportData` that's
schema-faithful but cheap to materialize — no BQ, no pickle reads, no
artifact directory walks. The qmd consumes this exactly the same way
it consumes `load()` output, so the styling sandbox exercises the
same code paths as the real report.

The numbers are not statistically meaningful — they're shaped to make
each visual element render with believable variety (a separation plot
that actually separates, a feature-importance bar chart with both
positive and negative bars across all feature families, etc.).
"""

from __future__ import annotations

import random
from typing import Sequence

import polars as pl

from src.reports.collection_data import CollectionReportData, OutcomeArtifacts

_DEFAULT_SEED = 42

# Plausible game-name fragments for synthetic titles. Mixed lengths so
# the truncation logic in tables/plots gets exercised.
_NAME_PREFIXES = [
    "Brass", "Wingspan", "Terraforming", "Azul", "Castles of",
    "Twilight", "Through the", "Underwater", "Crystal", "Forest of",
    "Ancient", "Skyline", "Voyage of", "Empire of", "Lost",
    "Frozen", "Burning", "Quest for", "Tales of", "Throne of",
]
_NAME_SUFFIXES = [
    "Birmingham", "Mars", "Burgundy", "Cities", "Trails",
    "Imperium", "Ages", "Rivers", "Stars", "Ruins",
    "Kingdoms", "Citadels", "Discovery", "Sand", "Light",
    "Skies", "Tides", "Stone", "Iron", "Rome",
]

_FEATURES_BY_GROUP: dict[str, list[str]] = {
    "category_": [
        "strategy_games", "family_games", "thematic", "abstract",
        "wargames", "party_games", "economic", "puzzle", "fantasy",
        "science_fiction", "historical", "mythology",
    ],
    "mechanic_": [
        "deck_building", "worker_placement", "area_control", "auction",
        "drafting", "tile_placement", "dice_rolling", "engine_building",
        "set_collection", "hand_management", "cooperative", "trick_taking",
    ],
    "designer_": [
        "uwe_rosenberg", "stefan_feld", "vital_lacerda", "reiner_knizia",
        "vlaada_chvatil", "martin_wallace", "phil_walker_harding",
        "elizabeth_hargrave", "matt_leacock", "alexander_pfister",
    ],
    "artist_": [
        "ian_obrien", "kwanchai_moriya", "vincent_dutrait",
        "naiade", "mihajlo_dimitrievski", "beth_sobel", "atha_kanaani",
    ],
    "publisher_": [
        "stonemaier", "z_man_games", "fantasy_flight", "asmodee",
        "rio_grande", "alea", "queen_games", "ravensburger",
    ],
    "family_": [
        "campaign_games", "legacy", "solo_playable", "expansion",
        "two_player_only", "deluxe_edition",
    ],
    "player_count_": [
        "1", "2", "3", "4", "5", "6",
    ],
    "missingindicator_": [
        "min_age", "average_weight", "playing_time", "year_published",
    ],
}


def _gen_name(rng: random.Random) -> str:
    return f"{rng.choice(_NAME_PREFIXES)} {rng.choice(_NAME_SUFFIXES)}"


def _build_games(rng: random.Random, n: int) -> pl.DataFrame:
    rows = []
    for gid in range(1, n + 1):
        year = rng.randint(2000, 2026)
        rows.append(
            {
                "game_id": gid,
                "name": _gen_name(rng),
                "year_published": year,
                "min_players": rng.choice([1, 2]),
                "max_players": rng.choice([2, 4, 5, 6]),
                "min_playtime": rng.choice([20, 30, 45, 60]),
                "max_playtime": rng.choice([60, 90, 120, 180]),
                "average_weight": round(rng.uniform(1.0, 4.5), 2),
                "users_rated": rng.randint(50, 50_000),
                "image": f"https://placehold.co/200x200/0f172a/e2e8f0?text=Game+{gid}",
                "description": (
                    "A satisfyingly thematic game of moderate complexity. "
                    "Features include resource management, route building, "
                    "and a satisfying late-game climax."
                ),
            }
        )
    return pl.DataFrame(rows)


def _build_collection(
    rng: random.Random, games: pl.DataFrame, n_owned: int = 80
) -> pl.DataFrame:
    """Pick n_owned games from `games` and tag them as owned, plus a
    handful in wishlist/preordered/want for status-column variety."""
    all_ids = games["game_id"].to_list()
    rng.shuffle(all_ids)
    owned_ids = all_ids[:n_owned]
    wishlist_ids = all_ids[n_owned : n_owned + 8]
    preordered_ids = all_ids[n_owned + 8 : n_owned + 11]
    want_ids = all_ids[n_owned + 11 : n_owned + 16]
    prev_owned_ids = all_ids[n_owned + 16 : n_owned + 22]

    rows = []
    for gid in owned_ids:
        rows.append(
            {
                "game_id": gid,
                "owned": True,
                "wishlist": False,
                "preordered": False,
                "want": False,
                "previously_owned": False,
                "user_rating": round(rng.uniform(5.0, 9.5), 1),
            }
        )
    for gid in wishlist_ids:
        rows.append(_collection_row(gid, wishlist=True))
    for gid in preordered_ids:
        rows.append(_collection_row(gid, preordered=True))
    for gid in want_ids:
        rows.append(_collection_row(gid, want=True))
    for gid in prev_owned_ids:
        rows.append(_collection_row(gid, previously_owned=True))
    return pl.DataFrame(rows)


def _collection_row(
    gid: int,
    *,
    owned: bool = False,
    wishlist: bool = False,
    preordered: bool = False,
    want: bool = False,
    previously_owned: bool = False,
) -> dict:
    return {
        "game_id": gid,
        "owned": owned,
        "wishlist": wishlist,
        "preordered": preordered,
        "want": want,
        "previously_owned": previously_owned,
        "user_rating": None,
    }


def _build_feature_importance(rng: random.Random) -> pl.DataFrame:
    """Generate coefficients across every feature family — both pos and neg —
    so the importance + partial-effects tabset have content for each tab."""
    rows = []
    for prefix, names in _FEATURES_BY_GROUP.items():
        group_label = {
            "category_": "Categories",
            "mechanic_": "Mechanics",
            "designer_": "Designers",
            "artist_": "Artists",
            "publisher_": "Publishers",
            "family_": "Families",
            "player_count_": "Players",
            "missingindicator_": "Missingness",
        }[prefix]
        for name in names:
            value = rng.uniform(-1.5, 1.8)
            rows.append(
                {
                    "feature": f"{prefix}{name}",
                    "value": value,
                    "group": group_label,
                }
            )
    return pl.DataFrame(rows)


def _build_predictions(
    rng: random.Random,
    games: pl.DataFrame,
    owned_ids: set[int],
    n: int,
) -> pl.DataFrame:
    """Build a predictions frame: every game gets a proba; owned games
    skew higher so the separation plot actually separates."""
    sample = games.sample(n=min(n, games.height), with_replacement=False, seed=rng.randint(0, 1_000_000))
    rows = []
    for r in sample.iter_rows(named=True):
        is_owned = r["game_id"] in owned_ids
        if is_owned:
            proba = rng.betavariate(6, 2)
        else:
            proba = rng.betavariate(1.5, 5)
        rows.append(
            {
                "game_id": r["game_id"],
                "name": r["name"],
                "year_published": r["year_published"],
                "proba": round(proba, 4),
                "label": is_owned,
            }
        )
    return pl.DataFrame(rows)


def _build_upcoming(
    rng: random.Random,
    games: pl.DataFrame,
    finalize_through: int,
    n: int = 40,
) -> pl.DataFrame:
    """Build upcoming-predictions frame: games strictly after finalize_through."""
    upcoming = games.filter(pl.col("year_published") > finalize_through)
    if upcoming.height == 0:
        # Manufacture a few future games if our random seed didn't give any.
        rows = []
        for i in range(n):
            rows.append(
                {
                    "game_id": 100_000 + i,
                    "predicted_prob": round(rng.betavariate(2, 2), 4),
                }
            )
        return pl.DataFrame(rows)
    sample = upcoming.sample(
        n=min(n, upcoming.height),
        with_replacement=False,
        seed=rng.randint(0, 1_000_000),
    )
    rows = []
    for r in sample.iter_rows(named=True):
        rows.append(
            {
                "game_id": r["game_id"],
                "predicted_prob": round(rng.betavariate(2, 2), 4),
            }
        )
    return pl.DataFrame(rows)


def build_fake_report_data(
    *,
    username: str = "fixture_user",
    outcome: str = "own",
    seed: int = _DEFAULT_SEED,
    n_games: int = 600,
    n_owned: int = 80,
    finalize_through: int = 2024,
) -> CollectionReportData:
    """Construct a synthetic CollectionReportData for styling work.

    All frames have the same columns the real loader produces so the
    qmd renders without needing a special path. Pass a different `seed`
    to get different fake data while still being deterministic per-run.
    """
    rng = random.Random(seed)

    games = _build_games(rng, n_games)
    collection = _build_collection(rng, games, n_owned=n_owned)
    owned_ids = set(
        collection.filter(pl.col("owned"))["game_id"].to_list()
    )

    feature_importance = _build_feature_importance(rng)

    # Three eval splits; smaller for val/test like the real ones.
    oof_pred = _build_predictions(rng, games, owned_ids, n=500)
    val_pred = _build_predictions(rng, games, owned_ids, n=200)
    test_pred = _build_predictions(rng, games, owned_ids, n=200)

    upcoming_pred = _build_upcoming(rng, games, finalize_through=finalize_through)

    registration = {
        "username": username,
        "task": "binary_classification",
        "tuning_strategy": "halving_grid_search",
        "splits_version": "v20260101",
        "finalize_through": finalize_through,
        "threshold": 0.42,
        "n_train_used": 45_000,
        "n_val": 9_500,
        "n_test": 6_300,
        "best_params": {"C": 0.1, "penalty": "l2"},
        "trained_at": "2026-04-15T14:30:21.444",
        "finalized_at": "2026-04-16T08:12:03.111",
        "git_sha": "abc1234567890def",
        "metrics": {"roc_auc": 0.8912, "pr_auc": 0.4231, "f1": 0.4612},
        "oof_metrics": {"overall": {"roc_auc": 0.8741}},
    }

    artifacts = OutcomeArtifacts(
        outcome=outcome,
        selected_candidate="logistic_row_norm",
        selected_version=3,
        pipeline=None,
        registration=registration,
        threshold=registration["threshold"],
        feature_importance=feature_importance,
        oof_predictions=oof_pred,
        val_predictions=val_pred,
        test_predictions=test_pred,
        upcoming_predictions=upcoming_pred,
    )

    return CollectionReportData(
        username=username,
        collection=collection,
        games=games,
        outcomes={outcome: artifacts},
    )
