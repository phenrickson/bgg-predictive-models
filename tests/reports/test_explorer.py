import polars as pl

from src.reports.explorer import build_explorer_payload


def _coll():
    return pl.DataFrame(
        {
            "game_id": [1, 2],
            "owned": [True, False],
            "wishlist": [False, True],
            "user_rating": [9.0, None],
        }
    )


def _games():
    return pl.DataFrame(
        {
            "game_id": [1, 2],
            "name": ["Alpha", "Beta"],
            "year_published": [2020, 2021],
            "min_players": [2, 1],
            "max_players": [4, 5],
            "min_playtime": [60, 30],
            "max_playtime": [120, 30],
            "average_weight": [3.1, 1.5],
            "best_player_counts": ["4, 3", "2"],
            "recommended_player_counts": ["2, 3, 4", "1, 2"],
        }
    )


def test_payload_columns_have_kinds():
    p = build_explorer_payload(_coll(), _games())
    kinds = {c["label"]: c["kind"] for c in p["columns"]}
    assert kinds["Status"] == "discrete"
    # Players (supported range) is kept and distinct from the suggestions column
    assert kinds["Players"] == "range-contains"
    # Best + Recommended merged into one badges column
    assert kinds["Player Counts"] == "badges"
    assert "Best" not in kinds and "Recommended" not in kinds
    assert kinds["Complexity"] == "range"
    assert kinds["Game"] == "none"


def test_players_column_preserved():
    p = build_explorer_payload(_coll(), _games())
    labels = [c["label"] for c in p["columns"]]
    pi = labels.index("Players")
    # supported range strings, untouched
    assert p["rows"][0][pi] == "2–4"
    assert p["rows"][1][pi] == "1–5"


def test_badges_cell_shape():
    p = build_explorer_payload(_coll(), _games())
    labels = [c["label"] for c in p["columns"]]
    pc = labels.index("Player Counts")
    # Alpha: best "4, 3" -> [3,4]; rec "2, 3, 4" -> union [2,3,4]
    assert p["rows"][0][pc] == {"best": [3, 4], "rec": [2, 3, 4]}
    # Beta: best "2" -> [2]; rec "1, 2" -> [1,2]
    assert p["rows"][1][pc] == {"best": [2], "rec": [1, 2]}


def test_payload_empty_collection():
    p = build_explorer_payload(pl.DataFrame(), _games())
    assert p["rows"] == []
