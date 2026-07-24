import polars as pl

from src.reports.game_cards import complexity_chip, game_cards_html, player_badges


def _games():
    return pl.DataFrame(
        {
            "game_id": [1, 2],
            "name": ["Alpha", "Beta"],
            "year_published": [2020, 2021],
            "image": ["http://x/1.png", None],
            "description": ["A farming game.", "War."],
            "average_weight": [3.1, 1.5],
            "min_playtime": [60, 30],
            "max_playtime": [120, 30],
            "best_player_counts": ["4, 3", "2"],
            "recommended_player_counts": ["2, 3, 4", "1, 2"],
        }
    )


def test_player_badges_best_filled_rec_outlined():
    html = player_badges("4, 3", "2, 3, 4")
    assert html.count("pc-best") == 2   # 3 and 4 are best
    assert html.count("pc-rec") == 1    # 2 is recommended-only
    # numeric order
    assert html.index(">2<") < html.index(">3<") < html.index(">4<")


def test_complexity_chip_empty():
    assert "badge-none" in complexity_chip(None)
    assert "weight-badge" in complexity_chip(2.5)


def test_cards_order_and_filter():
    html = game_cards_html(_games(), [2, 1, 99], proba={1: 0.83})
    assert html.index("Beta") < html.index("Alpha")   # order preserved
    assert "boardgamegeek.com/boardgame/1" in html
    assert "0.830" in html                              # Pr(Yes) for game 1
    assert "99" not in html.split("card-grid")[1][:5]   # absent id skipped


def test_cards_empty():
    assert game_cards_html(_games(), []) == ""
