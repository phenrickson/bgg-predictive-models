import pytest

from src.reports.selections import Selections, dump_selections, load_selections

_YAML = """
name: rocky_bilbao_2026
username: phenrickson
criteria: {players: [5]}
games:
  1: {selection: lock, status: yes}
  2: {selection: maybe, status: yes}
  3: {selection: maybe, status: no}
  4: {selection: no, status: no}
"""


def _write(tmp_path, text):
    p = tmp_path / "sel.yaml"
    p.write_text(text)
    return p


def test_sections(tmp_path):
    s = load_selections(_write(tmp_path, _YAML))
    assert s.locks() == [1]
    assert s.maybes() == [2, 3]
    assert s.others() == [4]


def test_menu_is_status_yes(tmp_path):
    s = load_selections(_write(tmp_path, _YAML))
    # menu is status==yes regardless of selection: lock#1 + maybe#2
    assert s.menu() == [1, 2]


def test_bad_label_raises(tmp_path):
    bad = "games:\n  1: {selection: perhaps, status: yes}\n"
    with pytest.raises(ValueError):
        load_selections(_write(tmp_path, bad))


def test_roundtrip(tmp_path):
    s = load_selections(_write(tmp_path, _YAML))
    out = tmp_path / "out.yaml"
    dump_selections(s, out)
    s2 = load_selections(out)
    assert s2.games == s.games
    assert s2.name == "rocky_bilbao_2026"


def test_empty(tmp_path):
    s = load_selections(_write(tmp_path, "name: x\nusername: y\n"))
    assert s.menu() == [] and s.locks() == []
