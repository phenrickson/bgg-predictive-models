from __future__ import annotations

import src.collection.train_model as tm


def test_runs_stages_in_order(monkeypatch):
    calls = []

    def make(name, rc=0):
        def _main(argv):
            calls.append((name, list(argv)))
            return rc
        return _main

    monkeypatch.setattr(tm, "_split_main", make("split"))
    monkeypatch.setattr(tm, "_train_main", make("train"))
    monkeypatch.setattr(tm, "_finalize_main", make("finalize"))
    monkeypatch.setattr(tm, "_register_main", make("register"))

    rc = tm.main([
        "--username", "Gyges",
        "--outcome", "own",
        "--candidate", "logistic_row_norm",
        "--environment", "dev",
        "--local-root", "models/collections",
    ])

    assert rc == 0
    assert [c[0] for c in calls] == ["split", "train", "finalize", "register"]


def test_aborts_on_failed_stage_and_returns_rc(monkeypatch):
    calls = []

    def make(name, rc=0):
        def _main(argv):
            calls.append(name)
            return rc
        return _main

    monkeypatch.setattr(tm, "_split_main", make("split"))
    monkeypatch.setattr(tm, "_train_main", make("train", rc=3))
    monkeypatch.setattr(tm, "_finalize_main", make("finalize"))
    monkeypatch.setattr(tm, "_register_main", make("register"))

    rc = tm.main(["--username", "Gyges"])

    assert rc == 3
    assert calls == ["split", "train"]


def test_defaults_candidate_and_outcome(monkeypatch):
    seen = {}

    def cap(name):
        def _main(argv):
            seen[name] = list(argv)
            return 0
        return _main

    for n in ("_split_main", "_train_main", "_finalize_main", "_register_main"):
        monkeypatch.setattr(tm, n, cap(n))

    assert tm.main(["--username", "Gyges"]) == 0
    assert "logistic_row_norm" in seen["_train_main"]
    assert "own" in seen["_split_main"]
    assert "logistic_row_norm for Gyges/own" in seen["_register_main"]


def test_finalize_through_passthrough(monkeypatch):
    seen = {}

    def cap(name):
        def _main(argv):
            seen[name] = list(argv)
            return 0
        return _main

    for n in ("_split_main", "_train_main", "_finalize_main", "_register_main"):
        monkeypatch.setattr(tm, n, cap(n))

    assert tm.main(["--username", "Gyges", "--finalize-through", "2023"]) == 0
    fa = seen["_finalize_main"]
    assert "--finalize-through" in fa
    assert fa[fa.index("--finalize-through") + 1] == "2023"
    # not passed to other stages
    assert "--finalize-through" not in seen["_split_main"]
    assert "--finalize-through" not in seen["_train_main"]


def test_explicit_description_override(monkeypatch):
    seen = {}

    def cap(name):
        def _main(argv):
            seen[name] = list(argv)
            return 0
        return _main

    for n in ("_split_main", "_train_main", "_finalize_main", "_register_main"):
        monkeypatch.setattr(tm, n, cap(n))

    assert tm.main(["--username", "Gyges", "--description", "my custom desc"]) == 0
    ra = seen["_register_main"]
    assert "--description" in ra
    assert ra[ra.index("--description") + 1] == "my custom desc"
