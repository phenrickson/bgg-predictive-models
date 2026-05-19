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
