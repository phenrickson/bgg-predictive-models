"""Cosine nearest-neighbour spot-check for a trained game embedding.

Stage A of the embedding-similarity eval: load a version's embeddings, L2
normalise, and print top-k neighbour lists for named games — optionally beside
another version so you can see how the lists move. Also lists the most
"isolated" games (lowest mean similarity to their own top-k), the tail where
sparse-tag pathologies show up.

    uv run python -m src.models.embeddings.neighbor_check \
        --experiment game-embeddings --version 4 \
        --compare svd-embeddings:5 \
        --games "Catan, Gloomhaven, Wingspan, The White Castle, System Gateway"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def load_embedding(
    experiment: str, version: Optional[int], experiments_dir: str = "./models/experiments"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ids, names, years, X) for a trained embedding version.

    X is L2-normalised so ``X @ X.T`` is cosine similarity. All splits
    (train / tune / test) are concatenated and de-duplicated on game_id.
    """
    from src.models.experiments import ExperimentTracker

    tracker = ExperimentTracker("embeddings", experiments_dir)
    exp = tracker.load_experiment(experiment, version)
    exp_dir = Path(exp.exp_dir)

    emb = pd.concat(
        [
            pd.read_parquet(exp_dir / f"{split}_embeddings.parquet")
            for split in ("train", "tune", "test")
            if (exp_dir / f"{split}_embeddings.parquet").exists()
        ],
        ignore_index=True,
    )
    data = pd.concat(
        [
            pd.read_parquet(exp_dir / f"{split}_data.parquet")
            for split in ("train", "tune", "test")
            if (exp_dir / f"{split}_data.parquet").exists()
        ],
        ignore_index=True,
    )
    emb = emb.drop_duplicates(subset="game_id").reset_index(drop=True)
    year_by_id = (
        data.drop_duplicates(subset="game_id").set_index("game_id")["year_published"]
        if "year_published" in data.columns
        else pd.Series(dtype="float64")
    )

    ids = emb["game_id"].to_numpy()
    names = emb["name"].astype(str).to_numpy()
    years = np.array([year_by_id.get(i, np.nan) for i in ids])
    X = np.vstack(emb["embedding"].to_numpy()).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
    return ids, names, years, X


def _resolve(query: str, ids: np.ndarray, names: np.ndarray) -> Optional[int]:
    """Row index for a query that is either a game_id or a name substring."""
    q = query.strip()
    if q.isdigit() and int(q) in set(ids.tolist()):
        return int(np.where(ids == int(q))[0][0])
    hits = np.where(np.char.lower(names.astype(str)) == q.lower())[0]
    if len(hits) == 0:
        mask = np.char.find(np.char.lower(names.astype(str)), q.lower()) >= 0
        hits = np.where(mask)[0]
    if len(hits) == 0:
        return None
    return int(hits[0])


def neighbors(
    idx: int, names: np.ndarray, ids: np.ndarray, years: np.ndarray, X: np.ndarray, k: int
) -> List[Tuple[float, int, str, float]]:
    sims = X @ X[idx]
    order = np.argsort(sims)[::-1]
    order = order[order != idx][:k]
    return [(float(sims[j]), int(ids[j]), str(names[j]), float(years[j])) for j in order]


def isolation(X: np.ndarray, k: int = 25) -> np.ndarray:
    """Mean cosine similarity of each game to its top-k neighbours (excl. self)."""
    sims = X @ X.T
    np.fill_diagonal(sims, -np.inf)
    part = np.partition(sims, -k, axis=1)[:, -k:]
    return part.mean(axis=1)


def _fmt_year(y: float) -> str:
    return f"{int(y)}" if np.isfinite(y) else "----"


def compare_lists(
    game: str,
    a: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    b: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    k: int,
    label_a: str,
    label_b: str,
) -> str:
    ids_a, names_a, years_a, X_a = a
    ia = _resolve(game, ids_a, names_a)
    if ia is None:
        return f"\n### {game!r}: not found in {label_a}\n"

    head = f"\n### {names_a[ia]} ({_fmt_year(years_a[ia])})  [id {ids_a[ia]}]\n"
    na = neighbors(ia, names_a, ids_a, years_a, X_a, k)

    if b is None:
        lines = [f"{s:5.3f}  {nm}  ({_fmt_year(yr)})" for s, _, nm, yr in na]
        return head + f"\n{label_a}:\n" + "\n".join(lines) + "\n"

    ids_b, names_b, years_b, X_b = b
    ib = _resolve(game, ids_b, names_b)
    nb = (
        neighbors(ib, names_b, ids_b, years_b, X_b, k)
        if ib is not None
        else []
    )
    set_b_ids = {gid for _, gid, _, _ in nb}
    set_a_ids = {gid for _, gid, _, _ in na}

    rows = [f"{'':>2} {label_a:<38}   {label_b:<38}"]
    for r in range(k):
        la = ""
        if r < len(na):
            s, gid, nm, yr = na[r]
            mark = " " if gid in set_b_ids else "*"
            la = f"{mark}{s:4.2f} {nm[:30]} ({_fmt_year(yr)})"
        lb = ""
        if r < len(nb):
            s, gid, nm, yr = nb[r]
            mark = " " if gid in set_a_ids else "*"
            lb = f"{mark}{s:4.2f} {nm[:30]} ({_fmt_year(yr)})"
        rows.append(f"{r + 1:>2} {la:<38}   {lb:<38}")
    rows.append("   (* = not in the other list)")
    return head + "\n".join(rows) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    try:  # Windows console defaults to cp1252 and chokes on game names
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", default="game-embeddings")
    p.add_argument("--version", type=int, default=None)
    p.add_argument(
        "--compare",
        default=None,
        help="another version as 'experiment:version' or ':version' (same experiment)",
    )
    p.add_argument("--experiments-dir", default="./models/experiments")
    p.add_argument("--games", default="", help="comma-separated names or game_ids")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--isolated", type=int, default=0, help="also list N most isolated games")
    p.add_argument("--isolated-k", type=int, default=25)
    args = p.parse_args(argv)

    a = load_embedding(args.experiment, args.version, args.experiments_dir)
    label_a = f"{args.experiment} v{args.version or 'latest'}"

    b = None
    label_b = ""
    if args.compare:
        exp_c, _, ver_c = args.compare.rpartition(":")
        exp_c = exp_c or args.experiment
        b = load_embedding(exp_c, int(ver_c), args.experiments_dir)
        label_b = f"{exp_c} v{ver_c}"

    games = [g for g in (s.strip() for s in args.games.split(",")) if g]
    for g in games:
        print(compare_lists(g, a, b, args.k, label_a, label_b))

    if args.isolated:
        ids_a, names_a, years_a, X_a = a
        iso = isolation(X_a, k=args.isolated_k)
        order = np.argsort(iso)[: args.isolated]
        print(f"\n### {args.isolated} most isolated games in {label_a} "
              f"(mean top-{args.isolated_k} cosine)\n")
        for j in order:
            print(f"{iso[j]:5.3f}  {names_a[j]}  ({_fmt_year(years_a[j])})  [id {ids_a[j]}]")


if __name__ == "__main__":
    main()
