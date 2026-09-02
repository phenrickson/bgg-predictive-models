"""Component-loadings x feature-prevalence diagnostic for a fitted game embedding.

For each PCA/SVD component: the top features by absolute loading, each with the
fraction of games that carry it, plus a concentration score
(``max(loading**2) / sum(loading**2)``). A component with high concentration
whose top features are all rare is a "rare-feature detector" — the pathology the
2-SD input scaling is meant to remove.

Pure function: ``summarize_components``. CLI::

    uv run python -m src.models.embeddings.diagnose_components \
        --experiment game-embeddings [--version N] [--top-k 10] [--flag-prevalence 0.005]
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def summarize_components(
    components: np.ndarray,
    feature_names: Sequence[str],
    prevalence: np.ndarray,
    top_k: int = 10,
    explained_variance_ratio: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Summarise each component by its top-loading features and their prevalence.

    Args:
        components: ``(n_components, n_features)`` loadings matrix.
        feature_names: length ``n_features``.
        prevalence: length ``n_features``; fraction of rows carrying each
            (binary) feature, or ``nan`` for continuous features.
        top_k: how many top features to report per component.
        explained_variance_ratio: optional length ``n_components``.

    Returns:
        One dict per component: ``component``, ``concentration``,
        ``top_features`` (feature / loading / abs_loading / prevalence),
        ``min_prevalence_in_top`` (min over the top features, ignoring nan),
        and ``explained_variance_ratio`` when supplied.
    """
    components = np.asarray(components, dtype=float)
    prevalence = np.asarray(prevalence, dtype=float)
    names = list(feature_names)

    out: List[Dict[str, Any]] = []
    for i, comp in enumerate(components):
        sq = comp ** 2
        denom = float(sq.sum()) or 1.0
        concentration = float(sq.max() / denom)

        order = np.argsort(np.abs(comp))[::-1][:top_k]
        top_features = [
            {
                "feature": names[j],
                "loading": float(comp[j]),
                "abs_loading": float(abs(comp[j])),
                "prevalence": float(prevalence[j]),
            }
            for j in order
        ]
        prev_in_top = [f["prevalence"] for f in top_features if not np.isnan(f["prevalence"])]
        row: Dict[str, Any] = {
            "component": i,
            "concentration": concentration,
            "top_features": top_features,
            "min_prevalence_in_top": min(prev_in_top) if prev_in_top else float("nan"),
        }
        if explained_variance_ratio is not None:
            row["explained_variance_ratio"] = float(explained_variance_ratio[i])
        out.append(row)
    return out


# --- CLI -----------------------------------------------------------------------


def _resolve_experiment_dir(experiment: str, version: Optional[int], experiments_dir: str) -> Path:
    from src.models.experiments import ExperimentTracker

    tracker = ExperimentTracker("embeddings", experiments_dir)
    exp = tracker.load_experiment(experiment, version)
    return Path(exp.exp_dir)


def _feature_prevalence(matrix, feature_names: Sequence[str]) -> np.ndarray:
    """Fraction of rows == 1 for binary columns; nan for the rest.

    Aligns positionally to ``feature_names`` (the preprocessor preserves column
    order), so it tolerates the ``SimpleImputer.get_feature_names_out`` mismatch
    that older pickled pipelines trip on.
    """
    import pandas as pd

    arr = matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)
    prev = np.full(len(feature_names), np.nan)
    for k in range(min(arr.shape[1], len(feature_names))):
        col = arr[:, k]
        col = col[~np.isnan(col.astype(float))] if col.dtype.kind in "fc" else col
        vals = set(np.unique(col).tolist())
        if vals.issubset({0, 1}):
            prev[k] = float(np.mean(col))
    return prev


def main(argv: Optional[Sequence[str]] = None) -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="game-embeddings")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--experiments-dir", default="./models/experiments")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--flag-prevalence",
        type=float,
        default=0.005,
        help="flag components whose top features are all rarer than this",
    )
    args = parser.parse_args(argv)
    exp_dir = _resolve_experiment_dir(args.experiment, args.version, args.experiments_dir)
    print(generate_report(exp_dir, top_k=args.top_k, flag_prevalence=args.flag_prevalence))


def generate_report(
    exp_dir: Path, top_k: int = 10, flag_prevalence: float = 0.005
) -> str:
    """Build the diagnostic report string for a trained experiment directory."""
    import json

    artifacts = json.loads((Path(exp_dir) / "artifacts.json").read_text())
    if "components" not in artifacts:
        raise ValueError("artifacts.json has no 'components' — diagnostic needs pca / svd.")
    components = np.asarray(artifacts["components"], dtype=float)
    feature_names = list(artifacts["feature_names"])
    evr = (
        np.asarray(artifacts["explained_variance_ratio"])
        if "explained_variance_ratio" in artifacts
        else None
    )

    prevalence = np.full(len(feature_names), np.nan)
    try:
        from src.models.embeddings.data import EmbeddingDataLoader

        with open(Path(exp_dir) / "embedding_pipeline.pkl", "rb") as f:
            preprocessor = pickle.load(f)["preprocessor"]
        # numpy output avoids sklearn's get_feature_names_out chain (older
        # pickled imputers raise there); align positionally to feature_names.
        preprocessor.set_output(transform="default")
        logger.info("Loading training features to compute prevalence...")
        df = EmbeddingDataLoader().load_embedding_data().to_pandas()
        prevalence = _feature_prevalence(preprocessor.transform(df), feature_names)
    except Exception as e:  # noqa: BLE001 — diagnostic degrades gracefully
        logger.warning("prevalence unavailable (%s); reporting concentration only", e)

    rows = summarize_components(
        components, feature_names, prevalence, top_k=top_k, explained_variance_ratio=evr
    )

    def _feat_label(f: Dict[str, Any]) -> str:
        p = f["prevalence"]
        tag = "cont" if np.isnan(p) else f"{p:.3f}"
        return f"{f['feature']}({tag})"

    lines = [
        f"component diagnostic - {Path(exp_dir).name}  ({len(rows)} components, sorted by concentration)",
        f"{'comp':>4}  {'conc':>6}  {'min_prev':>9}  {'evr':>7}  top features",
        "-" * 100,
    ]
    for r in sorted(rows, key=lambda r: r["concentration"], reverse=True):
        evr_s = f"{r.get('explained_variance_ratio', float('nan')):.4f}"
        mp = r["min_prevalence_in_top"]
        mp_s = "   nan   " if np.isnan(mp) else f"{mp:9.5f}"
        feats = ", ".join(_feat_label(f) for f in r["top_features"][:5])
        lines.append(
            f"{r['component']:>4}  {r['concentration']:6.3f}  {mp_s}  {evr_s:>7}  {feats}"
        )

    flagged = sorted(
        r["component"]
        for r in rows
        if not np.isnan(r["min_prevalence_in_top"])
        and r["min_prevalence_in_top"] < flag_prevalence
    )
    lines.append("")
    lines.append(
        f"{len(flagged)} component(s) with every top-{top_k} feature rarer than "
        f"{flag_prevalence:g}: {flagged}"
    )
    return "\n".join(lines)


def write_report(exp_dir: Path, **kwargs) -> Path:
    """Write ``component_diagnostic.txt`` into an experiment dir; return its path."""
    out = Path(exp_dir) / "component_diagnostic.txt"
    try:
        out.write_text(generate_report(exp_dir, **kwargs), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 — never fail a training run over the diagnostic
        logger.warning("component diagnostic skipped (%s)", e)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
