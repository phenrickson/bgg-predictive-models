"""CLI: register a trained collection model to GCS for production scoring."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

from services.collections.registered_model import RegisteredCollectionModel
from services.collections.registry_writer import RegistryWriter
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def _resolve_version(candidate_root: Path, version: Union[int, str]) -> int:
    if version != "latest":
        return int(version)
    versions = [
        int(p.name[1:]) for p in candidate_root.iterdir()
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
    ]
    if not versions:
        raise FileNotFoundError(f"No versions under {candidate_root}")
    return max(versions)


def register_collection(
    *,
    username: str,
    outcome: str,
    candidate: str,
    description: str,
    version: Union[int, str] = "latest",
    environment: str = "dev",
    local_root: str = "models/collections",
) -> dict:
    # Local layout matches src/collection/collection_artifact_storage.py:
    # {local_root}/{username}/{outcome}/{candidate}/v{N}/. The `environment`
    # arg controls the GCS prefix and registry context, not the local path.
    cand_root = Path(local_root) / username / outcome / candidate
    if not cand_root.is_dir():
        raise FileNotFoundError(f"Candidate dir not found: {cand_root}")

    resolved = _resolve_version(cand_root, version)
    version_dir = cand_root / f"v{resolved}"

    finalized = version_dir / "finalized.pkl"
    if not finalized.exists():
        raise FileNotFoundError(
            f"Finalized artifact not found: {finalized}. "
            f"Run finalize before promoting."
        )
    pipeline_path, kind = finalized, "finalized"

    pipeline = pickle.loads(pipeline_path.read_bytes())

    threshold = None
    if (version_dir / "threshold.json").exists():
        threshold = json.loads(
            (version_dir / "threshold.json").read_text()
        ).get("threshold")

    cand_reg = {}
    if (version_dir / "registration.json").exists():
        cand_reg = json.loads((version_dir / "registration.json").read_text())

    registry = RegisteredCollectionModel(username=username, outcome=outcome)
    registration = registry.register(
        pipeline=pipeline,
        threshold=threshold,
        source_metadata={
            "candidate": candidate,
            "candidate_version": resolved,
            "pipeline_kind": kind,
            "splits_version": cand_reg.get("splits_version"),
            "candidate_registration": cand_reg,
        },
        description=description,
    )
    logger.info(
        "Registered %s/%s as v%d (kind=%s, source v%d)",
        username, outcome, registration["version"], kind, resolved,
    )

    # Insert into the BQ registry so the scoring service can find this deployment.
    cfg = load_config()
    registry_table = cfg.get_collection_registry_table()
    bucket = cfg.get_bucket_name()
    env = cfg.get_environment_prefix()
    gcs_path = (
        f"gs://{bucket}/{env}/services/collections/"
        f"{username}/{outcome}/v{registration['version']}/"
    )
    RegistryWriter(registry_table).register_deployment(
        username=username,
        outcome=outcome,
        model_version=registration["version"],
        gcs_path=gcs_path,
        # The finalize step writes "finalize_through" to candidate
        # registration.json (see src/collection/collection_artifact_storage.py
        # save_finalized_pipeline). Map it to the registry's
        # finalize_through_year column.
        finalize_through_year=cand_reg.get("finalize_through"),
        registered_at=datetime.now(timezone.utc),
    )

    return registration


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--username", required=True)
    p.add_argument("--outcome", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--version", default="latest")
    p.add_argument("--environment", default="dev")
    p.add_argument("--local-root", default="models/collections")
    args = p.parse_args(argv)

    try:
        registration = register_collection(
            username=args.username,
            outcome=args.outcome,
            candidate=args.candidate,
            description=args.description,
            version=args.version,
            environment=args.environment,
            local_root=args.local_root,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(json.dumps({"registered": registration}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
