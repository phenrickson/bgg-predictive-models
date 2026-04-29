# Collection Model Registration Implementation Plan

**Goal:** Replace the local-only `promote` step with a GCS-backed registration flow that publishes a trained collection model (preferring its finalized refit) to the bucket the scoring service reads from.

**Architecture:** New `services/collections/` package mirrors `services/scoring/` but uses a sibling `RegisteredCollectionModel` class (no `ExperimentTracker` inheritance). CLI reads candidate run from local `models/collections/{env}/{username}/{outcome}/{candidate}/v{N}/`, prefers `finalized.pkl`, uploads `pipeline.pkl` + `threshold.json` + `registration.json` to `gs://{bucket}/{env}/services/collections/{username}/{outcome}/v{N}/`. Justfile `promote` is rewired; old `src/collection/promote.py` is deleted.

---

## Task 1: `RegisteredCollectionModel` class

Create `services/collections/__init__.py` (empty) and `services/collections/registered_model.py`:

```python
"""GCS-backed registration for trained collection models."""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from services.scoring.auth import (
    AuthenticationError,
    get_authenticated_storage_client,
)
from src.utils.config import load_config


class RegisteredCollectionModel:
    """Per-user collection model registration in GCS.

    Layout: {env}/services/collections/{username}/{outcome}/v{N}/
            {pipeline.pkl, threshold.json, registration.json}
    """

    def __init__(
        self,
        username: str,
        outcome: str,
        bucket_name: Optional[str] = None,
        environment_prefix: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        self.username = username
        self.outcome = outcome

        if bucket_name is None or environment_prefix is None:
            cfg = load_config()
            bucket_name = bucket_name or cfg.get_bucket_name()
            environment_prefix = environment_prefix or cfg.get_environment_prefix()

        try:
            self.storage_client = get_authenticated_storage_client(project_id)
        except AuthenticationError as e:
            raise ValueError(f"Authentication failed: {e}")

        self.bucket = self.storage_client.bucket(bucket_name)
        self.bucket_name = bucket_name
        self.environment_prefix = environment_prefix
        self.base_prefix = f"{environment_prefix}/services/collections/{username}/{outcome}"

    def list_versions(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for blob in self.bucket.list_blobs(prefix=self.base_prefix):
            if blob.name.endswith("/registration.json"):
                out.append(json.loads(blob.download_as_text()))
        return sorted(out, key=lambda v: v["version"])

    def register(
        self,
        pipeline: Any,
        threshold: Optional[float],
        source_metadata: Dict[str, Any],
        description: str,
    ) -> Dict[str, Any]:
        version = max((v["version"] for v in self.list_versions()), default=0) + 1
        prefix = f"{self.base_prefix}/v{version}"

        registration = {
            "username": self.username,
            "outcome": self.outcome,
            "version": version,
            "description": description,
            "source": source_metadata,
            "threshold": threshold,
            "registered_at": datetime.now().isoformat(),
        }

        self.bucket.blob(f"{prefix}/pipeline.pkl").upload_from_string(
            pickle.dumps(pipeline), content_type="application/octet-stream"
        )
        self.bucket.blob(f"{prefix}/threshold.json").upload_from_string(
            json.dumps({"threshold": threshold}), content_type="application/json"
        )
        self.bucket.blob(f"{prefix}/registration.json").upload_from_string(
            json.dumps(registration, indent=2), content_type="application/json"
        )
        return registration

    def load(
        self, version: Optional[int] = None
    ) -> Tuple[Any, Optional[float], Dict[str, Any]]:
        versions = self.list_versions()
        if not versions:
            raise ValueError(f"No registered versions for {self.username}/{self.outcome}")
        if version is None:
            version = max(v["version"] for v in versions)
        elif not any(v["version"] == version for v in versions):
            raise ValueError(f"Version {version} not registered")

        prefix = f"{self.base_prefix}/v{version}"
        pipeline = pickle.loads(
            self.bucket.blob(f"{prefix}/pipeline.pkl").download_as_string()
        )
        threshold = json.loads(
            self.bucket.blob(f"{prefix}/threshold.json").download_as_text()
        ).get("threshold")
        registration = json.loads(
            self.bucket.blob(f"{prefix}/registration.json").download_as_text()
        )
        return pipeline, threshold, registration
```

**Commit:**
```bash
git add services/collections/__init__.py services/collections/registered_model.py
git commit -m "feat(services): RegisteredCollectionModel for GCS-backed collection model registration"
```

---

## Task 2: CLI

Create `services/collections/register_model.py`:

```python
"""CLI: register a trained collection model to GCS for production scoring."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Union

from services.collections.registered_model import RegisteredCollectionModel

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
    cand_root = Path(local_root) / environment / username / outcome / candidate
    if not cand_root.is_dir():
        raise FileNotFoundError(f"Candidate dir not found: {cand_root}")

    resolved = _resolve_version(cand_root, version)
    version_dir = cand_root / f"v{resolved}"

    finalized = version_dir / "finalized.pkl"
    train_only = version_dir / "model.pkl"
    if finalized.exists():
        pipeline_path, kind = finalized, "finalized"
    elif train_only.exists():
        pipeline_path, kind = train_only, "train_only"
    else:
        raise FileNotFoundError(f"No finalized.pkl or model.pkl in {version_dir}")

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
```

**Sanity check:**
```bash
uv run python -m services.collections.register_model --help
```
Expected: prints usage with all six args.

**Commit:**
```bash
git add services/collections/register_model.py
git commit -m "feat(services): register_model CLI for collection models"
```

---

## Task 3: Rewire justfile, delete old promote

Replace the existing `promote` recipe in `justfile`:

```make
# Register a trained collection model to GCS for the standalone scoring
# service. Prefers the finalized refit if present.
promote outcome="own" candidate="lgbm_default" version="latest" description="":
    uv run python -m services.collections.register_model \
        --username {{username}} --environment {{environment}} --outcome {{outcome}} \
        --candidate {{candidate}} --version {{version}} \
        --local-root {{local_root}} \
        --description "$([ -n "{{description}}" ] && echo "{{description}}" || echo "{{candidate}} for {{username}}/{{outcome}}")"
```

Delete the old module:
```bash
git rm src/collection/promote.py
```

Verify nothing imports it:
```bash
grep -rn "src.collection.promote\|from src.collection import promote" .
```
Expected: no matches.

Smoke-test:
```bash
just --dry-run promote candidate=lgbm_row_norm
```
Expected: rendered command references `services.collections.register_model`.

**Commit:**
```bash
git add justfile
git commit -m "feat: rewire just promote to GCS-backed services.collections.register_model"
```
