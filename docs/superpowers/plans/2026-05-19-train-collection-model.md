# Train Collection Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One operation — given a username — trains, finalizes, and registers a `logistic_row_norm` collection model in a GCP Cloud Run job, then re-renders that user's model report.

**Architecture:** A new pure `src/collection/train_model.py` entrypoint chains the existing lifecycle modules' `main(argv)->int` functions in-process (split → train → finalize → register) against one local working dir. A new GitHub workflow is a thin trigger: it ensures a `collection-training` Cloud Run job exists (gcloud, mirroring `docker-collections-build.yml`), executes it for the username (the job gsutil-syncs the user's tree in/out of `gs://`), waits, then dispatches `build-model-reports.yml`. `docker-collections-build` gains a `src/collection/**` path trigger so the image carries current lifecycle code.

**Tech Stack:** Python 3.12, `uv`, existing `src.collection.*` + `services.collections.register_model` (all expose `main(argv: Optional[List[str]]=None) -> int`), GitHub Actions, `gcloud run jobs`, `gsutil`, `collections` Docker image, pytest.

**Branch:** `feature/train-collection-model` (already created; spec commit `ba4d753` is its base). All work here, never `main`.

---

## File Structure

**Created:**
- `src/collection/train_model.py` — the pure in-process orchestrator. One responsibility: run split→train→finalize→register in order by calling each module's `main([...])`, short-circuit + report which stage failed. No gsutil, no CI dispatch.
- `tests/test_collection_train_model.py` — unit tests: ordering, short-circuit-on-failure, arg construction. The four underlying `main` functions are monkeypatched (we test orchestration, not re-test split/train/finalize/register).
- `docker/collection-training-entrypoint.sh` — the Cloud Run job container command: gsutil rsync down → `uv run python -m src.collection.train_model` → gsutil rsync up. One responsibility: the gs:// boundary around the pure entrypoint.
- `.github/workflows/train-collection-model.yml` — thin trigger workflow.

**Modified:**
- `.github/workflows/docker-collections-build.yml` — add `src/collection/**` to `on.push.paths`.

**NOT modified:** `src/collection/split.py`, `train.py`, `finalize.py`, `services/collections/register_model.py` — they already expose the uniform `main(argv)->int` contract; `train_model.py` consumes it without changing them. `docker/collections.Dockerfile` — already `COPY src/` + `WORKDIR /app`; the job overrides the command, no Dockerfile change needed.

---

### Task 1: Pure orchestrator `train_model.py` — ordering

**Files:**
- Create: `src/collection/train_model.py`
- Test: `tests/test_collection_train_model.py`

Context: `src.collection.split`, `src.collection.train`, `src.collection.finalize`, and `services.collections.register_model` each define `def main(argv: Optional[List[str]] = None) -> int` returning `0` on success, non-zero on failure. `train_model.py` calls them in order with explicit argv lists, stopping at the first non-zero return and reporting which stage failed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collection_train_model.py
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
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_collection_train_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.collection.train_model'`.

- [ ] **Step 3: Implement the orchestrator**

```python
# src/collection/train_model.py
"""Single in-process entrypoint: split -> train -> finalize -> register
for one user / one candidate. Pure lifecycle — no gsutil, no CI
dispatch (those are the Cloud Run job wrapper's and the workflow's
job). Reuses each existing module's `main(argv)->int` contract; does
not modify or duplicate their logic.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from src.collection.split import main as _split_main
from src.collection.train import main as _train_main
from src.collection.finalize import main as _finalize_main
from services.collections.register_model import main as _register_main

logger = logging.getLogger("src.collection.train_model")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--username", required=True)
    p.add_argument("--outcome", default="own")
    p.add_argument("--candidate", default="logistic_row_norm")
    p.add_argument("--environment", default="dev")
    p.add_argument("--local-root", default="models/collections")
    p.add_argument("--finalize-through", default=None)
    p.add_argument("--description", default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _build_parser().parse_args(argv)

    common = [
        "--username", args.username,
        "--outcome", args.outcome,
        "--environment", args.environment,
        "--local-root", args.local_root,
    ]

    split_argv = list(common)
    train_argv = common + ["--candidate", args.candidate]
    finalize_argv = common + ["--candidate", args.candidate]
    if args.finalize_through:
        finalize_argv += ["--finalize-through", args.finalize_through]
    description = (
        args.description
        or f"{args.candidate} for {args.username}/{args.outcome}"
    )
    register_argv = common + [
        "--candidate", args.candidate,
        "--description", description,
    ]

    stages = [
        ("split", _split_main, split_argv),
        ("train", _train_main, train_argv),
        ("finalize", _finalize_main, finalize_argv),
        ("register", _register_main, register_argv),
    ]

    for name, fn, stage_argv in stages:
        logger.info("=== %s ===", name)
        rc = fn(stage_argv)
        if rc != 0:
            logger.error("Stage %r failed (rc=%s); aborting.", name, rc)
            return rc
    logger.info(
        "train_model complete: %s/%s/%s",
        args.username, args.outcome, args.candidate,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run it, verify it passes**

Run: `uv run pytest tests/test_collection_train_model.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/collection/train_model.py tests/test_collection_train_model.py
git commit -m "feat(collection): train_model orchestrator (split->train->finalize->register)"
```

---

### Task 2: Short-circuit on stage failure

**Files:**
- Modify: `tests/test_collection_train_model.py` (add tests; implementation already supports this from Task 1)

Context: Task 1's loop already returns the failing rc and stops. This task adds the tests that pin that behavior (TDD: the test proves the already-written behavior is correct and guards against regression).

- [ ] **Step 1: Add the failing-stage tests**

Append to `tests/test_collection_train_model.py`:

```python
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
    assert calls == ["split", "train"]  # stopped after train; no finalize/register


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
    # register gets a default description
    assert "logistic_row_norm for Gyges/own" in seen["_register_main"]
```

- [ ] **Step 2: Run, verify pass**

Run: `uv run pytest tests/test_collection_train_model.py -v`
Expected: PASS (3 passed total — ordering + abort + defaults).

- [ ] **Step 3: Commit**

```bash
git add tests/test_collection_train_model.py
git commit -m "test(collection): pin train_model short-circuit + defaults"
```

---

### Task 3: Cloud Run job entrypoint script (gs:// boundary)

**Files:**
- Create: `docker/collection-training-entrypoint.sh`
- Test: `tests/test_collection_training_entrypoint.py`

Context: the Cloud Run job runs the `collections` image with this script as its command. It is the gs:// boundary the spec keeps *outside* `train_model.py`: rsync the user's tree down, run the pure entrypoint, rsync results up. Args come from env vars set as Cloud Run job execution overrides (`TRAIN_USERNAME`, `TRAIN_OUTCOME`, `TRAIN_CANDIDATE`, `ENVIRONMENT`, `GCP_PROJECT_ID`). The test asserts the script is well-formed and references the required pieces (we cannot run a real gsutil/cloud job in unit tests; a real dispatch validates execution per the spec).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collection_training_entrypoint.py
from pathlib import Path
import stat


SCRIPT = Path("docker/collection-training-entrypoint.sh")


def test_entrypoint_exists_and_executable():
    assert SCRIPT.exists()
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "entrypoint must be executable"


def test_entrypoint_has_required_shape():
    text = SCRIPT.read_text()
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text
    # pull -> run -> push, in that order
    i_pull = text.index("rsync")
    i_run = text.index("src.collection.train_model")
    i_push = text.rindex("rsync")
    assert i_pull < i_run < i_push, "must rsync down, run, then rsync up"
    # uses the env-var contract
    for var in ("TRAIN_USERNAME", "ENVIRONMENT", "GCP_PROJECT_ID"):
        assert var in text
    # pure entrypoint invoked via uv run
    assert "uv run python -m src.collection.train_model" in text
```

- [ ] **Step 2: Run, verify it fails**

Run: `uv run pytest tests/test_collection_training_entrypoint.py -v`
Expected: FAIL — `assert SCRIPT.exists()` is False.

- [ ] **Step 3: Create the script**

```bash
# docker/collection-training-entrypoint.sh
#!/usr/bin/env bash
# Cloud Run job command for collection model training. The gs:// sync
# boundary kept OUT of train_model.py (which stays pure/testable):
# pull the user's artifact tree, run the in-process lifecycle, push the
# results back. Per-execution args arrive as env overrides.
set -euo pipefail

: "${TRAIN_USERNAME:?TRAIN_USERNAME required}"
TRAIN_OUTCOME="${TRAIN_OUTCOME:-own}"
TRAIN_CANDIDATE="${TRAIN_CANDIDATE:-logistic_row_norm}"
: "${ENVIRONMENT:?ENVIRONMENT required}"
: "${GCP_PROJECT_ID:?GCP_PROJECT_ID required}"

LOCAL_ROOT="/app/models/collections"
USER_LOCAL="${LOCAL_ROOT}/${TRAIN_USERNAME}"
USER_GS="gs://${GCP_PROJECT_ID}/${ENVIRONMENT}/collections/${TRAIN_USERNAME}"

mkdir -p "${USER_LOCAL}"

echo "Pulling ${USER_GS} -> ${USER_LOCAL}"
gsutil -m rsync -r "${USER_GS}" "${USER_LOCAL}" 2>&1 \
  || echo "(no prior artifacts for ${TRAIN_USERNAME}; first run)"

echo "Running train_model for ${TRAIN_USERNAME}/${TRAIN_OUTCOME}/${TRAIN_CANDIDATE}"
uv run python -m src.collection.train_model \
  --username "${TRAIN_USERNAME}" \
  --outcome "${TRAIN_OUTCOME}" \
  --candidate "${TRAIN_CANDIDATE}" \
  --environment "${ENVIRONMENT}" \
  --local-root "${LOCAL_ROOT}"

echo "Pushing ${USER_LOCAL} -> ${USER_GS}"
gsutil -m rsync -r "${USER_LOCAL}" "${USER_GS}"

echo "collection-training done: ${TRAIN_USERNAME}/${TRAIN_OUTCOME}/${TRAIN_CANDIDATE}"
```

Then make it executable:

```bash
chmod +x docker/collection-training-entrypoint.sh
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest tests/test_collection_training_entrypoint.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add docker/collection-training-entrypoint.sh tests/test_collection_training_entrypoint.py
git commit -m "feat(collection): Cloud Run job entrypoint (gs:// sync boundary)"
```

---

### Task 4: Make the script reachable in the image; add src/collection trigger

**Files:**
- Modify: `.github/workflows/docker-collections-build.yml`
- Modify: `docker/collections.Dockerfile`

Context: the job runs `collections:<env>` with the entrypoint script as its command. `collections.Dockerfile` does `COPY src/ /app/src/` but does NOT copy `docker/`. The script must be in the image. Also add `src/collection/**` to the build trigger so a lifecycle-code change rebuilds the image (accepted coupling: this also redeploys the scoring service — see spec Design Decision 3).

- [ ] **Step 1: Add the path trigger**

In `.github/workflows/docker-collections-build.yml`, the `on.push.paths` list currently is:

```yaml
    paths:
      - 'services/collections/**'
      - 'services/scoring/auth.py'
      - 'docker/collections.Dockerfile'
      - '.github/workflows/docker-collections-build.yml'
```

Change it to:

```yaml
    paths:
      - 'services/collections/**'
      - 'services/scoring/auth.py'
      - 'src/collection/**'
      - 'docker/collections.Dockerfile'
      - 'docker/collection-training-entrypoint.sh'
      - '.github/workflows/docker-collections-build.yml'
```

- [ ] **Step 2: COPY the entrypoint into the image**

Read `docker/collections.Dockerfile`. After the existing `COPY src/ /app/src/` line, add:

```dockerfile
COPY docker/collection-training-entrypoint.sh /app/collection-training-entrypoint.sh
RUN chmod +x /app/collection-training-entrypoint.sh
```

(Place it with the other `COPY` lines, before the `CMD`. Do not change `WORKDIR`, `CMD`, or anything else — the scoring service still starts via the existing `CMD`; the training job overrides the command to run this script.)

- [ ] **Step 3: Validate the workflow YAML**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/docker-collections-build.yml')); print('YAML OK')"`
Expected: `YAML OK`

- [ ] **Step 4: Confirm the Dockerfile change is minimal**

Run: `git diff docker/collections.Dockerfile`
Expected: only the two added lines (COPY + chmod of the entrypoint), nothing else changed.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/docker-collections-build.yml docker/collections.Dockerfile
git commit -m "build(collection): ship training entrypoint in image; trigger build on src/collection"
```

---

### Task 5: The thin trigger workflow

**Files:**
- Create: `.github/workflows/train-collection-model.yml`

Context: mirrors the auth/setup pattern of `docker-collections-build.yml` exactly (it is the project's established gcloud-in-workflow pattern). It ensures the `collection-training` Cloud Run job exists (idempotent create-or-update), executes it for the username with env overrides, waits, then dispatches the model report. The job uses the same service account and env-var convention as the scoring service.

- [ ] **Step 1: Create the workflow**

```yaml
# .github/workflows/train-collection-model.yml
name: Train Collection Model

on:
  workflow_dispatch:
    inputs:
      username:
        description: 'BGG username to train a collection model for'
        required: true

env:
  GCP_PROJECT_ID: bgg-predictive-models

jobs:
  train:
    name: Train + register collection model
    runs-on: ubuntu-latest
    environment: ${{ github.ref == 'refs/heads/main' && 'PROD' || 'DEV' }}
    permissions:
      contents: read
      id-token: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Environment
        run: |
          mkdir -p credentials
          echo '${{ secrets.GCP_SA_KEY_BGG_ML }}' > credentials/service-account-key.json

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY_BGG_ML }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Set environment name
        id: env
        run: |
          if [[ "${{ github.ref }}" == "refs/heads/main" ]]; then
            echo "env_name=prod" >> $GITHUB_OUTPUT
          else
            echo "env_name=dev" >> $GITHUB_OUTPUT
          fi

      - name: Ensure collection-training job exists
        run: |
          IMAGE="us-central1-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/bgg-predictive-models/collections:${{ steps.env.outputs.env_name }}"
          SA="bgg-predictive-models@${{ env.GCP_PROJECT_ID }}.iam.gserviceaccount.com"
          COMMON="--image $IMAGE \
            --region us-central1 \
            --service-account $SA \
            --command /app/collection-training-entrypoint.sh \
            --memory 4Gi --cpu 2 --task-timeout 1800 --max-retries 0 \
            --set-env-vars ENVIRONMENT=${{ steps.env.outputs.env_name }},GCP_PROJECT_ID=${{ env.GCP_PROJECT_ID }},GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account-key.json"
          gcloud run jobs describe collection-training --region us-central1 >/dev/null 2>&1 \
            && gcloud run jobs update collection-training $COMMON \
            || gcloud run jobs create collection-training $COMMON

      - name: Execute training job for user
        run: |
          gcloud run jobs execute collection-training \
            --region us-central1 \
            --wait \
            --update-env-vars TRAIN_USERNAME=${{ inputs.username }},TRAIN_OUTCOME=own,TRAIN_CANDIDATE=logistic_row_norm

      - name: Dispatch model report render
        run: |
          gh workflow run build-model-reports.yml \
            -f users=${{ inputs.username }} -f outcome=own \
            || echo "WARN: report dispatch failed; rerun manually: gh workflow run build-model-reports.yml -f users=${{ inputs.username }} -f outcome=own"
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

- [ ] **Step 2: Validate the workflow YAML**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/train-collection-model.yml')); print('YAML OK')"`
Expected: `YAML OK`

- [ ] **Step 3: Confirm key invariants by inspection**

Run: `grep -nE "workflow_dispatch|username|collection-training|--command /app/collection-training-entrypoint.sh|--wait|build-model-reports" .github/workflows/train-collection-model.yml`
Expected: shows the `username` dispatch input, the create/update/execute of `collection-training`, the entrypoint as `--command`, `--wait` on execute, and the report dispatch — confirming the thin-trigger shape from the spec.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/train-collection-model.yml
git commit -m "feat(ci): train-collection-model thin trigger workflow"
```

---

### Task 6: Full suite + branch push

**Files:** none (verification + integration)

- [ ] **Step 1: Run the full reports + collection test suite**

Run: `uv run pytest tests/test_collection_train_model.py tests/test_collection_training_entrypoint.py tests/test_collections_register_model.py -v`
Expected: all pass (the two new test files green; `test_collections_register_model` unaffected — we did not modify `register_model`).

- [ ] **Step 2: Confirm no unintended changes**

Run: `git status --short && git diff main --stat`
Expected: only the files this plan created/modified (`src/collection/train_model.py`, the two test files, `docker/collection-training-entrypoint.sh`, `docker/collections.Dockerfile`, the two workflow files, plus the spec/plan docs). No `.claude/`, no `src/collection/split|train|finalize.py`, no `register_model.py`.

- [ ] **Step 3: Push the branch**

```bash
git push -u origin feature/train-collection-model
```

- [ ] **Step 4: Note for the human (do NOT auto-do)**

The real validation is a deliberate dispatch for `Gyges` (spec: "the deliberate first run is the test"). Merging the branch to `main` triggers `docker-collections-build` (now path-watching `src/collection/**` and the entrypoint script), which rebuilds `collections:<env>` AND redeploys the scoring service (accepted coupling). Only after that image exists should `train-collection-model.yml` be dispatched for `Gyges`. This sequencing + the live dispatch is a human-gated step, surfaced at execution handoff — not performed automatically by the plan.

---

## Self-Review

**1. Spec coverage:**

| Spec element | Task |
|---|---|
| `src/collection/train_model.py` pure, in-process split→train→finalize→register | Task 1 |
| Reuses existing `main(argv)->int` (no modify/duplicate of the 4 modules) | Task 1 (imports `main` from each; verified all 5 share the signature) |
| Short-circuit + name failed stage; non-zero exit | Tasks 1 & 2 |
| Default candidate `logistic_row_norm`, outcome `own`, description default | Tasks 1 & 2 |
| gs:// sync is a boundary concern, NOT in the entrypoint | Task 3 (script does rsync; Task 1 entrypoint has no gsutil) |
| Cloud Run **job** (not service, not terraform), gcloud-in-workflow | Task 5 (`gcloud run jobs create/update/execute`) |
| Same image / SA / env-var convention as scoring deploy | Tasks 4 & 5 (collections image, `bgg-predictive-models@…iam`, ENVIRONMENT/GCP_PROJECT_ID/GOOGLE_APPLICATION_CREDENTIALS) |
| `docker-collections-build` gains `src/collection/**` trigger; accepted scoring-redeploy coupling | Task 4 |
| Report seam in the workflow (not entrypoint, not register_model) | Task 5 step "Dispatch model report render" |
| Thin trigger: auth → ensure job → execute --wait → dispatch report | Task 5 |
| `workflow_dispatch` + `username` only (trigger model deferred) | Task 5 (single required `username` input) |
| Testable off-CI | Tasks 1–3 (pytest, monkeypatched mains; no cloud needed) |
| Real `Gyges` dispatch is the integration test, human-gated | Task 6 step 4 |

No gaps.

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to". Every code/script/YAML step is complete and literal.

**3. Type/contract consistency:** `train_model.main(argv)->int` matches the project's uniform module contract (verified: split/train/finalize/register all `def main(argv: Optional[List[str]]=None)->int`). The monkeypatch targets in tests (`_split_main`, `_train_main`, `_finalize_main`, `_register_main`) exactly match the import aliases bound in Task 1's module. The entrypoint env-var names (`TRAIN_USERNAME`/`TRAIN_OUTCOME`/`TRAIN_CANDIDATE`/`ENVIRONMENT`/`GCP_PROJECT_ID`) are identical across Task 3's script and Task 5's `--update-env-vars`/`--set-env-vars`. `--command /app/collection-training-entrypoint.sh` (Task 5) matches the COPY destination in Task 4.

**Note for executor:** Tasks 1–3 are fully local/TDD (fast loop, the lesson from prior work). Tasks 4–5 are YAML/Dockerfile/script — validated by lint + inspection locally; their cloud behavior is proven only by the human-gated `Gyges` dispatch after merge (Task 6 step 4). Do not attempt to run `gcloud`/dispatch from the plan.
