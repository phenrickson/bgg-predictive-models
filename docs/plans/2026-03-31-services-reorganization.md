# Services Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `scoring_service/`, `embeddings_service/`, and `text_embeddings_service/` into a unified `services/` directory, renaming `embeddings_service` to `game_embeddings`.

**Architecture:** Git `mv` the three service directories under `services/`, then update all references: Python imports, Makefile targets, Dockerfiles, GitHub Actions workflows, cloudbuild.yaml, tests, and README.

**Tech Stack:** Python, Make, Docker, GitHub Actions, Google Cloud Build

---

## File Structure

After this plan, the following changes exist:

**Directories moved:**
- `scoring_service/` -> `services/scoring/`
- `embeddings_service/` -> `services/game_embeddings/`
- `text_embeddings_service/` -> `services/text_embeddings/`

**Files created:**
- `services/__init__.py` — Makes `services/` a Python package for import resolution

**Files modified (path references updated):**
- `Makefile`
- `register.py`
- `docker/scoring.Dockerfile`
- `docker/embeddings.Dockerfile`
- `docker/text_embeddings.Dockerfile`
- `.github/workflows/docker-scoring-build.yml`
- `.github/workflows/docker-embeddings-build.yml`
- `.github/workflows/docker-text-embeddings-build.yml`
- `services/scoring/register_model.py` (internal import)
- `services/scoring/score.py` (internal import)
- `services/game_embeddings/auth.py` (import from scoring)
- `services/game_embeddings/__init__.py` (module docstring)
- `services/game_embeddings/main.py` (internal import)
- `services/game_embeddings/register_model.py` (internal import)
- `services/text_embeddings/auth.py` (import from scoring)
- `services/text_embeddings/main.py` (internal import)
- `services/text_embeddings/register_model.py` (internal import)
- `services/scoring/cloudbuild.yaml`
- `src/models/text_embeddings/score.py`
- `tests/test_authentication.py`
- `tests/test_register.py`
- `tests/test_register_model.py`
- `README.md`

---

### Task 1: Move directories with git mv

**Files:**
- Move: `scoring_service/` -> `services/scoring/`
- Move: `embeddings_service/` -> `services/game_embeddings/`
- Move: `text_embeddings_service/` -> `services/text_embeddings/`

- [ ] **Step 1: Create services directory, add __init__.py, and move all three services**

```bash
mkdir -p services
touch services/__init__.py
git mv scoring_service services/scoring
git mv embeddings_service services/game_embeddings
git mv text_embeddings_service services/text_embeddings
```

- [ ] **Step 2: Verify the moves**

Run: `ls services/`
Expected: `game_embeddings  scoring  text_embeddings`

Run: `ls services/scoring/`
Expected: `__init__.py  auth.py  cloud_experiment_tracker.py  cloudbuild.yaml  main.py  register_model.py  registered_model.py  score.py  verify_models.py  README.md`

- [ ] **Step 3: Commit the directory moves**

```bash
git add -A
git commit -m "refactor: move service directories under services/"
```

---

### Task 2: Update Python imports in services/scoring/

**Files:**
- Modify: `services/scoring/register_model.py:15`
- Modify: `services/scoring/score.py:18-22`

- [ ] **Step 1: Update register_model.py import**

In `services/scoring/register_model.py`, change line 15:

Old:
```python
from scoring_service.registered_model import RegisteredModel  # noqa: E402
```

New:
```python
from services.scoring.registered_model import RegisteredModel  # noqa: E402
```

- [ ] **Step 2: Update score.py imports**

In `services/scoring/score.py`, replace lines 18-22:

Old:
```python
# Add scoring_service directory to path for auth module
scoring_service_path = os.path.dirname(__file__)
sys.path.insert(0, scoring_service_path)

from auth import get_authenticated_storage_client  # noqa: E402
```

New:
```python
from services.scoring.auth import get_authenticated_storage_client  # noqa: E402
```

- [ ] **Step 3: Commit**

```bash
git add services/scoring/register_model.py services/scoring/score.py
git commit -m "refactor: update imports in services/scoring"
```

---

### Task 3: Update Python imports in services/game_embeddings/

**Files:**
- Modify: `services/game_embeddings/__init__.py`
- Modify: `services/game_embeddings/auth.py:1-18`
- Modify: `services/game_embeddings/main.py:22`
- Modify: `services/game_embeddings/register_model.py:15`

- [ ] **Step 1: Update __init__.py docstring**

In `services/game_embeddings/__init__.py`, change the docstring:

Old:
```python
"""Embeddings service for generating and querying game embeddings."""
```

New:
```python
"""Game embeddings service for generating and querying game embeddings."""
```

- [ ] **Step 2: Update auth.py**

Replace the full content of `services/game_embeddings/auth.py`:

Old:
```python
"""Authentication utilities for the embeddings service.

Re-exports from scoring_service.auth for consistency.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from scoring_service.auth import (  # noqa: E402, F401
    AuthenticationError,
    GCPAuthenticator,
    get_authenticated_storage_client,
    verify_authentication,
)
```

New:
```python
"""Authentication utilities for the game embeddings service.

Re-exports from services.scoring.auth for consistency.
"""

from services.scoring.auth import (  # noqa: F401
    AuthenticationError,
    GCPAuthenticator,
    get_authenticated_storage_client,
    verify_authentication,
)
```

- [ ] **Step 3: Update main.py import**

In `services/game_embeddings/main.py`, change line 22:

Old:
```python
from embeddings_service.registered_model import RegisteredEmbeddingModel  # noqa: E402
```

New:
```python
from services.game_embeddings.registered_model import RegisteredEmbeddingModel  # noqa: E402
```

- [ ] **Step 4: Update register_model.py import**

In `services/game_embeddings/register_model.py`, change line 15:

Old:
```python
from embeddings_service.registered_model import RegisteredEmbeddingModel  # noqa: E402
```

New:
```python
from services.game_embeddings.registered_model import RegisteredEmbeddingModel  # noqa: E402
```

- [ ] **Step 5: Commit**

```bash
git add services/game_embeddings/
git commit -m "refactor: update imports in services/game_embeddings"
```

---

### Task 4: Update Python imports in services/text_embeddings/

**Files:**
- Modify: `services/text_embeddings/auth.py:1-18`
- Modify: `services/text_embeddings/main.py:21`
- Modify: `services/text_embeddings/register_model.py:17`

- [ ] **Step 1: Update auth.py**

Replace the full content of `services/text_embeddings/auth.py`:

Old:
```python
"""Authentication utilities for the text embeddings service.

Re-exports from scoring_service.auth for consistency.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from scoring_service.auth import (  # noqa: E402, F401
    AuthenticationError,
    GCPAuthenticator,
    get_authenticated_storage_client,
    verify_authentication,
)
```

New:
```python
"""Authentication utilities for the text embeddings service.

Re-exports from services.scoring.auth for consistency.
"""

from services.scoring.auth import (  # noqa: F401
    AuthenticationError,
    GCPAuthenticator,
    get_authenticated_storage_client,
    verify_authentication,
)
```

- [ ] **Step 2: Update main.py import**

In `services/text_embeddings/main.py`, change line 21:

Old:
```python
from text_embeddings_service.registered_model import RegisteredTextEmbeddingModel  # noqa: E402
```

New:
```python
from services.text_embeddings.registered_model import RegisteredTextEmbeddingModel  # noqa: E402
```

- [ ] **Step 3: Update register_model.py import**

In `services/text_embeddings/register_model.py`, change line 17:

Old:
```python
from text_embeddings_service.registered_model import RegisteredTextEmbeddingModel  # noqa: E402
```

New:
```python
from services.text_embeddings.registered_model import RegisteredTextEmbeddingModel  # noqa: E402
```

- [ ] **Step 4: Commit**

```bash
git add services/text_embeddings/
git commit -m "refactor: update imports in services/text_embeddings"
```

---

### Task 5: Update src/ references to service modules

**Files:**
- Modify: `src/models/text_embeddings/score.py:81`

- [ ] **Step 1: Update the import**

In `src/models/text_embeddings/score.py`, change line 81:

Old:
```python
    from text_embeddings_service.registered_model import RegisteredTextEmbeddingModel
```

New:
```python
    from services.text_embeddings.registered_model import RegisteredTextEmbeddingModel
```

- [ ] **Step 2: Commit**

```bash
git add src/models/text_embeddings/score.py
git commit -m "refactor: update text_embeddings_service import in src"
```

---

### Task 6: Update register.py

**Files:**
- Modify: `register.py:61,80,99,118,137`

- [ ] **Step 1: Replace all scoring_service.register_model references**

In `register.py`, replace all 5 occurrences of `"scoring_service.register_model"` with `"services.scoring.register_model"`:

Line 61:
```python
        "services.scoring.register_model",
```

Line 80:
```python
        "services.scoring.register_model",
```

Line 99:
```python
        "services.scoring.register_model",
```

Line 118:
```python
        "services.scoring.register_model",
```

Line 137:
```python
        "services.scoring.register_model",
```

- [ ] **Step 2: Commit**

```bash
git add register.py
git commit -m "refactor: update register.py to use services.scoring module path"
```

---

### Task 7: Update Makefile

**Files:**
- Modify: `Makefile:165,181,274,281`

- [ ] **Step 1: Update text_embeddings register target**

In `Makefile`, change line 165:

Old:
```makefile
	uv run -m text_embeddings_service.register_model \
```

New:
```makefile
	uv run -m services.text_embeddings.register_model \
```

- [ ] **Step 2: Update embeddings register target**

In `Makefile`, change line 181:

Old:
```makefile
	uv run -m embeddings_service.register_model \
```

New:
```makefile
	uv run -m services.game_embeddings.register_model \
```

- [ ] **Step 3: Update scoring-service target**

In `Makefile`, change line 274:

Old:
```makefile
	uv run -m scoring_service.score \
```

New:
```makefile
	uv run -m services.scoring.score \
```

- [ ] **Step 4: Update scoring-service-upload target**

In `Makefile`, change line 281:

Old:
```makefile
	uv run -m scoring_service.score \
```

New:
```makefile
	uv run -m services.scoring.score \
```

- [ ] **Step 5: Commit**

```bash
git add Makefile
git commit -m "refactor: update Makefile service module paths"
```

---

### Task 8: Update Dockerfiles

**Files:**
- Modify: `docker/scoring.Dockerfile:25-28`
- Modify: `docker/embeddings.Dockerfile:25-27,52`
- Modify: `docker/text_embeddings.Dockerfile:23-25,50`

- [ ] **Step 1: Update scoring.Dockerfile**

In `docker/scoring.Dockerfile`, change lines 25-28:

Old:
```dockerfile
COPY scoring_service/main.py .
COPY scoring_service/cloud_experiment_tracker.py .
COPY scoring_service/registered_model.py .
COPY scoring_service/auth.py .
```

New:
```dockerfile
COPY services/scoring/main.py .
COPY services/scoring/cloud_experiment_tracker.py .
COPY services/scoring/registered_model.py .
COPY services/scoring/auth.py .
```

- [ ] **Step 2: Update embeddings.Dockerfile**

In `docker/embeddings.Dockerfile`, change lines 25-27:

Old:
```dockerfile
COPY embeddings_service/ /app/embeddings_service/
COPY scoring_service/auth.py /app/scoring_service/auth.py
COPY scoring_service/__init__.py /app/scoring_service/__init__.py
```

New:
```dockerfile
COPY services/__init__.py /app/services/__init__.py
COPY services/game_embeddings/ /app/services/game_embeddings/
COPY services/scoring/auth.py /app/services/scoring/auth.py
COPY services/scoring/__init__.py /app/services/scoring/__init__.py
```

Change line 52:

Old:
```dockerfile
CMD ["uvicorn", "embeddings_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

New:
```dockerfile
CMD ["uvicorn", "services.game_embeddings.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 3: Update text_embeddings.Dockerfile**

In `docker/text_embeddings.Dockerfile`, change lines 23-25:

Old:
```dockerfile
COPY text_embeddings_service/ /app/text_embeddings_service/
COPY scoring_service/auth.py /app/scoring_service/auth.py
COPY scoring_service/__init__.py /app/scoring_service/__init__.py
```

New:
```dockerfile
COPY services/__init__.py /app/services/__init__.py
COPY services/text_embeddings/ /app/services/text_embeddings/
COPY services/scoring/auth.py /app/services/scoring/auth.py
COPY services/scoring/__init__.py /app/services/scoring/__init__.py
```

Change line 50:

Old:
```dockerfile
CMD ["uvicorn", "text_embeddings_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

New:
```dockerfile
CMD ["uvicorn", "services.text_embeddings.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 4: Commit**

```bash
git add docker/scoring.Dockerfile docker/embeddings.Dockerfile docker/text_embeddings.Dockerfile
git commit -m "refactor: update Dockerfiles for services/ directory structure"
```

---

### Task 9: Update GitHub Actions workflows

**Files:**
- Modify: `.github/workflows/docker-scoring-build.yml:9`
- Modify: `.github/workflows/docker-embeddings-build.yml:9`
- Modify: `.github/workflows/docker-text-embeddings-build.yml:9`

- [ ] **Step 1: Update scoring workflow path trigger**

In `.github/workflows/docker-scoring-build.yml`, change line 9:

Old:
```yaml
      - 'scoring_service/**'
```

New:
```yaml
      - 'services/scoring/**'
```

- [ ] **Step 2: Update embeddings workflow path trigger**

In `.github/workflows/docker-embeddings-build.yml`, change line 9:

Old:
```yaml
      - 'embeddings_service/**'
```

New:
```yaml
      - 'services/game_embeddings/**'
```

- [ ] **Step 3: Update text embeddings workflow path trigger**

In `.github/workflows/docker-text-embeddings-build.yml`, change line 9:

Old:
```yaml
      - 'text_embeddings_service/**'
```

New:
```yaml
      - 'services/text_embeddings/**'
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/docker-scoring-build.yml .github/workflows/docker-embeddings-build.yml .github/workflows/docker-text-embeddings-build.yml
git commit -m "refactor: update workflow path triggers for services/ directory"
```

---

### Task 10: Update cloudbuild.yaml

**Files:**
- Modify: `services/scoring/cloudbuild.yaml:49-50`

- [ ] **Step 1: Update artifact paths**

In `services/scoring/cloudbuild.yaml`, change lines 49-50:

Old:
```yaml
    - 'scoring_service/pyproject.toml'
    - 'scoring_service/Dockerfile'
```

New:
```yaml
    - 'services/scoring/pyproject.toml'
    - 'services/scoring/Dockerfile'
```

- [ ] **Step 2: Commit**

```bash
git add services/scoring/cloudbuild.yaml
git commit -m "refactor: update cloudbuild.yaml artifact paths"
```

---

### Task 11: Update test files

**Files:**
- Modify: `tests/test_authentication.py:12,90,109,118,126,139,150,166,347`
- Modify: `tests/test_register.py:116,139,155,171`
- Modify: `tests/test_register_model.py:7,50-53,96-99,137`

- [ ] **Step 1: Update test_authentication.py**

In `tests/test_authentication.py`, replace all occurrences of `scoring_service.auth` with `services.scoring.auth`:

Line 12:
```python
from services.scoring.auth import GCPAuthenticator, AuthenticationError
```

Line 90:
```python
    @patch("services.scoring.auth.default")
```

Line 109:
```python
                "services.scoring.auth.default", side_effect=DefaultCredentialsError()
```

Line 118:
```python
                "services.scoring.auth.default", side_effect=DefaultCredentialsError()
```

Line 126:
```python
    @patch("services.scoring.auth.storage.Client")
```

Line 139:
```python
    @patch("services.scoring.auth.storage.Client")
```

Line 150:
```python
    @patch("services.scoring.auth.storage.Client")
```

Line 166:
```python
    @patch("services.scoring.auth.storage.Client")
```

Line 347:
```python
    @patch("services.scoring.auth.storage.Client")
```

- [ ] **Step 2: Update test_register.py**

In `tests/test_register.py`, replace all occurrences of `"scoring_service.register_model"` with `"services.scoring.register_model"`:

Line 116:
```python
            "services.scoring.register_model",
```

Line 139:
```python
        assert "services.scoring.register_model" in cmd
```

Line 155:
```python
        assert "services.scoring.register_model" in cmd
```

Line 171:
```python
        assert "services.scoring.register_model" in cmd
```

- [ ] **Step 3: Update test_register_model.py**

In `tests/test_register_model.py`, replace all occurrences of `scoring_service.register_model` with `services.scoring.register_model`:

Line 7:
```python
from services.scoring.register_model import validate_environment, register_model
```

Line 50:
```python
    @patch("services.scoring.register_model.get_project_id")
```

Line 51:
```python
    @patch("services.scoring.register_model.load_config")
```

Line 52:
```python
    @patch("services.scoring.register_model.ExperimentTracker")
```

Line 53:
```python
    @patch("services.scoring.register_model.RegisteredModel")
```

Line 96:
```python
    @patch("services.scoring.register_model.get_project_id")
```

Line 97:
```python
    @patch("services.scoring.register_model.load_config")
```

Line 98:
```python
    @patch("services.scoring.register_model.ExperimentTracker")
```

Line 99:
```python
    @patch("services.scoring.register_model.RegisteredModel")
```

Line 137:
```python
    @patch("services.scoring.register_model.get_project_id")
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_authentication.py tests/test_register.py tests/test_register_model.py
git commit -m "refactor: update test imports for services/ directory"
```

---

### Task 12: Update README.md

**Files:**
- Modify: `README.md:29,33,48`

- [ ] **Step 1: Update project structure in README**

In `README.md`, replace the three service directory lines in the project structure:

Change line 29:
Old: `├── embeddings_service/        # Game embedding inference service (FastAPI)`
New: `├── services/                   # Production services (FastAPI)`

Change line 33:
Old: `├── scoring_service/           # Production scoring service (FastAPI)`
New: Remove this line (covered by services/ entry above)

Change line 48:
Old: `├── text_embeddings_service/   # Text embedding inference service`
New: Remove this line (covered by services/ entry above)

Add after the `services/` line:
```
│   ├── scoring/               # Scoring and prediction service
│   ├── game_embeddings/       # Game embedding inference service
│   └── text_embeddings/       # Text embedding inference service
```

- [ ] **Step 2: Update the Cloud Build reference**

In `README.md`, find and update (around line 328):

Old:
```
gcloud builds submit --config scoring_service/cloudbuild.yaml
```

New:
```
gcloud builds submit --config services/scoring/cloudbuild.yaml
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README for services/ directory structure"
```

---

### Task 13: Run tests and verify

- [ ] **Step 1: Run linting**

Run: `cd /Users/phenrickson/Documents/projects/bgg-predictive-models && uv run ruff check .`
Expected: No errors related to the moved services

- [ ] **Step 2: Run tests**

Run: `cd /Users/phenrickson/Documents/projects/bgg-predictive-models && uv run -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 3: Verify no stale references remain**

Run: `grep -r "scoring_service" --include="*.py" --include="*.yml" --include="*.yaml" --include="Makefile" --include="Dockerfile" --include="*.md" . | grep -v ".git/" | grep -v "docs/plans/" | grep -v "docs/specs/" | grep -v "__pycache__"`
Expected: No remaining references to `scoring_service` outside of docs/plans and docs/specs

Run: `grep -r "embeddings_service" --include="*.py" --include="*.yml" --include="*.yaml" --include="Makefile" --include="Dockerfile" --include="*.md" . | grep -v ".git/" | grep -v "docs/plans/" | grep -v "docs/specs/" | grep -v "__pycache__"`
Expected: No remaining references to `embeddings_service` outside of docs/plans and docs/specs

Run: `grep -r "text_embeddings_service" --include="*.py" --include="*.yml" --include="*.yaml" --include="Makefile" --include="Dockerfile" --include="*.md" . | grep -v ".git/" | grep -v "docs/plans/" | grep -v "docs/specs/" | grep -v "__pycache__"`
Expected: No remaining references to `text_embeddings_service` outside of docs/plans and docs/specs

- [ ] **Step 4: Fix any issues found and commit**

If any stale references or test failures are found, fix them and commit:
```bash
git add -A
git commit -m "fix: resolve remaining service path references"
```
