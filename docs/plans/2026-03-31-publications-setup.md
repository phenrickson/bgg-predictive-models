# Publications Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a `publications/` directory with Quarto infrastructure for producing publication-worthy documents, supporting both living (re-renderable) and point-in-time artifact patterns.

**Architecture:** Minimal Quarto project (`type: default`) with subdirectories for living docs and artifacts. Self-contained HTML output for maximum portability. Makefile targets for rendering. Quarto 1.8.25 is already installed on the system.

**Tech Stack:** Quarto, Python (Jupyter kernel), Make

---

## File Structure

**Files created:**
- `publications/_quarto.yml` — Quarto project configuration
- `publications/living/.gitkeep` — Placeholder for living documents directory
- `publications/artifacts/.gitkeep` — Placeholder for artifacts directory
- `publications/living/example.qmd` — Example living document to verify setup works

**Files modified:**
- `.gitignore` — Add `publications/_output/`
- `Makefile` — Add publication rendering targets

---

### Task 1: Create publications directory structure

**Files:**
- Create: `publications/_quarto.yml`
- Create: `publications/living/.gitkeep`
- Create: `publications/artifacts/.gitkeep`

- [ ] **Step 1: Create the directory structure**

```bash
mkdir -p publications/living publications/artifacts
```

- [ ] **Step 2: Create _quarto.yml**

Create `publications/_quarto.yml`:

```yaml
project:
  type: default
  output-dir: _output

format:
  html:
    theme: cosmo
    toc: true
    toc-depth: 3
    code-fold: true
    self-contained: true

execute:
  engine: jupyter
  kernel: python3
```

- [ ] **Step 3: Create .gitkeep files**

```bash
touch publications/living/.gitkeep
touch publications/artifacts/.gitkeep
```

- [ ] **Step 4: Commit**

```bash
git add publications/
git commit -m "feat: create publications directory with Quarto config"
```

---

### Task 2: Add .gitignore entry for _output

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add publications/_output/ to .gitignore**

Append to `.gitignore`:

```
# Quarto rendered output (living documents)
publications/_output/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore publications/_output"
```

---

### Task 3: Create example living document

**Files:**
- Create: `publications/living/example.qmd`

- [ ] **Step 1: Create the example document**

Create `publications/living/example.qmd`:

````markdown
---
title: "Example Living Document"
author: "Phil Henrickson"
date: today
format:
  html:
    self-contained: true
    code-fold: true
---

## Overview

This is an example living document for the BGG Predictive Models project. Living documents re-render with fresh data each time they are built.

## Setup

```{python}
import sys
from pathlib import Path

# Add project root to path
project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root))
```

## Example: Project Configuration

```{python}
import yaml

config_path = project_root / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

print(f"Project: {config.get('ml_project', {}).get('project_id', 'N/A')}")
print(f"Current year: {config.get('years', {}).get('current', 'N/A')}")
print(f"Model types: {list(config.get('models', {}).keys())}")
```

## Next Steps

Replace this example with real analysis documents. See `publications/artifacts/` for point-in-time snapshots.
````

- [ ] **Step 2: Commit**

```bash
git add publications/living/example.qmd
git commit -m "feat: add example living document"
```

---

### Task 4: Add Makefile targets

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add publication targets to Makefile help**

In the `Makefile`, add these lines to the help echo block (after the streamlit-test line, around line 47):

```makefile
	@echo '  make publications                Render all living publications'
	@echo '  make publication                 Render a single publication (FILE=path/to/doc.qmd)'
	@echo '  make publication-artifact        Render a point-in-time artifact (FILE=path/to/doc.qmd)'
```

- [ ] **Step 2: Add publication targets to Makefile**

Add the following at the end of the `Makefile` (after the streamlit-stop target):

```makefile

# Publications (Quarto)
.PHONY: publications publication publication-artifact

publications:  ## Render all living publications
	cd publications && quarto render living/

publication:  ## Render a single publication (FILE=living/example.qmd)
	cd publications && quarto render $(FILE)

publication-artifact:  ## Render a point-in-time artifact in place (FILE=artifacts/example.qmd)
	cd publications && quarto render $(FILE) --output-dir $(dir $(FILE))
```

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat: add Makefile targets for Quarto publications"
```

---

### Task 5: Verify Quarto rendering works

- [ ] **Step 1: Render the example document**

Run: `cd /Users/phenrickson/Documents/projects/bgg-predictive-models && make publications`
Expected: Quarto renders `publications/living/example.qmd` to `publications/_output/example.html`

- [ ] **Step 2: Verify the output exists**

Run: `ls publications/_output/`
Expected: `example.html` (or similar)

- [ ] **Step 3: Verify the output is self-contained HTML**

Run: `head -5 publications/_output/example.html`
Expected: HTML document headers (confirms rendering succeeded)

- [ ] **Step 4: Verify the output is gitignored**

Run: `cd /Users/phenrickson/Documents/projects/bgg-predictive-models && git status publications/`
Expected: `_output/` directory should NOT appear in untracked files

- [ ] **Step 5: Verify single file rendering**

Run: `cd /Users/phenrickson/Documents/projects/bgg-predictive-models && make publication FILE=living/example.qmd`
Expected: Renders successfully

- [ ] **Step 6: Fix any issues and commit if needed**

If any issues were found, fix and commit:
```bash
git add -A
git commit -m "fix: resolve publications rendering issues"
```
