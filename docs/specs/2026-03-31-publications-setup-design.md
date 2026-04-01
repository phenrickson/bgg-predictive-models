# Publications Setup Design

## Goal

Create infrastructure for producing publication-worthy documents that detail and showcase the work in this project. Documents range from technical methodology write-ups to polished portfolio pieces with interactive visualizations.

## Structure

```
publications/
├── _quarto.yml          # Quarto project configuration
├── living/              # Re-renderable documents (execute code at render time)
├── artifacts/           # Point-in-time snapshots (rendered output is committed)
└── _output/             # Render output directory (gitignored)
```

## Two Document Patterns

### Living Documents
- Re-render with fresh data on demand
- Code cells execute at render time (pulling from BigQuery, loading models, etc.)
- Rendered output goes to `_output/` (gitignored)
- Examples: current model evaluation dashboards, latest prediction distributions

### Point-in-Time Artifacts
- Rendered once as a snapshot, then the output is committed
- Source `.qmd` lives in `artifacts/` alongside its rendered output
- Examples: "Model performance as of March 2026", release-specific methodology docs

## Quarto Configuration

**`_quarto.yml`** — Minimal project config:
- `type: default` (not website or book — maximum flexibility)
- Output format: `html` with `self-contained: true` for portable HTML files
- Optional PDF output via `format: pdf`
- Python execution engine (Jupyter kernel)

Self-contained HTML is the key output format. These files:
- Can be served via GitHub Pages
- Can be loaded into a frontend application
- Can be shared as standalone files
- Work without any external dependencies

## Dependencies

- `quarto` CLI (installed separately, not a Python package)
- Python packages already in the project (pandas, plotly, matplotlib, etc.) are available to document code cells
- No new Python dependencies required

## Makefile Targets

```makefile
# Render all living documents
make publications

# Render a specific document
make publication FILE=living/model-evaluation.qmd

# Render artifacts (point-in-time)
make publication-artifact FILE=artifacts/2026-03-methodology.qmd
```

## Gitignore

```
publications/_output/
```

Artifact rendered outputs are committed intentionally. Living document outputs are not.

## Out of Scope

- GitHub Pages deployment automation (future work)
- Frontend integration for loading rendered docs (future work)
- Migrating existing markdown docs to Quarto
- Quarto website or book project structure (can upgrade later if needed)

## Verification

- `quarto render publications/` succeeds
- Living documents produce self-contained HTML in `_output/`
- Artifact documents produce output alongside their source
- Makefile targets work
