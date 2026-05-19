# Reports image: uses the collections image only for its OS / Python /
# uv / installed-dependency layers, then overlays the CURRENT project
# source. We do NOT inherit src/ from the base: docker-collections-build
# only triggers on services/collections paths, so collections:prod is
# frozen w.r.t. src/reports/** changes — relying on it for src/ silently
# ships stale code (caused a `ModuleNotFoundError: No module named
# 'src.reports'` at render time). This image therefore copies src/,
# config/, and the dependency lock fresh from the building commit.

ARG BASE_IMAGE=us-central1-docker.pkg.dev/bgg-predictive-models/bgg-predictive-models/collections:prod
FROM ${BASE_IMAGE}

# Install Quarto. Pinned to a recent stable version; bump explicitly
# when we want to upgrade rather than tracking floating "latest".
ARG QUARTO_VERSION=1.6.42
RUN curl -L -o /tmp/quarto.deb \
        "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb" \
 && apt-get update --allow-releaseinfo-change \
 && apt-get install -y /tmp/quarto.deb \
 && rm /tmp/quarto.deb \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Overlay the current project source fresh from the building commit
# (mirrors collections.Dockerfile's copy set). Do not rely on the base
# for these — see the header comment.
# README.md is required: pyproject.toml has `readme = "README.md"`, so
# `uv sync` (which builds this project) fails without it.
COPY pyproject.toml uv.lock README.md /app/
COPY src/ /app/src/
COPY config/ /app/config/
COPY config.yaml /app/config.yaml
COPY reports/ /app/reports/

# Reconcile deps against the (possibly newer) lock copied above. The
# base image's venv supplies the heavy wheels; this is incremental and
# a near-no-op when nothing changed.
RUN uv sync

# QUARTO_PYTHON points the Quarto kernel at the same venv we use for
# everything else.
ENV QUARTO_PYTHON=/app/.venv/bin/python

# Default entrypoint: invoke the renderer. Override args at `docker run`:
#   docker run image --username rahdo --source gs://... --output-dir /out
ENTRYPOINT ["python", "-m", "reports.render"]
