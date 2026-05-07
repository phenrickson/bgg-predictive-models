# Reports image: layered on top of the collections image, which already
# carries Python 3.12 + uv + project deps + src/ + config/. We add Quarto
# and the reports/ tree, then point the entrypoint at the renderer.
#
# The base tag uses :prod so this image is always in sync with whatever
# was last deployed for collections — the small consistency lag (when
# collections rebuilds first, this image picks up the new base on its
# next build) is acceptable.

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

# The reports source. Everything else (src/, config/, pyproject.toml,
# uv.lock) is already in the base image.
COPY reports/ /app/reports/

# Re-sync deps in case the reports build adds anything not already in
# the base — uv is incremental, so this is a no-op when nothing changed.
RUN uv sync

# QUARTO_PYTHON points the Quarto kernel at the same venv we use for
# everything else.
ENV QUARTO_PYTHON=/app/.venv/bin/python

# Default entrypoint: invoke the renderer. Override args at `docker run`:
#   docker run image --username rahdo --source gs://... --output-dir /out
ENTRYPOINT ["python", "-m", "reports.render"]
