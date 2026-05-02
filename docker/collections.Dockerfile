FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies including OpenMP for LightGBM
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        libgomp1 \
        libomp-dev \
        make && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Copy dependency files first for layer caching
COPY pyproject.toml .
COPY uv.lock .

# Collection service source
COPY services/collections/ /app/services/collections/

# Scoring service auth (re-exported by collections/auth.py)
COPY services/scoring/__init__.py /app/services/scoring/__init__.py
COPY services/scoring/auth.py /app/services/scoring/auth.py

# Shared project source
COPY src/ /app/src/

# Copy config files
COPY config/ /app/config/
COPY config.yaml /app/config.yaml

# Copy credentials (will be created by GitHub Actions)
COPY credentials/service-account-key.json /app/credentials/

# Create virtual environment and install dependencies using lock file
RUN uv venv && uv sync

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account-key.json

# Expose port for FastAPI
EXPOSE 8080

# Note: import path is services.collections.main (not bare "main") because
# we keep the package layout intact rather than flattening like scoring does.
CMD ["uvicorn", "services.collections.main:app", "--host", "0.0.0.0", "--port", "8080"]
