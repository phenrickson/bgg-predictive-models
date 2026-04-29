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

# Copy required files
COPY pyproject.toml .
COPY uv.lock .
COPY services/__init__.py /app/services/__init__.py
COPY services/game_embeddings/ /app/services/game_embeddings/
COPY services/scoring/auth.py /app/services/scoring/auth.py
COPY services/scoring/__init__.py /app/services/scoring/__init__.py
COPY src/ /app/src/

# Copy config files
COPY config/ /app/config/
COPY config.yaml /app/config.yaml

# Copy credentials (will be created by GitHub Actions)
COPY credentials/service-account-key.json /app/credentials/

# Create necessary directories
RUN mkdir -p data/predictions models/experiments

# Create virtual environment and install dependencies using lock file
RUN uv venv && uv sync

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account-key.json

# Expose port for FastAPI
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "services.game_embeddings.main:app", "--host", "0.0.0.0", "--port", "8080"]
