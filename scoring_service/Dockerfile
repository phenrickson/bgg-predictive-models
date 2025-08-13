FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY src /app/src
COPY main.py .

# Create virtual environment and install dependencies
RUN uv venv /venv
RUN /venv/bin/uv pip install -e .

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH="/venv/bin:$PATH"

# Expose port for FastAPI
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
