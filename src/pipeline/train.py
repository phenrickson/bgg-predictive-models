"""Training entry point for outcome models.

Usage:
    uv run -m src.pipeline.train --model hurdle --experiment my-experiment
    uv run -m src.pipeline.train --model complexity --experiment my-experiment --use-embeddings
"""

from src.models.outcomes.train import main

if __name__ == "__main__":
    main()
