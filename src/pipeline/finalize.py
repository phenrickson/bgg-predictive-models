"""Finalization entry point for outcome models.

Usage:
    uv run -m src.pipeline.finalize --model hurdle --experiment my-experiment
    uv run -m src.pipeline.finalize --model complexity --experiment my-experiment --use-embeddings
"""

from src.models.outcomes.train import main_finalize as main

if __name__ == "__main__":
    main()
