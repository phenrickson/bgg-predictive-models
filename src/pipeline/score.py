"""Scoring entry point for outcome models.

Usage:
    uv run -m src.pipeline.score --model-type hurdle --experiment my-experiment
    uv run -m src.pipeline.score --model-type complexity --experiment my-experiment
"""

from src.models.score import main

if __name__ == "__main__":
    main()
