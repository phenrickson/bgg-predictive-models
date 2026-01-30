"""Scoring entry point for outcome models.

Usage:
    uv run -m src.pipeline.score --model hurdle
    uv run -m src.pipeline.score --model complexity
"""

from src.models.score import main

if __name__ == "__main__":
    main()
