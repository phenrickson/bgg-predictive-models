"""Pipeline entry points for training, finalizing, and scoring models."""

from src.pipeline.train import main as train
from src.pipeline.finalize import main as finalize
from src.pipeline.score import main as score

__all__ = ["train", "finalize", "score"]
