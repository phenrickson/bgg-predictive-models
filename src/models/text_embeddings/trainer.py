"""Trainer for text embedding models."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from src.models.experiments import ExperimentTracker
from src.models.training import load_data
from src.utils.config import Config, load_config

from .base import WordEmbedding
from .document import DocumentEmbedding
from .pmi import PMIEmbedding

logger = logging.getLogger(__name__)

ALGORITHMS = {
    "pmi": PMIEmbedding,
}


class TextEmbeddingTrainer:
    """Orchestrates training of text embedding models."""

    def __init__(
        self,
        config: Optional[Config] = None,
        output_dir: str = "./models/experiments",
    ):
        """Initialize the trainer.

        Args:
            config: Configuration object. If None, loads from config.yaml.
            output_dir: Directory for storing experiment artifacts.
        """
        self.config = config or load_config()
        self.output_dir = Path(output_dir)
        self.tracker = ExperimentTracker("text_embeddings", str(self.output_dir))

    def get_config_defaults(self) -> Dict[str, Any]:
        """Get default settings from config.yaml."""
        if self.config.text_embeddings is None:
            return {}

        te = self.config.text_embeddings
        return {
            "algorithm": te.algorithm,
            "embedding_dim": te.embedding_dim,
            "experiment_name": te.experiment_name,
            "document_method": te.document_method,
        }

    def get_algorithm_params(self, algorithm: str) -> Dict[str, Any]:
        """Get algorithm-specific parameters from config.yaml."""
        if self.config.text_embeddings is None:
            return {}

        return self.config.text_embeddings.get_algorithm_params(algorithm)

    def train(
        self,
        documents: List[str],
        algorithm: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        experiment_name: Optional[str] = None,
        description: Optional[str] = None,
        document_method: Optional[str] = None,
        algorithm_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train a text embedding model.

        Args:
            documents: List of text documents to train on.
            algorithm: Algorithm to use ('pmi'). If None, uses config.yaml.
            embedding_dim: Dimension of word embeddings. If None, uses config.yaml.
            experiment_name: Name for the experiment. If None, uses config.yaml.
            description: Description of the experiment.
            document_method: Method for aggregating word vectors. If None, uses config.yaml.
            algorithm_params: Additional parameters for the algorithm. Merged with config.yaml.

        Returns:
            Dictionary with training info and metrics.
        """
        # Get defaults from config
        defaults = self.get_config_defaults()

        # Apply defaults where args are None
        algorithm = algorithm or defaults.get("algorithm", "pmi")
        embedding_dim = embedding_dim or defaults.get("embedding_dim", 100)
        experiment_name = experiment_name or defaults.get("experiment_name", "text-embeddings")
        document_method = document_method or defaults.get("document_method", "mean")

        # Get algorithm params from config, then override with provided params
        config_algo_params = self.get_algorithm_params(algorithm)
        if algorithm_params:
            config_algo_params.update(algorithm_params)
        algorithm_params = config_algo_params

        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(ALGORITHMS.keys())}")

        logger.info(f"Training {algorithm} text embeddings")
        logger.info(f"  Documents: {len(documents)}")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Document method: {document_method}")

        # Create word embedding model
        word_model = ALGORITHMS[algorithm](
            embedding_dim=embedding_dim,
            **algorithm_params,
        )

        # Fit word embeddings
        logger.info("Fitting word embeddings...")
        word_model.fit(documents)

        # Create document embedding
        logger.info("Fitting document embeddings...")
        doc_model = DocumentEmbedding(word_model, method=document_method)
        doc_model.fit(documents)

        # Create experiment
        experiment = self.tracker.create_experiment(
            name=experiment_name,
            description=description or f"{algorithm.upper()} text embeddings",
            metadata={
                "algorithm": algorithm,
                "embedding_dim": embedding_dim,
                "document_method": document_method,
                "n_documents": len(documents),
                "vocab_size": len(word_model.get_vocabulary()),
            },
            config={
                "algorithm_params": algorithm_params,
            },
        )

        # Save models
        word_model_path = experiment.exp_dir / "word_embedding.pkl"
        with open(word_model_path, "wb") as f:
            pickle.dump(word_model, f)
        logger.info(f"Saved word embedding to {word_model_path}")

        doc_model_path = experiment.exp_dir / "document_embedding.pkl"
        with open(doc_model_path, "wb") as f:
            pickle.dump(doc_model, f)
        logger.info(f"Saved document embedding to {doc_model_path}")

        # Log parameters
        experiment.log_parameters({
            "algorithm": algorithm,
            "embedding_dim": embedding_dim,
            "document_method": document_method,
            **algorithm_params,
        })

        # Log model info
        experiment.log_model_info({
            "algorithm": algorithm,
            "embedding_dim": embedding_dim,
            "vocab_size": len(word_model.get_vocabulary()),
            "n_documents": len(documents),
        })

        # Save interpretability artifacts
        logger.info("Saving interpretability artifacts...")
        self._save_artifacts(word_model, experiment)

        logger.info(f"Experiment saved to {experiment.exp_dir}")

        return {
            "experiment_dir": str(experiment.exp_dir),
            "vocab_size": len(word_model.get_vocabulary()),
            "embedding_dim": embedding_dim,
        }

    def _save_artifacts(self, word_model: WordEmbedding, experiment) -> None:
        """Save interpretability artifacts for the trained model."""
        exp_dir = experiment.exp_dir

        # 1. Vocabulary statistics
        self._save_vocab_stats(word_model, exp_dir)

        # 2. Word similarities for common game terms
        self._save_word_similarities(word_model, exp_dir)

        # 3. SVD analysis (if available)
        self._save_svd_analysis(word_model, exp_dir)

        # 4. Component loadings
        self._save_component_loadings(word_model, exp_dir)

        # 5. 2D projection for visualization
        self._save_2d_projection(word_model, exp_dir)

    def _save_vocab_stats(self, word_model: WordEmbedding, exp_dir: Path) -> None:
        """Save vocabulary statistics."""
        vocab = word_model.get_vocabulary()

        # Get word counts if available (PMI model)
        word_counts = {}
        if hasattr(word_model, "get_word_counts"):
            word_counts = word_model.get_word_counts()

        # Top 100 words by frequency
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:100]

        stats = {
            "vocab_size": len(vocab),
            "top_words": [{"word": w, "count": c} for w, c in top_words],
        }

        with open(exp_dir / "vocab_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved vocabulary stats to {exp_dir / 'vocab_stats.json'}")

    def _save_word_similarities(self, word_model: WordEmbedding, exp_dir: Path) -> None:
        """Save similar words for common game-related terms."""
        seed_words = [
            "strategy", "card", "dice", "cooperative", "war", "fantasy",
            "economic", "trading", "building", "combat", "adventure",
            "puzzle", "party", "family", "abstract", "miniatures",
            "deck", "worker", "area", "control", "auction", "drafting",
        ]

        similarities = {}
        for word in seed_words:
            similar = word_model.most_similar(word, n=10)
            if similar:
                similarities[word] = [
                    {"word": w, "similarity": round(s, 4)} for w, s in similar
                ]

        with open(exp_dir / "word_similarities.json", "w") as f:
            json.dump(similarities, f, indent=2)
        logger.info(f"Saved word similarities to {exp_dir / 'word_similarities.json'}")

    def _save_svd_analysis(self, word_model: WordEmbedding, exp_dir: Path) -> None:
        """Save SVD analysis (singular values, variance explained)."""
        if not hasattr(word_model, "get_singular_values"):
            return

        singular_values = word_model.get_singular_values()
        if singular_values is None:
            return

        # Compute explained variance ratio
        variance = singular_values ** 2
        total_variance = variance.sum()
        explained_ratio = (variance / total_variance).tolist() if total_variance > 0 else []
        cumulative_ratio = np.cumsum(explained_ratio).tolist()

        analysis = {
            "singular_values": singular_values.tolist(),
            "explained_variance_ratio": explained_ratio,
            "cumulative_variance_ratio": cumulative_ratio,
        }

        with open(exp_dir / "svd_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Saved SVD analysis to {exp_dir / 'svd_analysis.json'}")

    def _save_component_loadings(self, word_model: WordEmbedding, exp_dir: Path) -> None:
        """Save top words for each component/dimension."""
        if not hasattr(word_model, "get_all_vectors"):
            return

        vectors = word_model.get_all_vectors()
        if vectors is None:
            return

        vocab = word_model.get_vocabulary()
        n_components = vectors.shape[1]

        rows = []
        for comp_idx in range(n_components):
            # Get component values for all words
            comp_values = vectors[:, comp_idx]

            # Top 50 positive loadings
            top_pos_idx = np.argsort(comp_values)[-50:][::-1]
            for idx in top_pos_idx:
                rows.append({
                    "component": comp_idx,
                    "word": vocab[idx],
                    "loading": float(comp_values[idx]),
                    "direction": "positive",
                })

            # Top 50 negative loadings
            top_neg_idx = np.argsort(comp_values)[:50]
            for idx in top_neg_idx:
                rows.append({
                    "component": comp_idx,
                    "word": vocab[idx],
                    "loading": float(comp_values[idx]),
                    "direction": "negative",
                })

        df = pl.DataFrame(rows)
        df.write_csv(exp_dir / "component_loadings.csv")
        logger.info(f"Saved component loadings to {exp_dir / 'component_loadings.csv'}")

    def _save_2d_projection(self, word_model: WordEmbedding, exp_dir: Path) -> None:
        """Save 2D PCA projection of word embeddings for visualization."""
        if not hasattr(word_model, "get_all_vectors"):
            return

        vectors = word_model.get_all_vectors()
        if vectors is None or vectors.shape[0] < 3:
            return

        vocab = word_model.get_vocabulary()

        # Get word counts for coloring
        word_counts = {}
        if hasattr(word_model, "get_word_counts"):
            word_counts = word_model.get_word_counts()

        # PCA to 2D
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(vectors)

        # Create dataframe
        df = pl.DataFrame({
            "word": vocab,
            "x": coords_2d[:, 0],
            "y": coords_2d[:, 1],
            "frequency": [word_counts.get(w, 0) for w in vocab],
        })

        df.write_parquet(exp_dir / "word_embeddings_2d.parquet")
        logger.info(f"Saved 2D projection to {exp_dir / 'word_embeddings_2d.parquet'}")

        # Also save PCA metadata
        pca_info = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "n_words": len(vocab),
        }
        with open(exp_dir / "pca_2d_info.json", "w") as f:
            json.dump(pca_info, f, indent=2)


class TextEmbeddingGenerator:
    """Generates embeddings using a trained model."""

    def __init__(
        self,
        experiment_name: str,
        version: Optional[int] = None,
        experiments_dir: str = "./models/experiments",
    ):
        """Initialize the generator.

        Args:
            experiment_name: Name of the experiment to load.
            version: Specific version to load. If None, loads latest.
            experiments_dir: Directory containing experiments.
        """
        self.tracker = ExperimentTracker("text_embeddings", experiments_dir)
        self.experiment = self.tracker.load_experiment(experiment_name, version)

        # Load models
        self._load_models()

    def _load_models(self):
        """Load word and document embedding models."""
        word_path = self.experiment.exp_dir / "word_embedding.pkl"
        doc_path = self.experiment.exp_dir / "document_embedding.pkl"

        if not word_path.exists():
            raise ValueError(f"Word embedding not found at {word_path}")

        with open(word_path, "rb") as f:
            self.word_model: WordEmbedding = pickle.load(f)

        if doc_path.exists():
            with open(doc_path, "rb") as f:
                self.doc_model: DocumentEmbedding = pickle.load(f)
        else:
            # Create default document embedding
            self.doc_model = DocumentEmbedding(self.word_model, method="mean")

        logger.info(f"Loaded text embedding model from {self.experiment.exp_dir}")

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Generate embeddings for documents.

        Args:
            documents: List of text documents.

        Returns:
            Array of shape (n_documents, embedding_dim).
        """
        return self.doc_model.transform(documents)

    def embed_document(self, document: str) -> np.ndarray:
        """Generate embedding for a single document.

        Args:
            document: Text document.

        Returns:
            Embedding vector.
        """
        return self.doc_model.transform([document])[0]

    def most_similar_words(self, word: str, n: int = 10) -> List[tuple[str, float]]:
        """Find words most similar to the given word.

        Args:
            word: Query word.
            n: Number of results.

        Returns:
            List of (word, similarity) tuples.
        """
        return self.word_model.most_similar(word, n)

    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get the embedding vector for a word.

        Args:
            word: The word to look up.

        Returns:
            Embedding vector, or None if not in vocabulary.
        """
        return self.word_model.get_vector(word)
