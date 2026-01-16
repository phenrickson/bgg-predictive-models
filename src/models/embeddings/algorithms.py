"""Embedding algorithms for dimension reduction."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD

logger = logging.getLogger(__name__)


class BaseEmbeddingAlgorithm(ABC):
    """Abstract base class for embedding algorithms."""

    def __init__(self, embedding_dim: int, **kwargs):
        """Initialize the embedding algorithm.

        Args:
            embedding_dim: Target embedding dimension.
            **kwargs: Algorithm-specific parameters.
        """
        self.embedding_dim = embedding_dim
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "BaseEmbeddingAlgorithm":
        """Fit the dimension reduction model.

        Args:
            X: Input features DataFrame.

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data to embedding space.

        Args:
            X: Input features DataFrame.

        Returns:
            Embedding array of shape (n_samples, embedding_dim).
        """
        pass

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X: Input features DataFrame.

        Returns:
            Embedding array of shape (n_samples, embedding_dim).
        """
        self.fit(X)
        return self.transform(X)

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get algorithm-specific metrics after fitting.

        Returns:
            Dictionary of metrics (e.g., explained variance for PCA).
        """
        pass

    def get_artifacts(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get algorithm-specific artifacts for visualization and analysis.

        Args:
            feature_names: Optional list of input feature names.

        Returns:
            Dictionary of artifacts (e.g., components matrix for PCA).
        """
        return {}


class PCAEmbedding(BaseEmbeddingAlgorithm):
    """PCA-based embeddings using sklearn."""

    def __init__(self, embedding_dim: int, whiten: bool = True, **kwargs):
        """Initialize PCA embedding.

        Args:
            embedding_dim: Number of principal components.
            whiten: Whether to whiten the output.
            **kwargs: Additional PCA parameters.
        """
        super().__init__(embedding_dim)
        self.whiten = whiten
        self.model = PCA(n_components=embedding_dim, whiten=whiten, **kwargs)

    def fit(self, X: pd.DataFrame) -> "PCAEmbedding":
        """Fit PCA on the input data.

        Note: Data is assumed to be already centered/scaled by the preprocessor.
        """
        self.model.fit(X)
        self.is_fitted = True
        logger.info(
            f"PCA fitted with {self.embedding_dim} components, "
            f"explained variance: {self.model.explained_variance_ratio_.sum():.4f}"
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted PCA."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")
        return self.model.transform(X)

    def get_metrics(self) -> Dict[str, Any]:
        """Get PCA metrics including explained variance."""
        if not self.is_fitted:
            return {}
        return {
            "explained_variance_ratio": self.model.explained_variance_ratio_.tolist(),
            "total_explained_variance": float(
                self.model.explained_variance_ratio_.sum()
            ),
            "n_components": self.model.n_components_,
        }

    def get_artifacts(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get PCA artifacts including component loadings."""
        if not self.is_fitted:
            return {}

        artifacts = {
            "components": self.model.components_.tolist(),  # (n_components, n_features)
            "explained_variance": self.model.explained_variance_.tolist(),
            "explained_variance_ratio": self.model.explained_variance_ratio_.tolist(),
            "singular_values": self.model.singular_values_.tolist(),
            "mean": self.model.mean_.tolist(),
            "n_features_in": self.model.n_features_in_,
        }

        # Add feature names if provided
        if feature_names is not None:
            artifacts["feature_names"] = feature_names

            # Compute top features per component for easier interpretation
            top_features_per_component = []
            for i, component in enumerate(self.model.components_):
                # Get indices of top 10 features by absolute loading
                top_indices = np.argsort(np.abs(component))[-10:][::-1]
                top_features = [
                    {
                        "feature": feature_names[idx],
                        "loading": float(component[idx]),
                        "abs_loading": float(abs(component[idx])),
                    }
                    for idx in top_indices
                ]
                top_features_per_component.append(top_features)
            artifacts["top_features_per_component"] = top_features_per_component

        return artifacts


class SVDEmbedding(BaseEmbeddingAlgorithm):
    """TruncatedSVD-based embeddings for potentially sparse data."""

    def __init__(self, embedding_dim: int, n_iter: int = 5, **kwargs):
        """Initialize SVD embedding.

        Args:
            embedding_dim: Number of components.
            n_iter: Number of iterations for randomized SVD.
            **kwargs: Additional TruncatedSVD parameters.
        """
        super().__init__(embedding_dim)
        self.n_iter = n_iter
        self.model = TruncatedSVD(
            n_components=embedding_dim, n_iter=n_iter, random_state=42, **kwargs
        )

    def fit(self, X: pd.DataFrame) -> "SVDEmbedding":
        """Fit TruncatedSVD on the input data.

        Note: Data is assumed to be already centered/scaled by the preprocessor.
        """
        self.model.fit(X)
        self.is_fitted = True
        logger.info(
            f"SVD fitted with {self.embedding_dim} components, "
            f"explained variance: {self.model.explained_variance_ratio_.sum():.4f}"
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted SVD."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")
        return self.model.transform(X)

    def get_metrics(self) -> Dict[str, Any]:
        """Get SVD metrics including explained variance."""
        if not self.is_fitted:
            return {}
        return {
            "explained_variance_ratio": self.model.explained_variance_ratio_.tolist(),
            "total_explained_variance": float(
                self.model.explained_variance_ratio_.sum()
            ),
            "n_components": self.model.n_components,
        }

    def get_artifacts(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get SVD artifacts including component loadings."""
        if not self.is_fitted:
            return {}

        artifacts = {
            "components": self.model.components_.tolist(),  # (n_components, n_features)
            "explained_variance": self.model.explained_variance_.tolist(),
            "explained_variance_ratio": self.model.explained_variance_ratio_.tolist(),
            "singular_values": self.model.singular_values_.tolist(),
        }

        # Add feature names if provided
        if feature_names is not None:
            artifacts["feature_names"] = feature_names

            # Compute top features per component
            top_features_per_component = []
            for i, component in enumerate(self.model.components_):
                top_indices = np.argsort(np.abs(component))[-10:][::-1]
                top_features = [
                    {
                        "feature": feature_names[idx],
                        "loading": float(component[idx]),
                        "abs_loading": float(abs(component[idx])),
                    }
                    for idx in top_indices
                ]
                top_features_per_component.append(top_features)
            artifacts["top_features_per_component"] = top_features_per_component

        return artifacts


class UMAPEmbedding(BaseEmbeddingAlgorithm):
    """UMAP-based embeddings for non-linear dimension reduction."""

    def __init__(
        self,
        embedding_dim: int,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        **kwargs,
    ):
        """Initialize UMAP embedding.

        Args:
            embedding_dim: Target dimension.
            n_neighbors: Number of neighbors for local structure.
            min_dist: Minimum distance between points in embedding.
            metric: Distance metric to use.
            **kwargs: Additional UMAP parameters.
        """
        super().__init__(embedding_dim)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric

        try:
            from umap import UMAP

            self.model = UMAP(
                n_components=embedding_dim,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=42,
                **kwargs,
            )
        except ImportError:
            raise ImportError("umap-learn is required for UMAPEmbedding")

    def fit(self, X: pd.DataFrame) -> "UMAPEmbedding":
        """Fit UMAP on the input data.

        Note: Data is assumed to be already centered/scaled by the preprocessor.
        """
        self.model.fit(X)
        self.is_fitted = True
        logger.info(
            f"UMAP fitted with {self.embedding_dim} components, "
            f"n_neighbors={self.n_neighbors}, min_dist={self.min_dist}"
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted UMAP."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")
        return self.model.transform(X)

    def get_metrics(self) -> Dict[str, Any]:
        """Get UMAP metrics."""
        if not self.is_fitted:
            return {}
        return {
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "n_components": self.embedding_dim,
        }

    def get_artifacts(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get UMAP artifacts."""
        if not self.is_fitted:
            return {}

        artifacts = {
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "n_components": self.embedding_dim,
        }

        # Add feature names if provided
        if feature_names is not None:
            artifacts["feature_names"] = feature_names

        return artifacts


class AutoencoderEmbedding(BaseEmbeddingAlgorithm):
    """PyTorch autoencoder for learned embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_layers: Optional[List[int]] = None,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        dropout: float = 0.2,
        **kwargs,
    ):
        """Initialize Autoencoder embedding.

        Args:
            embedding_dim: Size of the bottleneck layer (embedding dimension).
            hidden_layers: List of hidden layer sizes for encoder.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            learning_rate: Learning rate for optimizer.
            dropout: Dropout rate for regularization.
            **kwargs: Additional parameters.
        """
        super().__init__(embedding_dim)
        self.hidden_layers = hidden_layers or [512, 256, 128]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.encoder = None
        self.decoder = None
        self.input_dim = None
        self.training_history = []

    def _build_model(self, input_dim: int):
        """Build the autoencoder architecture."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch is required for AutoencoderEmbedding")

        self.input_dim = input_dim

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_layers:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                ]
            )
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, self.embedding_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = self.embedding_dim
        for hidden_dim in reversed(self.hidden_layers):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                ]
            )
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def fit(self, X: pd.DataFrame) -> "AutoencoderEmbedding":
        """Train the autoencoder on input data.

        Note: Data is assumed to be already centered/scaled by the preprocessor.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X_array = X.values if hasattr(X, "values") else X
        self._build_model(X_array.shape[1])

        X_tensor = torch.FloatTensor(X_array)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate,
        )
        criterion = nn.MSELoss()

        self.encoder.train()
        self.decoder.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, _ in dataloader:
                optimizer.zero_grad()
                encoded = self.encoder(batch_X)
                decoded = self.decoder(encoded)
                loss = criterion(decoded, batch_X)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.training_history.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

        self.encoder.eval()
        self.decoder.eval()
        self.is_fitted = True

        logger.info(
            f"Autoencoder trained for {self.epochs} epochs, "
            f"final loss: {self.training_history[-1]:.6f}"
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Generate embeddings using the trained encoder."""
        import torch

        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")

        X_array = X.values if hasattr(X, "values") else X
        X_tensor = torch.FloatTensor(X_array)

        with torch.no_grad():
            embeddings = self.encoder(X_tensor).numpy()

        return embeddings

    def get_metrics(self) -> Dict[str, Any]:
        """Get autoencoder training metrics."""
        if not self.is_fitted:
            return {}
        return {
            "final_loss": self.training_history[-1] if self.training_history else None,
            "epochs_trained": len(self.training_history),
            "hidden_layers": self.hidden_layers,
            "embedding_dim": self.embedding_dim,
            "input_dim": self.input_dim,
        }

    def get_artifacts(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get autoencoder artifacts including training history."""
        if not self.is_fitted:
            return {}

        artifacts = {
            "training_history": self.training_history,
            "hidden_layers": self.hidden_layers,
            "embedding_dim": self.embedding_dim,
            "input_dim": self.input_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
        }

        # Add feature names if provided
        if feature_names is not None:
            artifacts["feature_names"] = feature_names

        return artifacts


def create_embedding_algorithm(
    algorithm: str, embedding_dim: int, **kwargs
) -> BaseEmbeddingAlgorithm:
    """Factory function to create embedding algorithms.

    Args:
        algorithm: Algorithm name ('pca', 'svd', 'umap', 'autoencoder').
        embedding_dim: Target embedding dimension.
        **kwargs: Algorithm-specific parameters.

    Returns:
        Configured embedding algorithm instance.

    Raises:
        ValueError: If algorithm is not recognized.
    """
    algorithms = {
        "pca": PCAEmbedding,
        "svd": SVDEmbedding,
        "umap": UMAPEmbedding,
        "autoencoder": AutoencoderEmbedding,
    }

    if algorithm not in algorithms:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Available: {list(algorithms.keys())}"
        )

    return algorithms[algorithm](embedding_dim=embedding_dim, **kwargs)
