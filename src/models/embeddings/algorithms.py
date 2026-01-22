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
        n_neighbors: int = 100,
        min_dist: float = 0.5,
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
        patience: int = 10,
        min_delta: float = 1e-6,
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
            patience: Early stopping patience (epochs without improvement).
            min_delta: Minimum change to qualify as improvement.
            **kwargs: Additional parameters.
        """
        super().__init__(embedding_dim)
        self.hidden_layers = hidden_layers or [512, 256, 128]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        self.encoder = None
        self.decoder = None
        self.input_dim = None
        self.training_history = []
        self.early_stopped = False
        self.stopped_epoch = None

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
        Uses early stopping if loss doesn't improve for `patience` epochs.
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

        # Early stopping tracking
        best_loss = float('inf')
        epochs_without_improvement = 0
        best_encoder_state = None
        best_decoder_state = None

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

            # Early stopping check
            if avg_loss < best_loss - self.min_delta:
                best_loss = avg_loss
                epochs_without_improvement = 0
                # Save best model state
                best_encoder_state = {k: v.clone() for k, v in self.encoder.state_dict().items()}
                best_decoder_state = {k: v.clone() for k, v in self.decoder.state_dict().items()}
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

            # Early stopping trigger
            if epochs_without_improvement >= self.patience:
                self.early_stopped = True
                self.stopped_epoch = epoch + 1
                logger.info(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {self.patience} epochs)"
                )
                # Restore best model
                if best_encoder_state is not None:
                    self.encoder.load_state_dict(best_encoder_state)
                    self.decoder.load_state_dict(best_decoder_state)
                break

        self.encoder.eval()
        self.decoder.eval()
        self.is_fitted = True

        epochs_trained = len(self.training_history)
        final_msg = f"Autoencoder trained for {epochs_trained} epochs"
        if self.early_stopped:
            final_msg += f" (early stopped, best loss: {best_loss:.6f})"
        else:
            final_msg += f", final loss: {self.training_history[-1]:.6f}"
        logger.info(final_msg)

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
        metrics = {
            "final_loss": self.training_history[-1] if self.training_history else None,
            "best_loss": min(self.training_history) if self.training_history else None,
            "epochs_trained": len(self.training_history),
            "hidden_layers": self.hidden_layers,
            "embedding_dim": self.embedding_dim,
            "input_dim": self.input_dim,
            "early_stopped": self.early_stopped,
        }
        if self.early_stopped:
            metrics["stopped_epoch"] = self.stopped_epoch
        return metrics

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
            "patience": self.patience,
            "min_delta": self.min_delta,
            "early_stopped": self.early_stopped,
        }

        if self.early_stopped:
            artifacts["stopped_epoch"] = self.stopped_epoch

        # Add feature names if provided
        if feature_names is not None:
            artifacts["feature_names"] = feature_names

        return artifacts


class VAEEmbedding(BaseEmbeddingAlgorithm):
    """Variational Autoencoder for well-structured learned embeddings.

    Unlike regular autoencoders, VAEs enforce structure on the latent space
    by pushing it toward a standard Gaussian distribution. This prevents
    latent collapse and creates smooth, continuous embeddings.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_layers: Optional[List[int]] = None,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        beta: float = 1.0,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        patience: int = 10,
        min_delta: float = 1e-6,
        **kwargs,
    ):
        """Initialize VAE embedding.

        Args:
            embedding_dim: Size of the latent space (embedding dimension).
            hidden_layers: List of hidden layer sizes for encoder/decoder.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            learning_rate: Learning rate for optimizer.
            beta: Weight for KL divergence loss (beta-VAE). Higher = more regularization.
                  beta=1.0 is standard VAE, beta>1 pushes harder toward Gaussian.
            dropout: Dropout rate for regularization.
            use_batch_norm: Whether to use batch normalization.
            patience: Early stopping patience (epochs without improvement).
            min_delta: Minimum change to qualify as improvement.
            **kwargs: Additional parameters.
        """
        super().__init__(embedding_dim)
        self.hidden_layers = hidden_layers or [512, 256, 128]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.patience = patience
        self.min_delta = min_delta

        self.encoder = None
        self.fc_mu = None
        self.fc_logvar = None
        self.decoder = None
        self.input_dim = None
        self.training_history = []
        self.kl_history = []
        self.recon_history = []
        self.early_stopped = False
        self.stopped_epoch = None

    def _build_model(self, input_dim: int):
        """Build the VAE architecture."""
        try:
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch is required for VAEEmbedding")

        self.input_dim = input_dim

        # Build encoder (outputs to hidden, then split to mu and logvar)
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            if self.dropout > 0:
                encoder_layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Separate heads for mean and log-variance
        self.fc_mu = nn.Linear(prev_dim, self.embedding_dim)
        self.fc_logvar = nn.Linear(prev_dim, self.embedding_dim)

        # Build decoder
        decoder_layers = []
        prev_dim = self.embedding_dim
        for hidden_dim in reversed(self.hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.ReLU())
            if self.dropout > 0:
                decoder_layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def _reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon."""
        import torch

        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def _kl_divergence(self, mu, logvar):
        """KL divergence from N(mu, sigma) to N(0, 1)."""
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

    def fit(self, X: pd.DataFrame) -> "VAEEmbedding":
        """Train the VAE on input data.

        Uses early stopping if loss doesn't improve for `patience` epochs.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X_array = X.values if hasattr(X, "values") else X
        self._build_model(X_array.shape[1])

        X_tensor = torch.FloatTensor(X_array)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Collect all parameters
        params = (
            list(self.encoder.parameters())
            + list(self.fc_mu.parameters())
            + list(self.fc_logvar.parameters())
            + list(self.decoder.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        self.encoder.train()
        self.decoder.train()

        # Early stopping tracking
        best_loss = float('inf')
        epochs_without_improvement = 0
        best_state = None

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0

            for (batch_X,) in dataloader:
                optimizer.zero_grad()

                # Encode
                h = self.encoder(batch_X)
                mu = self.fc_mu(h)
                logvar = self.fc_logvar(h)

                # Reparameterize
                z = self._reparameterize(mu, logvar)

                # Decode
                x_recon = self.decoder(z)

                # Losses
                recon_loss = nn.functional.mse_loss(x_recon, batch_X, reduction="mean")
                kl_loss = self._kl_divergence(mu, logvar)
                loss = recon_loss + self.beta * kl_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()

            n_batches = len(dataloader)
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_kl = epoch_kl / n_batches

            self.training_history.append(avg_loss)
            self.recon_history.append(avg_recon)
            self.kl_history.append(avg_kl)

            # Early stopping check
            if avg_loss < best_loss - self.min_delta:
                best_loss = avg_loss
                epochs_without_improvement = 0
                # Save best model state
                best_state = {
                    'encoder': {k: v.clone() for k, v in self.encoder.state_dict().items()},
                    'fc_mu': {k: v.clone() for k, v in self.fc_mu.state_dict().items()},
                    'fc_logvar': {k: v.clone() for k, v in self.fc_logvar.state_dict().items()},
                    'decoder': {k: v.clone() for k, v in self.decoder.state_dict().items()},
                }
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs}, "
                    f"Loss: {avg_loss:.6f} (Recon: {avg_recon:.6f}, KL: {avg_kl:.6f})"
                )

            # Early stopping trigger
            if epochs_without_improvement >= self.patience:
                self.early_stopped = True
                self.stopped_epoch = epoch + 1
                logger.info(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {self.patience} epochs)"
                )
                # Restore best model
                if best_state is not None:
                    self.encoder.load_state_dict(best_state['encoder'])
                    self.fc_mu.load_state_dict(best_state['fc_mu'])
                    self.fc_logvar.load_state_dict(best_state['fc_logvar'])
                    self.decoder.load_state_dict(best_state['decoder'])
                break

        self.encoder.eval()
        self.decoder.eval()
        self.is_fitted = True

        epochs_trained = len(self.training_history)
        final_msg = f"VAE trained for {epochs_trained} epochs"
        if self.early_stopped:
            final_msg += f" (early stopped, best loss: {best_loss:.6f})"
        else:
            final_msg += (
                f", final loss: {self.training_history[-1]:.6f} "
                f"(recon: {self.recon_history[-1]:.6f}, kl: {self.kl_history[-1]:.6f})"
            )
        logger.info(final_msg)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Generate embeddings using the trained encoder (returns mu, the mean)."""
        import torch

        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")

        X_array = X.values if hasattr(X, "values") else X
        X_tensor = torch.FloatTensor(X_array)

        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder(X_tensor)
            mu = self.fc_mu(h)

        return mu.numpy()

    def get_metrics(self) -> Dict[str, Any]:
        """Get VAE training metrics."""
        if not self.is_fitted:
            return {}
        metrics = {
            "final_loss": self.training_history[-1] if self.training_history else None,
            "best_loss": min(self.training_history) if self.training_history else None,
            "final_recon_loss": self.recon_history[-1] if self.recon_history else None,
            "final_kl_loss": self.kl_history[-1] if self.kl_history else None,
            "epochs_trained": len(self.training_history),
            "hidden_layers": self.hidden_layers,
            "embedding_dim": self.embedding_dim,
            "input_dim": self.input_dim,
            "beta": self.beta,
            "early_stopped": self.early_stopped,
        }
        if self.early_stopped:
            metrics["stopped_epoch"] = self.stopped_epoch
        return metrics

    def get_artifacts(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get VAE artifacts including training history."""
        if not self.is_fitted:
            return {}

        artifacts = {
            "training_history": self.training_history,
            "recon_history": self.recon_history,
            "kl_history": self.kl_history,
            "hidden_layers": self.hidden_layers,
            "embedding_dim": self.embedding_dim,
            "input_dim": self.input_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "beta": self.beta,
            "dropout": self.dropout,
            "use_batch_norm": self.use_batch_norm,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "early_stopped": self.early_stopped,
        }

        if self.early_stopped:
            artifacts["stopped_epoch"] = self.stopped_epoch

        if feature_names is not None:
            artifacts["feature_names"] = feature_names

        return artifacts


def create_embedding_algorithm(
    algorithm: str, embedding_dim: int, **kwargs
) -> BaseEmbeddingAlgorithm:
    """Factory function to create embedding algorithms.

    Args:
        algorithm: Algorithm name ('pca', 'svd', 'umap', 'autoencoder', 'vae').
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
        "vae": VAEEmbedding,
    }

    if algorithm not in algorithms:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Available: {list(algorithms.keys())}"
        )

    return algorithms[algorithm](embedding_dim=embedding_dim, **kwargs)
