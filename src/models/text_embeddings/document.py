"""Document embedding via word vector aggregation."""

import logging
from typing import List, Optional

import numpy as np

from .base import WordEmbedding
from .tokenizer import tokenize

logger = logging.getLogger(__name__)


class DocumentEmbedding:
    """Aggregate word embeddings into document embeddings."""

    def __init__(
        self,
        word_embedding: WordEmbedding,
        method: str = "mean",
        sif_alpha: float = 1e-3,
    ):
        """Initialize document embedding.

        Args:
            word_embedding: Fitted word embedding model.
            method: Aggregation method ('mean', 'tfidf', or 'sif').
            sif_alpha: SIF smoothing parameter (default 1e-3).
        """
        self.word_embedding = word_embedding
        self.method = method
        self.sif_alpha = sif_alpha

        if method not in ("mean", "tfidf", "sif"):
            raise ValueError(f"Unknown method: {method}. Use 'mean', 'tfidf', or 'sif'.")

        self._idf: Optional[dict] = None
        self._word_probs: Optional[dict] = None
        self._principal_component: Optional[np.ndarray] = None

    def fit(self, documents: List[str]) -> "DocumentEmbedding":
        """Fit document embedding.

        For tfidf: computes IDF weights.
        For sif: computes word probabilities and principal component.

        Args:
            documents: List of documents.

        Returns:
            Self for method chaining.
        """
        if self.method == "tfidf":
            self._compute_idf(documents)
        elif self.method == "sif":
            self._fit_sif(documents)
        return self

    def _compute_idf(self, documents: List[str]) -> None:
        """Compute IDF weights for vocabulary."""
        n_docs = len(documents)
        doc_freq = {}

        for doc in documents:
            tokens = set(tokenize(doc))
            for word in tokens:
                doc_freq[word] = doc_freq.get(word, 0) + 1

        self._idf = {
            word: np.log(n_docs / (df + 1)) for word, df in doc_freq.items()
        }

    def _fit_sif(self, documents: List[str]) -> None:
        """Fit SIF model: compute word probabilities and principal component.

        SIF (Smooth Inverse Frequency) weights words by a / (a + p(w))
        where p(w) is the word probability, then removes the first
        principal component (common discourse vector).
        """
        # Step 1: Compute word probabilities
        word_counts: dict = {}
        total_words = 0

        for doc in documents:
            tokens = tokenize(doc)
            for word in tokens:
                word_counts[word] = word_counts.get(word, 0) + 1
                total_words += 1

        self._word_probs = {
            word: count / total_words for word, count in word_counts.items()
        }

        logger.info(f"SIF: computed probabilities for {len(self._word_probs)} words")

        # Step 2: Compute weighted embeddings for all documents
        dim = self.word_embedding.get_embedding_dim()
        embeddings = np.zeros((len(documents), dim))

        for i, doc in enumerate(documents):
            embeddings[i] = self._sif_weighted_embedding(tokenize(doc))

        # Step 3: Compute first principal component
        # Center the embeddings
        mean_embedding = embeddings.mean(axis=0)
        centered = embeddings - mean_embedding

        # SVD to get first principal component
        # Use randomized SVD for efficiency on large corpora
        try:
            u, s, vt = np.linalg.svd(centered, full_matrices=False)
            self._principal_component = vt[0]  # First right singular vector
            logger.info("SIF: computed principal component for removal")
        except np.linalg.LinAlgError:
            logger.warning("SIF: SVD failed, skipping principal component removal")
            self._principal_component = None

    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to embedding vectors.

        Args:
            documents: List of documents.

        Returns:
            Array of shape (n_documents, embedding_dim).
        """
        dim = self.word_embedding.get_embedding_dim()
        embeddings = np.zeros((len(documents), dim))

        for i, doc in enumerate(documents):
            embeddings[i] = self._embed_document(doc)

        return embeddings

    def _embed_document(self, document: str) -> np.ndarray:
        """Embed a single document."""
        dim = self.word_embedding.get_embedding_dim()
        tokens = tokenize(document)

        if not tokens:
            return np.zeros(dim)

        if self.method == "mean":
            return self._mean_embedding(tokens)
        elif self.method == "tfidf":
            return self._tfidf_embedding(tokens)
        elif self.method == "sif":
            return self._sif_embedding(tokens)

        return np.zeros(dim)

    def _mean_embedding(self, tokens: List[str]) -> np.ndarray:
        """Simple average of word vectors."""
        dim = self.word_embedding.get_embedding_dim()
        vectors = []

        for word in tokens:
            vec = self.word_embedding.get_vector(word)
            if vec is not None:
                vectors.append(vec)

        if not vectors:
            return np.zeros(dim)

        return np.mean(vectors, axis=0)

    def _tfidf_embedding(self, tokens: List[str]) -> np.ndarray:
        """TF-IDF weighted average of word vectors."""
        dim = self.word_embedding.get_embedding_dim()

        if self._idf is None:
            raise ValueError("Must call fit() before using tfidf method")

        # Compute TF
        tf = {}
        for word in tokens:
            tf[word] = tf.get(word, 0) + 1
        max_tf = max(tf.values()) if tf else 1

        # Weighted sum
        weighted_sum = np.zeros(dim)
        total_weight = 0

        for word, count in tf.items():
            vec = self.word_embedding.get_vector(word)
            if vec is not None:
                # TF-IDF weight
                term_freq = 0.5 + 0.5 * (count / max_tf)  # Augmented TF
                idf = self._idf.get(word, 0)
                weight = term_freq * idf

                weighted_sum += weight * vec
                total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight

        return np.zeros(dim)

    def _sif_weighted_embedding(self, tokens: List[str]) -> np.ndarray:
        """Compute SIF-weighted average (before principal component removal)."""
        dim = self.word_embedding.get_embedding_dim()

        # If word_probs not fitted, fall back to uniform weighting (like mean)
        if self._word_probs is None:
            logger.warning("SIF: _word_probs is None, falling back to mean embedding")
            return self._mean_embedding(tokens)

        weighted_sum = np.zeros(dim)
        total_weight = 0
        words_with_vectors = 0

        for word in tokens:
            vec = self.word_embedding.get_vector(word)
            if vec is not None:
                words_with_vectors += 1
                # SIF weight: a / (a + p(w))
                # Use a small default probability for OOV words
                p_w = self._word_probs.get(word, self.sif_alpha)
                weight = self.sif_alpha / (self.sif_alpha + p_w)

                weighted_sum += weight * vec
                total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight

        return np.zeros(dim)

    def _sif_embedding(self, tokens: List[str]) -> np.ndarray:
        """SIF embedding with principal component removal."""
        embedding = self._sif_weighted_embedding(tokens)

        # Check if we got a valid embedding before PC removal
        pre_removal_norm = np.linalg.norm(embedding)
        if pre_removal_norm < 1e-10:
            logger.warning(
                f"SIF: weighted embedding has near-zero norm ({pre_removal_norm:.6f}) "
                f"before PC removal. tokens={len(tokens)}"
            )

        # Remove projection onto first principal component
        if self._principal_component is not None:
            projection = np.dot(embedding, self._principal_component)
            embedding = embedding - projection * self._principal_component

            post_removal_norm = np.linalg.norm(embedding)
            if pre_removal_norm > 1e-6 and post_removal_norm < 1e-10:
                logger.warning(
                    f"SIF: PC removal zeroed out embedding. "
                    f"pre_norm={pre_removal_norm:.4f}, post_norm={post_removal_norm:.6f}, "
                    f"projection={projection:.4f}"
                )

        return embedding
