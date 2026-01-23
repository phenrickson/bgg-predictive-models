"""Word embeddings via PMI and SVD."""

import logging
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from .base import WordEmbedding
from .tokenizer import tokenize_documents

logger = logging.getLogger(__name__)


class PMIEmbedding(WordEmbedding):
    """Word embeddings using Pointwise Mutual Information and SVD.

    Process:
    1. Build co-occurrence matrix from skipgram windows
    2. Compute PPMI (Positive PMI) to normalize for word frequency
    3. Apply SVD to reduce to target dimensions
    """

    def __init__(
        self,
        embedding_dim: int = 100,
        window_size: int = 5,
        min_count: int = 5,
    ):
        """Initialize PMI embedding model.

        Args:
            embedding_dim: Number of dimensions for word vectors.
            window_size: Context window size for co-occurrence.
            min_count: Minimum word frequency to include in vocabulary.
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count

        self._word_to_idx: Dict[str, int] = {}
        self._idx_to_word: Dict[int, str] = {}
        self._vectors: Optional[np.ndarray] = None
        self._word_counts: Optional[Counter] = None
        self._singular_values: Optional[np.ndarray] = None

    def fit(self, documents: List[str]) -> "PMIEmbedding":
        """Fit embeddings on a corpus.

        Args:
            documents: List of text documents.

        Returns:
            Self for method chaining.
        """
        # Tokenize
        logger.info(f"Tokenizing {len(documents)} documents...")
        tokenized = tokenize_documents(documents)

        # Build vocabulary
        logger.info("Building vocabulary...")
        self._word_counts = Counter()
        for tokens in tokenized:
            self._word_counts.update(tokens)

        vocab = [
            word for word, count in self._word_counts.items() if count >= self.min_count
        ]
        vocab.sort()  # Consistent ordering

        self._word_to_idx = {word: i for i, word in enumerate(vocab)}
        self._idx_to_word = {i: word for word, i in self._word_to_idx.items()}

        vocab_size = len(vocab)
        logger.info(f"Vocabulary size: {vocab_size} words (min_count={self.min_count})")

        # Build co-occurrence matrix
        logger.info(f"Building co-occurrence matrix (window_size={self.window_size})...")
        cooccur = self._build_cooccurrence_matrix(tokenized)

        # Compute PPMI
        logger.info("Computing PPMI...")
        ppmi = self._compute_ppmi(cooccur)

        # Apply SVD
        logger.info(f"Applying SVD (embedding_dim={self.embedding_dim})...")
        self._vectors = self._apply_svd(ppmi)

        logger.info(f"Embeddings complete: {self._vectors.shape}")
        return self

    def _build_cooccurrence_matrix(
        self, tokenized_docs: List[List[str]]
    ) -> csr_matrix:
        """Build word-word co-occurrence matrix from skipgram windows."""
        vocab_size = len(self._word_to_idx)
        cooccur = Counter()

        for tokens in tokenized_docs:
            for i, word in enumerate(tokens):
                if word not in self._word_to_idx:
                    continue

                word_idx = self._word_to_idx[word]

                # Look at context window
                start = max(0, i - self.window_size)
                end = min(len(tokens), i + self.window_size + 1)

                for j in range(start, end):
                    if i == j:
                        continue
                    context_word = tokens[j]
                    if context_word not in self._word_to_idx:
                        continue

                    context_idx = self._word_to_idx[context_word]
                    cooccur[(word_idx, context_idx)] += 1

        # Convert to sparse matrix
        rows, cols, data = [], [], []
        for (i, j), count in cooccur.items():
            rows.append(i)
            cols.append(j)
            data.append(count)

        return csr_matrix(
            (data, (rows, cols)), shape=(vocab_size, vocab_size), dtype=np.float64
        )

    def _compute_ppmi(self, cooccur: csr_matrix) -> csr_matrix:
        """Compute Positive Pointwise Mutual Information.

        PMI(w, c) = log(P(w,c) / (P(w) * P(c)))
                  = log(count(w,c) * total / (count(w) * count(c)))

        PPMI = max(0, PMI)
        """
        # Total co-occurrences
        total = cooccur.sum()
        if total == 0:
            return cooccur

        # Word frequencies (row sums and column sums)
        word_freq = np.array(cooccur.sum(axis=1)).flatten()
        context_freq = np.array(cooccur.sum(axis=0)).flatten()

        # Compute PPMI for non-zero entries
        rows, cols = cooccur.nonzero()
        data = []

        for i, j in zip(rows, cols):
            count_wc = cooccur[i, j]
            count_w = word_freq[i]
            count_c = context_freq[j]

            if count_w > 0 and count_c > 0:
                pmi = np.log2((count_wc * total) / (count_w * count_c))
                ppmi = max(0, pmi)
                data.append(ppmi)
            else:
                data.append(0)

        return csr_matrix(
            (data, (rows, cols)), shape=cooccur.shape, dtype=np.float64
        )

    def _apply_svd(self, ppmi: csr_matrix) -> np.ndarray:
        """Apply truncated SVD to get word vectors."""
        # Ensure we don't request more components than matrix allows
        k = min(self.embedding_dim, min(ppmi.shape) - 1)

        if k < 1:
            logger.warning("Matrix too small for SVD, returning zeros")
            return np.zeros((ppmi.shape[0], self.embedding_dim))

        # Truncated SVD
        u, s, vt = svds(ppmi, k=k)

        # Sort by singular values (svds returns in ascending order)
        idx = np.argsort(s)[::-1]
        u = u[:, idx]
        s = s[idx]

        # Store singular values for analysis
        self._singular_values = s

        # Word vectors: U * sqrt(S)
        vectors = u * np.sqrt(s)

        # Pad if necessary
        if vectors.shape[1] < self.embedding_dim:
            padding = np.zeros((vectors.shape[0], self.embedding_dim - vectors.shape[1]))
            vectors = np.hstack([vectors, padding])

        return vectors

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get embedding vector for a word."""
        if self._vectors is None:
            raise ValueError("Model not fitted")

        idx = self._word_to_idx.get(word)
        if idx is None:
            return None

        return self._vectors[idx]

    def get_vocabulary(self) -> List[str]:
        """Get list of words in vocabulary."""
        return list(self._word_to_idx.keys())

    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self.embedding_dim

    def get_word_counts(self) -> Dict[str, int]:
        """Get word frequency counts (only for words in vocabulary)."""
        if self._word_counts is None:
            return {}
        return {
            word: self._word_counts[word]
            for word in self._word_to_idx.keys()
        }

    def get_singular_values(self) -> Optional[np.ndarray]:
        """Get singular values from SVD decomposition."""
        return self._singular_values

    def get_all_vectors(self) -> Optional[np.ndarray]:
        """Get all word vectors as a matrix (vocab_size x embedding_dim)."""
        return self._vectors
