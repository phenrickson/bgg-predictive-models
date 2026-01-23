"""Base class for word embedding algorithms."""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class WordEmbedding(ABC):
    """Abstract base class for word embedding models."""

    @abstractmethod
    def fit(self, documents: List[str]) -> "WordEmbedding":
        """Fit the embedding model on a corpus of documents.

        Args:
            documents: List of text documents.

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get the embedding vector for a word.

        Args:
            word: The word to look up.

        Returns:
            Embedding vector, or None if word not in vocabulary.
        """
        pass

    @abstractmethod
    def get_vocabulary(self) -> List[str]:
        """Get the vocabulary (list of words with embeddings)."""
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings."""
        pass

    def get_vectors(self, words: List[str]) -> np.ndarray:
        """Get embedding vectors for multiple words.

        Args:
            words: List of words to look up.

        Returns:
            Array of shape (n_words, embedding_dim).
            Words not in vocabulary get zero vectors.
        """
        dim = self.get_embedding_dim()
        vectors = np.zeros((len(words), dim))
        for i, word in enumerate(words):
            vec = self.get_vector(word)
            if vec is not None:
                vectors[i] = vec
        return vectors

    def most_similar(
        self, word: str, n: int = 10
    ) -> List[tuple[str, float]]:
        """Find words most similar to the given word.

        Args:
            word: The query word.
            n: Number of similar words to return.

        Returns:
            List of (word, similarity) tuples, sorted by similarity descending.
        """
        vec = self.get_vector(word)
        if vec is None:
            return []

        vocab = self.get_vocabulary()
        similarities = []

        for other_word in vocab:
            if other_word == word:
                continue
            other_vec = self.get_vector(other_word)
            if other_vec is not None:
                sim = np.dot(vec, other_vec) / (
                    np.linalg.norm(vec) * np.linalg.norm(other_vec) + 1e-10
                )
                similarities.append((other_word, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
