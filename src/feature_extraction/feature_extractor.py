"""
Sentence embedding utilities.

Uses Sentence-BERT when available, otherwise falls back to TF-IDF.
"""

import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _has_sentence_transformers = True
except ImportError:
    _has_sentence_transformers = False
    SentenceTransformer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _has_sklearn = True
except ImportError:
    _has_sklearn = False
    TfidfVectorizer = None


logger = logging.getLogger(__name__)


class SentenceEmbedder:
    """Generate sentence embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", fallback_dim: int = 384):
        self.model_name = model_name
        self.fallback_dim = fallback_dim
        self.model = None

        if _has_sentence_transformers:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                logger.warning("Could not load Sentence-BERT model: %s", e)
                self.model = None

    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Embed a list of sentences.
        """
        if not sentences:
            return np.empty((0, self.fallback_dim), dtype=float)

        if self.model is not None:
            try:
                return self.model.encode(
                    sentences,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
            except Exception as e:
                logger.warning("Sentence-BERT encoding failed: %s", e)

        return self._embed_with_tfidf(sentences)

    def _embed_with_tfidf(self, sentences: List[str]) -> np.ndarray:
        """
        Fallback embedding method using TF-IDF.
        """
        if not sentences:
            return np.empty((0, self.fallback_dim), dtype=float)

        if not _has_sklearn:
            return np.zeros((len(sentences), self.fallback_dim), dtype=float)

        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=self.fallback_dim,
                stop_words="english"
            )
            X = vectorizer.fit_transform(sentences)
            arr = X.toarray()

            if arr.shape[1] < self.fallback_dim:
                pad = np.zeros((arr.shape[0], self.fallback_dim - arr.shape[1]), dtype=float)
                arr = np.hstack([arr, pad])

            return arr[:, :self.fallback_dim]
        except Exception as e:
            logger.warning("TF-IDF embedding failed: %s", e)
            return np.zeros((len(sentences), self.fallback_dim), dtype=float)


def embed_sentences(
    sentences: List[str],
    model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Convenience function for sentence embeddings.
    """
    embedder = SentenceEmbedder(model_name=model_name)
    return embedder.embed_sentences(sentences)


def save_embeddings(
    embeddings: np.ndarray,
    output_path: str,
    metadata: Optional[Dict] = None
):
    """
    Save embeddings to disk.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if output_path.endswith(".pkl"):
        data = {
            "embeddings": embeddings,
            "metadata": metadata or {},
            "shape": embeddings.shape
        }
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
    else:
        np.save(output_path, embeddings)

        if metadata:
            meta_path = output_path.replace(".npy", "_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)


def load_embeddings(input_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Load embeddings and metadata from disk.
    """
    if input_path.endswith(".pkl"):
        with open(input_path, "rb") as f:
            data = pickle.load(f)
        return data["embeddings"], data.get("metadata", {})

    embeddings = np.load(input_path)

    meta_path = input_path.replace(".npy", "_meta.json")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return embeddings, metadata