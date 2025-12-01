# src/feature_extraction/feature_extractor.py
"""
Feature Extraction Utilities

Provides utilities for:
- Sentence embedding generation using Sentence-BERT
- Embedding storage and retrieval
- Integration with keyphrase extraction
"""

from typing import List, Optional, Dict
import os
import json
import numpy as np
import pickle

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


class SentenceEmbedder:
    """Handles sentence embedding generation using Sentence-BERT."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the sentence embedder.
        
        Args:
            model_name: Name of the Sentence-BERT model to use
        """
        self.model_name = model_name
        self.model = None
        
        if _has_sentence_transformers:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Warning: Could not load Sentence-BERT model: {e}")
    
    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of sentences.
        
        Args:
            sentences: List of sentences to embed
            
        Returns:
            NumPy array of embeddings (shape: [num_sentences, embedding_dim])
        """
        if not sentences:
            return np.array([])
        
        # Use Sentence-BERT if available
        if self.model is not None:
            try:
                embeddings = self.model.encode(
                    sentences,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                return embeddings
            except Exception as e:
                print(f"Warning: Sentence-BERT encoding failed: {e}")
        
        # Fallback to TF-IDF
        return self._embed_with_tfidf(sentences)
    
    def _embed_with_tfidf(self, sentences: List[str]) -> np.ndarray:
        """
        Fallback embedding using TF-IDF.
        
        Args:
            sentences: List of sentences
            
        Returns:
            TF-IDF matrix as dense array
        """
        if not _has_sklearn:
            # Ultimate fallback: random embeddings for demo
            return np.random.randn(len(sentences), 384)
        
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=384,  # Match SBERT dimension
                stop_words='english'
            )
            X = vectorizer.fit_transform(sentences)
            return X.toarray()
        except Exception:
            return np.random.randn(len(sentences), 384)


def embed_sentences(sentences: List[str], 
                   model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Convenience function to generate sentence embeddings.
    
    Args:
        sentences: List of sentences to embed
        model_name: Name of Sentence-BERT model
        
    Returns:
        NumPy array of embeddings
    """
    embedder = SentenceEmbedder(model_name)
    return embedder.embed_sentences(sentences)


def save_embeddings(embeddings: np.ndarray, 
                   output_path: str,
                   metadata: Optional[Dict] = None):
    """
    Save embeddings to file with metadata.
    
    Args:
        embeddings: NumPy array of embeddings
        output_path: Path to save embeddings (.pkl or .npy)
        metadata: Optional metadata dictionary
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Save as pickle (includes metadata)
    if output_path.endswith('.pkl'):
        data = {
            'embeddings': embeddings,
            'metadata': metadata or {},
            'shape': embeddings.shape
        }
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        # Save as numpy
        np.save(output_path, embeddings)
        
        # Save metadata separately
        if metadata:
            meta_path = output_path.replace('.npy', '_meta.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)


def load_embeddings(input_path: str) -> tuple:
    """
    Load embeddings and metadata from file.
    
    Args:
        input_path: Path to embeddings file
        
    Returns:
        Tuple of (embeddings, metadata)
    """
    if input_path.endswith('.pkl'):
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        return data['embeddings'], data.get('metadata', {})
    else:
        embeddings = np.load(input_path)
        
        # Try to load metadata
        meta_path = input_path.replace('.npy', '_meta.json')
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        
        return embeddings, metadata
