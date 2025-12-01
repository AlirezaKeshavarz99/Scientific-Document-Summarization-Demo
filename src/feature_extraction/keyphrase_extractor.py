# src/feature_extraction/keyphrase_extractor.py
"""
Keyphrase Extraction Module

Extracts important keyphrases from scientific text using KeyBERT with 
Maximal Marginal Relevance (MMR) for diversity. Falls back to TF-IDF if KeyBERT
is not available.
"""

from typing import List, Tuple
import math

try:
    from keybert import KeyBERT
    _has_keybert = True
except ImportError:
    _has_keybert = False
    
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    _has_sklearn = True
except ImportError:
    _has_sklearn = False


class ScientificKeyphraseExtractor:
    """
    Extracts keyphrases from scientific text with dynamic top_n calculation
    based on text length.
    """
    
    def __init__(self, 
                 min_keyphrases: int = 3,
                 max_keyphrases: int = 15,
                 chars_per_keyphrase: int = 300,
                 diversity: float = 0.7):
        """
        Initialize the keyphrase extractor.
        
        Args:
            min_keyphrases: Minimum number of keyphrases to extract
            max_keyphrases: Maximum number of keyphrases to extract
            chars_per_keyphrase: Target characters per keyphrase for dynamic calculation
            diversity: MMR diversity parameter (0-1, higher = more diverse)
        """
        self.min_keyphrases = min_keyphrases
        self.max_keyphrases = max_keyphrases
        self.chars_per_keyphrase = chars_per_keyphrase
        self.diversity = diversity
        self.model = None
        
        # Try to initialize KeyBERT
        if _has_keybert:
            try:
                self.model = KeyBERT()
            except Exception as e:
                print(f"Warning: Could not initialize KeyBERT: {e}")
                print("Falling back to TF-IDF extraction")

    def calculate_dynamic_top_n(self, text: str) -> int:
        """
        Calculate optimal number of keyphrases based on text length.
        
        Args:
            text: Input text
            
        Returns:
            Optimal number of keyphrases
        """
        char_count = len(text)
        calculated_n = math.ceil(char_count / self.chars_per_keyphrase)
        return max(self.min_keyphrases, min(self.max_keyphrases, calculated_n))

    def extract_keyphrases(self, 
                          text: str, 
                          top_n: int = None,
                          ngram_range: Tuple[int, int] = (1, 3)) -> List[Tuple[str, float]]:
        """
        Extract keyphrases from text.
        
        Args:
            text: Input text
            top_n: Number of keyphrases to extract (None for dynamic calculation)
            ngram_range: N-gram range for extraction (default: unigrams to trigrams)
            
        Returns:
            List of (keyphrase, score) tuples
        """
        if not text or len(text.split()) < 5:
            return []
        
        # Calculate dynamic top_n if not specified
        if top_n is None:
            top_n = self.calculate_dynamic_top_n(text)
        
        # Try KeyBERT first
        if self.model is not None:
            try:
                keyphrases = self.model.extract_keywords(
                    text,
                    keyphrase_ngram_range=ngram_range,
                    stop_words='english',
                    use_mmr=True,
                    diversity=self.diversity,
                    top_n=top_n
                )
                return keyphrases
            except Exception as e:
                print(f"KeyBERT extraction failed: {e}. Using TF-IDF fallback.")
        
        # TF-IDF fallback
        return self._extract_with_tfidf(text, top_n, ngram_range)
    
    def _extract_with_tfidf(self, 
                           text: str, 
                           top_n: int,
                           ngram_range: Tuple[int, int]) -> List[Tuple[str, float]]:
        """
        Fallback extraction using TF-IDF.
        
        Args:
            text: Input text
            top_n: Number of keyphrases to extract
            ngram_range: N-gram range
            
        Returns:
            List of (keyphrase, score) tuples
        """
        if not _has_sklearn:
            # Ultimate fallback: simple word frequency
            words = [w for w in text.lower().split() if len(w) > 3 and w.isalpha()]
            from collections import Counter
            freq = Counter(words)
            return [(word, float(count)) for word, count in freq.most_common(top_n)]
        
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                stop_words='english',
                max_features=top_n * 3  # Extract more, then filter
            )
            X = vectorizer.fit_transform([text])
            features = np.array(vectorizer.get_feature_names_out())
            scores = X.toarray()[0]
            
            # Get top N
            top_idx = scores.argsort()[-top_n:][::-1]
            results = [(features[i], float(scores[i])) for i in top_idx if scores[i] > 0]
            return results
        except Exception as e:
            # Simple fallback
            words = [w for w in text.lower().split() if len(w) > 3 and w.isalpha()]
            from collections import Counter
            freq = Counter(words)
            return [(word, float(count)) for word, count in freq.most_common(top_n)]
