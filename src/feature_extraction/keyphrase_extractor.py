"""
Keyphrase extraction for scientific text.

Uses KeyBERT when available, otherwise falls back to TF-IDF or a simple
frequency-based approach.
"""

import logging
import math
from collections import Counter
from typing import List, Tuple

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
    np = None


logger = logging.getLogger(__name__)


class ScientificKeyphraseExtractor:
    """
    Extract keyphrases from scientific text.
    """

    def __init__(
        self,
        min_keyphrases: int = 3,
        max_keyphrases: int = 15,
        chars_per_keyphrase: int = 300,
        diversity: float = 0.7,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.min_keyphrases = min_keyphrases
        self.max_keyphrases = max_keyphrases
        self.chars_per_keyphrase = chars_per_keyphrase
        self.diversity = diversity
        self.model = None

        if _has_keybert:
            try:
                self.model = KeyBERT(model_name)
            except Exception as e:
                logger.warning("Could not initialize KeyBERT: %s", e)
                self.model = None

    def calculate_dynamic_top_n(self, text: str) -> int:
        """
        Estimate how many keyphrases to extract based on text length.
        """
        char_count = len(text or "")
        calculated_n = math.ceil(char_count / self.chars_per_keyphrase)
        return max(self.min_keyphrases, min(self.max_keyphrases, calculated_n))

    def extract_keyphrases(
        self,
        text: str,
        top_n: int = None,
        ngram_range: Tuple[int, int] = (1, 3)
    ) -> List[Tuple[str, float]]:
        """
        Extract keyphrases from text.
        """
        if not text or len(text.split()) < 5:
            return []

        if top_n is None:
            top_n = self.calculate_dynamic_top_n(text)

        if self.model is not None:
            try:
                return self.model.extract_keywords(
                    text,
                    keyphrase_ngram_range=ngram_range,
                    stop_words="english",
                    use_mmr=True,
                    diversity=self.diversity,
                    top_n=top_n
                )
            except Exception as e:
                logger.warning("KeyBERT extraction failed: %s. Using fallback.", e)

        return self._extract_with_tfidf(text, top_n, ngram_range)

    def _extract_with_tfidf(
        self,
        text: str,
        top_n: int,
        ngram_range: Tuple[int, int]
    ) -> List[Tuple[str, float]]:
        """
        Fallback extraction using TF-IDF or simple frequency counts.
        """
        if not _has_sklearn:
            return self._simple_frequency_fallback(text, top_n)

        try:
            vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                stop_words="english",
                max_features=max(top_n * 3, 10)
            )
            X = vectorizer.fit_transform([text])
            features = np.array(vectorizer.get_feature_names_out())
            scores = X.toarray()[0]

            top_idx = scores.argsort()[-top_n:][::-1]
            results = [(features[i], float(scores[i])) for i in top_idx if scores[i] > 0]

            if results:
                return results

            return self._simple_frequency_fallback(text, top_n)
        except Exception as e:
            logger.warning("TF-IDF fallback failed: %s", e)
            return self._simple_frequency_fallback(text, top_n)

    def _simple_frequency_fallback(self, text: str, top_n: int) -> List[Tuple[str, float]]:
        """
        Last-resort fallback based on word frequency.
        """
        words = [w for w in text.lower().split() if len(w) > 3 and w.isalpha()]
        freq = Counter(words)
        return [(word, float(count)) for word, count in freq.most_common(top_n)]