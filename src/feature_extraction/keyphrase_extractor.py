# src/feature_extraction/keyphrase_extractor.py
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple
import numpy as np

class ScientificKeyphraseExtractor:
    def __init__(self):
        # demo: TF-IDF fallback; KeyBERT can be used in private full environment
        self.vectorizer = None

    def extract_keyphrases(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        if not text or len(text.split()) < 5:
            return []
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1,3), stop_words='english')
            X = vectorizer.fit_transform([text])
            features = np.array(vectorizer.get_feature_names_out())
            scores = X.toarray()[0]
            top_idx = scores.argsort()[-top_n:][::-1]
            results = [(features[i], float(scores[i])) for i in top_idx if scores[i] > 0]
            return results
        except Exception:
            tokens = [t for t in text.split() if len(t) > 3]
            return [(t, 1.0) for t in tokens[:top_n]]
