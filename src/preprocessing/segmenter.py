# src/preprocessing/segmenter.py

import re
import logging
from typing import List, Dict, Tuple

import nltk

try:
import spacy
_has_spacy = True
except ImportError:
_has_spacy = False

logger = logging.getLogger(**name**)

# Ensure punkt is available

try:
nltk.data.find("tokenizers/punkt")
except LookupError:
nltk.download("punkt", quiet=True)

class DocumentSegmenter:
"""
Basic segmenter for scientific papers.
Handles section splitting and sentence segmentation.
"""

```
def __init__(self, spacy_model: str = "en_core_web_sm"):
    self.nlp = None

    if _has_spacy:
        try:
            self.nlp = spacy.load(spacy_model)
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
        except Exception as e:
            logger.warning(f"spaCy model not loaded: {e}")
            self.nlp = None

def clean_scientific_text(self, text: str) -> Tuple[str, Dict]:
    """
    Light cleaning for scientific text.
    Keeps citations, numbers, and references mostly intact.
    """

    if not text:
        return "", {}

    meta = {
        "figures": [],
        "tables": [],
        "urls": []
    }

    # Figures
    meta["figures"] = re.findall(r"(Fig\.?\s*\d+[A-Za-z]?)", text)
    text = re.sub(r"Fig\.?\s*(\d+)", r"Figure \1", text)

    # Tables
    meta["tables"] = re.findall(r"(Table\s*\d+)", text)

    # URLs
    meta["urls"] = re.findall(r"https?://[^\s]+", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Fix spacing around punctuation (light touch)
    text = re.sub(r"(\w)([,.;:!?])(\w)", r"\1\2 \3", text)

    return text, meta

def extract_sections(self, document: str) -> Dict[str, str]:
    """
    Try to split a scientific paper into standard sections.
    Works best with well-formatted articles.
    """

    sections = {}

    patterns = {
        "abstract": r"abstract\s*[:\-]?\s*(.*?)(?=\n\s*\n(introduction|methods?|$))",
        "introduction": r"introduction\s*[:\-]?\s*(.*?)(?=\n\s*\n(methods?|results?|$))",
        "methods": r"(methods?|methodology)\s*[:\-]?\s*(.*?)(?=\n\s*\n(results?|discussion|$))",
        "results": r"(results?|findings?)\s*[:\-]?\s*(.*?)(?=\n\s*\n(discussion|conclusion|$))",
        "discussion": r"discussion\s*[:\-]?\s*(.*?)(?=\n\s*\n(conclusion|references?|$))",
        "conclusion": r"conclusions?\s*[:\-]?\s*(.*?)(?=\n\s*\n(references?|$))",
    }

    for name, pattern in patterns.items():
        match = re.search(pattern, document, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            cleaned, _ = self.clean_scientific_text(content)
            if cleaned:
                sections[name] = cleaned

    # fallback: treat full text as one section
    if not sections:
        cleaned, _ = self.clean_scientific_text(document)
        sections["fulltext"] = cleaned

    return sections

def segment_sentences(self, text: str) -> List[str]:
    """
    Split text into sentences.
    Uses spaCy if available, otherwise falls back to NLTK.
    """

    if not text or not text.strip():
        return []

    # spaCy (preferred)
    if self.nlp:
        try:
            doc = self.nlp(text)
            sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 8]
            if sents:
                return sents
        except Exception:
            pass

    # NLTK fallback
    try:
        sents = nltk.sent_tokenize(text)
        return [s.strip() for s in sents if len(s.strip()) > 8]
    except Exception:
        # last resort split
        return [s.strip() + "." for s in text.split(".") if s.strip()]

def segment_document(self, document: str) -> Dict[str, List[str]]:
    """
    Full pipeline: sectioning + sentence splitting.
    """

    sections = self.extract_sections(document)

    return {
        name: self.segment_sentences(content)
        for name, content in sections.items()
    }
```
