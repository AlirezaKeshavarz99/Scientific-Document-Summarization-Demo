# src/preprocessing/segmenter.py
import re
from typing import List, Dict
import nltk

try:
    import spacy
    _has_spacy = True
except Exception:
    _has_spacy = False

nltk.download('punkt', quiet=True)

class ScientificDocumentSegmenter:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.spacy_model = spacy_model
        if _has_spacy:
            try:
                self.nlp = spacy.load(spacy_model)
            except Exception:
                self.nlp = None
        else:
            self.nlp = None

    def clean_text_for_scientific_literature(self, text: str) -> str:
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\([Ff]ig\.?\s*\d+\)', '', text)
        text = re.sub(r'\([Tt]able\s*\d+\)', '', text)
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_sections(self, document: str) -> Dict[str, str]:
        sections = {}
        # Lower for matching but keep original for content
        doc = document
        doc_lower = document.lower()
        section_patterns = {
            'abstract': r'abstract\s*[:\-]?\s*(.*?)(?=\n\s*\n|introduction|$)',
            'introduction': r'introduction\s*[:\-]?\s*(.*?)(?=\n\s*\nmethods|methodology|$)',
            'methods': r'methods?\s*[:\-]?\s*(.*?)(?=\n\s*\nresults|findings|$)',
            'results': r'results\s*[:\-]?\s*(.*?)(?=\n\s*\ndiscussion|$)',
            'discussion': r'discussion\s*[:\-]?\s*(.*?)(?=\n\s*\nconclusion|$)',
            'conclusion': r'conclusion\s*[:\-]?\s*(.*?)(?=\n\s*\nreferences|$)'
        }
        for section_name, pattern in section_patterns.items():
            m = re.search(pattern, doc_lower, re.IGNORECASE | re.DOTALL)
            if m:
                start, end = m.span()
                # Extract original text by mapping span to original
                # Simpler: use matched group from original document using non-lowered regex
                try:
                    m_orig = re.search(pattern, doc, re.IGNORECASE | re.DOTALL)
                    content = m_orig.group(1).strip()
                except Exception:
                    content = m.group(1).strip()
                sections[section_name] = self.clean_text_for_scientific_literature(content)
        # If no sections found, return full text as 'fulltext'
        if not sections:
            sections['fulltext'] = self.clean_text_for_scientific_literature(document)
        return sections

    def segment_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        # Prefer spaCy if available and model loaded
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= 6]
            if sentences:
                return sentences
        # Fallback to NLTK
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 6]
        return sentences
