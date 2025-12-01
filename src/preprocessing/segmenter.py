# src/preprocessing/segmenter.py
"""
Scientific Document Segmentation Module

This module handles text cleaning, section extraction, and sentence segmentation
for scientific documents. It preserves scientific notation, citations, and technical terms
while normalizing whitespace and removing artifacts.
"""

import re
from typing import List, Dict, Tuple
import nltk

try:
    import spacy
    _has_spacy = True
except ImportError:
    _has_spacy = False

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class ScientificDocumentSegmenter:
    """
    Segmenter for scientific documents that handles section extraction
    and sentence segmentation with scientific text awareness.
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the segmenter.
        
        Args:
            spacy_model: Name of spaCy model to use for sentence segmentation
        """
        self.spacy_model = spacy_model
        self.nlp = None
        
        if _has_spacy:
            try:
                self.nlp = spacy.load(spacy_model)
                # Add sentencizer if not present
                if "sentencizer" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("sentencizer")
            except Exception as e:
                print(f"Warning: Could not load spaCy model '{spacy_model}': {e}")
                print("Falling back to NLTK for sentence segmentation")
                self.nlp = None

    def clean_text_for_scientific_literature(self, text: str) -> Tuple[str, Dict]:
        """
        Clean scientific text while preserving important elements.
        
        Preserves:
        - Scientific notation (e.g., 1.5e-10)
        - P-values (e.g., p < 0.05)
        - Citations (e.g., [1, 2, 3])
        - Greek letters and subscripts
        - Numerical values and percentages
        
        Args:
            text: Raw text to clean
            
        Returns:
            Tuple of (cleaned_text, extracted_elements)
        """
        if not text:
            return "", {}
        
        elements = {
            'figures': [],
            'tables': [],
            'urls': []
        }
        
        # Standardize figure references
        figures = re.findall(r'(Fig\.?\s*\d+[A-Za-z]?)', text)
        elements['figures'] = figures
        text = re.sub(r'Fig\.?\s*(\d+)', r'Figure \1', text)
        
        # Standardize table references
        tables = re.findall(r'(Table\s*\d+)', text)
        elements['tables'] = tables
        
        # Extract URLs
        urls = re.findall(r'https?://[^\s]+', text)
        elements['urls'] = urls
        
        # Normalize whitespace (preserve single spaces)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'(\w)([,.;:!?])(\w)', r'\1\2 \3', text)
        
        return text, elements

    def extract_sections(self, document: str) -> Dict[str, str]:
        """
        Extract standard scientific paper sections from document.
        
        Sections extracted: Abstract, Introduction, Methods, Results, 
        Discussion, Conclusion
        
        Args:
            document: Full document text
            
        Returns:
            Dictionary mapping section names to their content
        """
        sections = {}
        
        # Define section patterns (case-insensitive matching)
        section_patterns = {
            'abstract': r'abstract\s*[:\-]?\s*(.*?)(?=\n\s*\n(?:introduction|methods?|$))',
            'introduction': r'introduction\s*[:\-]?\s*(.*?)(?=\n\s*\n(?:methods?|methodology|materials?\s+and\s+methods?|$))',
            'methods': r'(?:methods?|methodology|materials?\s+and\s+methods?)\s*[:\-]?\s*(.*?)(?=\n\s*\n(?:results?|findings?|$))',
            'results': r'(?:results?|findings?)\s*[:\-]?\s*(.*?)(?=\n\s*\n(?:discussion|conclusions?|$))',
            'discussion': r'discussion\s*[:\-]?\s*(.*?)(?=\n\s*\n(?:conclusions?|acknowledgments?|references?|$))',
            'conclusion': r'conclusions?\s*[:\-]?\s*(.*?)(?=\n\s*\n(?:acknowledgments?|references?|$))'
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, document, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                cleaned_content, _ = self.clean_text_for_scientific_literature(content)
                if cleaned_content:
                    sections[section_name] = cleaned_content
        
        # If no sections found, treat entire document as content
        if not sections:
            cleaned_text, _ = self.clean_text_for_scientific_literature(document)
            sections['fulltext'] = cleaned_text
        
        return sections

    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences using spaCy or NLTK.
        
        Args:
            text: Text to segment
            
        Returns:
            List of sentences
        """
        if not text or not text.strip():
            return []
        
        # Try spaCy first (better for scientific text)
        if self.nlp:
            try:
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents 
                           if len(sent.text.strip()) >= 8]
                if sentences:
                    return sentences
            except Exception:
                pass
        
        # Fallback to NLTK
        try:
            sentences = nltk.tokenize.sent_tokenize(text)
            sentences = [s.strip() for s in sentences if len(s.strip()) >= 8]
            return sentences
        except Exception:
            # Ultimate fallback: split on periods
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
            return sentences
