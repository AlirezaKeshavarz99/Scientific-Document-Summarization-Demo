# tests/test_segmenter.py
"""
Tests for the document segmentation module.

These tests cover:
- text cleaning
- section extraction
- sentence segmentation
- handling of empty and unstructured input
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.segmenter import DocumentSegmenter


def test_text_cleaning():
    """Text cleaning should preserve scientific content and extract metadata."""
    segmenter = DocumentSegmenter()

    text = "Figure 1 shows results. See Table 2. Visit https://example.com for more info."
    cleaned, elements = segmenter.clean_scientific_text(text)

    assert isinstance(cleaned, str)
    assert cleaned

    assert isinstance(elements, dict)
    assert "figures" in elements
    assert "tables" in elements
    assert "urls" in elements

    assert len(elements["urls"]) == 1


def test_section_extraction():
    """Section extraction should detect common scientific sections."""
    segmenter = DocumentSegmenter()

    document = """
    Abstract

    This is the abstract text with important findings.

    Introduction

    This is the introduction providing background information.

    Methods

    This describes the experimental methodology used.

    Results

    This section presents the findings.

    Discussion

    This discusses the implications of the results.

    Conclusion

    This provides concluding remarks.
    """

    sections = segmenter.extract_sections(document)

    assert isinstance(sections, dict)
    assert len(sections) > 0

    expected_sections = [
        "abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
    ]

    found_sections = [name for name in expected_sections if name in sections]
    assert len(found_sections) >= 2


def test_sentence_segmentation():
    """Sentence segmentation should return a list of non-empty sentences."""
    segmenter = DocumentSegmenter()

    text = "This is the first sentence. This is the second sentence. This is the third."
    sentences = segmenter.segment_sentences(text)

    assert isinstance(sentences, list)
    assert len(sentences) >= 2

    for sentence in sentences:
        assert isinstance(sentence, str)
        assert sentence.strip()


def test_empty_input():
    """Empty input should return empty outputs without errors."""
    segmenter = DocumentSegmenter()

    cleaned, elements = segmenter.clean_scientific_text("")
    assert cleaned == ""
    assert isinstance(elements, dict)

    sentences = segmenter.segment_sentences("")
    assert isinstance(sentences, list)
    assert len(sentences) == 0


def test_section_extraction_no_structure():
    """Documents without clear section markers should fall back to full text."""
    segmenter = DocumentSegmenter()

    document = "This is a plain text paragraph without any section markers."
    sections = segmenter.extract_sections(document)

    assert isinstance(sections, dict)
    assert len(sections) == 1
    assert "fulltext" in sections
    assert sections["fulltext"]


def test_preservation_of_scientific_notation():
    """Scientific notation and basic numeric expressions should remain readable."""
    segmenter = DocumentSegmenter()

    text = "The p-value was p < 0.05. Results showed 1.5e-10 concentration."
    cleaned, _ = segmenter.clean_scientific_text(text)

    assert "0.05" in cleaned
    assert "1.5e-10" in cleaned or "1.5e" in cleaned


def test_multiple_sections_same_document():
    """Multiple sections should be detected in a structured document."""
    segmenter = DocumentSegmenter()

    document = """
    Abstract

    Abstract content here.

    Introduction

    Introduction content here.

    Methods

    Methods content here.
    """

    sections = segmenter.extract_sections(document)

    assert isinstance(sections, dict)
    assert len(sections) >= 2

    for section_name, section_text in sections.items():
        assert isinstance(section_name, str)
        assert isinstance(section_text, str)
        assert section_text.strip()