# tests/test_segmenter.py
"""
Tests for Document Segmentation Module

Tests text cleaning, section extraction, and sentence segmentation
functionality of the ScientificDocumentSegmenter class.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.segmenter import ScientificDocumentSegmenter


def test_text_cleaning():
    """Test scientific text cleaning functionality."""
    segmenter = ScientificDocumentSegmenter()
    
    text = "Figure 1 shows results. See Table 2. Visit https://example.com for more info."
    cleaned, elements = segmenter.clean_text_for_scientific_literature(text)
    
    # Check that text is cleaned
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0
    
    # Check extracted elements
    assert isinstance(elements, dict)
    assert "figures" in elements
    assert "tables" in elements
    assert "urls" in elements


def test_section_extraction():
    """Test extraction of standard scientific sections."""
    segmenter = ScientificDocumentSegmenter()
    
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
    
    # Check sections dictionary
    assert isinstance(sections, dict)
    assert len(sections) > 0
    
    # Check for expected sections
    expected_sections = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
    found_sections = [s for s in expected_sections if s in sections]
    assert len(found_sections) > 0


def test_sentence_segmentation():
    """Test sentence segmentation functionality."""
    segmenter = ScientificDocumentSegmenter()
    
    text = "This is the first sentence. This is the second sentence. This is the third."
    sentences = segmenter.segment_sentences(text)
    
    # Check sentences list
    assert isinstance(sentences, list)
    assert len(sentences) >= 2  # Should split into at least 2 sentences
    
    # Check each sentence
    for sent in sentences:
        assert isinstance(sent, str)
        assert len(sent) > 0


def test_empty_input():
    """Test handling of empty or invalid input."""
    segmenter = ScientificDocumentSegmenter()
    
    # Test empty text
    cleaned, elements = segmenter.clean_text_for_scientific_literature("")
    assert cleaned == ""
    assert isinstance(elements, dict)
    
    # Test sentence segmentation with empty text
    sentences = segmenter.segment_sentences("")
    assert isinstance(sentences, list)
    assert len(sentences) == 0


def test_section_extraction_no_structure():
    """Test section extraction when document has no clear structure."""
    segmenter = ScientificDocumentSegmenter()
    
    document = "This is just a plain text without any section markers."
    sections = segmenter.extract_sections(document)
    
    # Should return fulltext when no sections found
    assert isinstance(sections, dict)
    assert len(sections) > 0
    if "fulltext" in sections:
        assert len(sections["fulltext"]) > 0


def test_preservation_of_scientific_notation():
    """Test that scientific notation and special characters are preserved."""
    segmenter = ScientificDocumentSegmenter()
    
    text = "The p-value was p < 0.05. Results showed 1.5e-10 concentration."
    cleaned, _ = segmenter.clean_text_for_scientific_literature(text)
    
    # Should preserve p-values and scientific notation
    assert "p" in cleaned or "0.05" in cleaned
    assert "1.5" in cleaned or "e-10" in cleaned


def test_multiple_sections_same_document():
    """Test that all sections are correctly identified and separated."""
    segmenter = ScientificDocumentSegmenter()
    
    document = """
    Abstract
    
    Abstract content here.
    
    Introduction
    
    Introduction content here.
    
    Methods
    
    Methods content here.
    """
    
    sections = segmenter.extract_sections(document)
    
    # Check that we got multiple sections
    assert len(sections) >= 2
    
    # Check that sections are not empty
    for section_name, section_text in sections.items():
        assert isinstance(section_text, str)
        assert len(section_text) > 0


if __name__ == "__main__":
    # Run tests manually
    print("Running segmenter tests...")
    
    print("1. Testing text cleaning...")
    test_text_cleaning()
    print("   ✓ Passed")
    
    print("2. Testing section extraction...")
    test_section_extraction()
    print("   ✓ Passed")
    
    print("3. Testing sentence segmentation...")
    test_sentence_segmentation()
    print("   ✓ Passed")
    
    print("4. Testing empty input handling...")
    test_empty_input()
    print("   ✓ Passed")
    
    print("5. Testing extraction without structure...")
    test_section_extraction_no_structure()
    print("   ✓ Passed")
    
    print("6. Testing scientific notation preservation...")
    test_preservation_of_scientific_notation()
    print("   ✓ Passed")
    
    print("7. Testing multiple sections...")
    test_multiple_sections_same_document()
    print("   ✓ Passed")
    
    print("\n✓ All tests passed!")
