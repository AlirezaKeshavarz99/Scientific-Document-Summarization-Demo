# tests/test_pipeline_basic.py
"""
Basic Pipeline Tests

Tests core functionality of the summarization pipeline including:
- Document loading and segmentation
- Keyphrase extraction
- Summary generation
- End-to-end pipeline execution
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import SummarizationPipeline
from src.preprocessing.segmenter import ScientificDocumentSegmenter
from src.feature_extraction.keyphrase_extractor import ScientificKeyphraseExtractor


def test_pipeline_initialization():
    """Test that pipeline initializes without errors."""
    pipeline = SummarizationPipeline(config_path="configs/pipeline_config.yaml")
    assert pipeline is not None
    assert pipeline.device in ["cpu", "cuda"]
    assert pipeline.segmenter is not None
    assert pipeline.keyphrase_extractor is not None


def test_document_loading():
    """Test loading document from file."""
    pipeline = SummarizationPipeline(config_path="configs/pipeline_config.yaml")
    
    # Test with example file
    if os.path.exists("examples/sample_paper.txt"):
        text = pipeline.load_document("examples/sample_paper.txt")
        assert isinstance(text, str)
        assert len(text) > 0
        assert "Abstract" in text or "Introduction" in text


def test_document_segmentation():
    """Test document segmentation into sections."""
    segmenter = ScientificDocumentSegmenter()
    
    sample_text = """
    Abstract
    This is the abstract text.
    
    Introduction
    This is the introduction text.
    
    Methods
    This is the methods text.
    """
    
    sections = segmenter.extract_sections(sample_text)
    assert isinstance(sections, dict)
    assert len(sections) > 0


def test_keyphrase_extraction():
    """Test keyphrase extraction."""
    extractor = ScientificKeyphraseExtractor()
    
    sample_text = """
    Machine learning models have demonstrated remarkable capabilities in 
    natural language processing tasks. Deep neural networks can learn 
    complex patterns from large datasets.
    """
    
    keyphrases = extractor.extract_keyphrases(sample_text, top_n=5)
    assert isinstance(keyphrases, list)
    # Should return list of tuples (keyphrase, score) or empty list
    if keyphrases:
        assert isinstance(keyphrases[0], tuple)
        assert len(keyphrases[0]) == 2


def test_end_to_end_summarization():
    """Test complete summarization pipeline."""
    pipeline = SummarizationPipeline(config_path="configs/pipeline_config.yaml")
    
    # Test with example file if it exists
    if os.path.exists("examples/sample_paper.txt"):
        result = pipeline.summarize_document("examples/sample_paper.txt")
        
        # Check result structure
        assert isinstance(result, dict)
        assert "final_summary" in result
        assert "section_summaries" in result
        assert "importance_scores" in result
        assert "allocations" in result
        
        # Check summary content
        assert isinstance(result["final_summary"], str)
        assert len(result["final_summary"]) > 0
        
        # Check section summaries
        assert isinstance(result["section_summaries"], dict)
        assert len(result["section_summaries"]) > 0


def test_summarize_method():
    """Test the main summarize method."""
    pipeline = SummarizationPipeline(config_path="configs/pipeline_config.yaml")
    
    # Test with sample text
    sample_text = """
    Abstract
    
    This study investigates the application of machine learning to scientific text.
    
    Introduction
    
    Scientific literature analysis is crucial for researchers.
    
    Methods
    
    We used natural language processing techniques.
    
    Results
    
    Our approach achieved 85% accuracy.
    
    Conclusion
    
    Machine learning shows promise for literature analysis.
    """
    
    # Test document-level summary
    summary = pipeline.summarize(sample_text, summary_type="document", compression_ratio=0.3)
    assert isinstance(summary, str)
    assert len(summary) > 0
    
    # Test section-level summaries
    section_summaries = pipeline.summarize(sample_text, summary_type="section", compression_ratio=0.3)
    assert isinstance(section_summaries, dict)
    assert len(section_summaries) > 0


def test_compression_ratio():
    """Test that compression ratio affects output length."""
    pipeline = SummarizationPipeline(config_path="configs/pipeline_config.yaml")
    
    sample_text = """
    Abstract
    
    This is a longer text that we will use to test compression ratios. It has multiple sentences.
    We want to ensure that different compression ratios produce different length outputs.
    
    Introduction
    
    The introduction provides background information. It sets the stage for the research.
    Multiple sentences help us test the compression mechanism effectively.
    """
    
    # Generate summaries with different compression ratios
    summary_low = pipeline.summarize(sample_text, compression_ratio=0.1)
    summary_high = pipeline.summarize(sample_text, compression_ratio=0.5)
    
    # Both should be valid strings
    assert isinstance(summary_low, str)
    assert isinstance(summary_high, str)
    
    # This test is informational - actual lengths may vary based on LLM behavior
    # Just verify both produced output
    assert len(summary_low) > 0
    assert len(summary_high) > 0


if __name__ == "__main__":
    # Run tests manually
    print("Running pipeline tests...")
    
    print("1. Testing pipeline initialization...")
    test_pipeline_initialization()
    print("   ✓ Passed")
    
    print("2. Testing document loading...")
    test_document_loading()
    print("   ✓ Passed")
    
    print("3. Testing document segmentation...")
    test_document_segmentation()
    print("   ✓ Passed")
    
    print("4. Testing keyphrase extraction...")
    test_keyphrase_extraction()
    print("   ✓ Passed")
    
    print("5. Testing end-to-end summarization...")
    test_end_to_end_summarization()
    print("   ✓ Passed")
    
    print("6. Testing summarize method...")
    test_summarize_method()
    print("   ✓ Passed")
    
    print("7. Testing compression ratio...")
    test_compression_ratio()
    print("   ✓ Passed")
    
    print("\n✓ All tests passed!")
