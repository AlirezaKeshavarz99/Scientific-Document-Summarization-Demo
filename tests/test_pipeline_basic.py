# tests/test_pipeline_basic.py
"""
Basic tests for the summarization pipeline.

These tests focus on the main pieces of the repository:
- pipeline initialization
- document loading
- document segmentation
- keyphrase extraction
- end-to-end summarization
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import SummarizationPipeline
from src.preprocessing.segmenter import DocumentSegmenter
from src.feature_extraction.keyphrase_extractor import ScientificKeyphraseExtractor


def test_pipeline_initialization():
    """Pipeline should initialize with the expected components."""
    pipeline = SummarizationPipeline(config_path="configs/pipeline_config.yaml")

    assert pipeline is not None
    assert pipeline.device in ["cpu", "cuda"]
    assert pipeline.segmenter is not None
    assert pipeline.keyphrase_extractor is not None


def test_document_loading():
    """Document loading should return non-empty text for valid files."""
    sample_path = Path("examples/sample_paper.txt")

    if not sample_path.exists():
        return

    pipeline = SummarizationPipeline(config_path="configs/pipeline_config.yaml")
    text = pipeline.load_document(str(sample_path))

    assert isinstance(text, str)
    assert text.strip()
    assert "Abstract" in text or "Introduction" in text


def test_document_segmentation():
    """Segmenter should extract sections from a structured document."""
    segmenter = DocumentSegmenter()

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
    """Keyphrase extraction should return a list of phrase-score pairs."""
    extractor = ScientificKeyphraseExtractor()

    sample_text = """
    Machine learning models have demonstrated strong performance in
    natural language processing tasks. Deep neural networks can learn
    complex patterns from large datasets.
    """

    keyphrases = extractor.extract_keyphrases(sample_text, top_n=5)

    assert isinstance(keyphrases, list)

    if keyphrases:
        first_item = keyphrases[0]
        assert isinstance(first_item, tuple)
        assert len(first_item) == 2
        assert isinstance(first_item[0], str)
        assert isinstance(first_item[1], float)


def test_end_to_end_summarization():
    """End-to-end summarization should produce a structured result."""
    sample_path = Path("examples/sample_paper.txt")

    if not sample_path.exists():
        return

    pipeline = SummarizationPipeline(config_path="configs/pipeline_config.yaml")
    result = pipeline.summarize_document(str(sample_path))

    assert isinstance(result, dict)
    assert "final_summary" in result
    assert "section_summaries" in result
    assert "importance_scores" in result
    assert "allocations" in result

    assert isinstance(result["final_summary"], str)
    assert result["final_summary"].strip()

    assert isinstance(result["section_summaries"], dict)
    assert len(result["section_summaries"]) > 0


def test_summarize_method():
    """The summarize method should work for both document and section modes."""
    pipeline = SummarizationPipeline(config_path="configs/pipeline_config.yaml")

    sample_text = """
    Abstract

    This study investigates the application of machine learning to scientific text.

    Introduction

    Scientific literature analysis is important for researchers.

    Methods

    We used natural language processing techniques.

    Results

    Our approach achieved 85% accuracy.

    Conclusion

    Machine learning shows promise for literature analysis.
    """

    document_summary = pipeline.summarize(
        sample_text,
        summary_type="document",
        compression_ratio=0.3,
    )
    assert isinstance(document_summary, str)
    assert document_summary.strip()

    section_summaries = pipeline.summarize(
        sample_text,
        summary_type="section",
        compression_ratio=0.3,
    )
    assert isinstance(section_summaries, dict)
    assert len(section_summaries) > 0


def test_compression_ratio():
    """Different compression ratios should still produce valid summaries."""
    pipeline = SummarizationPipeline(config_path="configs/pipeline_config.yaml")

    sample_text = """
    Abstract

    This is a longer text that we use to test compression ratios. It has multiple sentences.
    We want to ensure that different compression ratios produce valid outputs.

    Introduction

    The introduction provides background information. It sets the stage for the research.
    Multiple sentences help us test the compression mechanism.
    """

    summary_low = pipeline.summarize(sample_text, compression_ratio=0.1)
    summary_high = pipeline.summarize(sample_text, compression_ratio=0.5)

    assert isinstance(summary_low, str)
    assert isinstance(summary_high, str)
    assert summary_low.strip()
    assert summary_high.strip()