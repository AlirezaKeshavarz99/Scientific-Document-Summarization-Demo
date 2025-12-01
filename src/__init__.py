"""
Scientific Document Summarization Framework

A self-reliant framework for hierarchical summarization of scientific documents
using pre-trained language models.
"""

__version__ = "1.0.0"
__author__ = "Alireza Keshavarz"
__email__ = "a.keshavarz@khu.ac.ir"

from src.pipeline import SummarizationPipeline
from src.preprocessing.segmenter import ScientificDocumentSegmenter
from src.feature_extraction.keyphrase_extractor import ScientificKeyphraseExtractor
from src.evaluation.metrics import evaluate_pair, calculate_rouge

__all__ = [
    "SummarizationPipeline",
    "ScientificDocumentSegmenter",
    "ScientificKeyphraseExtractor",
    "evaluate_pair",
    "calculate_rouge",
]
