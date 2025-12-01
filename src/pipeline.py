# src/pipeline.py
"""
Main Summarization Pipeline

Orchestrates the complete summarization workflow:
1. Document preprocessing and segmentation
2. Feature extraction (keyphrases, embeddings)
3. Section importance scoring
4. Summary generation (section-level and document-level)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml

from src.preprocessing.segmenter import ScientificDocumentSegmenter
from src.feature_extraction.keyphrase_extractor import ScientificKeyphraseExtractor
from src.feature_extraction.feature_extractor import SentenceEmbedder
from src.summarization.llm_integration import summarize_text, batch_summarize
from src.contrastive.contrastive import compute_section_importance, exponential_allocation
from src.utils.helpers import load_config, ensure_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationPipeline:
    """
    Complete pipeline for scientific document summarization.
    
    Supports both section-level and document-level summarization with
    hierarchical processing and importance-based content allocation.
    """
    
    def __init__(self, 
                 config_path: str = "configs/pipeline_config.yaml",
                 device: str = None):
        """
        Initialize the summarization pipeline.
        
        Args:
            config_path: Path to pipeline configuration file
            device: Device to use ('cpu' or 'cuda'). Overrides config if provided.
        """
        self.config = load_config(config_path)
        self.device = device or self.config["pipeline"]["device"]
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        
        self.segmenter = ScientificDocumentSegmenter(
            spacy_model=self.config["preprocessing"]["spacy_model"]
        )
        
        self.keyphrase_extractor = ScientificKeyphraseExtractor(
            min_keyphrases=self.config["feature_extraction"]["keyphrase"].get("min", 3),
            max_keyphrases=self.config["feature_extraction"]["keyphrase"].get("max", 15),
            diversity=self.config["feature_extraction"]["keyphrase"]["diversity"]
        )
        
        self.embedder = SentenceEmbedder(
            model_name=self.config["feature_extraction"]["sentence_model"]
        )
        
        # Ensure output directory exists
        ensure_directory("data/outputs")
        
        logger.info("Pipeline initialized successfully")
    
    def load_document(self, path: str) -> str:
        """
        Load document from file.
        
        Args:
            path: Path to document file
            
        Returns:
            Document text
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        
        return p.read_text(encoding='utf-8')
    
    def preprocess_document(self, text: str) -> Dict[str, str]:
        """
        Segment document into sections.
        
        Args:
            text: Raw document text
            
        Returns:
            Dictionary mapping section names to their content
        """
        logger.info("Segmenting document into sections...")
        sections = self.segmenter.extract_sections(text)
        logger.info(f"Found {len(sections)} sections: {list(sections.keys())}")
        return sections
    
    def extract_section_keyphrases(self, 
                                  sections: Dict[str, str]) -> Dict[str, List[tuple]]:
        """
        Extract keyphrases from each section.
        
        Args:
            sections: Dictionary of section texts
            
        Returns:
            Dictionary mapping section names to lists of (keyphrase, score) tuples
        """
        logger.info("Extracting keyphrases from sections...")
        section_keyphrases = {}
        
        for section_name, section_text in sections.items():
            keyphrases = self.keyphrase_extractor.extract_keyphrases(
                section_text,
                ngram_range=tuple(self.config["feature_extraction"]["keyphrase"]["n_grams"])
            )
            section_keyphrases[section_name] = keyphrases
            logger.info(f"  {section_name}: {len(keyphrases)} keyphrases")
        
        return section_keyphrases
    
    def compute_section_scores(self, 
                              sections: Dict[str, str]) -> Dict[str, float]:
        """
        Compute importance scores for sections.
        
        Args:
            sections: Dictionary of section texts
            
        Returns:
            Dictionary mapping section names to importance scores
        """
        logger.info("Computing section importance scores...")
        
        section_names = list(sections.keys())
        section_texts = list(sections.values())
        
        scores = compute_section_importance(section_texts, self.config)
        
        score_dict = dict(zip(section_names, scores))
        
        for name, score in score_dict.items():
            logger.info(f"  {name}: {score:.3f}")
        
        return score_dict
    
    def allocate_summary_budget(self,
                               sections: Dict[str, str],
                               scores: Dict[str, float],
                               compression_ratio: float = 0.2) -> Dict[str, int]:
        """
        Allocate summary budget to sections based on importance.
        
        Args:
            sections: Dictionary of section texts
            scores: Section importance scores
            compression_ratio: Target compression ratio (0-1)
            
        Returns:
            Dictionary mapping section names to sentence budgets
        """
        logger.info("Allocating summary budget...")
        
        # Calculate total budget based on document size and compression ratio
        total_sentences = sum(len(self.segmenter.segment_sentences(text)) 
                            for text in sections.values())
        total_budget = max(3, int(total_sentences * compression_ratio))
        
        # Get scores in same order as sections
        section_names = list(sections.keys())
        score_list = [scores[name] for name in section_names]
        
        # Compute allocations
        allocations = exponential_allocation(score_list, total_budget)
        
        allocation_dict = dict(zip(section_names, allocations))
        
        logger.info(f"Total budget: {total_budget} sentences")
        for name, alloc in allocation_dict.items():
            logger.info(f"  {name}: {alloc} sentences")
        
        return allocation_dict
    
    def generate_section_summaries(self,
                                  sections: Dict[str, str],
                                  allocations: Optional[Dict[str, int]] = None) -> Dict[str, str]:
        """
        Generate summaries for each section.
        
        Args:
            sections: Dictionary of section texts
            allocations: Optional budget allocations per section
            
        Returns:
            Dictionary mapping section names to their summaries
        """
        logger.info("Generating section summaries...")
        
        summaries = {}
        cfg = self.config["summarization"]
        
        for section_name, section_text in sections.items():
            # Determine max length based on allocation
            if allocations and section_name in allocations:
                # Rough estimate: 20 tokens per sentence
                max_length = allocations[section_name] * 20
            else:
                max_length = cfg["max_length"]
            
            max_length = max(30, min(max_length, cfg["max_length"]))
            
            logger.info(f"  Summarizing {section_name} (max_length={max_length})...")
            
            summary = summarize_text(
                section_text,
                model_name=cfg["model_name"],
                max_length=max_length,
                device=self.device
            )
            
            summaries[section_name] = summary
        
        return summaries
    
    def generate_document_summary(self,
                                 section_summaries: Dict[str, str]) -> str:
        """
        Combine section summaries into final document summary.
        
        Args:
            section_summaries: Dictionary of section summaries
            
        Returns:
            Final document summary
        """
        logger.info("Generating final document summary...")
        
        # Format: Section Name:\nSummary\n\n
        summary_parts = []
        for section_name, summary in section_summaries.items():
            summary_parts.append(f"**{section_name.upper()}**\n{summary}")
        
        final_summary = "\n\n".join(summary_parts)
        
        return final_summary
    
    def summarize(self,
                 document: Union[str, Path],
                 summary_type: str = "document",
                 compression_ratio: float = 0.2) -> Union[str, Dict[str, str]]:
        """
        Main summarization method.
        
        Args:
            document: Document text or path to document file
            summary_type: Type of summary ('section' or 'document')
            compression_ratio: Target compression ratio (0-1)
            
        Returns:
            Summary text (if document-level) or dict of section summaries
        """
        # Load document if path provided
        if isinstance(document, (str, Path)):
            if Path(document).exists():
                text = self.load_document(document)
            else:
                text = document
        else:
            text = str(document)
        
        # Preprocess
        sections = self.preprocess_document(text)
        
        # Extract keyphrases (for potential use)
        keyphrases = self.extract_section_keyphrases(sections)
        
        # Compute importance scores
        scores = self.compute_section_scores(sections)
        
        # Allocate budget
        allocations = self.allocate_summary_budget(sections, scores, compression_ratio)
        
        # Generate section summaries
        section_summaries = self.generate_section_summaries(sections, allocations)
        
        # Return based on type
        if summary_type == "section":
            return section_summaries
        else:
            return self.generate_document_summary(section_summaries)
    
    def summarize_document(self, path: str) -> Dict:
        """
        Legacy method for backward compatibility.
        
        Args:
            path: Path to document file
            
        Returns:
            Dictionary with summaries and metadata
        """
        text = self.load_document(path)
        sections = self.preprocess_document(text)
        scores = self.compute_section_scores(sections)
        allocations = self.allocate_summary_budget(sections, scores)
        section_summaries = self.generate_section_summaries(sections, allocations)
        final_summary = self.generate_document_summary(section_summaries)
        
        return {
            "section_summaries": section_summaries,
            "final_summary": final_summary,
            "importance_scores": scores,
            "allocations": allocations
        }
