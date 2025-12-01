# src/evaluation/metrics.py
"""
Evaluation Metrics Module

Provides comprehensive evaluation metrics for text summarization:
- ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
- BERTScore (semantic similarity)
- METEOR (if available via NLTK)
- Compression ratio and other statistics
"""

import logging
from typing import Dict, List, Optional

try:
    from rouge_score import rouge_scorer
    _has_rouge = True
except ImportError:
    _has_rouge = False

try:
    import bert_score
    _has_bertscore = True
except ImportError:
    _has_bertscore = False

try:
    import nltk
    from nltk.translate import meteor_score
    _has_meteor = True
    # Try to download required data
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
except ImportError:
    _has_meteor = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_rouge(reference: str, 
                   hypothesis: str,
                   rouge_types: Optional[List[str]] = None) -> Dict:
    """
    Calculate ROUGE scores between reference and hypothesis.
    
    Args:
        reference: Reference (gold) summary
        hypothesis: Generated (system) summary
        rouge_types: List of ROUGE types to compute (default: ['rouge1', 'rouge2', 'rougeL'])
        
    Returns:
        Dictionary of ROUGE scores with precision, recall, and F-measure
    """
    if not _has_rouge:
        logger.warning("rouge-score package not available")
        return {}
    
    if rouge_types is None:
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
    
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    
    # Format scores
    result = {}
    for key, value in scores.items():
        result[key] = {
            "precision": value.precision,
            "recall": value.recall,
            "fmeasure": value.fmeasure
        }
    
    return result


def calculate_bertscore(reference: str,
                       hypothesis: str,
                       lang: str = "en",
                       model_type: str = "microsoft/deberta-xlarge-mnli") -> Dict:
    """
    Calculate BERTScore between reference and hypothesis.
    
    Args:
        reference: Reference summary
        hypothesis: Generated summary
        lang: Language code
        model_type: BERT model to use for scoring
        
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    if not _has_bertscore:
        logger.warning("bert-score package not available")
        return {}
    
    try:
        # Use a lighter model for demo if the default is too large
        # Common choices: "microsoft/deberta-base-mnli", "roberta-large"
        model_type = "microsoft/deberta-base-mnli"  # Lighter model for demo
        
        P, R, F1 = bert_score.score(
            [hypothesis],
            [reference],
            lang=lang,
            model_type=model_type,
            verbose=False
        )
        
        return {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean())
        }
    except Exception as e:
        logger.warning(f"BERTScore calculation failed: {e}")
        return {}


def calculate_meteor(reference: str, hypothesis: str) -> float:
    """
    Calculate METEOR score between reference and hypothesis.
    
    Args:
        reference: Reference summary
        hypothesis: Generated summary
        
    Returns:
        METEOR score (0-1)
    """
    if not _has_meteor:
        logger.warning("NLTK METEOR not available")
        return 0.0
    
    try:
        # Tokenize
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        score = meteor_score.single_meteor_score(ref_tokens, hyp_tokens)
        return score
    except Exception as e:
        logger.warning(f"METEOR calculation failed: {e}")
        return 0.0


def calculate_compression_ratio(original: str, summary: str) -> Dict:
    """
    Calculate compression statistics.
    
    Args:
        original: Original text
        summary: Summary text
        
    Returns:
        Dictionary with compression metrics
    """
    orig_chars = len(original)
    orig_words = len(original.split())
    summ_chars = len(summary)
    summ_words = len(summary.split())
    
    char_ratio = summ_chars / orig_chars if orig_chars > 0 else 0
    word_ratio = summ_words / orig_words if orig_words > 0 else 0
    
    return {
        "original_characters": orig_chars,
        "summary_characters": summ_chars,
        "original_words": orig_words,
        "summary_words": summ_words,
        "character_compression": 1 - char_ratio,
        "word_compression": 1 - word_ratio,
        "compression_ratio": char_ratio
    }


def evaluate_pair(reference: str,
                 hypothesis: str,
                 metrics: Optional[List[str]] = None) -> Dict:
    """
    Comprehensive evaluation of reference-hypothesis pair.
    
    Args:
        reference: Reference (gold) summary
        hypothesis: Generated (system) summary
        metrics: List of metrics to compute (default: all available)
        
    Returns:
        Dictionary with all computed metrics
    """
    if metrics is None:
        metrics = ["rouge", "bertscore", "meteor", "compression"]
    
    results = {}
    
    # ROUGE scores
    if "rouge" in metrics:
        rouge_scores = calculate_rouge(reference, hypothesis)
        results["rouge"] = rouge_scores
    
    # BERTScore
    if "bertscore" in metrics:
        bert_scores = calculate_bertscore(reference, hypothesis)
        if bert_scores:
            results["bertscore"] = bert_scores
    
    # METEOR
    if "meteor" in metrics:
        meteor = calculate_meteor(reference, hypothesis)
        results["meteor"] = meteor
    
    # Compression statistics
    if "compression" in metrics:
        compression = calculate_compression_ratio(reference, hypothesis)
        results["compression"] = compression
    
    return results


def format_evaluation_results(results: Dict) -> str:
    """
    Format evaluation results for display.
    
    Args:
        results: Dictionary of evaluation results
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 60)
    
    # ROUGE scores
    if "rouge" in results:
        lines.append("\nROUGE Scores:")
        lines.append("-" * 60)
        for rouge_type, scores in results["rouge"].items():
            lines.append(f"  {rouge_type.upper()}:")
            lines.append(f"    Precision: {scores['precision']:.4f}")
            lines.append(f"    Recall:    {scores['recall']:.4f}")
            lines.append(f"    F-measure: {scores['fmeasure']:.4f}")
    
    # BERTScore
    if "bertscore" in results:
        lines.append("\nBERTScore:")
        lines.append("-" * 60)
        lines.append(f"  Precision: {results['bertscore']['precision']:.4f}")
        lines.append(f"  Recall:    {results['bertscore']['recall']:.4f}")
        lines.append(f"  F1:        {results['bertscore']['f1']:.4f}")
    
    # METEOR
    if "meteor" in results:
        lines.append("\nMETEOR Score:")
        lines.append("-" * 60)
        lines.append(f"  Score: {results['meteor']:.4f}")
    
    # Compression
    if "compression" in results:
        comp = results["compression"]
        lines.append("\nCompression Statistics:")
        lines.append("-" * 60)
        lines.append(f"  Original:  {comp['original_words']} words, {comp['original_characters']} chars")
        lines.append(f"  Summary:   {comp['summary_words']} words, {comp['summary_characters']} chars")
        lines.append(f"  Compression: {comp['word_compression']:.1%} (words), {comp['character_compression']:.1%} (chars)")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
