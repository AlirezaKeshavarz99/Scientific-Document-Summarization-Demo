"""
Evaluation metrics for summarization.

Provides ROUGE, BERTScore, METEOR, and compression statistics.
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

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
except ImportError:
    _has_meteor = False


logger = logging.getLogger(__name__)


def _zero_rouge_result(rouge_types: List[str]) -> Dict[str, Dict[str, float]]:
    return {
        rouge_type: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
        for rouge_type in rouge_types
    }


def calculate_rouge(
    reference: str,
    hypothesis: str,
    rouge_types: Optional[List[str]] = None
) -> Dict:
    """
    Calculate ROUGE scores between a reference and a hypothesis.
    """
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    if not reference or not reference.strip() or not hypothesis or not hypothesis.strip():
        return _zero_rouge_result(rouge_types)

    if not _has_rouge:
        logger.warning("rouge-score package is not available.")
        return {}

    try:
        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        scores = scorer.score(reference, hypothesis)

        result = {}
        for key, value in scores.items():
            result[key] = {
                "precision": value.precision,
                "recall": value.recall,
                "fmeasure": value.fmeasure,
            }
        return result
    except Exception as e:
        logger.warning("ROUGE calculation failed: %s", e)
        return {}


def calculate_bertscore(
    reference: str,
    hypothesis: str,
    lang: str = "en",
    model_type: str = "microsoft/deberta-base-mnli"
) -> Dict:
    """
    Calculate BERTScore between reference and hypothesis.
    """
    if not reference or not reference.strip() or not hypothesis or not hypothesis.strip():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not _has_bertscore:
        logger.warning("bert-score package is not available.")
        return {}

    try:
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
            "f1": float(F1.mean()),
        }
    except Exception as e:
        logger.warning("BERTScore calculation failed: %s", e)
        return {}


def calculate_meteor(reference: str, hypothesis: str) -> float:
    """
    Calculate METEOR score between reference and hypothesis.
    """
    if not reference or not reference.strip() or not hypothesis or not hypothesis.strip():
        return 0.0

    if not _has_meteor:
        logger.warning("NLTK METEOR is not available.")
        return 0.0

    try:
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        return float(meteor_score.single_meteor_score(ref_tokens, hyp_tokens))
    except Exception as e:
        logger.warning("METEOR calculation failed: %s", e)
        return 0.0


def calculate_compression_ratio(original: str, summary: str) -> Dict:
    """
    Calculate compression statistics.
    """
    orig_chars = len(original or "")
    summ_chars = len(summary or "")
    orig_words = len((original or "").split())
    summ_words = len((summary or "").split())

    char_ratio = summ_chars / orig_chars if orig_chars > 0 else 0.0
    word_ratio = summ_words / orig_words if orig_words > 0 else 0.0

    return {
        "original_characters": orig_chars,
        "summary_characters": summ_chars,
        "original_words": orig_words,
        "summary_words": summ_words,
        "character_compression": 1 - char_ratio,
        "word_compression": 1 - word_ratio,
        "compression_ratio": char_ratio,
    }


def evaluate_pair(
    reference: str,
    hypothesis: str,
    metrics: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate a reference-hypothesis pair using selected metrics.
    """
    if metrics is None:
        metrics = ["rouge", "bertscore", "meteor", "compression"]

    metrics = [m.lower() for m in metrics]
    results = {}

    if "rouge" in metrics:
        results["rouge"] = calculate_rouge(reference, hypothesis)

    if "bertscore" in metrics:
        bert_scores = calculate_bertscore(reference, hypothesis)
        if bert_scores:
            results["bertscore"] = bert_scores

    if "meteor" in metrics:
        results["meteor"] = calculate_meteor(reference, hypothesis)

    if "compression" in metrics:
        results["compression"] = calculate_compression_ratio(reference, hypothesis)

    return results


def format_evaluation_results(results: Dict) -> str:
    """
    Format evaluation results for display.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 60)

    if "rouge" in results:
        lines.append("\nROUGE Scores:")
        lines.append("-" * 60)
        for rouge_type, scores in results["rouge"].items():
            lines.append(f"  {rouge_type.upper()}:")
            lines.append(f"    Precision: {scores['precision']:.4f}")
            lines.append(f"    Recall:    {scores['recall']:.4f}")
            lines.append(f"    F-measure: {scores['fmeasure']:.4f}")

    if "bertscore" in results:
        lines.append("\nBERTScore:")
        lines.append("-" * 60)
        lines.append(f"  Precision: {results['bertscore']['precision']:.4f}")
        lines.append(f"  Recall:    {results['bertscore']['recall']:.4f}")
        lines.append(f"  F1:        {results['bertscore']['f1']:.4f}")

    if "meteor" in results:
        lines.append("\nMETEOR Score:")
        lines.append("-" * 60)
        lines.append(f"  Score: {results['meteor']:.4f}")

    if "compression" in results:
        comp = results["compression"]
        lines.append("\nCompression Statistics:")
        lines.append("-" * 60)
        lines.append(
            f"  Original:  {comp['original_words']} words, {comp['original_characters']} chars"
        )
        lines.append(
            f"  Summary:   {comp['summary_words']} words, {comp['summary_characters']} chars"
        )
        lines.append(
            f"  Compression: {comp['word_compression']:.1%} (words), "
            f"{comp['character_compression']:.1%} (chars)"
        )

    lines.append("=" * 60)
    return "\n".join(lines)