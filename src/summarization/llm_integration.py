"""
LLM integration for abstractive summarization.

Uses Hugging Face summarization models when available and falls back to a
simple extractive summary when they are not.
"""

import logging
import re
from typing import List, Optional

try:
    from transformers import pipeline
    _has_transformers = True
except ImportError:
    _has_transformers = False


logger = logging.getLogger(__name__)
_model_cache = {}


def _get_summarizer(model_name: str, device: str = "cpu"):
    """
    Load or reuse a summarization pipeline.
    """
    cache_key = f"{model_name}_{device}"

    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if not model_name or model_name.lower() == "none":
        _model_cache[cache_key] = None
        return None

    if not _has_transformers:
        logger.warning("Transformers is not available. Using fallback summarizer.")
        _model_cache[cache_key] = None
        return None

    try:
        hf_device = -1 if device == "cpu" else 0
        logger.info("Loading summarization model: %s", model_name)

        summarizer = pipeline(
            "summarization",
            model=model_name,
            device=hf_device
        )
        _model_cache[cache_key] = summarizer
        return summarizer
    except Exception as e:
        logger.warning("Could not load model %s: %s", model_name, e)
        _model_cache[cache_key] = None
        return None


def _extractive_fallback(
    text: str,
    max_sentences: int = 3,
    max_length: int = 300
) -> str:
    """
    Simple extractive fallback summary.
    """
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return text[:max_length]

    summary = " ".join(sentences[:max_sentences]).strip()

    if summary and not summary.endswith((".", "!", "?")):
        summary += "."

    if len(summary) > max_length:
        summary = summary[:max_length].rsplit(" ", 1)[0] + "..."

    return summary


def summarize_text(
    text: str,
    model_name: str = "facebook/bart-large-cnn",
    max_length: int = 128,
    min_length: int = 30,
    device: str = "cpu",
    num_beams: int = 4
) -> str:
    """
    Generate a summary for the input text.
    """
    if not text or not text.strip():
        return ""

    summarizer = _get_summarizer(model_name, device=device)

    if summarizer is not None:
        try:
            words = text.split()
            if len(words) > 1024:
                text = " ".join(words[:1024])

            result = summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=num_beams
            )
            return result[0]["summary_text"].strip()
        except Exception as e:
            logger.warning("Summarization failed: %s. Using fallback.", e)

    return _extractive_fallback(text, max_sentences=3, max_length=max_length)


def summarize_with_context(
    text: str,
    context: Optional[str] = None,
    keyphrases: Optional[List[str]] = None,
    model_name: str = "facebook/bart-large-cnn",
    max_length: int = 128,
    device: str = "cpu"
) -> str:
    """
    Generate a summary using a little extra context.
    """
    parts = []

    if context:
        parts.append(context.strip())

    if keyphrases:
        selected = [kp.strip() for kp in keyphrases if kp and kp.strip()]
        if selected:
            parts.append("Keyphrases: " + ", ".join(selected[:10]))

    parts.append(text)

    guided_text = "\n\n".join(parts)
    return summarize_text(
        guided_text,
        model_name=model_name,
        max_length=max_length,
        device=device
    )


def batch_summarize(
    texts: List[str],
    model_name: str = "facebook/bart-large-cnn",
    max_length: int = 128,
    device: str = "cpu"
) -> List[str]:
    """
    Summarize a list of texts.
    """
    if not texts:
        return []

    summaries = []
    for text in texts:
        summaries.append(
            summarize_text(
                text,
                model_name=model_name,
                max_length=max_length,
                device=device
            )
        )

    return summaries