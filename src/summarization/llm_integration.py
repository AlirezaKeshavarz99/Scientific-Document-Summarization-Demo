# src/summarization/llm_integration.py
"""
LLM Integration Module

Provides abstractive summarization using pre-trained language models.
Supports BART, T5, and other Hugging Face summarization models with
graceful fallbacks for demo purposes.
"""

import logging
from typing import List, Optional
import re

try:
    from transformers import pipeline, AutoTokenizer
    _has_transformers = True
except ImportError:
    _has_transformers = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model cache to avoid reloading
_model_cache = {}


def _get_summarizer(model_name: str, device: str = "cpu"):
    """
    Get or create a summarization pipeline.
    
    Args:
        model_name: Name of the Hugging Face model
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Summarization pipeline or None if unavailable
    """
    cache_key = f"{model_name}_{device}"
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    if model_name is None or model_name.lower() == "none":
        _model_cache[cache_key] = None
        return None
    
    if not _has_transformers:
        logger.warning("Transformers not available. Using fallback summarizer.")
        _model_cache[cache_key] = None
        return None
    
    try:
        # Convert device string to pipeline format
        hf_device = -1 if device == "cpu" else 0
        
        logger.info(f"Loading summarization model: {model_name}")
        pipe = pipeline(
            "summarization",
            model=model_name,
            device=hf_device
        )
        _model_cache[cache_key] = pipe
        return pipe
    except Exception as e:
        logger.warning(f"Could not load model {model_name}: {e}")
        logger.warning("Using fallback extractive summarizer")
        _model_cache[cache_key] = None
        return None


def _extractive_fallback(text: str, 
                         max_sentences: int = 3,
                         max_length: int = 300) -> str:
    """
    Fallback extractive summarization by selecting first N sentences.
    
    Args:
        text: Input text
        max_sentences: Maximum number of sentences to include
        max_length: Maximum character length
        
    Returns:
        Extractive summary
    """
    # Clean and split into sentences
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if not sentences:
        return text[:max_length]
    
    # Select first N sentences
    selected = sentences[:max_sentences]
    summary = '. '.join(selected)
    
    # Ensure proper ending
    if summary and not summary.endswith('.'):
        summary += '.'
    
    # Truncate if too long
    if len(summary) > max_length:
        summary = summary[:max_length].rsplit(' ', 1)[0] + '...'
    
    return summary


def summarize_text(text: str,
                  model_name: str = "facebook/bart-large-cnn",
                  max_length: int = 128,
                  min_length: int = 30,
                  device: str = "cpu",
                  num_beams: int = 4) -> str:
    """
    Generate abstractive summary of input text.
    
    Args:
        text: Input text to summarize
        model_name: Name of Hugging Face model to use
        max_length: Maximum length of generated summary
        min_length: Minimum length of generated summary
        device: Device to run on ('cpu' or 'cuda')
        num_beams: Number of beams for beam search
        
    Returns:
        Generated summary text
    """
    if not text or not text.strip():
        return ""
    
    # Get summarizer
    summarizer = _get_summarizer(model_name, device=device)
    
    # Use LLM if available
    if summarizer is not None:
        try:
            # Truncate input if too long (model-dependent max length)
            max_input_length = 1024
            if len(text.split()) > max_input_length:
                text = ' '.join(text.split()[:max_input_length])
            
            result = summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=num_beams
            )
            
            summary = result[0]["summary_text"]
            return summary.strip()
            
        except Exception as e:
            logger.warning(f"Summarization failed: {e}. Using fallback.")
    
    # Fallback to extractive summary
    return _extractive_fallback(text, max_sentences=3, max_length=max_length)


def summarize_with_context(text: str,
                          context: Optional[str] = None,
                          keyphrases: Optional[List[str]] = None,
                          model_name: str = "facebook/bart-large-cnn",
                          max_length: int = 128,
                          device: str = "cpu") -> str:
    """
    Generate summary with additional context and keyphrases.
    
    This function can incorporate keyphrases and context to guide
    the summarization process.
    
    Args:
        text: Input text to summarize
        context: Additional context (e.g., document title, section name)
        keyphrases: Important keyphrases to preserve
        model_name: Name of Hugging Face model
        max_length: Maximum summary length
        device: Device to run on
        
    Returns:
        Generated summary
    """
    # For now, use standard summarization
    # In full implementation, this would incorporate context and keyphrases
    # into the prompt or model input
    
    summary = summarize_text(
        text,
        model_name=model_name,
        max_length=max_length,
        device=device
    )
    
    return summary


def batch_summarize(texts: List[str],
                   model_name: str = "facebook/bart-large-cnn",
                   max_length: int = 128,
                   device: str = "cpu") -> List[str]:
    """
    Summarize multiple texts in batch.
    
    Args:
        texts: List of texts to summarize
        model_name: Name of Hugging Face model
        max_length: Maximum summary length
        device: Device to run on
        
    Returns:
        List of summaries
    """
    if not texts:
        return []
    
    summaries = []
    for text in texts:
        summary = summarize_text(
            text,
            model_name=model_name,
            max_length=max_length,
            device=device
        )
        summaries.append(summary)
    
    return summaries
