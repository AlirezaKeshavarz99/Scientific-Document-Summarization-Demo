# src/summarization/llm_integration.py
import logging
from transformers import pipeline
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
_model_cache = {}

def _get_summarizer(model_name: str, device: str = "cpu"):
    key = f"{model_name}_{device}"
    if key in _model_cache:
        return _model_cache[key]
    if model_name is None or model_name.lower() == "none":
        _model_cache[key] = None
        return None
    try:
        # device mapping: device == "cpu" -> device=-1 in pipeline; device != "cpu" -> 0
        hf_device = -1 if device == "cpu" else 0
        pipe = pipeline("summarization", model=model_name, device=hf_device)
        _model_cache[key] = pipe
        return pipe
    except Exception as e:
        logging.warning(f"Could not load model {model_name}: {e}. Using deterministic fallback summarizer.")
        _model_cache[key] = None
        return None

def summarize_text(text: str, model_name: str = "facebook/bart-large-cnn", max_length: int = 128, device: str = "cpu") -> str:
    summarizer = _get_summarizer(model_name, device=device)
    if summarizer is None:
        # Deterministic fallback: first 2 sentences or first 300 characters
        split = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        if len(split) >= 2:
            return ". ".join(split[:2]).strip() + "."
        return text.strip()[:300]
    try:
        out = summarizer(text, max_length=max_length, min_length=20, do_sample=False)
        return out[0]["summary_text"]
    except Exception as e:
        logging.warning(f"Summarizer runtime failure: {e}. Using fallback.")
        split = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        return (". ".join(split[:2]) + ".") if split else text[:300]
