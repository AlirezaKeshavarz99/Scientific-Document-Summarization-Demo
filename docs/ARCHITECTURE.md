# ARCHITECTURE & METHODOLOGY

This document contains the full technical description of the two-phase, eight-step summarization approach described in the MSc thesis. Some experimental scripts, dataset preparation, and final model weights are private and are marked as "PROTECTED".

## Overview
- Phase 1: Preparation (contrastive learning, keyphrase extraction, section importance scoring, exponential allocation).
- Phase 2: Generation (intra-section fusion with LLM, final document synthesis).

## Protected components (not included in demo)
- Full contrastive training on PubMed dataset (scripts and trained weights).
- Mistral-7B full experiments using 8-bit quantization (private model weights).
- Large-scale evaluation scripts that require private datasets.

## Demo mapping
List of demo files and what they demonstrate:
- `src/contrastive/demo_contrastive.py` — toy InfoNCE implementation showing the contrastive principle.
- `src/feature_extraction/keyphrase_extractor.py` — TF-IDF / KeyBERT wrapper for keyphrase extraction (demo uses TF-IDF fallback).
- `src/summarization/llm_integration.py` — summarizer wrapper with deterministic fallback.
- `src/pipeline.py` — orchestrator covering segmentation, importance scoring (demo), allocation, and synthesis.

(Include further methodological details from the thesis as needed; mark private lines.)
