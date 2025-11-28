# Framework Architecture Documentation

## Overview
This document describes the architecture of the Multi-Stage Scientific Document Summarization Framework presented in the MSc thesis.

## Two-Phase Architecture

### Phase 1: Section-Level Summarization
1. **Document Segmentation**
   - Identifies logical sections (Abstract, Introduction, Methods, etc.)
   - Uses pattern recognition and semantic analysis

2. **Keyphrase Extraction**
   - Extracts salient terms using KeyBERT
   - Identifies domain-specific terminology

3. **Semantic Representation**
   - Applies contrastive learning with InfoNCE loss
   - Generates refined sentence embeddings

4. **Section Summarization**
   - Uses prompt engineering with LLMs (Llama-2-13B, Mistral-7B)
   - Generates concise, factual section summaries

### Phase 2: Complete Document Summarization
1. **Section Importance Analysis**
   - Computes cosine similarity between sections and abstract
   - Applies Gini-based distinctiveness analysis

2. **Summary Length Allocation**
   - Uses exponential allocation algorithm
   - Distributes summary length based on section importance

3. **Multi-Stage Synthesis**
   - Refines section summaries through iterative processing
   - Applies critical n-gram fusion to preserve terminology

4. **Final Summary Generation**
   - Concatenates and refines section summaries
   - Ensures coherence and completeness

## Key Innovations

### Technical Innovations
- **Contrastive Learning**: Semantic distinction between sections
- **Gini-based Analysis**: Quantitative section importance measurement
- **Exponential Allocation**: Dynamic summary length distribution
- **Critical N-gram Fusion**: Technical terminology preservation

### Methodological Innovations
- **Self-Reliant Operation**: No external training data required
- **Fine-Tuning-Free**: Uses pre-trained models effectively
- **Multi-Stage Evaluation**: Comprehensive assessment protocol

## Performance Highlights
- **ROUGE-1**: 0.50 (25% improvement over baseline)
- **ROUGE-2**: 0.25
- **BERTScore**: 0.88
- **Human Evaluation**: 4.3/5.0

*Note: This demo version showcases the architecture. The complete implementation with novel algorithms is part of ongoing research.*