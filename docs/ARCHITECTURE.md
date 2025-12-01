# System Architecture

## Overview

This document describes the architectural design of the Scientific Document Summarization Framework.

## High-Level Architecture

```
┌─────────────────────────────────────────────┐
│          INPUT DOCUMENT                      │
│     (Scientific Paper - Text)                │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│      PREPROCESSING MODULE                    │
│  • Text Cleaning                             │
│  • Section Segmentation                      │
│  • Sentence Tokenization                     │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│    FEATURE EXTRACTION MODULE                 │
│  • Keyphrase Extraction (KeyBERT)          │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│    CONTRASTIVE LEARNING MODULE               │
│  • Embedding Enhancement                     │
│  • InfoNCE Loss Optimization                 │
└──────────────┬──────────────────────────────┘
               │
      ┌────────┴────────┐
      ▼                  ▼
  PHASE 1            PHASE 2
  Section         Document
  Summaries       Summary
```

## Component Details

### 1. Preprocessing Module
- Scientific text cleaning
- Section segmentation
- Sentence tokenization with spaCy

### 2. Feature Extraction
- Dynamic keyphrase extraction
- MMR diversity optimization

### 3. Contrastive Learning
- Sentence embedding refinement
- Positive/negative pair generation
- InfoNCE loss optimization

### 4. Summarization Engine
- LLM-based sentence fusion
- Multi-attempt validation

### 5. Evaluation Framework
- ROUGE, BERTScore, METEOR metrics

