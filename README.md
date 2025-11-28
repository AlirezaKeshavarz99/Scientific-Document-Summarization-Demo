<<<<<<< HEAD
=======
# Scientific Document Summarization Framework (Demo)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Demo Implementation of MSc Thesis:** *"An Innovative Self-Rellant Framework for Multi-Stage Summarization of Long Scientific Documents"*

## 🔬 Overview

This repository contains a **demonstration version** of the multi-stage summarization framework developed during my MSc research. The framework addresses the challenge of efficiently summarizing long scientific documents without requiring external training data or fine-tuning.

> **📝 Publication Status**: The complete implementation with novel algorithms is part of ongoing research being prepared for ACL 2026. This demo showcases the architecture and software engineering approach.

## 🏗️ Architecture

The framework employs a sophisticated two-phase approach:

### Phase 1: Section-Level Summarization
- **Document Segmentation**: Identifies logical sections (Abstract, Introduction, Methods, etc.)
- **Keyphrase Extraction**: Uses KeyBERT to extract salient terms
- **Semantic Representation**: Contrastive learning for refined embeddings
- **LLM Summarization**: Prompt engineering with large language models

### Phase 2: Complete Document Summarization  
- **Section Importance**: Cosine similarity and Gini-based analysis
- **Length Allocation**: Exponential allocation algorithm
- **Multi-Stage Synthesis**: Critical n-gram fusion and refinement
- **Final Summary**: Coherent document-level overview

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/AlirezaKeshavarz99/Scientific-Document-Summarization-Demo.git
cd Scientific-Document-Summarization-Demo
pip install -r requirements.txt
>>>>>>> 835666ce5e97729aba823ea4462c55e65b6cc296
