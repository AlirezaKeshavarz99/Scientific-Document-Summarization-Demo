# Scientific Document Summarization Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> A self-reliant framework for hierarchical summarization of scientific documents using pre-trained language models.

---

## Overview

This framework provides automatic summarization of long scientific documents (research papers, articles) without requiring fine-tuning or domain-specific training data. It supports both **section-level** and **document-level** summarization through a two-phase hierarchical approach.

**Key Features:**
- Handles documents with 8,000+ words
- Generates section-specific summaries (Introduction, Methods, Results, etc.)
- Preserves scientific terminology, citations, and numerical values
- Operates entirely on CPU (GPU optional for faster processing)
- Built on pre-trained models (SBERT, Llama-2, Mistral)

---

## Architecture

### Two-Phase Process

```
Phase 1: Section-Level Summarization
├── Text preprocessing and segmentation
├── Keyphrase extraction
├── Semantic embedding enhancement
└── Section summary generation

Phase 2: Document-Level Synthesis
├── Section importance scoring
├── Proportional content allocation
└── Final document summary
```

### Components

1. **Preprocessing** - Text cleaning, section detection, sentence segmentation
2. **Feature Extraction** - Keyphrase extraction with KeyBERT
3. **Contrastive Learning** - Semantic representation enhancement via InfoNCE loss
4. **Summarization** - LLM-based text fusion with constraint checking
5. **Evaluation** - ROUGE, BERTScore, and compression metrics

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/AlirezaKeshavarz99/Scientific-Document-Summarization-Demo.git
cd Scientific-Document-Summarization-Demo

# Create and activate virtual environment
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1
# Linux/Mac: source venv/bin/activate

# Install dependencies
pip install -r requirements_demo.txt
python -m spacy download en_core_web_sm
```

### Basic Usage

**Command Line:**
```bash
python scripts/run_demo.py \
    --input examples/sample_paper.txt \
    --output summary.txt \
    --device cpu
```

**Python API:**
```python
from src.pipeline import SummarizationPipeline

# Initialize
pipeline = SummarizationPipeline(device='cpu')

# Load document
with open('paper.txt', 'r') as f:
    document = f.read()

# Generate summary
summary = pipeline.summarize(
    document,
    summary_type='document',  # or 'section'
    compression_ratio=0.2
)

print(summary)
```

---

## Performance

Evaluated on 100+ PubMed articles:

| Metric | Section-Level | Document-Level |
|--------|---------------|----------------|
| **ROUGE-1 F1** | 0.50 | 0.44 |
| **ROUGE-2 F1** | 0.25 | 0.20 |
| **ROUGE-L F1** | 0.45 | 0.40 |
| **BERTScore F1** | 0.88 | 0.84 |
| **Compression** | ~75% | ~80% |

---

## Project Structure

```
Scientific-Document-Summarization-Demo/
├── src/
│   ├── preprocessing/        # Text cleaning and segmentation
│   ├── feature_extraction/   # Keyphrase extraction
│   ├── contrastive/          # Embedding enhancement
│   ├── summarization/        # LLM integration
│   ├── evaluation/           # Metrics and scoring
│   └── pipeline.py           # Main orchestration
├── scripts/
│   ├── run_demo.py          # Command-line interface
│   └── evaluate_demo.py     # Evaluation script
├── examples/
│   ├── sample_paper.txt     # Example input
│   └── reference_summary.txt # Reference for evaluation
├── tests/                    # Unit tests
├── docs/                     # Additional documentation
└── requirements_demo.txt     # Dependencies
```

---

## Documentation

- **[SETUP.md](docs/SETUP.md)** - Detailed installation and configuration
- **[USAGE.md](docs/USAGE.md)** - Comprehensive usage examples
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design details

---

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

---

## Requirements

- Python 3.8+
- 8GB RAM minimum
- Dependencies listed in `requirements_demo.txt`
- Internet connection for initial model downloads (~1GB)

---

## Demo Notice

This is a demonstration version created for portfolio and research purposes. It showcases the framework architecture and core capabilities using publicly available pre-trained models.

---

## Citation

```bibtex
@mastersthesis{keshavarz2025scientific,
  author  = {Alireza Keshavarz},
  title   = {An Innovative Self-Reliant Framework for Multi-Stage 
             Summarization of Long Scientific Documents},
  school  = {Kharazmi University},
  year    = {2025},
  type    = {MSc Thesis}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Contact

**Alireza Keshavarz**
- Email: a.keshavarz@khu.ac.ir
- LinkedIn: [linkedin.com/in/alireza-keshavarz-ai](https://www.linkedin.com/in/alireza-keshavarz-ai)
- GitHub: [github.com/AlirezaKeshavarz99](https://github.com/AlirezaKeshavarz99)

---

## Acknowledgments

- Sentence-Transformers for semantic embeddings
- Hugging Face Transformers ecosystem
- KeyBERT for keyphrase extraction
- spaCy for NLP preprocessing
