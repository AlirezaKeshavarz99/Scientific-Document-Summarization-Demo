# Scientific Document Summarization Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> A self-reliant framework for hierarchical summarization of scientific documents using pre-trained language models.

---

## Overview

This framework provides automatic summarization of long scientific documents (research papers, articles) without requiring fine-tuning or domain-specific training data. It introduces a novel **hierarchical approach** that performs **section-level summarization** (a new task for PubMed articles) followed by **document-level synthesis**.

**Key Innovations & Features:**
*   **Novel Section-Level Summarization**: Introduces and evaluates the task of generating summaries for individual paper sections (Introduction, Methods, Results, etc.) on PubMed articles.
*   **Document-Level Synthesis**: Produces a coherent final summary by integrating section-level insights.
*   **No Fine-Tuning Required**: Operates effectively on CPU using pre-trained models (SBERT, Llama-2, Mistral).
*   **Preserves Scientific Integrity**: Maintains key terminology, citations, and numerical values.
*   **Handles Long Documents**: Processes documents with 8,000+ words through a structured two-phase pipeline.

---

## Architecture

### Two-Phase Hierarchical Process


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

The framework was evaluated on a corpus of 100+ PubMed articles. The results are presented separately for the novel **section-level** task and the standard **document-level** task.

**1. Section-Level Summarization (Novel Contribution)**

This framework introduces and evaluates the task of generating individual summaries for key sections of a scientific paper (e.g., Introduction, Methods, Results, Discussion). Performance on this novel task is as follows:

| Metric | Section-Level |
|--------|---------------|
| **ROUGE-1 F1** | 0.50 |
| **ROUGE-2 F1** | 0.25 |
| **ROUGE-L F1** | 0.45 |
| **BERTScore F1** | 0.88 |
| **Compression** | ~75% |

These results establish a baseline for the novel task of section-level summarization on PubMed articles.


**2. Document-Level Summarization (Benchmark Comparison)**

For the standard document-level summarization task, the framework's performance is competitive with state-of-the-art models fine-tuned on PubMed, as shown in the comparison below.

| Model / Framework | Training Data | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |
|--------|---------------|----------------|----------------|----------------| ----------------|
| **This Framework** | **No Fine-Tuning** (Zero-shot) | **44.0** | **20.0** | **40.0** | **84.0** |
| BART (Benchmark) | In-domain (PubMed only) | 42.0 | 18.2 | 37.9 | 90.1 |
| BART (Benchmark) | Mixed-domain | 41.7 | 18.1 | 37.6 | 89.9 |
| PEGASUS (Benchmark) | In-domain (PubMed only) | 39.8 | 15.8 | 36.5 | 89.5 |
| T5 (Benchmark) | In-domain (PubMed only) | 40.2 | 16.3 | 36.6 | 89.4 |

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
