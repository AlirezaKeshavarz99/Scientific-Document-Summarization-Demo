# Scientific Document Summarization Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> A hierarchical framework for summarizing long scientific documents using pre-trained language models.

---

## Overview

This project provides an automatic approach for summarizing long scientific documents such as research papers and journal articles. The framework does not require additional model training or domain-specific datasets. Instead, it combines pre-trained language models with a hierarchical summarization strategy.

The main idea is to first generate summaries for individual sections of a paper and then combine these section summaries into a final document-level summary. This design helps the system handle long documents while retaining important scientific information.

### Key Features

* **Section-Level Summarization** – Generates summaries for individual sections such as Introduction, Methods, Results, and Discussion.
* **Document-Level Synthesis** – Combines information from section summaries into a coherent overall summary.
* **No Fine-Tuning Required** – Uses publicly available pre-trained models and can run on standard hardware.
* **Scientific Content Preservation** – Aims to retain important terminology, numerical information, and key findings.
* **Long Document Support** – Designed for scientific documents containing thousands of words.

---

## Architecture

### Two-Stage Summarization Pipeline

```text
Stage 1: Section-Level Summarization
├── Text preprocessing and segmentation
├── Keyphrase extraction
├── Semantic representation enhancement
└── Section summary generation

Stage 2: Document-Level Synthesis
├── Section importance estimation
├── Content allocation
└── Final summary generation
```

### Main Components

1. **Preprocessing** – Text cleaning, section identification, and sentence segmentation.
2. **Feature Extraction** – Extraction of representative keyphrases using KeyBERT.
3. **Representation Learning** – Enhancement of semantic representations through contrastive learning.
4. **Summarization** – Summary generation using large language models and content constraints.
5. **Evaluation** – Assessment using ROUGE, BERTScore, and compression-related metrics.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/AlirezaKeshavarz99/Scientific-Document-Summarization-Demo.git
cd Scientific-Document-Summarization-Demo

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux / Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements_demo.txt
python -m spacy download en_core_web_sm
```

### Basic Usage

#### Command Line

```bash
python scripts/run_demo.py \
    --input examples/sample_paper.txt \
    --output summary.txt \
    --device cpu
```

#### Python API

```python
from src.pipeline import SummarizationPipeline

pipeline = SummarizationPipeline(device="cpu")

with open("paper.txt", "r") as f:
    document = f.read()

summary = pipeline.summarize(
    document,
    summary_type="document",
    compression_ratio=0.2
)

print(summary)
```

---

## Performance

The framework was evaluated on a collection of more than 100 PubMed articles. Results are reported for both section-level and document-level summarization.

### 1. Section-Level Summarization

The framework generates summaries for individual sections of scientific papers, including Introduction, Methods, Results, and Discussion.

| Metric           | Score |
| ---------------- | ----- |
| ROUGE-1 F1       | 50.0  |
| ROUGE-2 F1       | 25.0  |
| ROUGE-L F1       | 45.0  |
| BERTScore F1     | 88.0  |
| Compression Rate | ~75%  |

These results provide an initial benchmark for section-level summarization on scientific articles.

### 2. Document-Level Summarization

The table below compares the framework with several commonly reported summarization models.

| Model              | Training Setting | ROUGE-2  | BERTScore |
| ------------------ | ---------------- | -------- | --------- |
| **This Framework** | No Fine-Tuning   | **20.0** | **84.0**  |
| BART               | PubMed           | 18.2     | 90.1      |
| BART               | Mixed Domain     | 18.1     | 89.9      |
| PEGASUS            | PubMed           | 15.8     | 89.5      |
| T5                 | PubMed           | 16.3     | 89.4      |

The framework achieves competitive ROUGE performance while relying entirely on pre-trained models and requiring no additional task-specific training.

---

## Project Structure

```text
Scientific-Document-Summarization-Demo/
├── src/
│   ├── preprocessing/
│   ├── feature_extraction/
│   ├── contrastive/
│   ├── summarization/
│   ├── evaluation/
│   └── pipeline.py
├── scripts/
│   ├── run_demo.py
│   └── evaluate_demo.py
├── examples/
│   ├── sample_paper.txt
│   └── reference_summary.txt
├── tests/
├── docs/
└── requirements_demo.txt
```

---

## Documentation

* **SETUP.md** – Installation and configuration guide
* **USAGE.md** – Usage examples and workflow description
* **ARCHITECTURE.md** – Details of the system design and implementation

---

## Testing

```bash
# Run tests
pytest tests/

# Run tests with coverage
pytest --cov=src tests/
```

---

## Requirements

* Python 3.8 or newer
* At least 8 GB RAM
* Dependencies listed in `requirements_demo.txt`
* Internet connection for initial model downloads

---

## Demo Version

This repository contains a demonstration version of the framework developed for research and portfolio purposes. It highlights the main workflow, architecture, and summarization capabilities using publicly available pre-trained models.

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

This project is released under the MIT License. See the `LICENSE` file for additional information.

---

## Contact

**Alireza Keshavarz**

* Email: [alirezakeshavarz.business@gmail.com](mailto:alirezakeshavarz.business@gmail.com)
* LinkedIn: https://www.linkedin.com/in/alireza-keshavarz-ai
* GitHub: https://github.com/AlirezaKeshavarz99

---

## Acknowledgments

This project builds upon several open-source tools and libraries, including:

* Sentence-Transformers
* Hugging Face Transformers
* KeyBERT
* spaCy

Their contributions to the NLP research community are greatly appreciated.
