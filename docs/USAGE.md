# Usage Guide

This guide covers basic and advanced usage of the Scientific Document Summarization Framework.

---

## Quick Start

### Basic Command-Line Usage

```bash
python scripts/run_demo.py --input examples/sample_paper.txt --output summary.txt
```

### View Output

```bash
# Windows
type summary.txt

# Linux/Mac
cat summary.txt
```

---

## Command-Line Options

### `run_demo.py`

Generate summaries from scientific documents.

**Usage:**
```bash
python scripts/run_demo.py [OPTIONS]
```

**Options:**
- `--input PATH` - Input text file (required)
- `--output PATH` - Output summary file (required)
- `--summary-type {section,document}` - Summary type (default: document)
- `--compression-ratio FLOAT` - Target compression 0.1-0.5 (default: 0.2)
- `--device {cpu,cuda}` - Processing device (default: cpu)
- `--verbose` - Enable detailed logging

**Examples:**
```bash
# Document-level summary
python scripts/run_demo.py \
    --input paper.txt \
    --output summary.txt \
    --summary-type document \
    --compression-ratio 0.2

# Section-level summaries
python scripts/run_demo.py \
    --input paper.txt \
    --output sections.txt \
    --summary-type section \
    --compression-ratio 0.3
```

### `evaluate_demo.py`

Evaluate generated summaries against reference.

**Usage:**
```bash
python scripts/evaluate_demo.py --reference ref.txt --hypothesis gen.txt
```

**Options:**
- `--reference PATH` - Reference summary file (required)
- `--hypothesis PATH` - Generated summary file (required)
- `--metrics` - Comma-separated list: rouge,bertscore,meteor (default: all)

---

## Python API

### Basic Pipeline

```python
from src.pipeline import SummarizationPipeline

# Initialize pipeline
pipeline = SummarizationPipeline(device='cpu')

# Load document
with open('paper.txt', 'r', encoding='utf-8') as f:
    document = f.read()

# Generate summary
summary = pipeline.summarize(
    document,
    summary_type='document',
    compression_ratio=0.2
)

print(summary)
```

### Section-Level Summaries

```python
# Generate section-specific summaries
section_summaries = pipeline.summarize(
    document,
    summary_type='section',
    compression_ratio=0.25
)

# Access individual sections
for section_name, summary_text in section_summaries.items():
    print(f"\n## {section_name}")
    print(summary_text)
```

### Batch Processing

```python
from pathlib import Path

pipeline = SummarizationPipeline(device='cpu')

# Process multiple files
input_dir = Path('papers/')
output_dir = Path('summaries/')
output_dir.mkdir(exist_ok=True)

for paper_file in input_dir.glob('*.txt'):
    with open(paper_file, 'r') as f:
        document = f.read()
    
    summary = pipeline.summarize(document)
    
    output_file = output_dir / f"{paper_file.stem}_summary.txt"
    with open(output_file, 'w') as f:
        f.write(summary)
    
    print(f"Processed: {paper_file.name}")
```

---

## Use Cases

### 1. Literature Review

Quickly extract key information from multiple papers:

```python
papers = ['paper1.txt', 'paper2.txt', 'paper3.txt']
pipeline = SummarizationPipeline()

for paper_path in papers:
    with open(paper_path, 'r') as f:
        summary = pipeline.summarize(f.read(), compression_ratio=0.15)
    print(f"\n{paper_path}:\n{summary}\n")
```

### 2. Methods Extraction

Extract just the methodology section:

```python
section_summaries = pipeline.summarize(document, summary_type='section')

if 'Methods' in section_summaries:
    methods_summary = section_summaries['Methods']
    print(methods_summary)
```

### 3. Abstract Generation

Generate concise abstracts:

```python
abstract = pipeline.summarize(
    document,
    summary_type='document',
    compression_ratio=0.1  # Highly compressed
)
```

---

## Best Practices

### Input Formatting

- Use plain text files (.txt)
- UTF-8 encoding recommended
- Keep original formatting (sections, paragraphs)
- Citations can be included (will be preserved)

### Compression Ratios

- **0.1-0.15**: Very concise, abstract-like
- **0.2-0.25**: Balanced summary (recommended)
- **0.3-0.4**: Detailed summary
- **0.4-0.5**: Light compression

### Performance Tips

- Use CPU for single documents (< 10 pages)
- Use CUDA for batch processing or long documents
- Larger compression ratios process faster
- Section-level summarization is faster than document-level

---

## Output Formats

### Text Output

Standard output is plain text:

```
The study investigated... Key findings include... 
Methods involved... Results showed...
```

### JSON Output (Python API)

```python
import json

result = {
    'input_file': 'paper.txt',
    'summary_type': 'document',
    'summary': summary,
    'compression_ratio': 0.2,
    'original_length': len(document),
    'summary_length': len(summary)
}

with open('output.json', 'w') as f:
    json.dump(result, f, indent=2)
```

---

## Troubleshooting

### Common Issues

**ImportError: No module named 'src'**
- Solution: Run from repository root directory
- Or: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

**spaCy model not found**
- Solution: `python -m spacy download en_core_web_sm`

**Out of memory**
- Solution: Use smaller compression ratio
- Or: Process shorter documents
- Or: Use CPU instead of CUDA

**Empty or poor quality summary**
- Check input file formatting
- Ensure document has clear section structure
- Try different compression ratios

---

## Examples Directory

The `examples/` directory contains:

- `sample_paper.txt` - Example scientific paper
- `reference_summary.txt` - Reference summary for evaluation

Try running:
```bash
python scripts/run_demo.py \
    --input examples/sample_paper.txt \
    --output my_first_summary.txt
```

---

## Advanced Topics

### Custom Pipelines

For advanced users who need custom behavior, see the source code in `src/pipeline.py`.

### Evaluation

Compare generated summaries:

```bash
python scripts/evaluate_demo.py \
    --reference examples/reference_summary.txt \
    --hypothesis my_first_summary.txt
```

This outputs ROUGE, BERTScore, and METEOR metrics.

---

## Getting Help

If you encounter issues:

1. Check this documentation
2. Review `docs/SETUP.md` for installation issues
3. Check `docs/ARCHITECTURE.md` for system design
4. Open an issue on GitHub
5. Contact: a.keshavarz@khu.ac.ir

---

**Next Steps:**
- Try the basic examples above
- Experiment with different compression ratios
- Process your own scientific papers
- Review evaluation metrics for quality assessment
