# Scientific Document Summarization Framework (Demo)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]

**Demo Implementation of MSc Thesis:** *"An Innovative Self-Reliant Framework for Multi-Stage Summarization of Long Scientific Documents"*

---

**Demo Notice.** This repository contains a **cleaned, simplified demo** of the core pipeline used in my MSc thesis. The full research code, private datasets and trained weights are withheld due to an ongoing paper submission (ACL 2026) and dataset restrictions. This demo is runnable locally in CPU mode and includes fallback implementations so reviewers can run it without GPUs or large downloads.

## Elevator pitch
A two-phase summarization pipeline for long scientific documents:
1. Section-level summarization using semantic representations and contrastive learning (demo).
2. Document-level synthesis using LLM-assisted fusion (demo uses a small HF model or a deterministic fallback).

---

## Quickstart (Windows PowerShell)

1. Create & activate venv:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation: run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

2. Install demo dependencies:
```powershell
pip install -r requirements_demo.txt
```

3. (Optional) download spaCy fallback model:
```powershell
python -m spacy download en_core_web_sm
```

4. Run the demo:
```powershell
python -m scripts.run_demo --input examples\sample_paper.txt --output data\outputs\demo_summary.txt --device cpu
type data\outputs\demo_summary.txt
```

5. Run the contrastive demo:
```powershell
python -m src.contrastive.demo_contrastive
```

6. Evaluate (optional):
```powershell
python -m scripts.evaluate_demo --reference examples\reference_summary.txt --hypothesis data\outputs\demo_summary.txt
```

## Contact / Full code access

The full training scripts, dataset extraction, and final model weights are private until publication. If you need private access for review, please contact me and I can provide a private GitHub repository or run the code during a screen-share.

Best,
Alireza Keshavarz