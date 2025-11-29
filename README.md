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


If PowerShell blocks activation: run:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1


Install demo dependencies:

pip install -r requirements_demo.txt


(Optional) download spaCy fallback model:

python -m spacy download en_core_web_sm


Run the demo:

python -m scripts.run_demo --input examples\sample_paper.txt --output data\outputs\demo_summary.txt --device cpu
type data\outputs\demo_summary.txt


Run the contrastive demo:

python -m src.contrastive.demo_contrastive


Evaluate (optional):

python -m scripts.evaluate_demo --reference examples\reference_summary.txt --hypothesis data\outputs\demo_summary.txt


Contact / Full code access

The full training scripts, dataset extraction, and final model weights are private until publication. If you need private access for review, please contact me and I can provide a private GitHub repository or run the code during a screen-share.

Best,
Alireza Keshavarz