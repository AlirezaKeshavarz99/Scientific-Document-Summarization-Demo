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

## Quickstart (Windows)

1. Create & activate venv (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
