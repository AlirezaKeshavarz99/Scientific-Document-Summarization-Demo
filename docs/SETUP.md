# Setup Guide

## System Requirements

### Hardware
- **CPU:** Multi-core processor (4+ cores recommended)
- **RAM:** Minimum 8GB, 16GB recommended
- **Storage:** 5GB free space for models and dependencies
- **GPU:** Optional (provides 5-10x speedup)

### Software
- **Operating System:** Windows 10/11, Linux (Ubuntu 20.04+), macOS 10.15+
- **Python:** Version 3.8, 3.9, 3.10, or 3.11
- **Git:** For cloning the repository

---

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AlirezaKeshavarz99/Scientific-Document-Summarization-Demo.git
cd Scientific-Document-Summarization-Demo
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you encounter execution policy errors:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements_demo.txt
```

### 4. Download Required Models

**spaCy Language Model:**
```bash
python -m spacy download en_core_web_sm
```

**Optional - Scientific spaCy Model (better accuracy):**
```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
```

### 5. Verify Installation

```bash
python -c "import src; print('Installation successful!')"
```

---

## Configuration

### Environment Variables (Optional)

Create a `.env` file in the project root:

```env
# Model configurations
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=mistral-7b-instruct
DEVICE=cpu  # or 'cuda' for GPU

# Processing parameters
MAX_SECTION_LENGTH=5000
BATCH_SIZE=16
COMPRESSION_RATIO=0.2

# Output settings
OUTPUT_DIR=data/outputs
LOG_LEVEL=INFO
```

### Model Cache Directory

By default, Hugging Face models are cached in:
- **Windows:** `C:\Users\<username>\.cache\huggingface`
- **Linux/macOS:** `~/.cache/huggingface`

To change the cache location, set:
```bash
export HF_HOME=/path/to/cache  # Linux/macOS
set HF_HOME=C:\path\to\cache   # Windows CMD
$env:HF_HOME="C:\path\to\cache" # Windows PowerShell
```

---

## Testing the Installation

### Quick Test

```bash
python scripts/run_demo.py --input examples/sample_paper.txt --output test_output.txt
```

If successful, you should see:
```
Preprocessing document...
Extracting keyphrases...
Generating embeddings...
Creating summary...
Summary saved to test_output.txt
```

### Run Unit Tests

```bash
pytest tests/ -v
```

---

## Troubleshooting

### Common Issues

#### 1. **Module Not Found Error**

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Make sure you're in the project root directory
cd Scientific-Document-Summarization-Demo

# Install the package in editable mode
pip install -e .
```

#### 2. **CUDA Out of Memory (GPU)**

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**
- Reduce batch size in configuration
- Use CPU mode: `--device cpu`
- Use 8-bit quantization (automatically enabled for demo)

#### 3. **spaCy Model Not Found**

**Problem:** `OSError: [E050] Can't find model 'en_core_web_sm'`

**Solution:**
```bash
python -m spacy download en_core_web_sm
```

#### 4. **SSL Certificate Error**

**Problem:** `SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]`

**Solution:**
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements_demo.txt
```

#### 5. **Slow Model Download**

**Problem:** Hugging Face model download is very slow

**Solution:**
- Use a mirror (if in China):
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```
- Or download models manually from https://huggingface.co/

#### 6. **Permission Denied (Windows PowerShell)**

**Problem:** Cannot run activation script

**Solution:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## GPU Setup (Optional)

### NVIDIA CUDA Setup

**1. Check GPU Availability:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

**2. Install CUDA-enabled PyTorch:**

Visit https://pytorch.org/ and select your configuration, then install:

```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**3. Run with GPU:**
```bash
python scripts/run_demo.py --input examples/sample_paper.txt --output output.txt --device cuda
```

---

## Development Setup

For contributing or modifying the code:

### 1. Install Development Dependencies

```bash
pip install -r requirements_dev.txt  # If available
# Or manually:
pip install pytest pytest-cov black flake8 mypy
```

### 2. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### 3. Code Formatting

```bash
# Format code
black src/ tests/ scripts/

# Check linting
flake8 src/ tests/ scripts/

# Type checking
mypy src/
```

---

## Updating the Framework

To get the latest updates:

```bash
git pull origin main
pip install --upgrade -r requirements_demo.txt
```

---

## Uninstallation

To completely remove the framework:

```bash
# Deactivate virtual environment
deactivate

# Remove the project directory
cd ..
rm -rf Scientific-Document-Summarization-Demo  # Linux/macOS
Remove-Item -Recurse -Force Scientific-Document-Summarization-Demo  # Windows PowerShell
```

---

## Performance Optimization Tips

### 1. Use GPU if Available
- 5-10x faster inference
- Enable with `--device cuda`

### 2. Adjust Batch Size
- Larger batches = faster processing
- May require more memory
- Configure in settings or pass `--batch-size 32`

### 3. Use Smaller Models
- Trade-off: Speed vs. accuracy
- Edit `src/config.py` to change model selection

### 4. Enable Mixed Precision (GPU only)
```python
# In your code
torch.set_float32_matmul_precision('medium')
```

### 5. Cache Embeddings
- Pre-compute embeddings for frequently used documents
- Store in `data/cache/`

---

## Additional Resources

- **Hugging Face Documentation:** https://huggingface.co/docs
- **spaCy Documentation:** https://spacy.io/usage
- **PyTorch Documentation:** https://pytorch.org/docs/

---

## Support

If you encounter issues not covered here:

1. Check existing GitHub issues
2. Review the troubleshooting section
3. Contact: a.keshavarz@khu.ac.ir

---

**Document Version:** 1.0  
**Last Updated:** December 2025  
**Author:** Alireza Keshavarz
