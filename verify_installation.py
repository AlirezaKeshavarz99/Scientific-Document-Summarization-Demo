#!/usr/bin/env python
"""
Installation Verification Script

Checks that all dependencies are properly installed and the framework is ready to use.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_dependencies():
    """Check required dependencies."""
    print("\nChecking dependencies...")
    
    dependencies = {
        'numpy': 'NumPy',
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'sentence_transformers': 'Sentence-Transformers',
        'spacy': 'spaCy',
        'nltk': 'NLTK',
        'yaml': 'PyYAML',
        'sklearn': 'scikit-learn',
    }
    
    optional_dependencies = {
        'keybert': 'KeyBERT',
        'bert_score': 'BERTScore',
    }
    
    all_good = True
    
    # Check required dependencies
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (required)")
            all_good = False
    
    # Check optional dependencies
    print("\nChecking optional dependencies...")
    for module, name in optional_dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ○ {name} (optional, recommended)")
    
    return all_good


def check_spacy_models():
    """Check if spaCy models are installed."""
    print("\nChecking spaCy models...")
    
    try:
        import spacy
        
        # Try to load en_core_web_sm
        try:
            nlp = spacy.load("en_core_web_sm")
            print("  ✓ en_core_web_sm")
        except OSError:
            print("  ✗ en_core_web_sm (run: python -m spacy download en_core_web_sm)")
            return False
        
        return True
    except ImportError:
        print("  ✗ spaCy not installed")
        return False


def check_nltk_data():
    """Check if NLTK data is available."""
    print("\nChecking NLTK data...")
    
    try:
        import nltk
        
        required_data = ['punkt', 'stopwords', 'wordnet']
        all_available = True
        
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}' if data_name == 'punkt' 
                              else f'corpora/{data_name}')
                print(f"  ✓ {data_name}")
            except LookupError:
                print(f"  ○ {data_name} (will be downloaded automatically)")
        
        return True
    except ImportError:
        print("  ✗ NLTK not installed")
        return False


def check_project_structure():
    """Check project structure."""
    print("\nChecking project structure...")
    
    required_dirs = [
        'src',
        'scripts',
        'tests',
        'examples',
        'configs',
        'docs',
    ]
    
    all_good = True
    project_root = Path(__file__).parent
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (missing)")
            all_good = False
    
    return all_good


def check_src_imports():
    """Check if src modules can be imported."""
    print("\nChecking src module imports...")
    
    modules = [
        'src.pipeline',
        'src.preprocessing.segmenter',
        'src.feature_extraction.keyphrase_extractor',
        'src.summarization.llm_integration',
        'src.evaluation.metrics',
    ]
    
    all_good = True
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except Exception as e:
            print(f"  ✗ {module_name} ({str(e)[:50]}...)")
            all_good = False
    
    return all_good


def check_example_files():
    """Check if example files exist."""
    print("\nChecking example files...")
    
    example_files = [
        'examples/sample_paper.txt',
        'examples/reference_summary.txt',
    ]
    
    all_good = True
    project_root = Path(__file__).parent
    
    for file_path in example_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            all_good = False
    
    return all_good


def check_gpu_availability():
    """Check GPU availability."""
    print("\nChecking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✓ CUDA available: {gpu_name}")
            print(f"    PyTorch CUDA version: {torch.version.cuda}")
        else:
            print("  ○ CUDA not available (CPU mode will be used)")
        return True
    except Exception as e:
        print(f"  ○ Could not check GPU ({str(e)})")
        return True


def main():
    """Main verification function."""
    print("="*70)
    print("SCIENTIFIC DOCUMENT SUMMARIZATION - INSTALLATION VERIFICATION")
    print("="*70)
    
    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'spaCy Models': check_spacy_models(),
        'NLTK Data': check_nltk_data(),
        'Project Structure': check_project_structure(),
        'Module Imports': check_src_imports(),
        'Example Files': check_example_files(),
        'GPU': check_gpu_availability(),
    }
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name:.<50} {status}")
    
    all_passed = all(results.values())
    
    print("="*70)
    
    if all_passed:
        print("\n✓ All checks passed! The framework is ready to use.")
        print("\nNext steps:")
        print("  1. Try the demo: python scripts/run_demo.py --input examples/sample_paper.txt --output test.txt")
        print("  2. Run tests: pytest tests/")
        print("  3. Read the documentation: docs/USAGE.md")
    else:
        print("\n✗ Some checks failed. Please review the errors above.")
        print("\nCommon fixes:")
        print("  - Install missing dependencies: pip install -r requirements_demo.txt")
        print("  - Download spaCy model: python -m spacy download en_core_web_sm")
        print("  - Check Python version: python --version (requires 3.8+)")
    
    print("\n" + "="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
