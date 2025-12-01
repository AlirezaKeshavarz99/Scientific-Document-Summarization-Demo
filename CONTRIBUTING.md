# Contributing to Scientific Document Summarization Framework

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other contributors

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in Issues
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce the bug
   - Expected vs. actual behavior
   - Your environment (OS, Python version, etc.)
   - Error messages or screenshots

### Suggesting Enhancements

1. Check existing issues and discussions
2. Create a new issue with:
   - Clear description of the enhancement
   - Rationale for the change
   - Potential implementation approach
   - Any relevant examples or references

### Pull Requests

1. Fork the repository
2. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes following the coding standards below
4. Add or update tests as needed
5. Update documentation if required
6. Commit with clear, descriptive messages:
   ```bash
   git commit -m "Add: brief description of changes"
   ```
7. Push to your fork and submit a pull request

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/your-username/Scientific-Document-Summarization-Demo.git
   cd Scientific-Document-Summarization-Demo
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements_demo.txt
   pip install pytest pytest-cov black flake8
   ```

4. Install in editable mode:
   ```bash
   pip install -e .
   ```

## Coding Standards

### Style Guidelines

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Maximum line length: 100 characters
- Use type hints where appropriate

### Code Formatting

Format code with Black before committing:
```bash
black src/ tests/ scripts/
```

Check linting:
```bash
flake8 src/ tests/ scripts/ --max-line-length=100
```

### Documentation

- Add docstrings to all functions, classes, and modules
- Use Google-style docstrings:
  ```python
  def function_name(arg1: type, arg2: type) -> return_type:
      """
      Brief description.
      
      Longer description if needed.
      
      Args:
          arg1: Description of arg1
          arg2: Description of arg2
          
      Returns:
          Description of return value
          
      Raises:
          ExceptionType: When this exception is raised
      """
  ```

- Update README.md if adding new features
- Update relevant documentation in `docs/`

### Testing

- Write tests for new functionality
- Maintain or improve test coverage
- Run tests before submitting PR:
  ```bash
  pytest tests/ -v
  ```

- Run with coverage:
  ```bash
  pytest --cov=src tests/
  ```

### Commit Messages

Use clear, descriptive commit messages:

- **Add:** New features or files
  - `Add: KeyBERT integration for keyphrase extraction`
- **Fix:** Bug fixes
  - `Fix: Handle empty document sections gracefully`
- **Update:** Changes to existing functionality
  - `Update: Improve error messages in pipeline`
- **Refactor:** Code restructuring without changing behavior
  - `Refactor: Simplify section importance calculation`
- **Docs:** Documentation changes
  - `Docs: Add usage examples for API`
- **Test:** Adding or updating tests
  - `Test: Add unit tests for segmenter module`

## Project Structure

```
Scientific-Document-Summarization-Demo/
├── src/                      # Main source code
│   ├── preprocessing/        # Text preprocessing
│   ├── feature_extraction/   # Feature extraction
│   ├── contrastive/          # Contrastive learning
│   ├── summarization/        # Summarization engine
│   ├── evaluation/           # Evaluation metrics
│   └── utils/                # Utility functions
├── scripts/                  # Command-line scripts
├── tests/                    # Unit tests
├── examples/                 # Example files
├── docs/                     # Documentation
└── configs/                  # Configuration files
```

## Areas for Contribution

### High Priority
- Additional language support
- Improved evaluation metrics
- Performance optimizations
- Better error handling

### Medium Priority
- Additional example documents
- Enhanced visualization tools
- Integration with other frameworks
- Documentation improvements

### Ideas Welcome
- Novel summarization techniques
- Alternative model integrations
- Specialized domain adaptations
- User interface improvements

## Questions?

- Open an issue for questions
- Join discussions in the Issues section
- Contact: a.keshavarz@khu.ac.ir

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be acknowledged in:
- README.md (Contributors section)
- CHANGELOG.md (for significant contributions)
- Release notes

Thank you for contributing to the Scientific Document Summarization Framework!
