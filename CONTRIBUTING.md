# Contributing to FRA Diagnostics Platform

First off, thank you for considering contributing to the FRA Diagnostics Platform! 🎉

This document provides guidelines for contributing to this project. Following these guidelines helps maintain code quality and makes the review process smoother for everyone.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Issue Reporting](#issue-reporting)

---

## Code of Conduct

This project adheres to a code of conduct that we expect all participants to follow. By participating, you are expected to uphold this code:

- **Be respectful** and inclusive
- **Be constructive** in feedback
- **Focus on the best** for the community
- **Show empathy** towards other contributors

---

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**When submitting a bug report, include:**

- Clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages
- Screenshots if applicable

**Use this template:**

```markdown
**Description:**
Brief description of the bug

**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10.5]
- Dependencies: [relevant package versions]

**Logs/Screenshots:**
[Attach relevant information]
```

### ✨ Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- Clear use case and motivation
- Detailed description of the proposed functionality
- Examples of how it would work
- Potential impact on existing features
- Alternative solutions considered

### 🔧 Contributing Code

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Make** your changes following our coding standards
4. **Test** your changes thoroughly
5. **Commit** with descriptive messages
6. **Push** to your fork
7. **Submit** a Pull Request

---

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment tool (venv or conda)

### Setup Steps

```bash
# 1. Fork and clone
git clone https://github.com/jineshgoud45/fra-diagnostics.git
cd fra-diagnostics

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Install pre-commit hooks
pre-commit install

# 5. Verify setup
pytest --cov=.
```

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test frequently
pytest

# Run code quality checks
black .
isort .
mypy . --ignore-missing-imports
flake8 .

# Commit changes
git add .
git commit -m "feat: add amazing feature"

# Push and create PR
git push origin feature/your-feature-name
```

---

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length**: 100 characters (not 79)
- **Quotes**: Use single quotes for strings
- **Imports**: Organized with isort
- **Formatting**: Automated with Black

### Type Hints

**Required** for all functions and methods:

```python
def process_data(
    df: pd.DataFrame,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Process FRA data with given threshold."""
    ...
```

### Docstrings

Use **Google-style** docstrings:

```python
def calculate_dtw(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Calculate Dynamic Time Warping distance.
    
    Args:
        seq1: First sequence array
        seq2: Second sequence array
    
    Returns:
        DTW distance as float
        
    Raises:
        ValueError: If sequences are empty
        
    Examples:
        >>> seq1 = np.array([1, 2, 3])
        >>> seq2 = np.array([1, 2, 4])
        >>> dtw_dist = calculate_dtw(seq1, seq2)
        >>> print(f"Distance: {dtw_dist}")
        Distance: 1.0
    """
    ...
```

### Code Organization

- **One class per file** (unless tightly coupled)
- **Group related functions** together
- **Use meaningful names** (avoid abbreviations)
- **Keep functions small** (<50 lines preferred)
- **Avoid deep nesting** (max 3 levels)

### Error Handling

```python
# ✅ Good: Specific exceptions
try:
    df = pd.read_csv(filepath)
except FileNotFoundError:
    logger.error(f"File not found: {filepath}")
    raise FRAParserError(f"Cannot find file: {filepath}")
except pd.errors.EmptyDataError:
    logger.error(f"Empty file: {filepath}")
    raise FRAParserError(f"File is empty: {filepath}")

# ❌ Bad: Bare except
try:
    df = pd.read_csv(filepath)
except:
    pass  # Silent failures are dangerous!
```

### Logging

Use appropriate log levels:

```python
import logging

logger = logging.getLogger(__name__)

logger.debug("Detailed diagnostic information")
logger.info("Confirmation that things are working")
logger.warning("Something unexpected but recoverable")
logger.error("Serious problem, function failed")
logger.critical("Program may crash")
```

---

## Testing Requirements

### Test Coverage

- **Minimum**: 85% coverage required
- **Target**: 90%+ coverage preferred
- **Critical paths**: 100% coverage mandatory

### Test Organization

```
tests/
├── unit/              # Unit tests (fast, isolated)
├── integration/       # Integration tests (slower)
├── security/          # Security tests
└── performance/       # Performance benchmarks
```

### Writing Tests

```python
import unittest
import pytest
from hypothesis import given, strategies as st

class TestParser(unittest.TestCase):
    """Test suite for FRA parser."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = UniversalFRAParser()
    
    def test_parse_omicron_csv(self):
        """Test Omicron CSV parsing."""
        df = self.parser.parse_file('data/omicron.csv')
        self.assertIsNotNone(df)
        self.assertIn('frequency_hz', df.columns)
    
    @pytest.mark.security
    def test_path_traversal_protection(self):
        """Test protection against path traversal attacks."""
        with self.assertRaises(SecurityError):
            self.parser.parse_file('../../../etc/passwd')
    
    @given(n_sections=st.integers(min_value=10, max_value=1000))
    def test_simulator_never_crashes(self, n_sections):
        """Property-based test: simulator should never crash."""
        sim = TransformerSimulator(n_sections=n_sections)
        freq, mag, phase = sim.generate_normal()
        self.assertEqual(len(freq), len(mag))
```

### Running Tests

```bash
# All tests
pytest -v

# Specific marker
pytest -m unit
pytest -m security
pytest -m integration

# With coverage
pytest --cov=. --cov-report=html

# Watch mode (during development)
pytest-watch
```

---

## Pull Request Process

### Before Submitting

1. ✅ **Update documentation** if needed
2. ✅ **Add/update tests** for your changes
3. ✅ **Run full test suite** and ensure it passes
4. ✅ **Check code coverage** doesn't decrease
5. ✅ **Run linters** (black, isort, mypy, flake8)
6. ✅ **Update CHANGELOG.md** with your changes
7. ✅ **Rebase** on latest main branch

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing locally
- [ ] Coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Backward compatibility maintained

## Related Issues
Closes #123
Relates to #456
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by at least one maintainer
3. **Discussion** of any concerns or suggestions
4. **Approval** from maintainer
5. **Merge** by maintainer (squash and merge preferred)

---

## Commit Message Guidelines

Follow **Conventional Commits** specification:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring (no functional changes)
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `chore`: Maintenance tasks (dependencies, build, etc.)
- `ci`: CI/CD changes

### Examples

```bash
# Feature
git commit -m "feat(parser): add support for Doble XML format"

# Bug fix
git commit -m "fix(simulator): correct inductance calculation for deformation"

# Documentation
git commit -m "docs(readme): update installation instructions"

# Breaking change
git commit -m "feat(api)!: change predict() return format

BREAKING CHANGE: predict() now returns dict instead of tuple"
```

---

## Issue Reporting

### Security Issues

**Do NOT** open public issues for security vulnerabilities.

Instead:
1. Email: security@example.com
2. Include detailed description
3. Provide proof of concept if possible
4. Allow reasonable time for fix before disclosure

See [SECURITY.md](SECURITY.md) for details.

### Bug Reports

Use the bug report template provided in GitHub Issues.

### Feature Requests

Use the feature request template provided in GitHub Issues.

---

## Recognition

Contributors will be:
- Listed in [CHANGELOG.md](CHANGELOG.md)
- Credited in release notes
- Added to contributors list

---

## Questions?

- **Discord**: [Join our server](#)
- **Discussions**: [GitHub Discussions](https://github.com/jineshgoud45/fra-diagnostics/discussions)
- **Email**: 23eg109a16@anurag.edu.in

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

<p align="center">
  <b>Thank you for contributing! 🎉</b>
  <br>
  <sub>Your efforts make this project better for everyone</sub>
</p>
