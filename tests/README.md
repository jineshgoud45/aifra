# FRA Diagnostics Platform - Test Suite
**SIH 2025 PS 25190**

Comprehensive test suite for the Frequency Response Analysis (FRA) diagnostics platform.

## 📋 Test Coverage

### Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `test_simulator.py` | 40+ tests | Physics, fault generation, validation |
| `test_parser.py` | 45+ tests | Parsing, QA checks, security (XXE) |
| `test_security.py` | 35+ tests | File validation, path traversal, XXE |

### Coverage Areas

✅ **Unit Tests**
- Simulator initialization and validation
- Physics correctness (energy conservation, causality)
- Fault signature generation (all 6 fault types)
- Parser vendor detection and format handling
- IEC 60076-18 QA compliance checks

✅ **Integration Tests**
- End-to-end file parsing
- Multi-vendor format support
- DataFrame normalization

✅ **Security Tests**
- XXE (XML External Entity) attack prevention
- Path traversal protection
- File size limit enforcement
- Magic byte validation
- Binary file rejection

✅ **Edge Cases**
- Empty files
- Invalid data formats
- Extreme parameter values
- Boundary conditions

## 🚀 Running Tests

### Run All Tests
```bash
pytest
```

### Run with Coverage Report
```bash
pytest --cov=. --cov-report=html
```

### Run Specific Test File
```bash
pytest tests/test_simulator.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_simulator.py::TestPhysicsCorrectness -v
```

### Run Single Test
```bash
pytest tests/test_simulator.py::TestPhysicsCorrectness::test_magnitude_is_finite -v
```

### Run Tests by Marker
```bash
# Run only security tests
pytest -m security

# Run only unit tests
pytest -m unit

# Run all except slow tests
pytest -m "not slow"
```

### Run with Verbose Output
```bash
pytest -vv --tb=long --showlocals
```

## 📊 Test Markers

Tests are organized with markers for easy selection:

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests across modules
- `@pytest.mark.security` - Security-focused tests
- `@pytest.mark.slow` - Time-consuming tests (requires `--run-slow`)
- `@pytest.mark.physics` - Physics validation tests
- `@pytest.mark.parser` - Parser-specific tests
- `@pytest.mark.simulator` - Simulator-specific tests
- `@pytest.mark.app` - Application-level tests

## 🔧 Test Configuration

Configuration is in `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v --cov=. --cov-report=html
```

## 📈 Viewing Coverage Report

After running tests with coverage:

```bash
# Generate HTML report
pytest --cov=. --cov-report=html

# Open in browser
firefox htmlcov/index.html  # Linux
open htmlcov/index.html     # macOS
start htmlcov/index.html    # Windows
```

## ✅ Pre-Commit Testing

Before committing code, run:

```bash
# Quick test (skip slow tests)
pytest -v

# Full test with coverage
pytest --cov=. --cov-report=term-missing

# Security tests only
pytest -m security -v
```

## 🐛 Debugging Failed Tests

### Show Full Traceback
```bash
pytest --tb=long
```

### Show Local Variables
```bash
pytest --showlocals
```

### Stop on First Failure
```bash
pytest -x
```

### Run Last Failed Tests Only
```bash
pytest --lf
```

### Drop into Debugger on Failure
```bash
pytest --pdb
```

## 📝 Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_simulator.py        # Simulator unit tests
├── test_parser.py           # Parser integration tests
├── test_security.py         # Security tests
└── README.md                # This file
```

## 🎯 Test Quality Standards

All tests follow these standards:

1. **Descriptive names**: Test names clearly describe what is being tested
2. **AAA pattern**: Arrange, Act, Assert structure
3. **Isolation**: Each test is independent and can run in any order
4. **Fast**: Unit tests complete in <100ms (slow tests marked)
5. **Deterministic**: Same input always produces same output (use seeds)
6. **Comprehensive**: Cover happy path, edge cases, and error conditions

## 📚 Writing New Tests

### Template for New Test

```python
class TestYourFeature:
    """Test your feature description."""
    
    def test_feature_works_correctly(self):
        """Test that feature produces expected output."""
        # Arrange
        simulator = TransformerSimulator(seed=42)
        
        # Act
        result = simulator.generate_normal_signature()
        
        # Assert
        assert result is not None
        assert len(result) == 3
```

### Running Tests During Development

```bash
# Watch mode (re-run on file changes) - requires pytest-watch
ptw -- -v

# Or use entr (Linux/macOS)
ls tests/*.py | entr pytest tests/
```

## 🔍 Continuous Integration

Tests run automatically on:
- Every commit (pre-commit hook)
- Pull requests
- Nightly builds

## 📞 Support

For issues or questions about tests:
1. Check test output and error messages
2. Review test documentation above
3. Examine similar passing tests
4. Contact development team

## 📈 Test Metrics

Target metrics:
- **Coverage**: >85% for all modules
- **Pass Rate**: 100% on main branch
- **Speed**: <5s for unit tests, <30s for full suite
- **Maintenance**: Update tests when functionality changes

---

**Last Updated**: 2025-10-13  
**Test Framework**: pytest 7.4+  
**Python Version**: 3.8+
