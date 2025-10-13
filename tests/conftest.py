"""
Pytest Configuration and Fixtures
SIH 2025 PS 25190

Shared fixtures and configuration for test suite.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_fra_csv(temp_dir):
    """Create a sample FRA CSV file for testing."""
    csv_file = temp_dir / "sample.csv"
    
    # Generate realistic FRA data
    freqs = np.logspace(np.log10(20), np.log10(2e6), 100)
    mags = -20 * np.log10(freqs / 20) + np.random.randn(100) * 0.5
    phases = -90 + np.random.randn(100) * 10
    
    # Write CSV
    with open(csv_file, 'w') as f:
        f.write("frequency,magnitude,phase\n")
        for freq, mag, phase in zip(freqs, mags, phases):
            f.write(f"{freq},{mag},{phase}\n")
    
    return csv_file


@pytest.fixture
def sample_fra_xml(temp_dir):
    """Create a sample FRA XML file for testing."""
    xml_file = temp_dir / "sample.xml"
    
    xml_content = """<?xml version="1.0"?>
<FRAMeasurement>
  <TestID>TEST-001</TestID>
  <TestDate>2025-01-01 12:00:00</TestDate>
  <Device>FRANEO 800</Device>
  <Measurement>
    <Frequency>20</Frequency>
    <Magnitude>0.5</Magnitude>
    <Phase>-10</Phase>
  </Measurement>
  <Measurement>
    <Frequency>100</Frequency>
    <Magnitude>0.3</Magnitude>
    <Phase>-20</Phase>
  </Measurement>
  <Measurement>
    <Frequency>1000</Frequency>
    <Magnitude>0.1</Magnitude>
    <Phase>-30</Phase>
  </Measurement>
</FRAMeasurement>
"""
    xml_file.write_text(xml_content)
    return xml_file


@pytest.fixture
def sample_fra_dataframe():
    """Create a sample FRA DataFrame for testing."""
    freqs = np.logspace(np.log10(20), np.log10(2e6), 100)
    mags = -20 * np.log10(freqs / 20)
    phases = -90 * np.ones_like(freqs)
    
    df = pd.DataFrame({
        'frequency_hz': freqs,
        'magnitude_db': mags,
        'phase_deg': phases,
        'test_id': 'TEST-001',
        'vendor': 'test',
        'timestamp': pd.Timestamp('2025-01-01 12:00:00')
    })
    
    return df


@pytest.fixture
def simulator():
    """Create a TransformerSimulator instance with fixed seed."""
    from simulator import TransformerSimulator
    return TransformerSimulator(seed=42)


@pytest.fixture
def parser():
    """Create a UniversalFRAParser instance."""
    from parser import UniversalFRAParser
    return UniversalFRAParser()


# Pytest hooks
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests across components"
    )
    config.addinivalue_line(
        "markers", "security: Security-focused tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take significant time"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Auto-mark security tests
        if "security" in item.nodeid.lower():
            item.add_marker(pytest.mark.security)
        
        # Auto-mark tests by module
        if "test_simulator" in item.nodeid:
            item.add_marker(pytest.mark.simulator)
        elif "test_parser" in item.nodeid:
            item.add_marker(pytest.mark.parser)
        elif "test_app" in item.nodeid or "test_security" in item.nodeid:
            item.add_marker(pytest.mark.app)


# Pytest command line options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is specified."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
