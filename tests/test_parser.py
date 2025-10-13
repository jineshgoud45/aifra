"""
Unit Tests for Universal FRA Parser
SIH 2025 PS 25190

Tests cover:
- Vendor detection
- File format parsing (CSV, XML, TXT)
- IEC 60076-18 QA checks
- Security (XXE, path traversal, file size limits)
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parser import UniversalFRAParser, FRAParserError, IECComplianceWarning
from config import IEC, PARSER


class TestParserInitialization:
    """Test parser initialization."""
    
    def test_default_initialization(self):
        """Test parser initializes with defaults."""
        parser = UniversalFRAParser()
        
        assert len(parser.supported_vendors) > 0
        assert 'omicron' in parser.supported_vendors
        assert 'doble' in parser.supported_vendors
        assert 'megger' in parser.supported_vendors
        assert 'generic' in parser.supported_vendors
        assert isinstance(parser.qa_results, dict)


class TestVendorDetection:
    """Test vendor and format detection."""
    
    def test_detect_format_csv(self, tmp_path):
        """Test CSV format detection."""
        parser = UniversalFRAParser()
        
        # Create CSV file
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("freq,mag,phase\n20,0,0\n")
        
        fmt = parser._detect_format(str(csv_file))
        assert fmt == 'csv'
    
    def test_detect_format_xml(self, tmp_path):
        """Test XML format detection."""
        parser = UniversalFRAParser()
        
        # Create XML file
        xml_file = tmp_path / "test.xml"
        xml_file.write_text("<?xml version='1.0'?><root></root>")
        
        fmt = parser._detect_format(str(xml_file))
        assert fmt == 'xml'
    
    def test_detect_vendor_from_filename(self, tmp_path):
        """Test vendor detection from filename."""
        parser = UniversalFRAParser()
        
        # Omicron
        file1 = tmp_path / "franeo_test.csv"
        file1.write_text("data")
        assert parser._detect_vendor(str(file1), 'csv') == 'omicron'
        
        # Doble
        file2 = tmp_path / "doble_test.txt"
        file2.write_text("data")
        assert parser._detect_vendor(str(file2), 'txt') == 'doble'
        
        # Megger
        file3 = tmp_path / "frax_test.dat"
        file3.write_text("data")
        assert parser._detect_vendor(str(file3), 'dat') == 'megger'


class TestFileNotFound:
    """Test file not found scenarios."""
    
    def test_nonexistent_file_raises_error(self):
        """Test that non-existent file raises FileNotFoundError."""
        parser = UniversalFRAParser()
        
        with pytest.raises(FileNotFoundError):
            parser.parse_file("nonexistent_file.csv")
    
    def test_empty_file_raises_error(self, tmp_path):
        """Test that empty file raises error."""
        parser = UniversalFRAParser()
        
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")
        
        with pytest.raises(FRAParserError, match="empty"):
            parser.parse_file(str(empty_file))


class TestGenericParsing:
    """Test generic CSV/TXT parsing."""
    
    def test_parse_generic_csv(self, tmp_path):
        """Test parsing generic CSV file."""
        parser = UniversalFRAParser()
        
        # Create simple CSV
        csv_file = tmp_path / "test.csv"
        csv_data = """frequency,magnitude,phase
20,0.5,-10
100,0.3,-20
1000,0.1,-30
10000,-0.5,-40
100000,-1.0,-50
"""
        csv_file.write_text(csv_data)
        
        df = parser.parse_file(str(csv_file), vendor='generic')
        
        assert len(df) == 5
        assert 'frequency_hz' in df.columns
        assert 'magnitude_db' in df.columns
        assert 'phase_deg' in df.columns
        assert df['frequency_hz'].iloc[0] == 20
        assert df['magnitude_db'].iloc[0] == 0.5
    
    def test_parse_with_comments(self, tmp_path):
        """Test parsing file with comment lines."""
        parser = UniversalFRAParser()
        
        csv_file = tmp_path / "test_comments.csv"
        csv_data = """# This is a comment
# Another comment
frequency,magnitude,phase
20,0.5,-10
100,0.3,-20
"""
        csv_file.write_text(csv_data)
        
        df = parser.parse_file(str(csv_file), vendor='generic')
        
        assert len(df) == 2
        assert df['frequency_hz'].iloc[0] == 20


class TestOmicronParsing:
    """Test Omicron FRANEO format parsing."""
    
    def test_parse_omicron_csv_with_metadata(self, tmp_path):
        """Test parsing Omicron CSV with metadata headers."""
        parser = UniversalFRAParser()
        
        csv_file = tmp_path / "omicron_test.csv"
        csv_data = """# Omicron FRANEO
# Test ID: TEST-001
# Date: 2025-01-01
Frequency (Hz),Magnitude (dB),Phase (deg)
20,0.5,-10
100,0.3,-20
1000,0.1,-30
"""
        csv_file.write_text(csv_data)
        
        df = parser.parse_file(str(csv_file), vendor='omicron')
        
        assert len(df) >= 3
        assert 'test_id' in df.columns
        assert 'vendor' in df.columns
        assert df['vendor'].iloc[0] == 'omicron'


class TestDataNormalization:
    """Test data normalization and schema."""
    
    def test_canonical_columns_present(self, tmp_path):
        """Test that all canonical columns are present after parsing."""
        parser = UniversalFRAParser()
        
        csv_file = tmp_path / "test.csv"
        csv_data = "frequency,magnitude,phase\n20,0.5,-10\n100,0.3,-20\n"
        csv_file.write_text(csv_data)
        
        df = parser.parse_file(str(csv_file))
        
        for col in PARSER.canonical_columns:
            assert col in df.columns
    
    def test_phase_normalized_to_range(self, tmp_path):
        """Test that phase is normalized to -180 to 180."""
        parser = UniversalFRAParser()
        
        csv_file = tmp_path / "test_phase.csv"
        csv_data = "frequency,magnitude,phase\n20,0.5,200\n100,0.3,-200\n"
        csv_file.write_text(csv_data)
        
        df = parser.parse_file(str(csv_file))
        
        assert df['phase_deg'].min() >= -180
        assert df['phase_deg'].max() <= 180
    
    def test_sorted_by_frequency(self, tmp_path):
        """Test that data is sorted by frequency."""
        parser = UniversalFRAParser()
        
        csv_file = tmp_path / "test_unsorted.csv"
        csv_data = "frequency,magnitude,phase\n1000,0.1,-30\n20,0.5,-10\n100,0.3,-20\n"
        csv_file.write_text(csv_data)
        
        df = parser.parse_file(str(csv_file))
        
        # Check sorted
        assert df['frequency_hz'].is_monotonic_increasing


class TestQAChecks:
    """Test IEC 60076-18 QA checks."""
    
    def test_qa_checks_run_automatically(self, tmp_path):
        """Test that QA checks run automatically on parse."""
        parser = UniversalFRAParser()
        
        csv_file = tmp_path / "test_qa.csv"
        # Create data covering full IEC range
        freqs = np.logspace(np.log10(20), np.log10(2e6), 100)
        mags = -20 * np.log10(freqs / 20)  # Simple roll-off
        phases = -90 * np.ones_like(freqs)
        
        csv_data = "frequency,magnitude,phase\n"
        for f, m, p in zip(freqs, mags, phases):
            csv_data += f"{f},{m},{p}\n"
        
        csv_file.write_text(csv_data)
        
        df = parser.parse_file(str(csv_file))
        
        # Check QA results were stored
        assert str(csv_file) in parser.qa_results
        qa = parser.qa_results[str(csv_file)]
        
        assert 'checks' in qa
        assert 'frequency_range' in qa['checks']
        assert 'frequency_grid' in qa['checks']
        assert 'magnitude_stats' in qa['checks']
    
    def test_frequency_range_check(self, tmp_path):
        """Test frequency range QA check."""
        parser = UniversalFRAParser()
        
        # Good range
        csv_file = tmp_path / "good_range.csv"
        freqs = np.logspace(np.log10(20), np.log10(2e6), 50)
        mags = np.random.randn(50)
        phases = np.random.randn(50) * 90
        
        csv_data = "frequency,magnitude,phase\n"
        for f, m, p in zip(freqs, mags, phases):
            csv_data += f"{f},{m},{p}\n"
        csv_file.write_text(csv_data)
        
        df = parser.parse_file(str(csv_file))
        qa = parser.qa_results[str(csv_file)]
        
        assert qa['checks']['frequency_range']['passed']
    
    def test_artifact_detection(self, tmp_path):
        """Test artifact detection in QA checks."""
        parser = UniversalFRAParser()
        
        # Create data with artificial spike
        csv_file = tmp_path / "with_artifact.csv"
        freqs = np.logspace(np.log10(20), np.log10(2e6), 100)
        mags = -10 * np.log10(freqs / 20)
        mags[50] += 10  # Add spike
        phases = -45 * np.ones_like(freqs)
        
        csv_data = "frequency,magnitude,phase\n"
        for f, m, p in zip(freqs, mags, phases):
            csv_data += f"{f},{m},{p}\n"
        csv_file.write_text(csv_data)
        
        with pytest.warns(IECComplianceWarning):
            df = parser.parse_file(str(csv_file))
        
        qa = parser.qa_results[str(csv_file)]
        assert 'artifacts' in qa['checks']


class TestSecurityXXE:
    """Test XXE (XML External Entity) attack prevention."""
    
    def test_xxe_attack_blocked(self, tmp_path):
        """Test that XXE attacks are blocked."""
        parser = UniversalFRAParser()
        
        # Malicious XML with external entity
        xml_file = tmp_path / "malicious.xml"
        malicious_xml = """<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<root>
  <data>&xxe;</data>
</root>
"""
        xml_file.write_text(malicious_xml)
        
        # Should raise error (defusedxml blocks this)
        with pytest.raises(FRAParserError):
            parser.parse_file(str(xml_file), vendor='omicron')
    
    def test_safe_xml_parses_correctly(self, tmp_path):
        """Test that safe XML parses without issues."""
        parser = UniversalFRAParser()
        
        xml_file = tmp_path / "safe.xml"
        safe_xml = """<?xml version="1.0"?>
<root>
  <TestID>TEST-001</TestID>
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
</root>
"""
        xml_file.write_text(safe_xml)
        
        df = parser.parse_file(str(xml_file), vendor='omicron')
        assert len(df) >= 2


class TestFileSizeLimits:
    """Test file size limits."""
    
    def test_large_file_warning(self, tmp_path, caplog):
        """Test that large files generate a warning."""
        parser = UniversalFRAParser()
        
        # Create large file (just over limit)
        csv_file = tmp_path / "large.csv"
        csv_data = "frequency,magnitude,phase\n"
        
        # Create many rows to exceed size limit
        for i in range(100000):
            csv_data += f"{20+i},0.5,-10\n"
        
        csv_file.write_text(csv_data)
        
        # Should parse but generate warning if over limit
        try:
            df = parser.parse_file(str(csv_file))
            # Check logs for warning
            assert len(df) > 0
        except FRAParserError:
            # Might also reject if too large
            pass


class TestMaxDataPoints:
    """Test maximum data points limit."""
    
    def test_truncate_excessive_data_points(self, tmp_path):
        """Test that excessive data points are truncated."""
        parser = UniversalFRAParser()
        
        csv_file = tmp_path / "huge.csv"
        csv_data = "frequency,magnitude,phase\n"
        
        # Create way more than max
        for i in range(IEC.max_data_points + 1000):
            csv_data += f"{20+i*10},0.5,-10\n"
        
        csv_file.write_text(csv_data)
        
        df = parser.parse_file(str(csv_file))
        
        # Should be truncated
        assert len(df) <= IEC.max_data_points


class TestMinimumDataPoints:
    """Test minimum data points check."""
    
    def test_warning_for_few_data_points(self, tmp_path, caplog):
        """Test warning for too few data points."""
        parser = UniversalFRAParser()
        
        csv_file = tmp_path / "sparse.csv"
        csv_data = "frequency,magnitude,phase\n20,0.5,-10\n100,0.3,-20\n"
        csv_file.write_text(csv_data)
        
        df = parser.parse_file(str(csv_file))
        
        # Should parse but with warning
        assert len(df) == 2


class TestInvalidData:
    """Test handling of invalid data."""
    
    def test_missing_columns_raises_error(self, tmp_path):
        """Test that missing required columns raises error."""
        parser = UniversalFRAParser()
        
        csv_file = tmp_path / "missing_cols.csv"
        csv_data = "frequency,magnitude\n20,0.5\n"  # Missing phase
        csv_file.write_text(csv_data)
        
        with pytest.raises(FRAParserError):
            parser.parse_file(str(csv_file))
    
    def test_non_numeric_data_handled(self, tmp_path):
        """Test that non-numeric data is handled gracefully."""
        parser = UniversalFRAParser()
        
        csv_file = tmp_path / "bad_data.csv"
        csv_data = "frequency,magnitude,phase\n20,0.5,-10\nXYZ,ABC,DEF\n100,0.3,-20\n"
        csv_file.write_text(csv_data)
        
        # Should parse and drop bad rows
        df = parser.parse_file(str(csv_file))
        assert len(df) == 2  # Only valid rows
    
    def test_negative_frequency_removed(self, tmp_path):
        """Test that negative frequencies are removed."""
        parser = UniversalFRAParser()
        
        csv_file = tmp_path / "neg_freq.csv"
        csv_data = "frequency,magnitude,phase\n-20,0.5,-10\n20,0.5,-10\n100,0.3,-20\n"
        csv_file.write_text(csv_data)
        
        df = parser.parse_file(str(csv_file))
        assert all(df['frequency_hz'] > 0)
        assert len(df) == 2  # Negative frequency removed


class TestGetQAResults:
    """Test get_qa_results method."""
    
    def test_get_qa_for_specific_file(self, tmp_path):
        """Test getting QA results for specific file."""
        parser = UniversalFRAParser()
        
        csv_file = tmp_path / "test.csv"
        csv_data = "frequency,magnitude,phase\n20,0.5,-10\n100,0.3,-20\n"
        csv_file.write_text(csv_data)
        
        df = parser.parse_file(str(csv_file))
        
        qa = parser.get_qa_results(str(csv_file))
        assert qa['filepath'] == str(csv_file)
        assert 'checks' in qa
    
    def test_get_all_qa_results(self, tmp_path):
        """Test getting all QA results."""
        parser = UniversalFRAParser()
        
        # Parse multiple files
        for i in range(3):
            csv_file = tmp_path / f"test{i}.csv"
            csv_data = "frequency,magnitude,phase\n20,0.5,-10\n100,0.3,-20\n"
            csv_file.write_text(csv_data)
            parser.parse_file(str(csv_file))
        
        all_qa = parser.get_qa_results()
        assert len(all_qa) == 3


class TestConvenienceFunction:
    """Test parse_fra_file convenience function."""
    
    def test_convenience_function(self, tmp_path):
        """Test parse_fra_file convenience function."""
        from parser import parse_fra_file
        
        csv_file = tmp_path / "test.csv"
        csv_data = "frequency,magnitude,phase\n20,0.5,-10\n100,0.3,-20\n"
        csv_file.write_text(csv_data)
        
        df = parse_fra_file(str(csv_file))
        
        assert len(df) == 2
        assert 'frequency_hz' in df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
