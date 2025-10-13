"""
Unit Tests for Universal FRA/SFRA Parser
SIH 2025 PS 25190

Tests cover:
- Omicron FRANEO CSV and XML formats
- Doble SFRA suite format
- Megger FRAX analyzer format
- Generic format parsing
- IEC 60076-18 compliance checks
- Error handling
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from datetime import datetime
from parser import (
    UniversalFRAParser, 
    FRAParserError, 
    IECComplianceWarning,
    parse_fra_file
)


class TestUniversalFRAParser(unittest.TestCase):
    """Test suite for Universal FRA Parser."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = UniversalFRAParser()
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_sample_omicron_csv(self) -> str:
        """Create sample Omicron FRANEO CSV file."""
        filepath = os.path.join(self.test_dir, 'omicron_test.csv')
        content = """# Omicron FRANEO FRA Measurement
# Device: FRANEO 800
# Test ID: TF001_H1-H2
# Date: 2025-01-15 10:30:00
# Impedance: 50 Ohm
# Configuration: End-to-End Open Circuit
Frequency (Hz),Magnitude (dB),Phase (deg)
20.0,-45.2,85.3
25.0,-44.8,84.1
31.5,-44.3,82.9
40.0,-43.8,81.5
50.0,-43.2,80.0
63.0,-42.5,78.2
79.4,-41.7,76.3
100.0,-40.8,74.1
125.9,-39.8,71.8
158.5,-38.6,69.2
200.0,-37.3,66.5
251.2,-35.9,63.6
316.2,-34.3,60.5
398.1,-32.6,57.2
501.2,-30.8,53.7
631.0,-28.9,50.0
794.3,-26.9,46.2
1000.0,-24.8,42.3
1258.9,-22.6,38.3
1584.9,-20.3,34.2
1995.3,-18.0,30.1
2511.9,-15.6,26.0
3162.3,-13.2,21.9
3981.1,-10.8,17.9
5011.9,-8.4,14.0
6309.6,-6.1,10.3
7943.3,-3.8,6.8
10000.0,-1.6,3.5
12589.3,0.5,0.5
15848.9,2.5,-2.3
19952.6,4.4,-4.9
25118.9,6.2,-7.3
31622.8,7.9,-9.5
39810.7,9.5,-11.5
50118.7,10.9,-13.3
63095.7,12.2,-14.9
79432.8,13.4,-16.3
100000.0,14.5,-17.6
125892.5,15.5,-18.7
158489.3,16.4,-19.7
199526.2,17.2,-20.6
251188.6,17.9,-21.4
316227.8,18.5,-22.1
398107.2,19.0,-22.7
501187.2,19.5,-23.2
630957.3,19.9,-23.7
794328.2,20.2,-24.1
1000000.0,20.5,-24.4
1258925.4,20.7,-24.7
1584893.2,20.9,-24.9
2000000.0,21.0,-25.1
"""
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath
    
    def create_sample_omicron_xml(self) -> str:
        """Create sample Omicron FRANEO XML file."""
        filepath = os.path.join(self.test_dir, 'omicron_test.xml')
        content = """<?xml version="1.0" encoding="UTF-8"?>
<FRAMeasurement>
    <TestID>TF001_H1-H2_XML</TestID>
    <TestDate>2025-01-15 10:30:00</TestDate>
    <Device>FRANEO 800</Device>
    <Configuration>
        <Impedance>50</Impedance>
        <TestType>End-to-End Open</TestType>
    </Configuration>
    <Measurements>
        <Measurement>
            <Frequency>20.0</Frequency>
            <Magnitude>-45.2</Magnitude>
            <Phase>85.3</Phase>
        </Measurement>
        <Measurement>
            <Frequency>50.0</Frequency>
            <Magnitude>-43.2</Magnitude>
            <Phase>80.0</Phase>
        </Measurement>
        <Measurement>
            <Frequency>100.0</Frequency>
            <Magnitude>-40.8</Magnitude>
            <Phase>74.1</Phase>
        </Measurement>
        <Measurement>
            <Frequency>1000.0</Frequency>
            <Magnitude>-24.8</Magnitude>
            <Phase>42.3</Phase>
        </Measurement>
        <Measurement>
            <Frequency>10000.0</Frequency>
            <Magnitude>-1.6</Magnitude>
            <Phase>3.5</Phase>
        </Measurement>
        <Measurement>
            <Frequency>100000.0</Frequency>
            <Magnitude>14.5</Magnitude>
            <Phase>-17.6</Phase>
        </Measurement>
        <Measurement>
            <Frequency>1000000.0</Frequency>
            <Magnitude>20.5</Magnitude>
            <Phase>-24.4</Phase>
        </Measurement>
        <Measurement>
            <Frequency>2000000.0</Frequency>
            <Magnitude>21.0</Magnitude>
            <Phase>-25.1</Phase>
        </Measurement>
    </Measurements>
</FRAMeasurement>
"""
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath
    
    def create_sample_doble_file(self) -> str:
        """Create sample Doble SFRA suite file."""
        filepath = os.path.join(self.test_dir, 'doble_test.txt')
        content = """Doble Engineering SFRA Test Data
Test ID: XFMR_2345_Phase_A
Date: 01/15/2025
Serial Number: 2345-A
Operator: J. Smith
Test Configuration: H1-H2 Open Circuit
Impedance: 50 Ohms

Frequency (Hz)\tMagnitude (dB)\tPhase (deg)
20.0\t-44.9\t84.8
40.0\t-43.5\t81.2
80.0\t-41.4\t75.8
160.0\t-38.1\t68.4
320.0\t-33.8\t59.1
640.0\t-28.2\t48.5
1280.0\t-21.9\t36.8
2560.0\t-14.8\t24.3
5120.0\t-7.2\t11.9
10240.0\t0.2\t0.1
20480.0\t7.1\t-11.2
40960.0\t13.2\t-21.5
81920.0\t18.3\t-30.1
163840.0\t22.1\t-36.8
327680.0\t24.8\t-41.9
655360.0\t26.5\t-45.6
1310720.0\t27.6\t-48.2
2000000.0\t28.1\t-49.8
"""
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath
    
    def create_sample_megger_file(self) -> str:
        """Create sample Megger FRAX analyzer file."""
        filepath = os.path.join(self.test_dir, 'megger_frax_test.dat')
        content = """[MEGGER FRAX 101]
DeviceType=FRAX101
SerialNumber=12345678
TestID=TRAFO_001_LV
DateTime=2025-01-15 14:25:30
Operator=Engineer_1
Configuration=LV Winding H1-H2
Impedance=50
CableLength=10m
[MEASUREMENT_DATA]
Frequency (Hz),Magnitude (dB),Phase (deg)
20.0,-46.1,86.2
30.0,-45.4,84.5
50.0,-44.2,81.8
80.0,-42.6,78.1
125.0,-40.5,73.5
200.0,-37.8,67.9
315.0,-34.5,61.2
500.0,-30.6,53.5
800.0,-26.2,44.8
1250.0,-21.3,35.2
2000.0,-15.9,25.1
3150.0,-10.2,14.8
5000.0,-4.3,4.9
8000.0,1.5,-4.5
12500.0,7.1,-13.2
20000.0,12.3,-21.1
31500.0,16.8,-27.9
50000.0,20.5,-33.5
80000.0,23.4,-37.8
125000.0,25.7,-41.2
200000.0,27.3,-43.8
315000.0,28.4,-45.7
500000.0,29.1,-47.1
800000.0,29.6,-48.2
1250000.0,29.9,-48.9
2000000.0,30.1,-49.4
"""
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath
    
    def test_omicron_csv_parsing(self):
        """Test parsing Omicron FRANEO CSV format."""
        filepath = self.create_sample_omicron_csv()
        df = self.parser.parse_file(filepath, vendor='omicron')
        
        # Check DataFrame structure
        self.assertEqual(list(df.columns), self.parser.CANONICAL_COLUMNS)
        
        # Check data content
        self.assertGreater(len(df), 0)
        self.assertEqual(df['vendor'].iloc[0], 'omicron')
        self.assertIn('TF001', df['test_id'].iloc[0])
        
        # Check frequency range
        self.assertAlmostEqual(df['frequency_hz'].min(), 20.0, places=1)
        self.assertAlmostEqual(df['frequency_hz'].max(), 2000000.0, places=1)
        
        # Check data types
        self.assertTrue(pd.api.types.is_float_dtype(df['frequency_hz']))
        self.assertTrue(pd.api.types.is_float_dtype(df['magnitude_db']))
        self.assertTrue(pd.api.types.is_float_dtype(df['phase_deg']))
        
        # Check sorting
        self.assertTrue(df['frequency_hz'].is_monotonic_increasing)
    
    def test_omicron_xml_parsing(self):
        """Test parsing Omicron FRANEO XML format."""
        filepath = self.create_sample_omicron_xml()
        df = self.parser.parse_file(filepath, vendor='omicron')
        
        self.assertEqual(list(df.columns), self.parser.CANONICAL_COLUMNS)
        self.assertGreater(len(df), 0)
        self.assertEqual(df['vendor'].iloc[0], 'omicron')
        self.assertIn('TF001', df['test_id'].iloc[0])
    
    def test_doble_parsing(self):
        """Test parsing Doble SFRA suite format."""
        filepath = self.create_sample_doble_file()
        df = self.parser.parse_file(filepath, vendor='doble')
        
        self.assertEqual(list(df.columns), self.parser.CANONICAL_COLUMNS)
        self.assertGreater(len(df), 0)
        self.assertEqual(df['vendor'].iloc[0], 'doble')
        self.assertIn('2345', df['test_id'].iloc[0])
        
        # Verify frequency range
        self.assertLess(df['frequency_hz'].min(), 50.0)
        self.assertGreater(df['frequency_hz'].max(), 1000000.0)
    
    def test_megger_parsing(self):
        """Test parsing Megger FRAX analyzer format."""
        filepath = self.create_sample_megger_file()
        df = self.parser.parse_file(filepath, vendor='megger')
        
        self.assertEqual(list(df.columns), self.parser.CANONICAL_COLUMNS)
        self.assertGreater(len(df), 0)
        self.assertEqual(df['vendor'].iloc[0], 'megger')
        self.assertIn('TRAFO', df['test_id'].iloc[0])
    
    def test_vendor_auto_detection(self):
        """Test automatic vendor detection."""
        # Test Omicron detection
        omicron_file = self.create_sample_omicron_csv()
        df = self.parser.parse_file(omicron_file)  # No vendor specified
        self.assertEqual(df['vendor'].iloc[0], 'omicron')
        
        # Test Doble detection
        doble_file = self.create_sample_doble_file()
        df = self.parser.parse_file(doble_file)
        self.assertEqual(df['vendor'].iloc[0], 'doble')
        
        # Test Megger detection
        megger_file = self.create_sample_megger_file()
        df = self.parser.parse_file(megger_file)
        self.assertEqual(df['vendor'].iloc[0], 'megger')
    
    def test_iec_frequency_range_check(self):
        """Test IEC 60076-18 frequency range compliance."""
        filepath = self.create_sample_omicron_csv()
        df = self.parser.parse_file(filepath)
        
        qa_results = self.parser.get_qa_results(filepath)
        
        self.assertIn('frequency_range', qa_results['checks'])
        freq_check = qa_results['checks']['frequency_range']
        
        # Should pass as range covers 20 Hz to 2 MHz
        self.assertTrue(freq_check['passed'])
        self.assertLessEqual(freq_check['min_freq'], 40.0)  # Within 2x of 20 Hz
        self.assertGreaterEqual(freq_check['max_freq'], 1000000.0)  # Within 2x of 2 MHz
    
    def test_iec_artifact_detection(self):
        """Test IEC 60076-18 artifact detection (>3 dB deviation)."""
        filepath = self.create_sample_omicron_csv()
        df = self.parser.parse_file(filepath)
        
        qa_results = self.parser.get_qa_results(filepath)
        
        self.assertIn('artifacts', qa_results['checks'])
        artifact_check = qa_results['checks']['artifacts']
        
        self.assertEqual(artifact_check['threshold_db'], 3.0)
        self.assertIn('num_artifacts', artifact_check)
        self.assertIn('max_deviation_db', artifact_check)
    
    def test_phase_normalization(self):
        """Test phase normalization to -180 to 180 range."""
        # Create file with phases outside -180 to 180
        filepath = os.path.join(self.test_dir, 'phase_test.csv')
        content = """Frequency,Magnitude,Phase
100,10,270
200,15,360
300,20,450
"""
        with open(filepath, 'w') as f:
            f.write(content)
        
        df = self.parser.parse_file(filepath, vendor='generic')
        
        # Check all phases are in -180 to 180 range
        self.assertTrue((df['phase_deg'] >= -180).all())
        self.assertTrue((df['phase_deg'] <= 180).all())
        
        # 270 should become -90, 360 -> 0, 450 -> 90
        self.assertAlmostEqual(df['phase_deg'].iloc[0], -90.0, places=1)
        self.assertAlmostEqual(df['phase_deg'].iloc[1], 0.0, places=1)
        self.assertAlmostEqual(df['phase_deg'].iloc[2], 90.0, places=1)
    
    def test_missing_file_error(self):
        """Test error handling for missing files."""
        with self.assertRaises(FileNotFoundError):
            self.parser.parse_file('nonexistent_file.csv')
    
    def test_invalid_csv_error(self):
        """Test error handling for invalid CSV content."""
        filepath = os.path.join(self.test_dir, 'invalid.csv')
        with open(filepath, 'w') as f:
            f.write("This is not valid FRA data\n")
            f.write("Random text with no structure\n")
        
        with self.assertRaises(FRAParserError):
            self.parser.parse_file(filepath, vendor='omicron')
    
    def test_invalid_xml_error(self):
        """Test error handling for invalid XML."""
        filepath = os.path.join(self.test_dir, 'invalid.xml')
        with open(filepath, 'w') as f:
            f.write("<InvalidXML>No closing tag\n")
        
        with self.assertRaises(FRAParserError):
            self.parser.parse_file(filepath, vendor='omicron')
    
    def test_missing_columns_error(self):
        """Test error handling for missing required columns."""
        filepath = os.path.join(self.test_dir, 'missing_cols.csv')
        content = """Frequency,SomeOtherColumn
100,50
200,60
"""
        with open(filepath, 'w') as f:
            f.write(content)
        
        with self.assertRaises(FRAParserError):
            self.parser.parse_file(filepath, vendor='generic')
    
    def test_non_numeric_data_handling(self):
        """Test handling of non-numeric data."""
        filepath = os.path.join(self.test_dir, 'non_numeric.csv')
        content = """Frequency,Magnitude,Phase
100,10.5,45.2
abc,def,ghi
300,15.2,50.1
"""
        with open(filepath, 'w') as f:
            f.write(content)
        
        df = self.parser.parse_file(filepath, vendor='generic')
        
        # Should have 2 valid rows (invalid row dropped)
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(df['frequency_hz'].iloc[0], 100.0)
        self.assertAlmostEqual(df['frequency_hz'].iloc[1], 300.0)
    
    def test_convenience_function(self):
        """Test convenience parse_fra_file function."""
        filepath = self.create_sample_omicron_csv()
        df = parse_fra_file(filepath, vendor='omicron')
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), UniversalFRAParser.CANONICAL_COLUMNS)
        self.assertGreater(len(df), 0)
    
    def test_qa_results_retrieval(self):
        """Test QA results retrieval."""
        filepath = self.create_sample_omicron_csv()
        df = self.parser.parse_file(filepath)
        
        # Get QA results for specific file
        qa = self.parser.get_qa_results(filepath)
        self.assertIn('checks', qa)
        self.assertIn('frequency_range', qa['checks'])
        
        # Get all QA results
        all_qa = self.parser.get_qa_results()
        self.assertIn(filepath, all_qa)
    
    def test_timestamp_parsing(self):
        """Test timestamp parsing from various formats."""
        test_cases = [
            '2025-01-15 10:30:00',
            '2025-01-15',
            '15/01/2025 10:30:00',
            '01/15/2025',
            '20250115_103000'
        ]
        
        for ts_str in test_cases:
            result = self.parser._parse_timestamp(ts_str)
            self.assertIsInstance(result, datetime)


class TestIECCompliance(unittest.TestCase):
    """Test IEC 60076-18 compliance features."""
    
    def setUp(self):
        self.parser = UniversalFRAParser()
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_frequency_grid_log_scale(self):
        """Test frequency grid log scale check."""
        # Create file with non-uniform log spacing
        filepath = os.path.join(self.test_dir, 'log_test.csv')
        freqs = [20, 50, 100, 150, 200, 10000, 20000, 1000000]  # Poor log spacing
        content = "Frequency,Magnitude,Phase\n"
        for f in freqs:
            content += f"{f},-30,45\n"
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        df = self.parser.parse_file(filepath, vendor='generic')
        qa = self.parser.get_qa_results(filepath)
        
        self.assertIn('frequency_grid', qa['checks'])
        self.assertIn('log_spacing_std', qa['checks']['frequency_grid'])
    
    def test_magnitude_statistics(self):
        """Test magnitude statistics calculation."""
        filepath = os.path.join(self.test_dir, 'mag_stats.csv')
        content = """Frequency,Magnitude,Phase
100,-40,80
200,-30,70
300,-20,60
400,-10,50
500,0,40
"""
        with open(filepath, 'w') as f:
            f.write(content)
        
        df = self.parser.parse_file(filepath, vendor='generic')
        qa = self.parser.get_qa_results(filepath)
        
        mag_stats = qa['checks']['magnitude_stats']
        self.assertIn('mean_db', mag_stats)
        self.assertIn('std_db', mag_stats)
        self.assertIn('min_db', mag_stats)
        self.assertIn('max_db', mag_stats)
        
        self.assertAlmostEqual(mag_stats['mean_db'], -20.0, places=1)


def run_tests():
    """Run all tests and display results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestUniversalFRAParser))
    suite.addTests(loader.loadTestsFromTestCase(TestIECCompliance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
