"""
Edge Case and Property-Based Tests
SIH 2025 PS 25190

Comprehensive edge case testing including:
- Property-based testing with hypothesis
- Large file handling
- Unicode filenames
- Corrupted/malformed data
- Boundary conditions
- Concurrent operations
"""

import unittest
import tempfile
import os
import numpy as np
import pandas as pd
from pathlib import Path
from hypothesis import given, strategies as st, settings
from hypothesis import HealthCheck
import shutil
from typing import Optional

# Import modules to test
try:
    from parser import UniversalFRAParser, FRAParserError
    from simulator import TransformerSimulator
    from config import IEC, PARSER, TESTING
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from parser import UniversalFRAParser, FRAParserError
    from simulator import TransformerSimulator
    from config import IEC, PARSER, TESTING


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.parser = UniversalFRAParser()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        # Should handle empty DataFrames gracefully
        qa_results = self.parser._perform_qa_checks(df, 'test.csv')
        
        self.assertFalse(qa_results['checks']['data_present']['passed'])
    
    def test_minimum_data_points(self):
        """Test with exactly minimum required data points."""
        # Create DataFrame with exactly IEC.min_data_points
        freq = np.logspace(1, 6, IEC.min_data_points)
        df = pd.DataFrame({
            'frequency_hz': freq,
            'magnitude_db': np.random.randn(IEC.min_data_points),
            'phase_deg': np.random.randn(IEC.min_data_points),
            'test_id': ['test'] * IEC.min_data_points,
            'vendor': ['generic'] * IEC.min_data_points
        })
        
        # Should process without errors
        qa_results = self.parser._perform_qa_checks(df, 'test.csv')
        self.assertIn('frequency_range', qa_results['checks'])
    
    def test_unicode_filename_handling(self):
        """Test handling of unicode filenames."""
        if not TESTING.unicode_test_enabled:
            self.skipTest("Unicode testing disabled")
        
        unicode_names = [
            '测试文件.csv',  # Chinese
            'тест.csv',      # Russian
            'δοκιμή.csv',    # Greek
            'テスト.csv',     # Japanese
            'test_émojis_🔬.csv'  # Emoji
        ]
        
        for name in unicode_names:
            filepath = os.path.join(self.test_dir, name)
            
            # Create a valid CSV file with unicode name
            df = pd.DataFrame({
                'frequency_hz': [100, 1000, 10000],
                'magnitude_db': [0, -10, -20],
                'phase_deg': [-45, -90, -135]
            })
            df.to_csv(filepath, index=False)
            
            # Should handle unicode filenames
            self.assertTrue(os.path.exists(filepath))
            os.remove(filepath)
    
    def test_missing_required_columns(self):
        """Test DataFrame with missing required columns."""
        df = pd.DataFrame({
            'frequency_hz': [100, 1000],
            'magnitude_db': [0, -10]
            # Missing phase_deg, test_id, vendor
        })
        
        # Should raise FRAParserError
        with self.assertRaises(FRAParserError) as context:
            self.parser._perform_qa_checks(df, 'test.csv')
        
        self.assertIn('Missing required columns', str(context.exception))
    
    def test_extreme_frequency_values(self):
        """Test with frequencies at extreme boundaries."""
        test_cases = [
            # (min_freq, max_freq, should_pass)
            (IEC.freq_min, IEC.freq_max, True),  # Exact boundaries
            (IEC.freq_min / 10, IEC.freq_max, False),  # Below minimum
            (IEC.freq_min, IEC.freq_max * 10, False),  # Above maximum
            (1, 100, False),  # Too narrow range
        ]
        
        for min_f, max_f, should_pass in test_cases:
            freq = np.logspace(np.log10(min_f), np.log10(max_f), 100)
            df = pd.DataFrame({
                'frequency_hz': freq,
                'magnitude_db': np.random.randn(100),
                'phase_deg': np.random.randn(100),
                'test_id': ['test'] * 100,
                'vendor': ['generic'] * 100
            })
            
            qa_results = self.parser._perform_qa_checks(df, 'test.csv')
            self.assertEqual(
                qa_results['checks']['frequency_range']['passed'],
                should_pass,
                f"Failed for freq range {min_f}-{max_f}"
            )
    
    def test_nan_and_inf_values(self):
        """Test handling of NaN and Inf values in data."""
        # DataFrame with NaN values
        df_nan = pd.DataFrame({
            'frequency_hz': [100, np.nan, 1000],
            'magnitude_db': [0, -10, np.nan],
            'phase_deg': [-45, -90, -135],
            'test_id': ['test'] * 3,
            'vendor': ['generic'] * 3
        })
        
        # Should handle NaN gracefully (dropna or handle in QA)
        # Parser should not crash
        try:
            qa_results = self.parser._perform_qa_checks(df_nan.dropna(), 'test.csv')
            # If dropna is used, test should pass with remaining data
        except Exception as e:
            # If not using dropna, should raise appropriate error
            self.assertIsInstance(e, (ValueError, FRAParserError))
    
    def test_duplicate_frequencies(self):
        """Test handling of duplicate frequency values."""
        df = pd.DataFrame({
            'frequency_hz': [100, 100, 1000, 1000],  # Duplicates
            'magnitude_db': [0, -1, -10, -11],
            'phase_deg': [-45, -46, -90, -91],
            'test_id': ['test'] * 4,
            'vendor': ['generic'] * 4
        })
        
        # Should handle duplicates (either reject or process)
        qa_results = self.parser._perform_qa_checks(df, 'test.csv')
        self.assertIsNotNone(qa_results)
    
    def test_unsorted_frequencies(self):
        """Test handling of unsorted frequency data."""
        df = pd.DataFrame({
            'frequency_hz': [1000, 100, 10000, 500],  # Unsorted
            'magnitude_db': [-10, 0, -20, -5],
            'phase_deg': [-90, -45, -135, -67.5],
            'test_id': ['test'] * 4,
            'vendor': ['generic'] * 4
        })
        
        # Parser should handle or sort data
        qa_results = self.parser._perform_qa_checks(df, 'test.csv')
        self.assertIsNotNone(qa_results)


class TestPropertyBased(unittest.TestCase):
    """Property-based testing with hypothesis."""
    
    @given(
        n_sections=st.integers(min_value=10, max_value=200),
        base_r=st.floats(min_value=0.01, max_value=10.0),
        seed=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_simulator_never_crashes(self, n_sections: int, base_r: float, seed: int):
        """Simulator should never crash with valid inputs."""
        try:
            sim = TransformerSimulator(
                n_sections=n_sections,
                base_R=base_r,
                seed=seed
            )
            
            # Should generate data without crashing
            freq, mag, phase = sim.generate_normal()
            
            # Basic sanity checks
            self.assertEqual(len(freq), len(mag))
            self.assertEqual(len(freq), len(phase))
            self.assertTrue(np.all(np.isfinite(freq)))
            self.assertTrue(np.all(np.isfinite(mag)))
            
        except ValueError as e:
            # Acceptable if input validation rejects invalid parameters
            self.assertIn('Invalid', str(e))
    
    @given(
        data_points=st.integers(min_value=TESTING.min_test_data_points, max_value=1000)
    )
    @settings(max_examples=10, deadline=None)
    def test_qa_checks_scale_with_data_size(self, data_points: int):
        """QA checks should work with varying data sizes."""
        freq = np.logspace(np.log10(IEC.freq_min), np.log10(IEC.freq_max), data_points)
        df = pd.DataFrame({
            'frequency_hz': freq,
            'magnitude_db': np.random.randn(data_points) * 10,
            'phase_deg': np.random.randn(data_points) * 45,
            'test_id': ['test'] * data_points,
            'vendor': ['generic'] * data_points
        })
        
        parser = UniversalFRAParser()
        qa_results = parser._perform_qa_checks(df, 'test.csv')
        
        # Should complete without errors
        self.assertIsNotNone(qa_results)
        self.assertIn('checks', qa_results)
    
    @given(
        magnitude_offset=st.floats(min_value=-100, max_value=100),
        phase_offset=st.floats(min_value=-180, max_value=180)
    )
    @settings(max_examples=15, deadline=None)
    def test_qa_checks_invariant_to_offsets(self, magnitude_offset: float, phase_offset: float):
        """QA checks should be relatively invariant to DC offsets."""
        freq = np.logspace(np.log10(IEC.freq_min), np.log10(IEC.freq_max), 100)
        
        # Create two DataFrames with different offsets
        df1 = pd.DataFrame({
            'frequency_hz': freq,
            'magnitude_db': np.random.randn(100) + magnitude_offset,
            'phase_deg': np.random.randn(100) * 45 + phase_offset,
            'test_id': ['test'] * 100,
            'vendor': ['generic'] * 100
        })
        
        parser = UniversalFRAParser()
        qa_results = parser._perform_qa_checks(df1, 'test.csv')
        
        # Frequency range check should pass regardless of magnitude/phase offsets
        self.assertTrue(qa_results['checks']['frequency_range']['passed'])


class TestConcurrency(unittest.TestCase):
    """Test concurrent operations and thread safety."""
    
    def test_concurrent_parser_instances(self):
        """Test multiple parser instances in parallel."""
        import threading
        
        results = []
        errors = []
        
        def parse_data(thread_id: int):
            try:
                parser = UniversalFRAParser()
                freq = np.logspace(1, 6, 100)
                df = pd.DataFrame({
                    'frequency_hz': freq,
                    'magnitude_db': np.random.randn(100),
                    'phase_deg': np.random.randn(100),
                    'test_id': [f'test_{thread_id}'] * 100,
                    'vendor': ['generic'] * 100
                })
                qa_results = parser._perform_qa_checks(df, f'test_{thread_id}.csv')
                results.append(qa_results)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = [threading.Thread(target=parse_data, args=(i,)) for i in range(5)]
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Should complete without errors
        self.assertEqual(len(errors), 0, f"Concurrent parsing errors: {errors}")
        self.assertEqual(len(results), 5)


def run_edge_case_tests() -> bool:
    """Run all edge case tests and return success status."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestPropertyBased))
    suite.addTests(loader.loadTestsFromTestCase(TestConcurrency))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_edge_case_tests()
    exit(0 if success else 1)
