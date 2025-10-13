"""
Unit Tests for Transformer FRA Simulator
SIH 2025 PS 25190

Tests cover:
- Transfer function computation
- All fault type generators
- Noise addition
- Dataset generation and balancing
- Multi-vendor export formats
- Output shape validation
- IEC 60076-18 compliance
- Reproducibility with random seed
"""

import unittest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from simulator import TransformerSimulator, generate_synthetic_dataset


class TestTransformerSimulator(unittest.TestCase):
    """Test suite for TransformerSimulator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = TransformerSimulator(
            n_sections=50,
            freq_points=100,
            seed=42
        )
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test simulator initialization."""
        self.assertEqual(self.simulator.n_sections, 50)
        self.assertEqual(self.simulator.freq_points, 100)
        self.assertEqual(len(self.simulator.frequencies), 100)
        self.assertEqual(self.simulator.seed, 42)
    
    def test_frequency_range_iec_compliance(self):
        """Test IEC 60076-18 frequency range compliance."""
        freq_min = self.simulator.frequencies.min()
        freq_max = self.simulator.frequencies.max()
        
        # Check frequency range
        self.assertAlmostEqual(freq_min, self.simulator.IEC_FREQ_MIN, delta=1.0)
        self.assertAlmostEqual(freq_max, self.simulator.IEC_FREQ_MAX, delta=1e5)
        
        # Check logarithmic spacing
        log_freq = np.log10(self.simulator.frequencies)
        log_diffs = np.diff(log_freq)
        
        # Should be relatively uniform in log space
        self.assertLess(np.std(log_diffs), 0.01)
    
    def test_normal_signature_output_shape(self):
        """Test normal signature output shapes."""
        freq, mag, phase = self.simulator.generate_normal_signature()
        
        self.assertEqual(len(freq), self.simulator.freq_points)
        self.assertEqual(len(mag), self.simulator.freq_points)
        self.assertEqual(len(phase), self.simulator.freq_points)
        
        # Check types
        self.assertIsInstance(freq, np.ndarray)
        self.assertIsInstance(mag, np.ndarray)
        self.assertIsInstance(phase, np.ndarray)
    
    def test_axial_deformation_output_shape(self):
        """Test axial deformation output shapes."""
        freq, mag, phase = self.simulator.generate_axial_deformation()
        
        self.assertEqual(len(freq), self.simulator.freq_points)
        self.assertEqual(len(mag), self.simulator.freq_points)
        self.assertEqual(len(phase), self.simulator.freq_points)
    
    def test_radial_deformation_output_shape(self):
        """Test radial deformation output shapes."""
        freq, mag, phase = self.simulator.generate_radial_deformation()
        
        self.assertEqual(len(freq), self.simulator.freq_points)
        self.assertEqual(len(mag), self.simulator.freq_points)
        self.assertEqual(len(phase), self.simulator.freq_points)
    
    def test_interturn_short_output_shape(self):
        """Test inter-turn short output shapes."""
        freq, mag, phase = self.simulator.generate_interturn_short()
        
        self.assertEqual(len(freq), self.simulator.freq_points)
        self.assertEqual(len(mag), self.simulator.freq_points)
        self.assertEqual(len(phase), self.simulator.freq_points)
    
    def test_core_grounding_output_shape(self):
        """Test core grounding output shapes."""
        freq, mag, phase = self.simulator.generate_core_grounding()
        
        self.assertEqual(len(freq), self.simulator.freq_points)
        self.assertEqual(len(mag), self.simulator.freq_points)
        self.assertEqual(len(phase), self.simulator.freq_points)
    
    def test_tapchanger_fault_output_shape(self):
        """Test tap-changer fault output shapes."""
        freq, mag, phase = self.simulator.generate_tapchanger_fault()
        
        self.assertEqual(len(freq), self.simulator.freq_points)
        self.assertEqual(len(mag), self.simulator.freq_points)
        self.assertEqual(len(phase), self.simulator.freq_points)
    
    def test_phase_normalization(self):
        """Test phase is normalized to -180 to 180 range."""
        for fault_method in [
            self.simulator.generate_normal_signature,
            self.simulator.generate_axial_deformation,
            self.simulator.generate_radial_deformation,
            self.simulator.generate_interturn_short,
            self.simulator.generate_core_grounding,
            self.simulator.generate_tapchanger_fault
        ]:
            _, _, phase = fault_method()
            
            self.assertTrue(np.all(phase >= -180))
            self.assertTrue(np.all(phase <= 180))
    
    def test_noise_addition(self):
        """Test measurement noise addition."""
        # Generate without noise
        _, mag_clean, phase_clean = self.simulator.generate_normal_signature(add_noise=False)
        
        # Generate with noise (should be different)
        _, mag_noisy, phase_noisy = self.simulator.generate_normal_signature(add_noise=True)
        
        # Should have differences due to noise
        self.assertFalse(np.allclose(mag_clean, mag_noisy))
        self.assertFalse(np.allclose(phase_clean, phase_noisy))
    
    def test_reproducibility_with_seed(self):
        """Test reproducibility with same random seed."""
        sim1 = TransformerSimulator(seed=123)
        sim2 = TransformerSimulator(seed=123)
        
        _, mag1, phase1 = sim1.generate_normal_signature()
        _, mag2, phase2 = sim2.generate_normal_signature()
        
        # Should be identical with same seed
        np.testing.assert_array_almost_equal(mag1, mag2, decimal=10)
        np.testing.assert_array_almost_equal(phase1, phase2, decimal=10)
    
    def test_different_seeds_produce_different_results(self):
        """Test different seeds produce different results."""
        sim1 = TransformerSimulator(seed=123)
        sim2 = TransformerSimulator(seed=456)
        
        _, mag1, _ = sim1.generate_normal_signature()
        _, mag2, _ = sim2.generate_normal_signature()
        
        # Should be different with different seeds
        self.assertFalse(np.allclose(mag1, mag2))
    
    def test_fault_severity_impact(self):
        """Test fault severity affects output."""
        # Low severity
        _, mag_low, _ = self.simulator.generate_axial_deformation(severity=0.05, add_noise=False)
        
        # High severity
        _, mag_high, _ = self.simulator.generate_axial_deformation(severity=0.30, add_noise=False)
        
        # Higher severity should produce more significant changes
        self.assertFalse(np.allclose(mag_low, mag_high))
    
    def test_generate_fault_data(self):
        """Test batch fault data generation."""
        n_samples = 5
        df = self.simulator.generate_fault_data('normal', n_samples=n_samples)
        
        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df['sample_id'].unique()), n_samples)
        self.assertEqual(len(df), n_samples * self.simulator.freq_points)
        
        # Check columns
        required_cols = ['frequency_hz', 'magnitude_db', 'phase_deg', 'fault_type', 'sample_id']
        for col in required_cols:
            self.assertIn(col, df.columns)
        
        # Check fault type
        self.assertTrue(all(df['fault_type'] == 'normal'))
    
    def test_generate_balanced_dataset(self):
        """Test balanced dataset generation."""
        total_samples = 60  # 10 per fault type (6 types)
        train_df, test_df = self.simulator.generate_balanced_dataset(
            total_samples=total_samples,
            train_ratio=0.8
        )
        
        # Check train/test split
        n_train_samples = len(train_df) // self.simulator.freq_points
        n_test_samples = len(test_df) // self.simulator.freq_points
        
        self.assertGreater(n_train_samples, 0)
        self.assertGreater(n_test_samples, 0)
        self.assertAlmostEqual(n_train_samples / (n_train_samples + n_test_samples), 0.8, delta=0.1)
        
        # Check class balance
        train_fault_counts = train_df.groupby('fault_type')['sample_id'].nunique()
        
        # All fault types should be present
        self.assertEqual(len(train_fault_counts), 6)
        
        # Should be relatively balanced
        self.assertLess(train_fault_counts.std(), train_fault_counts.mean() * 0.2)
    
    def test_export_omicron_csv(self):
        """Test Omicron CSV export."""
        df = self.simulator.generate_fault_data('normal', n_samples=2)
        files = self.simulator.export_omicron_csv(df, output_dir=self.test_dir)
        
        self.assertEqual(len(files), 2)
        
        for filepath in files:
            self.assertTrue(os.path.exists(filepath))
            self.assertTrue(filepath.endswith('.csv'))
            
            # Check file content
            with open(filepath, 'r') as f:
                content = f.read()
                self.assertIn('Omicron', content)
                self.assertIn('Frequency', content)
                self.assertIn('Magnitude', content)
    
    def test_export_doble_txt(self):
        """Test Doble TXT export."""
        df = self.simulator.generate_fault_data('axial_deformation', n_samples=2)
        files = self.simulator.export_doble_txt(df, output_dir=self.test_dir)
        
        self.assertEqual(len(files), 2)
        
        for filepath in files:
            self.assertTrue(os.path.exists(filepath))
            self.assertTrue(filepath.endswith('.txt'))
            
            with open(filepath, 'r') as f:
                content = f.read()
                self.assertIn('Doble', content)
                self.assertIn('Frequency', content)
    
    def test_export_megger_dat(self):
        """Test Megger DAT export."""
        df = self.simulator.generate_fault_data('interturn_short', n_samples=2)
        files = self.simulator.export_megger_dat(df, output_dir=self.test_dir)
        
        self.assertEqual(len(files), 2)
        
        for filepath in files:
            self.assertTrue(os.path.exists(filepath))
            self.assertTrue(filepath.endswith('.dat'))
            
            with open(filepath, 'r') as f:
                content = f.read()
                self.assertIn('MEGGER', content)
                self.assertIn('Frequency', content)
    
    def test_export_all_formats(self):
        """Test export to all vendor formats."""
        df = self.simulator.generate_fault_data('normal', n_samples=1)
        results = self.simulator.export_all_formats(df, output_dir=self.test_dir)
        
        self.assertIn('omicron', results)
        self.assertIn('doble', results)
        self.assertIn('megger', results)
        
        self.assertEqual(len(results['omicron']), 1)
        self.assertEqual(len(results['doble']), 1)
        self.assertEqual(len(results['megger']), 1)
    
    def test_invalid_fault_type(self):
        """Test error handling for invalid fault type."""
        with self.assertRaises(ValueError):
            self.simulator.generate_fault_data('invalid_fault_type', n_samples=1)
    
    def test_magnitude_values_reasonable(self):
        """Test magnitude values are in reasonable range."""
        freq, mag, phase = self.simulator.generate_normal_signature()
        
        # Magnitude should typically be within -100 to +50 dB for FRA
        self.assertTrue(np.all(mag > -100))
        self.assertTrue(np.all(mag < 50))
    
    def test_frequency_monotonic_increasing(self):
        """Test frequencies are monotonically increasing."""
        freq = self.simulator.frequencies
        
        self.assertTrue(np.all(np.diff(freq) > 0))
    
    def test_all_fault_types_different(self):
        """Test all fault types produce different signatures."""
        fault_types = [
            'normal',
            'axial_deformation',
            'radial_deformation',
            'interturn_short',
            'core_grounding',
            'tapchanger_fault'
        ]
        
        signatures = []
        for fault_type in fault_types:
            df = self.simulator.generate_fault_data(fault_type, n_samples=1)
            mag = df['magnitude_db'].values
            signatures.append(mag)
        
        # Check that signatures are different from each other
        for i in range(len(signatures)):
            for j in range(i + 1, len(signatures)):
                self.assertFalse(np.allclose(signatures[i], signatures[j], rtol=0.01))


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_generate_synthetic_dataset_basic(self):
        """Test basic synthetic dataset generation."""
        train_df, test_df = generate_synthetic_dataset(
            n_samples=60,  # Small for testing
            output_dir=self.test_dir,
            export_formats=False,
            visualize=False
        )
        
        # Check DataFrames are returned
        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertIsInstance(test_df, pd.DataFrame)
        
        # Check data files were created
        train_path = os.path.join(self.test_dir, 'train_dataset.csv')
        test_path = os.path.join(self.test_dir, 'test_dataset.csv')
        
        self.assertTrue(os.path.exists(train_path))
        self.assertTrue(os.path.exists(test_path))


class TestIECCompliance(unittest.TestCase):
    """Test IEC 60076-18 compliance features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = TransformerSimulator(seed=42)
    
    def test_impedance_standard(self):
        """Test 50Ω impedance standard."""
        self.assertEqual(self.simulator.IEC_IMPEDANCE, 50)
    
    def test_frequency_range_standard(self):
        """Test frequency range follows IEC standard."""
        self.assertEqual(self.simulator.IEC_FREQ_MIN, 20)
        self.assertEqual(self.simulator.IEC_FREQ_MAX, 2e6)
    
    def test_lead_inductance_effect(self):
        """Test lead inductance is included."""
        self.assertGreater(self.simulator.LEAD_INDUCTANCE, 0)
        self.assertEqual(self.simulator.LEAD_INDUCTANCE, 50e-9)  # 50 nH
    
    def test_noise_level_realistic(self):
        """Test noise level is in realistic range (1-2 dB)."""
        self.assertEqual(self.simulator.NOISE_LEVEL, (1.0, 2.0))


def run_tests():
    """Run all tests and display results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestTransformerSimulator))
    suite.addTests(loader.loadTestsFromTestCase(TestConvenienceFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestIECCompliance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("SIMULATOR TEST SUMMARY")
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
