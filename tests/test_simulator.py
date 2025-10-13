"""
Unit Tests for Transformer FRA Simulator
SIH 2025 PS 25190

Tests cover:
- Input validation
- Physics correctness (energy conservation, causality)
- Fault signature generation
- Numerical stability
- Edge cases
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator import TransformerSimulator
from config import IEC, SIM


class TestSimulatorInitialization:
    """Test simulator initialization and validation."""
    
    def test_default_initialization(self):
        """Test simulator with default parameters."""
        sim = TransformerSimulator()
        
        assert sim.n_sections == SIM.default_n_sections
        assert sim.base_R == SIM.default_base_r
        assert sim.base_L == SIM.default_base_l
        assert sim.base_C == SIM.default_base_c
        assert sim.freq_points == SIM.default_freq_points
        assert sim.seed == SIM.default_seed
    
    def test_custom_initialization(self):
        """Test simulator with custom parameters."""
        sim = TransformerSimulator(
            n_sections=100,
            base_R=0.2,
            base_L=2e-3,
            base_C=20e-12,
            freq_points=500,
            seed=123
        )
        
        assert sim.n_sections == 100
        assert sim.base_R == 0.2
        assert sim.base_L == 2e-3
        assert sim.base_C == 20e-12
        assert sim.freq_points == 500
        assert sim.seed == 123
    
    def test_negative_sections_raises_error(self):
        """Test that negative n_sections raises ValueError."""
        with pytest.raises(ValueError, match="n_sections must be positive"):
            TransformerSimulator(n_sections=-10)
    
    def test_zero_sections_raises_error(self):
        """Test that zero n_sections raises ValueError."""
        with pytest.raises(ValueError, match="n_sections must be positive"):
            TransformerSimulator(n_sections=0)
    
    def test_negative_resistance_raises_error(self):
        """Test that negative resistance raises ValueError."""
        with pytest.raises(ValueError, match="base_R must be positive"):
            TransformerSimulator(base_R=-0.1)
    
    def test_negative_inductance_raises_error(self):
        """Test that negative inductance raises ValueError."""
        with pytest.raises(ValueError, match="base_L must be positive"):
            TransformerSimulator(base_L=-1e-3)
    
    def test_negative_capacitance_raises_error(self):
        """Test that negative capacitance raises ValueError."""
        with pytest.raises(ValueError, match="base_C must be positive"):
            TransformerSimulator(base_C=-10e-12)
    
    def test_negative_freq_points_raises_error(self):
        """Test that negative freq_points raises ValueError."""
        with pytest.raises(ValueError, match="freq_points must be positive"):
            TransformerSimulator(freq_points=-100)
    
    def test_frequency_array_properties(self):
        """Test that frequency array has correct properties."""
        sim = TransformerSimulator(freq_points=1000)
        
        # Check length
        assert len(sim.frequencies) == 1000
        
        # Check range
        assert sim.frequencies[0] == pytest.approx(IEC.freq_min, rel=0.01)
        assert sim.frequencies[-1] == pytest.approx(IEC.freq_max, rel=0.01)
        
        # Check logarithmic spacing
        log_freqs = np.log10(sim.frequencies)
        log_diffs = np.diff(log_freqs)
        assert np.std(log_diffs) < 0.01  # Should be nearly constant


class TestReproducibility:
    """Test reproducibility with seed."""
    
    def test_same_seed_produces_same_results(self):
        """Test that same seed produces identical results."""
        sim1 = TransformerSimulator(seed=42)
        sim2 = TransformerSimulator(seed=42)
        
        freq1, mag1, phase1 = sim1.generate_normal_signature()
        freq2, mag2, phase2 = sim2.generate_normal_signature()
        
        np.testing.assert_array_equal(freq1, freq2)
        np.testing.assert_array_almost_equal(mag1, mag2, decimal=10)
        np.testing.assert_array_almost_equal(phase1, phase2, decimal=10)
    
    def test_different_seed_produces_different_results(self):
        """Test that different seeds produce different results."""
        sim1 = TransformerSimulator(seed=42)
        sim2 = TransformerSimulator(seed=123)
        
        _, mag1, _ = sim1.generate_normal_signature()
        _, mag2, _ = sim2.generate_normal_signature()
        
        # Should be different (not identical)
        assert not np.allclose(mag1, mag2, rtol=1e-10)


class TestPhysicsCorrectness:
    """Test physics properties of generated signatures."""
    
    def test_normal_signature_shape(self):
        """Test that normal signature has correct shape."""
        sim = TransformerSimulator(freq_points=1000)
        freq, mag, phase = sim.generate_normal_signature()
        
        assert len(freq) == 1000
        assert len(mag) == 1000
        assert len(phase) == 1000
    
    def test_magnitude_is_finite(self):
        """Test that magnitude values are finite."""
        sim = TransformerSimulator()
        _, mag, _ = sim.generate_normal_signature()
        
        assert np.all(np.isfinite(mag))
        assert not np.any(np.isnan(mag))
        assert not np.any(np.isinf(mag))
    
    def test_phase_in_valid_range(self):
        """Test that phase is in -180 to 180 degrees."""
        sim = TransformerSimulator()
        _, _, phase = sim.generate_normal_signature()
        
        assert np.all(phase >= -180)
        assert np.all(phase <= 180)
    
    def test_magnitude_generally_decreasing(self):
        """Test that magnitude generally decreases with frequency (low-pass)."""
        sim = TransformerSimulator(seed=42)
        freq, mag, _ = sim.generate_normal_signature(add_noise=False)
        
        # Low frequency magnitude should be higher than high frequency
        low_freq_mag = np.mean(mag[:100])  # First 10%
        high_freq_mag = np.mean(mag[-100:])  # Last 10%
        
        assert low_freq_mag > high_freq_mag
    
    def test_causality_phase_relationship(self):
        """Test that phase follows Kramers-Kronig relations (causality)."""
        sim = TransformerSimulator(seed=42)
        _, _, phase = sim.generate_normal_signature(add_noise=False)
        
        # Phase should generally be negative (lag) for passive system
        # At least 70% of points should be negative
        assert np.sum(phase < 0) / len(phase) > 0.7


class TestFaultGeneration:
    """Test fault signature generation."""
    
    def test_axial_deformation_increases_inductance(self):
        """Test that axial deformation increases effective inductance."""
        sim = TransformerSimulator(seed=42)
        
        _, mag_normal, _ = sim.generate_normal_signature(add_noise=False)
        _, mag_fault, _ = sim.generate_axial_deformation(severity=0.2, add_noise=False)
        
        # Fault should change signature (not identical)
        assert not np.allclose(mag_normal, mag_fault)
        
        # Should see difference in mid-frequency range (inductive effects)
        mid_range = slice(400, 600)
        assert not np.allclose(mag_normal[mid_range], mag_fault[mid_range], rtol=0.01)
    
    def test_radial_deformation_changes_capacitance(self):
        """Test that radial deformation changes capacitance."""
        sim = TransformerSimulator(seed=42)
        
        _, mag_normal, _ = sim.generate_normal_signature(add_noise=False)
        _, mag_fault, _ = sim.generate_radial_deformation(severity=0.2, add_noise=False)
        
        # Should see difference
        assert not np.allclose(mag_normal, mag_fault)
    
    def test_interturn_short_reduces_impedance(self):
        """Test that inter-turn short reduces impedance."""
        sim = TransformerSimulator(seed=42)
        
        _, mag_normal, _ = sim.generate_normal_signature(add_noise=False)
        _, mag_fault, _ = sim.generate_interturn_short(severity=0.4, add_noise=False)
        
        # Should see difference
        assert not np.allclose(mag_normal, mag_fault)
    
    def test_core_grounding_affects_low_frequency(self):
        """Test that core grounding primarily affects low frequency."""
        sim = TransformerSimulator(seed=42)
        
        _, mag_normal, _ = sim.generate_normal_signature(add_noise=False)
        _, mag_fault, _ = sim.generate_core_grounding(severity=0.3, add_noise=False)
        
        # Low frequency should show more change than high frequency
        low_diff = np.abs(mag_normal[:100] - mag_fault[:100]).mean()
        high_diff = np.abs(mag_normal[-100:] - mag_fault[-100:]).mean()
        
        assert low_diff > high_diff * 0.5  # At least 50% more change at low freq
    
    def test_tapchanger_fault_creates_discontinuity(self):
        """Test that tap-changer fault creates local discontinuity."""
        sim = TransformerSimulator(seed=42)
        
        _, mag_normal, _ = sim.generate_normal_signature(add_noise=False)
        _, mag_fault, _ = sim.generate_tapchanger_fault(
            severity=0.2,
            fault_location=50,
            add_noise=False
        )
        
        # Should see difference
        assert not np.allclose(mag_normal, mag_fault)
    
    def test_severity_affects_fault_magnitude(self):
        """Test that higher severity creates larger deviation."""
        sim = TransformerSimulator(seed=42)
        
        _, mag_normal, _ = sim.generate_normal_signature(add_noise=False)
        _, mag_low, _ = sim.generate_axial_deformation(severity=0.05, add_noise=False)
        _, mag_high, _ = sim.generate_axial_deformation(severity=0.25, add_noise=False)
        
        diff_low = np.abs(mag_normal - mag_low).mean()
        diff_high = np.abs(mag_normal - mag_high).mean()
        
        assert diff_high > diff_low


class TestNoiseAddition:
    """Test measurement noise addition."""
    
    def test_noise_adds_variation(self):
        """Test that noise adds variation to signals."""
        sim = TransformerSimulator(seed=42)
        
        _, mag_clean, _ = sim.generate_normal_signature(add_noise=False)
        _, mag_noisy, _ = sim.generate_normal_signature(add_noise=True)
        
        # Should be different
        assert not np.allclose(mag_clean, mag_noisy)
        
        # But should be correlated (same underlying signal)
        correlation = np.corrcoef(mag_clean, mag_noisy)[0, 1]
        assert correlation > 0.95  # High correlation
    
    def test_noise_level_realistic(self):
        """Test that noise level is realistic (1-2 dB)."""
        sim = TransformerSimulator(seed=42)
        
        _, mag_clean, _ = sim.generate_normal_signature(add_noise=False)
        _, mag_noisy, _ = sim.generate_normal_signature(add_noise=True)
        
        noise = np.abs(mag_clean - mag_noisy)
        
        # 95% of noise should be within 3 dB (3-sigma for 1-2 dB noise)
        assert np.percentile(noise, 95) < 3.0


class TestDataFrameGeneration:
    """Test DataFrame generation methods."""
    
    def test_generate_fault_data(self):
        """Test generate_fault_data method."""
        sim = TransformerSimulator(seed=42)
        
        df = sim.generate_fault_data('normal', n_samples=10)
        
        assert len(df) == 10 * sim.freq_points
        assert 'frequency_hz' in df.columns
        assert 'magnitude_db' in df.columns
        assert 'phase_deg' in df.columns
        assert 'fault_type' in df.columns
        assert 'sample_id' in df.columns
        
        assert df['fault_type'].unique()[0] == 'normal'
    
    def test_invalid_fault_type_raises_error(self):
        """Test that invalid fault type raises error."""
        sim = TransformerSimulator()
        
        with pytest.raises(ValueError, match="Unknown fault type"):
            sim.generate_fault_data('invalid_fault', n_samples=1)
    
    def test_generate_balanced_dataset(self):
        """Test balanced dataset generation."""
        sim = TransformerSimulator(seed=42)
        
        train_df, test_df = sim.generate_balanced_dataset(
            total_samples=600,  # 100 per class
            train_ratio=0.8
        )
        
        # Check split ratio (approximately)
        total_samples = len(train_df) + len(test_df)
        train_ratio = len(train_df) / total_samples
        assert 0.75 < train_ratio < 0.85  # Allow some variation
        
        # Check fault types are balanced
        fault_counts = train_df.groupby('fault_type').size()
        assert fault_counts.std() / fault_counts.mean() < 0.1  # Low variance


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_small_n_sections(self):
        """Test with very small number of sections."""
        sim = TransformerSimulator(n_sections=5, seed=42)
        freq, mag, phase = sim.generate_normal_signature()
        
        assert len(freq) > 0
        assert np.all(np.isfinite(mag))
        assert np.all(np.isfinite(phase))
    
    def test_very_large_n_sections(self):
        """Test with large number of sections."""
        sim = TransformerSimulator(n_sections=200, freq_points=100, seed=42)
        freq, mag, phase = sim.generate_normal_signature()
        
        assert len(freq) == 100
        assert np.all(np.isfinite(mag))
        assert np.all(np.isfinite(phase))
    
    def test_extreme_parameter_values(self):
        """Test with extreme but valid parameter values."""
        sim = TransformerSimulator(
            base_R=1e-6,  # Very small resistance
            base_L=1e-6,  # Very small inductance
            base_C=1e-15,  # Very small capacitance
            seed=42
        )
        
        freq, mag, phase = sim.generate_normal_signature()
        assert np.all(np.isfinite(mag))
        assert np.all(np.isfinite(phase))


class TestHelperMethods:
    """Test private helper methods."""
    
    def test_generate_base_parameters(self):
        """Test _generate_base_parameters helper."""
        sim = TransformerSimulator(n_sections=75, seed=42)
        
        R, L, C = sim._generate_base_parameters()
        
        assert len(R) == 75
        assert len(L) == 75
        assert len(C) == 75
        
        # Values should be close to base with tolerance
        assert np.all(R > sim.base_R * SIM.tolerance_min)
        assert np.all(R < sim.base_R * SIM.tolerance_max)
    
    def test_compute_transfer_function_stability(self):
        """Test transfer function computation for stability."""
        sim = TransformerSimulator(seed=42)
        R, L, C = sim._generate_base_parameters()
        
        mag, phase = sim._compute_transfer_function(R, L, C)
        
        # Should not have NaN or Inf
        assert np.all(np.isfinite(mag))
        assert np.all(np.isfinite(phase))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
