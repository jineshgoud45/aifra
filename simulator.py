"""
Transformer FRA Signature Simulator
SIH 2025 PS 25190

Implements lumped-parameter ladder network model per IEC 60076-18 for generating
synthetic FRA data with various transformer faults. Uses R-L-C ladder topology
to model distributed winding characteristics.

Physical Model:
    Transfer Function: H(f) = Vout(f) / Vin(f)
    
    For n-section ladder network:
    H(jω) = Product[i=1 to n]( Z_C(i) / (Z_L(i) + Z_R(i) + Z_C(i)) )
    
    Where:
    - Z_R(i) = R_i (resistive component)
    - Z_L(i) = jωL_i (inductive reactance)
    - Z_C(i) = 1/(jωC_i) (capacitive reactance)

Fault Types:
    1. Normal: Baseline healthy transformer
    2. Axial Deformation: +15-20% inductance in affected sections
    3. Radial Deformation: +15-20% capacitance shift
    4. Inter-turn Short: Reduced resistance/inductance in shorted sections
    5. Core Grounding: Low-frequency capacitance increase
    6. Tap-changer Fault: Local impedance discontinuity

IEC 60076-18 Compliance:
    - Frequency range: 20 Hz to 2 MHz (logarithmic)
    - Impedance: 50Ω measurement standard
    - Lead effects: Additional series inductance at high frequency
    - Noise: Gaussian 1-2 dB to simulate real measurements
"""

__all__ = [
    'TransformerSimulator',
    'generate_synthetic_dataset'
]

import numpy as np
import pandas as pd
from scipy import signal
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import os
import xml.etree.ElementTree as ET
import logging
import re

# Import centralized configuration
try:
    from config import IEC, SIM
except ImportError:
    # Fallback for backwards compatibility
    class IEC:
        freq_min = 20
        freq_max = 2e6
        impedance = 50
    
    class SIM:
        default_n_sections = 75
        default_base_r = 0.1
        default_base_l = 1e-3
        default_base_c = 10e-12
        default_freq_points = 1000
        default_seed = 42
        lead_inductance = 50e-9
        tolerance_min = 0.95
        tolerance_max = 1.05
        deformation_min = 0.15
        deformation_max = 0.20
        short_circuit_min = 0.30
        short_circuit_max = 0.50
        core_ground_min = 0.20
        core_ground_max = 0.40
        tap_changer_min = 0.10
        tap_changer_max = 0.25
        noise_level_min = 1.0
        noise_level_max = 2.0
        tap_position_min = 0.4
        tap_position_max = 0.6
        affected_sections_min = 0.2
        affected_sections_max = 0.4
        localized_fault_min = 0.05
        localized_fault_max = 0.15
        core_ground_sections = 0.2

# Configure logging
logger = logging.getLogger(__name__)

# Numerical stability constants
EPSILON = np.finfo(float).eps
MIN_IMPEDANCE = 1e-20  # Minimum impedance to prevent division by zero

class TransformerSimulator:
    """
    Lumped-parameter ladder network simulator for transformer FRA signatures.
    
    Models transformer windings as cascaded R-L-C sections with configurable
    fault injection capabilities per IEC 60076-18 guidelines.
    
    Attributes:
        n_sections: Number of ladder sections (50-100 typical)
        base_R: Base resistance per section (Ω)
        base_L: Base inductance per section (H)
        base_C: Base capacitance per section (F)
        freq_points: Number of frequency points (1000 default)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self, 
        n_sections: int = SIM.default_n_sections,
        base_R: float = SIM.default_base_r,
        base_L: float = SIM.default_base_l,
        base_C: float = SIM.default_base_c,
        freq_points: int = SIM.default_freq_points,
        seed: Optional[int] = SIM.default_seed
    ):
        """
        Initialize transformer FRA simulator.
        
        Args:
            n_sections: Number of ladder sections (10-1000 reasonable range)
            base_R: Base resistance per section in Ω (0.001-100 reasonable range)
            base_L: Base inductance per section in H (1e-6 to 1e-1 reasonable range)
            base_C: Base capacitance per section in F (1e-15 to 1e-9 reasonable range)
            freq_points: Number of frequency points for analysis (10-100000 reasonable range)
            seed: Random seed for reproducibility (0-2^32)
            
        Raises:
            TypeError: If parameters are not of correct type
            ValueError: If any parameter is outside valid range
        """
        # CRITICAL FIX: Type validation (prevent type confusion attacks)
        if not isinstance(n_sections, int):
            raise TypeError(f"n_sections must be int, got {type(n_sections).__name__}")
        if not isinstance(freq_points, int):
            raise TypeError(f"freq_points must be int, got {type(freq_points).__name__}")
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"seed must be int or None, got {type(seed).__name__}")
        
        # Type check for floats (but allow int -> float conversion for convenience)
        if not isinstance(base_R, (int, float)):
            raise TypeError(f"base_R must be numeric, got {type(base_R).__name__}")
        if not isinstance(base_L, (int, float)):
            raise TypeError(f"base_L must be numeric, got {type(base_L).__name__}")
        if not isinstance(base_C, (int, float)):
            raise TypeError(f"base_C must be numeric, got {type(base_C).__name__}")
        
        # CRITICAL FIX: Range validation with reasonable bounds (prevent resource exhaustion)
        # n_sections bounds: too few -> poor model, too many -> memory exhaustion
        if not 10 <= n_sections <= 1000:
            raise ValueError(
                f"n_sections must be in [10, 1000] for reasonable performance and accuracy, got {n_sections}"
            )
        
        # Resistance bounds (Ω)
        if not 0.001 <= base_R <= 100.0:
            raise ValueError(
                f"base_R must be in [0.001, 100.0] Ω for physical realism, got {base_R}"
            )
        
        # Inductance bounds (H) - typical transformer windings
        if not 1e-6 <= base_L <= 1e-1:
            raise ValueError(
                f"base_L must be in [1e-6, 1e-1] H for physical realism, got {base_L}"
            )
        
        # Capacitance bounds (F) - typical inter-turn capacitance
        if not 1e-15 <= base_C <= 1e-9:
            raise ValueError(
                f"base_C must be in [1e-15, 1e-9] F for physical realism, got {base_C}"
            )
        
        # Frequency points bounds (prevent memory issues and ensure meaningful analysis)
        if not 10 <= freq_points <= 100000:
            raise ValueError(
                f"freq_points must be in [10, 100000] for reasonable analysis, got {freq_points}"
            )
        
        # Seed validation (must fit in 32-bit unsigned int for numpy)
        if seed is not None and not 0 <= seed <= 2**32 - 1:
            raise ValueError(
                f"seed must be in [0, 2^32-1] for numpy compatibility, got {seed}"
            )
        
        self.n_sections = n_sections
        self.base_R = float(base_R)  # Ensure float type
        self.base_L = float(base_L)
        self.base_C = float(base_C)
        self.freq_points = freq_points
        self.seed = seed
        
        # CORRECTNESS: Use local RNG for reproducibility
        # Prevents interference with global numpy random state
        # Use modern Generator API instead of deprecated RandomState
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        
        # Generate logarithmic frequency array per IEC 60076-18
        self.frequencies = np.logspace(
            np.log10(IEC.freq_min),
            np.log10(IEC.freq_max),
            freq_points
        )
        
        self.angular_freq = 2 * np.pi * self.frequencies
    
    def _generate_base_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate baseline R-L-C parameters with manufacturing tolerance.
        
        This method creates nominally identical parameters for all sections,
        with small random variations (±5%) to simulate manufacturing tolerances.
        
        Returns:
            Tuple of (R_values, L_values, C_values) arrays
        """
        R_values = self.base_R * np.ones(self.n_sections) * self.rng.uniform(
            SIM.tolerance_min, SIM.tolerance_max, self.n_sections
        )
        L_values = self.base_L * np.ones(self.n_sections) * self.rng.uniform(
            SIM.tolerance_min, SIM.tolerance_max, self.n_sections
        )
        C_values = self.base_C * np.ones(self.n_sections) * self.rng.uniform(
            SIM.tolerance_min, SIM.tolerance_max, self.n_sections
        )
        
        return R_values, L_values, C_values
        
    def _compute_transfer_function(
        self,
        R_values: np.ndarray,
        L_values: np.ndarray,
        C_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute transfer function H(jω) = Vout/Vin for ladder network.
        
        Transfer Function Derivation:
            For each section i:
                Z_series(i) = R(i) + jωL(i)
                Z_shunt(i) = 1/(jωC(i))
                
            Voltage divider per section:
                V(i+1)/V(i) = Z_shunt(i) / (Z_series(i) + Z_shunt(i))
                
            Total transfer function:
                H(jω) = Product[i=1 to n](V(i+1)/V(i))
        
        Args:
            R_values: Resistance array for each section (Ω)
            L_values: Inductance array for each section (H)
            C_values: Capacitance array for each section (F)
        
        Returns:
            Tuple of (magnitude in dB, phase in degrees)
        """
        # Initialize transfer function (complex)
        H = np.ones(len(self.frequencies), dtype=complex)
        
        # Add lead inductance per IEC 60076-18 (high frequency effect)
        lead_impedance = 1j * self.angular_freq * SIM.lead_inductance
        
        for i in range(self.n_sections):
            # Series impedance: R + jωL + lead effect
            Z_series = R_values[i] + 1j * self.angular_freq * L_values[i]
            
            # Add lead effect for first and last sections
            if i == 0 or i == self.n_sections - 1:
                Z_series += lead_impedance
            
            # Shunt impedance: 1/(jωC)
            # Use maximum to ensure numerical stability
            Z_shunt_denom = 1j * self.angular_freq * C_values[i]
            Z_shunt = 1.0 / np.maximum(np.abs(Z_shunt_denom), MIN_IMPEDANCE) * np.exp(1j * np.angle(Z_shunt_denom))
            
            # Voltage divider for this section
            Z_total = Z_series + Z_shunt
            # Check for near-zero impedance
            if np.any(np.abs(Z_total) < MIN_IMPEDANCE):
                logger.warning(f"Near-zero total impedance detected in section {i}")
            
            # CORRECTNESS: Prevent division by zero - clamp to minimum impedance
            if np.any(np.abs(Z_total) < MIN_IMPEDANCE):
                logger.warning(f"Near-zero total impedance detected in section {i}, clamping to MIN_IMPEDANCE")
                Z_total = np.maximum(np.abs(Z_total), MIN_IMPEDANCE) * np.exp(1j * np.angle(Z_total))
            
            H *= Z_shunt / Z_total
        
        # Convert to magnitude (dB) and phase (degrees)
        # Use maximum to prevent log(0)
        H_magnitude = np.abs(H)
        magnitude_dB = 20 * np.log10(np.maximum(H_magnitude, MIN_IMPEDANCE))
        phase_deg = np.angle(H, deg=True)
        
        return magnitude_dB, phase_deg
    
    def _add_measurement_noise(
        self,
        magnitude_dB: np.ndarray,
        phase_deg: np.ndarray,
        noise_level: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add Gaussian measurement noise per IEC 60076-18 realistic conditions.
        
        Args:
            magnitude_dB: Clean magnitude response
            phase_deg: Clean phase response
            noise_level: Noise level in dB (random 1-2 dB if None)
        
        Returns:
            Tuple of (noisy magnitude, noisy phase)
            
        Raises:
            ValueError: If noise_level is negative or unreasonably high
        """
        # VALIDATION: Check noise_level bounds
        if noise_level is not None:
            if noise_level < 0:
                raise ValueError(f"noise_level must be non-negative, got {noise_level}")
            if noise_level > 20:
                logger.warning(f"Unusually high noise_level: {noise_level} dB (typical: 1-2 dB)")
        
        if noise_level is None:
            noise_level = self.rng.uniform(SIM.noise_level_min, SIM.noise_level_max)
        
        # Add Gaussian noise to magnitude
        mag_noise = self.rng.normal(0, noise_level, len(magnitude_dB))
        noisy_magnitude = magnitude_dB + mag_noise
        
        # Add smaller noise to phase (proportional)
        phase_noise_level = noise_level * 0.5  # Phase noise is typically smaller
        phase_noise = self.rng.normal(0, phase_noise_level, len(phase_deg))
        noisy_phase = phase_deg + phase_noise
        
        # Normalize phase to -180 to 180
        noisy_phase = ((noisy_phase + 180) % 360) - 180
        
        return noisy_magnitude, noisy_phase
    
    def generate_normal_signature(
        self,
        add_noise: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate FRA signature for healthy (normal) transformer.
        
        Returns:
            Tuple of (frequencies, magnitude_dB, phase_deg)
        """
        # Use helper method for base parameters
        R_values, L_values, C_values = self._generate_base_parameters()
        
        magnitude_dB, phase_deg = self._compute_transfer_function(R_values, L_values, C_values)
        
        if add_noise:
            magnitude_dB, phase_deg = self._add_measurement_noise(magnitude_dB, phase_deg)
        
        return self.frequencies, magnitude_dB, phase_deg
    
    def generate_axial_deformation(
        self,
        severity: Optional[float] = None,
        affected_sections: Optional[List[int]] = None,
        add_noise: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate FRA signature with axial winding deformation.
        
        Axial deformation compresses/expands winding axially, increasing
        inductance in affected sections (15-20% typical).
        
        Args:
            severity: Deformation severity (0.15-0.20), random if None
            affected_sections: List of affected section indices, random if None
            add_noise: Whether to add measurement noise
        
        Returns:
            Tuple of (frequencies, magnitude_dB, phase_deg)
        """
        if severity is None:
            severity = self.rng.uniform(SIM.deformation_min, SIM.deformation_max)
        
        # Use helper method for base parameters
        R_values, L_values, C_values = self._generate_base_parameters()
        
        # Apply axial deformation (increased inductance)
        if affected_sections is None:
            # Random 20-40% of sections affected
            n_affected = self.rng.integers(
                int(SIM.affected_sections_min * self.n_sections),
                int(SIM.affected_sections_max * self.n_sections)
            )
            affected_sections = self.rng.choice(self.n_sections, n_affected, replace=False)
        
        L_values[affected_sections] *= (1 + severity)
        
        magnitude_dB, phase_deg = self._compute_transfer_function(R_values, L_values, C_values)
        
        if add_noise:
            magnitude_dB, phase_deg = self._add_measurement_noise(magnitude_dB, phase_deg)
        
        return self.frequencies, magnitude_dB, phase_deg
    
    def generate_radial_deformation(
        self,
        severity: Optional[float] = None,
        affected_sections: Optional[List[int]] = None,
        add_noise: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate FRA signature with radial winding deformation.
        
        Radial deformation changes inter-turn spacing, affecting capacitance
        (15-20% typical shift).
        
        Args:
            severity: Deformation severity (0.15-0.20), random if None
            affected_sections: List of affected section indices, random if None
            add_noise: Whether to add measurement noise
        
        Returns:
            Tuple of (frequencies, magnitude_dB, phase_deg)
        """
        if severity is None:
            severity = self.rng.uniform(SIM.deformation_min, SIM.deformation_max)
        
        # Use helper method for base parameters
        R_values, L_values, C_values = self._generate_base_parameters()
        
        # Apply radial deformation (changed capacitance)
        if affected_sections is None:
            n_affected = self.rng.integers(
                int(SIM.affected_sections_min * self.n_sections),
                int(SIM.affected_sections_max * self.n_sections)
            )
            affected_sections = self.rng.choice(self.n_sections, n_affected, replace=False)
        
        C_values[affected_sections] *= (1 + severity)
        
        magnitude_dB, phase_deg = self._compute_transfer_function(R_values, L_values, C_values)
        
        if add_noise:
            magnitude_dB, phase_deg = self._add_measurement_noise(magnitude_dB, phase_deg)
        
        return self.frequencies, magnitude_dB, phase_deg
    
    def generate_interturn_short(
        self,
        severity: Optional[float] = None,
        affected_sections: Optional[List[int]] = None,
        add_noise: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate FRA signature with inter-turn short circuit.
        
        Short circuit reduces effective turns, decreasing both resistance
        and inductance in affected sections (30-50% reduction).
        
        Args:
            severity: Short severity (0.30-0.50), random if None
            affected_sections: List of affected section indices, random if None
            add_noise: Whether to add measurement noise
        
        Returns:
            Tuple of (frequencies, magnitude_dB, phase_deg)
        """
        if severity is None:
            severity = self.rng.uniform(SIM.short_circuit_min, SIM.short_circuit_max)
        
        # Use helper method for base parameters
        R_values, L_values, C_values = self._generate_base_parameters()
        
        # Apply inter-turn short (reduced R and L)
        if affected_sections is None:
            # Typically localized fault (5-15% of sections)
            n_affected = self.rng.integers(
                int(SIM.localized_fault_min * self.n_sections),
                int(SIM.localized_fault_max * self.n_sections)
            )
            affected_sections = self.rng.choice(self.n_sections, n_affected, replace=False)
        
        R_values[affected_sections] *= (1 - severity)
        L_values[affected_sections] *= (1 - severity)
        
        magnitude_dB, phase_deg = self._compute_transfer_function(R_values, L_values, C_values)
        
        if add_noise:
            magnitude_dB, phase_deg = self._add_measurement_noise(magnitude_dB, phase_deg)
        
        return self.frequencies, magnitude_dB, phase_deg
    
    def generate_core_grounding(
        self,
        severity: Optional[float] = None,
        add_noise: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate FRA signature with core grounding fault.
        
        Core grounding increases low-frequency capacitance to ground
        (20-40% increase, primarily affects low frequency response).
        
        Args:
            severity: Grounding fault severity (0.20-0.40), random if None
            add_noise: Whether to add measurement noise
        
        Returns:
            Tuple of (frequencies, magnitude_dB, phase_deg)
        """
        if severity is None:
            severity = self.rng.uniform(SIM.core_ground_min, SIM.core_ground_max)
        
        # Use helper method for base parameters
        R_values, L_values, C_values = self._generate_base_parameters()
        
        # Core grounding primarily affects first sections (ground connection)
        affected_sections = range(int(SIM.core_ground_sections * self.n_sections))
        C_values[list(affected_sections)] *= (1 + severity)
        
        magnitude_dB, phase_deg = self._compute_transfer_function(R_values, L_values, C_values)
        
        if add_noise:
            magnitude_dB, phase_deg = self._add_measurement_noise(magnitude_dB, phase_deg)
        
        return self.frequencies, magnitude_dB, phase_deg
    
    def generate_tapchanger_fault(
        self,
        severity: Optional[float] = None,
        fault_location: Optional[int] = None,
        add_noise: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate FRA signature with tap-changer fault.
        
        Tap-changer fault creates impedance discontinuity at specific location
        (10-25% local variation).
        
        Args:
            severity: Fault severity (0.10-0.25), random if None
            fault_location: Section index of fault, random if None
            add_noise: Whether to add measurement noise
        
        Returns:
            Tuple of (frequencies, magnitude_dB, phase_deg)
        """
        if severity is None:
            severity = self.rng.uniform(SIM.tap_changer_min, SIM.tap_changer_max)
        
        # Use helper method for base parameters
        R_values, L_values, C_values = self._generate_base_parameters()
        
        # Tap-changer fault at specific location
        if fault_location is None:
            # Typically at tap position (middle 40-60% of winding)
            fault_location = self.rng.integers(
                int(SIM.tap_position_min * self.n_sections),
                int(SIM.tap_position_max * self.n_sections)
            )
        
        # Create discontinuity (±2 sections around fault)
        affected_range = range(
            max(0, fault_location - 2),
            min(self.n_sections, fault_location + 3)
        )
        R_values[list(affected_range)] *= (1 + severity * self.rng.choice([-1, 1]))
        L_values[list(affected_range)] *= (1 + severity * self.rng.choice([-1, 1]))
        
        magnitude_dB, phase_deg = self._compute_transfer_function(R_values, L_values, C_values)
        
        if add_noise:
            magnitude_dB, phase_deg = self._add_measurement_noise(magnitude_dB, phase_deg)
        
        return self.frequencies, magnitude_dB, phase_deg
    
    def inject_fault(
        self,
        fault_type: str,
        severity: float = 0.2,
        affected_percentage: Optional[float] = None
    ) -> None:
        """
        Inject a specific fault into the transformer model.
        
        Args:
            fault_type: Type of fault (normal, axial_deformation, radial_deformation,
                       interturn_short, core_grounding, tapchanger_fault)
            severity: Severity of fault (0.0 to 1.0)
            affected_percentage: Percentage of sections affected (optional)
        
        Raises:
            ValueError: If fault_type is invalid or severity is out of range
        """
        # Input validation
        if not 0 <= severity <= 1:
            raise ValueError(f"Severity must be in range [0, 1], got {severity}")
        
        valid_faults = [
            'normal', 'axial_deformation', 'radial_deformation',
            'interturn_short', 'core_grounding', 'tapchanger_fault'
        ]
        if fault_type not in valid_faults:
            raise ValueError(f"Invalid fault_type '{fault_type}'. Must be one of: {valid_faults}")
        
        if affected_percentage is not None and not 0 < affected_percentage <= 1:
            raise ValueError(f"affected_percentage must be in range (0, 1], got {affected_percentage}")
    
    def generate_fault_data(
        self,
        fault_type: str,
        n_samples: int = 1000,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate multiple samples of specified fault type.
        
        Args:
            fault_type: One of 'normal', 'axial_deformation', 'radial_deformation',
                       'interturn_short', 'core_grounding', 'tapchanger_fault'
            n_samples: Number of samples to generate
            **kwargs: Additional arguments passed to specific fault generator
        
        Returns:
            pd.DataFrame: Normalized data with all samples
        """
        fault_methods = {
            'normal': self.generate_normal_signature,
            'axial_deformation': self.generate_axial_deformation,
            'radial_deformation': self.generate_radial_deformation,
            'interturn_short': self.generate_interturn_short,
            'core_grounding': self.generate_core_grounding,
            'tapchanger_fault': self.generate_tapchanger_fault
        }
        
        if fault_type not in fault_methods:
            raise ValueError(f"Unknown fault type: {fault_type}")
        
        all_data = []
        
        for sample_idx in range(n_samples):
            freq, mag, phase = fault_methods[fault_type](**kwargs)
            
            # Create DataFrame for this sample
            df = pd.DataFrame({
                'frequency_hz': freq,
                'magnitude_db': mag,
                'phase_deg': phase,
                'fault_type': fault_type,
                'sample_id': f"{fault_type}_{sample_idx:04d}",
                'timestamp': datetime.now()
            })
            
            all_data.append(df)
        
        return pd.concat(all_data, ignore_index=True)
    
    def generate_balanced_dataset(
        self,
        total_samples: int = 5000,
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate balanced dataset with all fault types.
        
        Args:
            total_samples: Total number of samples to generate
            train_ratio: Ratio of training samples (0.0-1.0)
        
        Returns:
            Tuple of (train_df, test_df)
        """
        fault_types = [
            'normal',
            'axial_deformation',
            'radial_deformation',
            'interturn_short',
            'core_grounding',
            'tapchanger_fault'
        ]
        
        samples_per_class = total_samples // len(fault_types)
        
        all_data = []
        
        logger.info(f"Generating {total_samples} balanced samples across {len(fault_types)} fault types...")
        
        for fault_type in fault_types:
            logger.info(f"  Generating {samples_per_class} samples for {fault_type}...")
            df = self.generate_fault_data(fault_type, n_samples=samples_per_class)
            all_data.append(df)
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Shuffle
        combined_df = combined_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        # Split train/test
        n_train = int(len(combined_df) // self.freq_points * train_ratio) * self.freq_points
        train_df = combined_df.iloc[:n_train].reset_index(drop=True)
        test_df = combined_df.iloc[n_train:].reset_index(drop=True)
        
        logger.info(f"\nDataset generated:")
        logger.info(f"  Train: {len(train_df) // self.freq_points} samples ({len(train_df)} rows)")
        logger.info(f"  Test: {len(test_df) // self.freq_points} samples ({len(test_df)} rows)")
        
        return train_df, test_df
    
    def generate_synthetic_dataset(
        self,
        n_samples: int,
        fault_distribution: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic dataset with multiple fault types.
        
        Args:
            n_samples: Number of samples to generate
            fault_distribution: Dictionary mapping fault types to proportions
        
        Returns:
            DataFrame with all synthetic samples
        """
        if fault_distribution is None:
            # Default balanced distribution
            fault_distribution = {
                'normal': 0.30,
                'axial_deformation': 0.15,
                'radial_deformation': 0.15,
                'interturn_short': 0.15,
                'core_grounding': 0.15,
                'tapchanger_fault': 0.10
            }
        
        logger.info(f"Generating {n_samples} synthetic samples...")
        
        all_data = []
        
        for fault_type, proportion in fault_distribution.items():
            n_fault_samples = int(n_samples * proportion)
            logger.info(f"  Generating {n_fault_samples} samples of {fault_type}")
            
            for i in range(n_fault_samples):
                if fault_type == 'normal':
                    freq, mag, phase = self.generate_normal_signature()
                elif fault_type == 'axial_deformation':
                    freq, mag, phase = self.generate_axial_deformation()
                elif fault_type == 'radial_deformation':
                    freq, mag, phase = self.generate_radial_deformation()
                elif fault_type == 'interturn_short':
                    freq, mag, phase = self.generate_interturn_short()
                elif fault_type == 'core_grounding':
                    freq, mag, phase = self.generate_core_grounding()
                elif fault_type == 'tapchanger_fault':
                    freq, mag, phase = self.generate_tapchanger_fault()
                else:
                    continue
                
                sample_df = pd.DataFrame({
                    'frequency_hz': freq,
                    'magnitude_db': mag,
                    'phase_deg': phase,
                    'sample_id': f'{fault_type}_{i:04d}',
                    'fault_type': fault_type
                })
                
                all_data.append(sample_df)
        
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"Generated {len(result)} total data points across {n_samples} samples")
        
        return result
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent directory traversal and invalid characters.
        
        Args:
            filename: Original filename or sample ID
        
        Returns:
            Sanitized filename safe for use in file paths
        """
        # Remove or replace dangerous characters
        safe_name = re.sub(r'[^\w\-.]', '_', filename)
        
        # Prevent directory traversal
        if safe_name in ('.', '..'):
            raise ValueError(f"Invalid filename: {filename}")
        
        # Remove leading dots to prevent hidden files
        safe_name = safe_name.lstrip('.')
        
        # Ensure name is not empty after sanitization
        if not safe_name:
            raise ValueError(f"Filename becomes empty after sanitization")
        
        return safe_name
    
    def export_omicron_csv(
        self,
        data: pd.DataFrame,
        output_dir: str = 'synthetic_data',
        prefix: str = 'omicron'
    ) -> List[str]:
        """
        Export data in Omicron FRANEO CSV format.
        
        Args:
            data: DataFrame with FRA data
            output_dir: Output directory path
            prefix: Filename prefix
        
        Returns:
            List of generated file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get unique samples
        sample_ids = data['sample_id'].unique()
        generated_files = []
        
        for sample_id in sample_ids:
            sample_data = data[data['sample_id'] == sample_id]
            safe_sample_id = self._sanitize_filename(str(sample_id))
            filepath = os.path.join(output_dir, f"{prefix}_{safe_sample_id}.csv")
            
            with open(filepath, 'w') as f:
                # Write Omicron header
                f.write("# Omicron FRANEO FRA Measurement\n")
                f.write("# Device: FRANEO 800 (Simulated)\n")
                f.write(f"# Test ID: {sample_id}\n")
                f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("# Impedance: 50 Ohm\n")
                f.write(f"# Fault Type: {sample_data['fault_type'].iloc[0]}\n")
                f.write("# Configuration: Synthetic Data Generation\n")
                f.write("Frequency (Hz),Magnitude (dB),Phase (deg)\n")
                
                # Write data
                for _, row in sample_data.iterrows():
                    f.write(f"{row['frequency_hz']:.6e},{row['magnitude_db']:.6f},{row['phase_deg']:.6f}\n")
            
            generated_files.append(filepath)
            logger.info(f"Exported Omicron CSV: {filepath}")
        
        return generated_files
    
    def export_doble_txt(
        self,
        data: pd.DataFrame,
        output_dir: str = 'synthetic_data',
        prefix: str = 'doble'
    ) -> List[str]:
        """
        Export data in Doble SFRA TXT format.
        
        Args:
            data: DataFrame with FRA data
            output_dir: Output directory path
            prefix: Filename prefix
        
        Returns:
            List of generated file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        sample_ids = data['sample_id'].unique()
        generated_files = []
        
        for sample_id in sample_ids:
            sample_data = data[data['sample_id'] == sample_id]
            safe_sample_id = self._sanitize_filename(str(sample_id))
            filepath = os.path.join(output_dir, f"{prefix}_{safe_sample_id}.txt")
            
            with open(filepath, 'w') as f:
                # Write Doble header
                f.write("Doble Engineering SFRA Test Data (Simulated)\n")
                f.write(f"Test ID: {sample_id}\n")
                f.write(f"Date: {datetime.now().strftime('%m/%d/%Y')}\n")
                f.write(f"Fault Type: {sample_data['fault_type'].iloc[0]}\n")
                f.write("Operator: Simulator\n")
                f.write("Test Configuration: Synthetic Data\n")
                f.write("Impedance: 50 Ohms\n")
                f.write("\n")
                f.write("Frequency (Hz)\tMagnitude (dB)\tPhase (deg)\n")
                
                # Write data
                for _, row in sample_data.iterrows():
                    f.write(f"{row['frequency_hz']:.6e}\t{row['magnitude_db']:.6f}\t{row['phase_deg']:.6f}\n")
            
            generated_files.append(filepath)
            logger.info(f"Exported Doble TXT: {filepath}")
        
        return generated_files
    
    def export_megger_dat(
        self,
        data: pd.DataFrame,
        output_dir: str = 'synthetic_data',
        prefix: str = 'megger'
    ) -> List[str]:
        """
        Export data in Megger FRAX DAT format.
        
        Args:
            data: DataFrame with FRA data
            output_dir: Output directory path
            prefix: Filename prefix
        
        Returns:
            List of generated file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        sample_ids = data['sample_id'].unique()
        generated_files = []
        
        for sample_id in sample_ids:
            sample_data = data[data['sample_id'] == sample_id]
            safe_sample_id = self._sanitize_filename(str(sample_id))
            filepath = os.path.join(output_dir, f"{prefix}_{safe_sample_id}.dat")
            
            with open(filepath, 'w') as f:
                # Write Megger header
                f.write("[MEGGER FRAX 101 - SIMULATED]\n")
                f.write("DeviceType=FRAX101\n")
                f.write("SerialNumber=SIM123456\n")
                f.write(f"TestID={sample_id}\n")
                f.write(f"DateTime={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Operator=Simulator\n")
                f.write(f"FaultType={sample_data['fault_type'].iloc[0]}\n")
                f.write("Configuration=Synthetic\n")
                f.write("Impedance=50\n")
                f.write("[MEASUREMENT_DATA]\n")
                f.write("Frequency (Hz),Magnitude (dB),Phase (deg)\n")
                
                # Write data
                for _, row in sample_data.iterrows():
                    f.write(f"{row['frequency_hz']:.6e},{row['magnitude_db']:.6f},{row['phase_deg']:.6f}\n")
            
            generated_files.append(filepath)
            logger.info(f"Exported Megger DAT: {filepath}")
        
        return generated_files
    
    def export_all_formats(
        self,
        data: pd.DataFrame,
        output_dir: str = 'synthetic_data',
        sample_limit: Optional[int] = None
    ) -> Dict[str, List[str]]:
        """
        Export data in all vendor formats.
        
        Args:
            data: DataFrame with FRA data
            output_dir: Output directory path
            sample_limit: Limit number of samples per format (None = all)
        
        Returns:
            Dictionary mapping format name to list of file paths
        """
        if sample_limit is not None:
            # Limit samples
            unique_samples = data['sample_id'].unique()[:sample_limit]
            data = data[data['sample_id'].isin(unique_samples)]
        
        logger.info(f"Exporting {len(data['sample_id'].unique())} samples to all vendor formats...")
        
        results = {}
        
        logger.info("  Exporting Omicron CSV...")
        results['omicron'] = self.export_omicron_csv(data, output_dir, 'omicron')
        
        logger.info("  Exporting Doble TXT...")
        results['doble'] = self.export_doble_txt(data, output_dir, 'doble')
        
        logger.info("  Exporting Megger DAT...")
        results['megger'] = self.export_megger_dat(data, output_dir, 'megger')
        
        logger.info(f"\nExported {sum(len(files) for files in results.values())} files total.")
        
        return results
    
    def visualize_signatures(
        self,
        signatures: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        title: str = "FRA Signatures Comparison",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Create Bode plot visualization comparing multiple FRA signatures.
        
        Args:
            signatures: Dict mapping label to (freq, mag, phase) tuples
            title: Plot title
            save_path: Path to save figure (None = display only)
            figsize: Figure size (width, height)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(signatures)))
        
        for (label, (freq, mag, phase)), color in zip(signatures.items(), colors):
            # Magnitude plot
            ax1.semilogx(freq, mag, label=label, linewidth=2, color=color, alpha=0.8)
            
            # Phase plot
            ax2.semilogx(freq, phase, label=label, linewidth=2, color=color, alpha=0.8)
        
        # Configure magnitude plot
        ax1.set_ylabel('Magnitude (dB)', fontsize=12, fontweight='bold')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, which='both', alpha=0.3, linestyle='--')
        ax1.legend(loc='best', fontsize=10)
        ax1.set_xlim(IEC.freq_min, IEC.freq_max)
        
        # Configure phase plot
        ax2.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Phase (degrees)', fontsize=12, fontweight='bold')
        ax2.grid(True, which='both', alpha=0.3, linestyle='--')
        ax2.legend(loc='best', fontsize=10)
        ax2.set_xlim(IEC.freq_min, IEC.freq_max)
        
        # Add IEC range annotations
        ax1.axvline(IEC.freq_min, color='red', linestyle=':', alpha=0.5, linewidth=1)
        ax1.axvline(IEC.freq_max, color='red', linestyle=':', alpha=0.5, linewidth=1)
        ax2.axvline(IEC.freq_min, color='red', linestyle=':', alpha=0.5, linewidth=1)
        ax2.axvline(IEC.freq_max, color='red', linestyle=':', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved: {save_path}")
        
        plt.show()
    
    def visualize_fault_comparison(
        self,
        save_dir: Optional[str] = 'visualizations'
    ):
        """
        Generate comparison plots for normal vs all fault types.
        
        Args:
            save_dir: Directory to save plots (None = display only)
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        logger.info("Generating fault comparison visualizations...")
        
        # Generate one sample of each fault type
        fault_types = [
            'normal',
            'axial_deformation',
            'radial_deformation',
            'interturn_short',
            'core_grounding',
            'tapchanger_fault'
        ]
        
        signatures = {}
        
        for fault_type in fault_types:
            logger.info(f"  Generating {fault_type}...")
            if fault_type == 'normal':
                freq, mag, phase = self.generate_normal_signature(add_noise=False)
            elif fault_type == 'axial_deformation':
                freq, mag, phase = self.generate_axial_deformation(add_noise=False)
            elif fault_type == 'radial_deformation':
                freq, mag, phase = self.generate_radial_deformation(add_noise=False)
            elif fault_type == 'interturn_short':
                freq, mag, phase = self.generate_interturn_short(add_noise=False)
            elif fault_type == 'core_grounding':
                freq, mag, phase = self.generate_core_grounding(add_noise=False)
            elif fault_type == 'tapchanger_fault':
                freq, mag, phase = self.generate_tapchanger_fault(add_noise=False)
            
            signatures[fault_type.replace('_', ' ').title()] = (freq, mag, phase)
        
        # Plot all faults together
        save_path = os.path.join(save_dir, 'all_faults_comparison.png') if save_dir else None
        self.visualize_signatures(
            signatures,
            title="Transformer FRA Signatures: Normal vs Fault Conditions",
            save_path=save_path
        )
        
        # Individual comparisons: Normal vs each fault
        normal_sig = signatures['Normal']
        
        for fault_name, fault_sig in signatures.items():
            if fault_name == 'Normal':
                continue
            
            comparison = {
                'Normal (Healthy)': normal_sig,
                f'{fault_name}': fault_sig
            }
            
            save_path = os.path.join(save_dir, f'normal_vs_{fault_name.lower().replace(" ", "_")}.png') if save_dir else None
            self.visualize_signatures(
                comparison,
                title=f"FRA Comparison: Normal vs {fault_name}",
                save_path=save_path,
                figsize=(12, 8)
            )
        
        logger.info(f"\nVisualization complete. {len(signatures) + len(signatures) - 1} plots generated.")


# Convenience functions for quick usage
def generate_synthetic_dataset(
    n_samples: int = 5000,
    output_dir: str = 'synthetic_data',
    export_formats: bool = True,
    visualize: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to generate complete synthetic FRA dataset.
    
    Args:
        n_samples: Total number of samples to generate
        output_dir: Output directory for exported files
        export_formats: Whether to export in vendor formats
        visualize: Whether to generate visualization plots
    
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info("="*70)
    logger.info("Transformer FRA Synthetic Data Generator")
    logger.info("SIH 2025 PS 25190")
    logger.info("="*70)
    logger.info("")
    
    # Initialize simulator
    simulator = TransformerSimulator(seed=42)
    
    # Generate balanced dataset
    train_df, test_df = simulator.generate_balanced_dataset(
        total_samples=n_samples,
        train_ratio=0.8
    )
    
    # Export samples in vendor formats
    if export_formats:
        logger.info("\nExporting samples to vendor formats...")
        # Export a subset (e.g., 10 samples per fault type)
        export_data = train_df.groupby('fault_type').head(10 * simulator.freq_points)
        simulator.export_all_formats(export_data, output_dir, sample_limit=None)
    
    # Generate visualizations
    if visualize:
        logger.info("\nGenerating visualizations...")
        simulator.visualize_fault_comparison(save_dir=os.path.join(output_dir, 'plots'))
    
    # Save complete datasets
    train_path = os.path.join(output_dir, 'train_dataset.csv')
    test_path = os.path.join(output_dir, 'test_dataset.csv')
    
    logger.info(f"\nSaving complete datasets...")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"  Train dataset: {train_path}")
    logger.info(f"  Test dataset: {test_path}")
    
    logger.info("\n" + "="*70)
    logger.info("Dataset generation complete!")
    logger.info("="*70)
    
    return train_df, test_df


if __name__ == '__main__':
    # Example usage
    train_df, test_df = generate_synthetic_dataset(
        n_samples=5000,
        output_dir='synthetic_data',
        export_formats=True,
        visualize=True
    )
    
    logger.info("\nDataset Summary:")
    logger.info(f"Train samples: {len(train_df['sample_id'].unique())}")
    logger.info(f"Test samples: {len(test_df['sample_id'].unique())}")
    logger.info(f"\nFault type distribution (train):")
    logger.info(train_df.groupby('fault_type')['sample_id'].nunique())
