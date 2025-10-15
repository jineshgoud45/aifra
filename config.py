
__all__ = [
    'str_to_bool',
    'IECStandards',
    'SimulatorConfig',
    'ParserConfig',
    'AppConfig',
    'MLConfig',
    'SecurityConfig',
    'LoggingConfig',
    'TestingConfig',
    'IEC',
    'SIM',
    'PARSER',
    'APP',
    'ML',
    'TESTING',
    'SECURITY',
    'LOGGING',
    'FAULT_TYPES',
    'FAULT_DISTRIBUTION_DEFAULT',
    'get_config_summary'
]

import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Base directory
BASE_DIR = Path(__file__).resolve().parent


def str_to_bool(value: str) -> bool:
    """
    Convert string to boolean for environment variables.
    
    Args:
        value: String value to convert
    
    Returns:
        bool: True for '1', 'true', 'yes', 'on' (case-insensitive), False otherwise
    
    Examples:
        >>> str_to_bool('true')
        True
        >>> str_to_bool('YES')
        True
        >>> str_to_bool('0')
        False
    """
    return value.lower() in ('1', 'true', 'yes', 'on')


@dataclass
class IECStandards:
    """IEC 60076-18 compliance standards."""
    freq_min: float = 20.0  # Hz
    freq_max: float = 2e6  # Hz (2 MHz)
    impedance: float = 50.0  # Ω
    frequency_tolerance: float = 2.0  # Allow 2x tolerance on boundaries
    
    # QA thresholds
    artifact_threshold_db: float = 3.0  # dB deviation threshold
    artifact_percentage_max: float = 0.05  # Maximum 5% artifacts
    log_spacing_threshold: float = 0.5  # Maximum std dev for log spacing
    min_data_points: int = 10
    max_data_points: int = 100000


@dataclass
class SimulatorConfig:
    """Transformer simulator configuration."""
    # Default simulator parameters
    default_n_sections: int = 75
    default_base_r: float = 0.1  # Ω
    default_base_l: float = 1e-3  # H
    default_base_c: float = 10e-12  # F
    default_freq_points: int = 1000
    default_seed: int = 42
    
    # Lead effects
    lead_inductance: float = 50e-9  # 50 nH typical lead inductance
    
    # Manufacturing tolerance
    tolerance_min: float = 0.95
    tolerance_max: float = 1.05
    
    # Fault severity ranges (percentage)
    deformation_min: float = 0.15
    deformation_max: float = 0.20
    short_circuit_min: float = 0.30
    short_circuit_max: float = 0.50
    core_ground_min: float = 0.20
    core_ground_max: float = 0.40
    tap_changer_min: float = 0.10
    tap_changer_max: float = 0.25
    
    # Noise parameters
    noise_level_min: float = 1.0  # dB
    noise_level_max: float = 2.0  # dB
    
    # Fault location ranges
    tap_position_min: float = 0.4  # 40% of winding
    tap_position_max: float = 0.6  # 60% of winding
    tap_affected_sections: int = 5  # ±2 sections around fault
    
    # Affected sections for deformation faults
    affected_sections_min: float = 0.2  # 20% of sections
    affected_sections_max: float = 0.4  # 40% of sections
    
    # Localized faults (inter-turn shorts)
    localized_fault_min: float = 0.05  # 5% of sections
    localized_fault_max: float = 0.15  # 15% of sections
    
    # Core grounding affected sections
    core_ground_sections: float = 0.2  # First 20% of sections


@dataclass
class ParserConfig:
    """Parser configuration."""
    max_file_size_mb: int = 50
    max_xml_iterations: int = 200000
    min_smoothing_window: int = 11  # Must be odd
    window_fraction_divisor: int = 5
    
    # Supported vendors
    supported_vendors: tuple = ('omicron', 'doble', 'megger', 'generic')
    
    # Supported file extensions
    allowed_extensions: tuple = ('.csv', '.xml', '.txt', '.dat')
    
    # Canonical schema columns
    canonical_columns: tuple = (
        'frequency_hz', 'magnitude_db', 'phase_deg',
        'test_id', 'vendor', 'timestamp'
    )


@dataclass
class AppConfig:
    """Web application configuration."""
    # Paths (use environment variables for deployment)
    model_dir: Path = field(default_factory=lambda: Path(
        os.getenv('FRA_MODEL_DIR', str(BASE_DIR / 'models'))
    ))
    temp_dir: Path = field(default_factory=lambda: Path(
        os.getenv('FRA_TEMP_DIR', str(BASE_DIR / 'temp'))
    ))
    data_dir: Path = field(default_factory=lambda: Path(
        os.getenv('FRA_DATA_DIR', str(BASE_DIR / 'data'))
    ))
    
    # Cache settings
    cache_ttl_seconds: int = 3600  # 1 hour
    
    # Upload limits
    max_upload_size_mb: int = 50
    max_upload_size_bytes: int = field(init=False)
    
    # Security
    require_auth: bool = field(default_factory=lambda: 
        str_to_bool(os.getenv('FRA_REQUIRE_AUTH', 'false'))
    )
    
    # Performance
    enable_profiling: bool = field(default_factory=lambda:
        str_to_bool(os.getenv('FRA_ENABLE_PROFILING', 'false'))
    )
    
    def __post_init__(self):
        """Calculate derived values."""
        self.max_upload_size_bytes = self.max_upload_size_mb * 1024 * 1024
        
        # Ensure directories exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # SECURITY: Validate auth configuration
        if self.require_auth:
            import warnings
            # Check if authentication secrets are configured
            auth_secret = os.getenv('FRA_AUTH_SECRET')
            if not auth_secret:
                warnings.warn(
                    "⚠️ SECURITY WARNING: Authentication is enabled (FRA_REQUIRE_AUTH=true) "
                    "but FRA_AUTH_SECRET is not set. Please configure authentication secrets "
                    "before deploying to production!",
                    UserWarning,
                    stacklevel=2
                )
            elif len(auth_secret) < 32:
                warnings.warn(
                    f"⚠️ SECURITY WARNING: FRA_AUTH_SECRET is too short ({len(auth_secret)} chars). "
                    f"Use at least 32 characters for production security.",
                    UserWarning,
                    stacklevel=2
                )


@dataclass
class MLConfig:
    """Machine learning configuration."""
    # Model architecture
    cnn_channels: tuple = (64, 128, 256)
    cnn_kernel_size: int = 5
    cnn_dropout_rate: float = 0.5  # Dropout rate for CNN layers
    resnet_version: str = 'resnet18'
    resnet_dropout_fc1: float = 0.5  # First FC layer dropout
    resnet_dropout_fc2: float = 0.3  # Second FC layer dropout
    
    # Training
    default_epochs: int = 50
    default_batch_size: int = 32
    default_learning_rate: float = 0.001
    train_test_split: float = 0.8
    
    # SVM
    svm_nu: float = 0.1
    svm_kernel: str = 'rbf'
    svm_gamma: str = 'scale'
    
    # DTW optimization parameters
    dtw_max_sequence_length: int = 500  # Subsample if sequences exceed this
    dtw_early_termination_threshold: float = 1000.0  # Stop if cost exceeds this
    dtw_subsample_points: int = 500  # Number of points when subsampling
    
    # Compute resources
    num_workers: int = field(default_factory=lambda: 
        int(os.getenv('FRA_NUM_WORKERS', '4'))
    )
    device: str = field(default_factory=lambda:
        os.getenv('FRA_DEVICE', 'cpu')  # 'cuda' or 'cpu'
    )


@dataclass
class TestingConfig:
    """Testing and benchmarking configuration."""
    # Test data limits
    max_test_data_points: int = 10000
    integration_test_timeout: int = 300  # seconds
    
    # Performance benchmarks
    max_parse_time_seconds: float = 5.0
    max_inference_time_seconds: float = 2.0
    max_report_generation_seconds: float = 10.0
    
    # Rate limiting for testing
    rate_limit_window_seconds: int = 60
    rate_limit_max_uploads: int = 10
    
    # Edge case test parameters
    max_test_file_size_mb: int = 100
    min_test_data_points: int = 5
    unicode_test_enabled: bool = True


@dataclass
class SecurityConfig:
    """Security configuration."""
    # File validation
    check_magic_bytes: bool = True
    max_xml_entity_expansions: int = 100
    
    # XXE protection
    use_defusedxml: bool = True
    
    # Path traversal protection
    allow_absolute_paths: bool = False
    max_path_length: int = 255
    
    # Rate limiting (if deployed as API)
    enable_rate_limiting: bool = field(default_factory=lambda:
        str_to_bool(os.getenv('FRA_ENABLE_RATE_LIMIT', 'false'))
    )
    requests_per_minute: int = 60


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = field(default_factory=lambda:
        os.getenv('FRA_LOG_LEVEL', 'INFO')
    )
    log_file: Path = field(default_factory=lambda: 
        BASE_DIR / 'logs' / 'fra_diagnostics.log'
    )
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    max_log_size_mb: int = 10
    backup_count: int = 5
    
    def __post_init__(self):
        """Ensure log directory exists."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


# Global configuration instances
IEC = IECStandards()
SIM = SimulatorConfig()
PARSER = ParserConfig()
APP = AppConfig()
ML = MLConfig()
TESTING = TestingConfig()
SECURITY = SecurityConfig()
LOGGING = LoggingConfig()


# Fault type mappings
FAULT_TYPES = {
    'normal': 'Normal (Healthy)',
    'axial_deformation': 'Axial Deformation',
    'radial_deformation': 'Radial Deformation',
    'interturn_short': 'Inter-turn Short Circuit',
    'core_grounding': 'Core Grounding Fault',
    'tapchanger_fault': 'Tap-changer Fault'
}

FAULT_DISTRIBUTION_DEFAULT = {
    'normal': 0.30,
    'axial_deformation': 0.15,
    'radial_deformation': 0.15,
    'interturn_short': 0.15,
    'core_grounding': 0.15,
    'tapchanger_fault': 0.10
}


def get_config_summary() -> Dict[str, str]:
    """Get configuration summary for debugging."""
    return {
        'iec_freq_range': f"{IEC.freq_min} Hz - {IEC.freq_max} Hz",
        'model_dir': str(APP.model_dir),
        'temp_dir': str(APP.temp_dir),
        'max_upload_mb': APP.max_upload_size_mb,
        'cache_ttl': f"{APP.cache_ttl_seconds}s",
        'require_auth': str(APP.require_auth),
        'device': ML.device,
        'log_level': LOGGING.log_level
    }


if __name__ == '__main__':
    # Print configuration summary
    logger.info("FRA Diagnostics Platform Configuration")
    logger.info("=" * 50)
    
    config = get_config_summary()
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    
    logger.info("=" * 50)
    logger.info("Configuration loaded successfully")
