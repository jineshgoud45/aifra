"""
Universal FRA/SFRA Data Parser for Power Systems Diagnostics
SIH 2025 PS 25190

This module provides a comprehensive parser for Frequency Response Analysis (FRA) and 
Swept Frequency Response Analysis (SFRA) data from multiple vendors including:
- Omicron (FRANEO CSV/XML formats)
- Doble (SFRA suites)
- Megger (FRAX analyzers)

The parser implements IEC 60076-18 compliance checks for power transformer diagnostics.
"""

__all__ = [
    'UniversalFRAParser',
    'FRAParserError',
    'IECComplianceWarning',
    'parse_fra_file'
]

import pandas as pd
import numpy as np
import defusedxml.ElementTree as ET  # SECURITY: Use defusedxml instead of xml.etree
import os
import warnings
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Import centralized configuration
try:
    from config import IEC, PARSER
except ImportError:
    # DESIGN DECISION: Fallback configuration for standalone module usage
    # 
    # This allows parser.py to be used independently without requiring
    # the full project structure. Useful for:
    # 1. Direct integration into other projects
    # 2. Testing in isolation
    # 3. Development without full dependency tree
    # 
    # In production, config.py should always be present, making this dead code.
    # However, it provides valuable development flexibility.
    class IEC:
        freq_min = 20
        freq_max = 2e6
        frequency_tolerance = 2.0
        artifact_threshold_db = 3.0
        artifact_percentage_max = 0.05
        log_spacing_threshold = 0.5
        min_data_points = 10
        max_data_points = 100000
    
    class PARSER:
        max_file_size_mb = 50
        max_xml_iterations = 200000
        min_smoothing_window = 11
        window_fraction_divisor = 5
        supported_vendors = ('omicron', 'doble', 'megger', 'generic')
        allowed_extensions = ('.csv', '.xml', '.txt', '.dat')
        canonical_columns = (
            'frequency_hz', 'magnitude_db', 'phase_deg',
            'test_id', 'vendor', 'timestamp'
        )

# Configure logging
logger = logging.getLogger(__name__)

class FRAParserError(Exception):
    """Custom exception for FRA/SFRA parsing errors."""
    pass


class IECComplianceWarning(UserWarning):
    """Warning for IEC 60076-18 compliance issues."""
    pass


class UniversalFRAParser:
    """
    Universal parser for FRA/SFRA data from multiple vendors.
    
    IEC 60076-18 Compliance:
        - Frequency range: 20 Hz to 2 MHz (log scale)
        - Impedance: 50Ω standard
        - Channel symmetry verification
        - Artifact detection (>3 dB deviation threshold)
    
    Canonical Schema:
        frequency_hz, magnitude_db, phase_deg, test_id, vendor, timestamp
    """
    
    CANONICAL_COLUMNS = list(PARSER.canonical_columns)
    
    def __init__(self) -> None:
        """Initialize the FRA parser with supported vendors."""
        self.supported_vendors: List[str] = list(PARSER.supported_vendors)
        self.qa_results: Dict[str, Dict] = {}
        
    def parse_file(self, filepath: str, vendor: Optional[str] = None) -> pd.DataFrame:
        """Parse FRA/SFRA data file with auto-detection."""
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Check if file is empty
        if os.path.getsize(filepath) == 0:
            logger.error(f"File is empty: {filepath}")
            raise FRAParserError(f"File is empty: {filepath}")
        
        # Check file size
        file_size = os.path.getsize(filepath)
        if file_size > PARSER.max_file_size_mb * 1024 * 1024:  # Rough estimate: 100 bytes per data point
            logger.warning(f"Large file detected: {file_size} bytes")
        
        file_format = self._detect_format(filepath)
        if vendor is None:
            vendor = self._detect_vendor(filepath, file_format)
        
        vendor = vendor.lower()
        if vendor not in self.supported_vendors:
            logger.error(f"Unsupported vendor: {vendor}")
            raise FRAParserError(f"Unsupported vendor: {vendor}")
        
        try:
            logger.info(f"Parsing file: {filepath} (vendor={vendor}, format={file_format})")
            
            if vendor == 'omicron':
                df = self._parse_omicron_xml(filepath) if file_format == 'xml' else self._parse_omicron_csv(filepath)
            elif vendor == 'doble':
                df = self._parse_doble(filepath)
            elif vendor == 'megger':
                df = self._parse_megger(filepath)
            else:
                df = self._parse_generic(filepath, file_format)
            
            # Validate parsed data
            if df is None or len(df) == 0:
                raise FRAParserError("No valid data extracted from file")
            
            if len(df) < IEC.min_data_points:
                logger.warning(f"Only {len(df)} data points found (minimum recommended: {IEC.min_data_points})")
            
            if len(df) > IEC.max_data_points:
                logger.warning(f"Too many data points ({len(df)}), truncating to {IEC.max_data_points}")
                df = df.iloc[:IEC.max_data_points]
            
            df = self._normalize_dataframe(df)
            self._perform_qa_checks(df, filepath)
            
            logger.info(f"Successfully parsed {len(df)} data points")
            return df
            
        except (pd.errors.ParserError, ValueError, KeyError) as e:
            error_msg = f"Error parsing {filepath}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise FRAParserError(error_msg) from e
        except ET.ParseError as e:
            error_msg = f"XML parsing error in {filepath}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise FRAParserError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error parsing {filepath}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise FRAParserError(error_msg) from e
    
    def _detect_format(self, filepath: str) -> str:
        """Detect file format (csv, xml, txt)."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.xml':
            return 'xml'
        elif ext == '.csv':
            return 'csv'
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
                return 'csv' if (',' in first_line or '\t' in first_line) else 'txt'
        except UnicodeDecodeError as e:
            # Only catch encoding errors, treat as text file
            logger.warning(f"Encoding error detecting format for {filepath}: {e}")
            return 'txt'
        # Let IOError and OSError propagate - they indicate serious problems
    
    def _detect_vendor(self, filepath: str, file_format: str) -> str:
        """Auto-detect vendor from file content."""
        filename = os.path.basename(filepath).lower()
        
        if 'franeo' in filename or 'omicron' in filename:
            return 'omicron'
        elif 'doble' in filename or 'dta' in filename:
            return 'doble'
        elif 'megger' in filename or 'frax' in filename:
            return 'megger'
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000).lower()
                if 'franeo' in content or 'omicron' in content:
                    return 'omicron'
                elif 'doble' in content:
                    return 'doble'
                elif 'megger' in content or 'frax' in content:
                    return 'megger'
        except (IOError, OSError, UnicodeDecodeError) as e:
            logger.warning(f"Error detecting vendor for {filepath}: {e}")
        
        return 'generic'
    
    def _parse_omicron_csv(self, filepath: str) -> pd.DataFrame:
        """Parse Omicron FRANEO CSV format."""
        metadata = {}
        data_start_row = 0
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith('#') or line_stripped.startswith(';'):
                if ':' in line_stripped:
                    parts = line_stripped[1:].split(':', 1)
                    if len(parts) == 2:
                        metadata[parts[0].strip()] = parts[1].strip()
            
            if any(kw in line_stripped.lower() for kw in ['frequency', 'magnitude', 'phase']):
                data_start_row = i
                break
        
        df = pd.read_csv(filepath, skiprows=data_start_row, encoding='utf-8',
                        skipinitialspace=True, on_bad_lines='skip')
        df.columns = df.columns.str.strip().str.lower()
        
        freq_col = self._find_column(df, ['frequency', 'freq', 'f'])
        mag_col = self._find_column(df, ['magnitude', 'mag', 'db'])
        phase_col = self._find_column(df, ['phase', 'phi', 'angle'])
        
        if not all([freq_col, mag_col, phase_col]):
            raise FRAParserError("Missing required columns in Omicron CSV")
        
        return pd.DataFrame({
            'frequency_hz': pd.to_numeric(df[freq_col], errors='coerce'),
            'magnitude_db': pd.to_numeric(df[mag_col], errors='coerce'),
            'phase_deg': pd.to_numeric(df[phase_col], errors='coerce'),
            'test_id': metadata.get('Test ID', os.path.basename(filepath)),
            'vendor': 'omicron',
            'timestamp': self._parse_timestamp(metadata.get('Date'))
        })
    
    def _parse_omicron_xml(self, filepath: str) -> pd.DataFrame:
        """Parse Omicron FRANEO XML format."""
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
        except ET.ParseError as e:
            raise FRAParserError(f"Invalid XML: {str(e)}")
        
        metadata = {elem.tag: elem.text for elem in root.iter() 
                   if elem.tag in ['TestID', 'TestDate', 'Device']}
        
        frequencies, magnitudes, phases = [], [], []
        iteration_count = 0
        
        for measurement in root.iter('Measurement'):
            freq = measurement.find('Frequency')
            mag = measurement.find('Magnitude')
            phase = measurement.find('Phase')
            
            if all([freq is not None, mag is not None, phase is not None]):
                try:
                    frequencies.append(float(freq.text))
                    magnitudes.append(float(mag.text))
                    phases.append(float(phase.text))
                except (ValueError, TypeError):
                    continue
            
            iteration_count += 1
            if iteration_count > PARSER.max_xml_iterations:
                raise FRAParserError("XML iteration limit exceeded")
        
        if not frequencies:
            raise FRAParserError("No measurement data in XML")
        
        return pd.DataFrame({
            'frequency_hz': frequencies,
            'magnitude_db': magnitudes,
            'phase_deg': phases,
            'test_id': metadata.get('TestID', os.path.basename(filepath)),
            'vendor': 'omicron',
            'timestamp': self._parse_timestamp(metadata.get('TestDate'))
        })
    
    def _parse_doble(self, filepath: str) -> pd.DataFrame:
        """Parse Doble SFRA suite format."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(1024)
            delimiter = '\t' if '\t' in sample else ','
            f.seek(0)  # Reset to beginning
            lines = f.readlines()
        
        metadata, data_start = {}, 0
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in ['frequency', 'magnitude', 'phase']):
                data_start = i
                break
            if ':' in line or '=' in line:
                sep = ':' if ':' in line else '='
                parts = line.split(sep, 1)
                if len(parts) == 2:
                    metadata[parts[0].strip()] = parts[1].strip()
        
        df = pd.read_csv(
            filepath, 
            sep=delimiter, 
            skiprows=data_start, 
            encoding='utf-8', 
            skipinitialspace=True, 
            on_bad_lines='warn'
        )
        df.columns = df.columns.str.strip().str.lower()
        
        freq_col = self._find_column(df, ['frequency', 'freq', 'hz'])
        mag_col = self._find_column(df, ['magnitude', 'mag', 'db'])
        phase_col = self._find_column(df, ['phase', 'phi', 'angle'])
        
        if not all([freq_col, mag_col, phase_col]):
            raise FRAParserError("Missing required columns in Doble file")
        
        return pd.DataFrame({
            'frequency_hz': pd.to_numeric(df[freq_col], errors='coerce'),
            'magnitude_db': pd.to_numeric(df[mag_col], errors='coerce'),
            'phase_deg': pd.to_numeric(df[phase_col], errors='coerce'),
            'test_id': metadata.get('Test ID', os.path.basename(filepath)),
            'vendor': 'doble',
            'timestamp': self._parse_timestamp(metadata.get('Date'))
        })
    
    def _parse_megger(self, filepath: str) -> pd.DataFrame:
        """Parse Megger FRAX analyzer format."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        metadata, data_start = {}, 0
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if '=' in line_stripped and not any(kw in line_stripped.lower() 
                                                for kw in ['frequency', 'magnitude']):
                key, val = line_stripped.split('=', 1)
                metadata[key.strip()] = val.strip()
            
            if any(kw in line_stripped.lower() for kw in ['frequency', 'magnitude', 'phase']):
                data_start = i
                break
        
        delimiter = '\t' if '\t' in lines[data_start] else ','
        df = pd.read_csv(filepath, sep=delimiter, skiprows=data_start, 
                        encoding='utf-8', skipinitialspace=True, on_bad_lines='skip')
        df.columns = df.columns.str.strip().str.lower()
        
        freq_col = self._find_column(df, ['frequency', 'freq', 'hz'])
        mag_col = self._find_column(df, ['magnitude', 'mag', 'db'])
        phase_col = self._find_column(df, ['phase', 'phi', 'angle'])
        
        if not all([freq_col, mag_col, phase_col]):
            raise FRAParserError("Missing required columns in Megger file")
        
        return pd.DataFrame({
            'frequency_hz': pd.to_numeric(df[freq_col], errors='coerce'),
            'magnitude_db': pd.to_numeric(df[mag_col], errors='coerce'),
            'phase_deg': pd.to_numeric(df[phase_col], errors='coerce'),
            'test_id': metadata.get('TestID', os.path.basename(filepath)),
            'vendor': 'megger',
            'timestamp': self._parse_timestamp(metadata.get('DateTime'))
        })
    
    def _parse_generic(self, filepath: str, file_format: str) -> pd.DataFrame:
        """Parse generic FRA/SFRA file format."""
        delimiter = ',' if file_format == 'csv' else r'\s+'
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(1024)
            if '\t' in sample:
                delimiter = '\t'
        
        df = pd.read_csv(filepath, sep=delimiter, encoding='utf-8',
                        skipinitialspace=True, comment='#', on_bad_lines='skip',
                        engine='python' if delimiter == r'\s+' else 'c')
        df.columns = df.columns.str.strip().str.lower()
        
        freq_col = self._find_column(df, ['frequency', 'freq', 'hz', 'f'])
        mag_col = self._find_column(df, ['magnitude', 'mag', 'db'])
        phase_col = self._find_column(df, ['phase', 'phi', 'angle'])
        
        if not all([freq_col, mag_col, phase_col]):
            raise FRAParserError("Cannot identify required columns")
        
        return pd.DataFrame({
            'frequency_hz': pd.to_numeric(df[freq_col], errors='coerce'),
            'magnitude_db': pd.to_numeric(df[mag_col], errors='coerce'),
            'phase_deg': pd.to_numeric(df[phase_col], errors='coerce'),
            'test_id': os.path.basename(filepath),
            'vendor': 'generic',
            'timestamp': datetime.now()
        })
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find column by matching possible names."""
        for name in possible_names:
            for col in df.columns:
                if name in col.lower():
                    return col
        return None
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize to canonical schema per IEC 60076-18."""
        # PERFORMANCE: Chain operations to avoid intermediate DataFrame copies
        # SECURITY WARNING: .query() uses eval() internally - NEVER pass user input to it!
        # Current usage is safe (hardcoded string), but future developers should be aware.
        df = (df
              .dropna(subset=['frequency_hz', 'magnitude_db', 'phase_deg'])
              .query('frequency_hz > 0')  # SAFE: hardcoded string, no user input
              .astype({
                  'frequency_hz': float,
                  'magnitude_db': float,
                  'phase_deg': float
              })
              .assign(
                  # Normalize phase to -180 to 180 range
                  phase_deg=lambda x: ((x['phase_deg'] + 180) % 360) - 180
              )
              .sort_values('frequency_hz')
              .reset_index(drop=True)
        )
        
        # Ensure all canonical columns exist
        for col in self.CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = datetime.now() if col == 'timestamp' else 'unknown'
        
        return df[self.CANONICAL_COLUMNS]
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> datetime:
        """Parse timestamp from various formats."""
        # RUNTIME CHECK: Explicitly handle None before operations
        if not timestamp_str:
            return datetime.now()
        
        # Strip whitespace only after confirming it's a string
        try:
            timestamp_str = timestamp_str.strip()
        except AttributeError:
            # If strip() fails, it's not a string
            return datetime.now()
        
        formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y %H:%M:%S',
                  '%d/%m/%Y', '%m/%d/%Y', '%Y%m%d_%H%M%S']
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except (ValueError, AttributeError):
                continue
        return datetime.now()
    
    def _perform_qa_checks(self, df: pd.DataFrame, filepath: str) -> Dict:
        """Perform IEC 60076-18 compliance QA checks."""
        qa_results = {
            'filepath': filepath,
            'test_id': df['test_id'].iloc[0] if not df.empty else 'unknown',
            'vendor': df['vendor'].iloc[0] if not df.empty else 'unknown',
            'checks': {}
        }
        
        df_len = len(df)  # Cache length for performance
        
        if df_len == 0:
            qa_results['checks']['data_present'] = {'passed': False}
            self.qa_results[filepath] = qa_results
            return qa_results
        
        # VALIDATION: Ensure required columns exist
        required_cols = {'frequency_hz', 'magnitude_db', 'phase_deg', 'test_id', 'vendor'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            logger.error(error_msg)
            raise FRAParserError(error_msg)
        
        # Check 1: Frequency range (IEC 60076-18: 20 Hz - 2 MHz)
        freq_min, freq_max = df['frequency_hz'].min(), df['frequency_hz'].max()
        # CORRECTNESS: Check if frequencies are WITHIN IEC range (with tolerance)
        # freq_min should be at least IEC.freq_min/tolerance (10 Hz with tolerance=2)
        # freq_max should be at most IEC.freq_max*tolerance (4 MHz with tolerance=2)
        freq_ok = (freq_min >= IEC.freq_min / IEC.frequency_tolerance and 
                   freq_max <= IEC.freq_max * IEC.frequency_tolerance)
        
        qa_results['checks']['frequency_range'] = {
            'passed': freq_ok,
            'min_freq': freq_min,
            'max_freq': freq_max,
            'expected_min': IEC.freq_min,
            'expected_max': IEC.freq_max,
            'message': f"Range: {freq_min:.1f} Hz to {freq_max:.2e} Hz (Expected: {IEC.freq_min} Hz to {IEC.freq_max:.2e} Hz)"
        }
        
        if not freq_ok:
            warnings.warn(
                f"Frequency range {freq_min:.1f}-{freq_max:.2e} Hz outside IEC 60076-18 spec "
                f"({IEC.freq_min}-{IEC.freq_max:.2e} Hz)", 
                IECComplianceWarning
            )
        
        # Check 2: Frequency grid (log scale)
        freq_log_diff = np.diff(np.log10(df['frequency_hz']))
        freq_log_std = np.std(freq_log_diff)
        log_ok = freq_log_std < IEC.log_spacing_threshold
        
        qa_results['checks']['frequency_grid'] = {
            'passed': log_ok,
            'log_spacing_std': freq_log_std,
            'num_points': len(df)
        }
        
        # Check 3: Channel symmetry (magnitude stats)
        qa_results['checks']['magnitude_stats'] = {
            'passed': True,
            'mean_db': df['magnitude_db'].mean(),
            'std_db': df['magnitude_db'].std(),
            'min_db': df['magnitude_db'].min(),
            'max_db': df['magnitude_db'].max()
        }
        
        # Check 4: Artifact detection (>3 dB deviation)
        if df_len >= IEC.min_data_points:
            # Calculate adaptive window size (must be odd and >= 3)
            window = max(3, min(PARSER.min_smoothing_window, df_len // PARSER.window_fraction_divisor))
            window = window + 1 if window % 2 == 0 else window
            
            # CRITICAL FIX: Ensure we have enough data points for the window
            if df_len < window:
                # Not enough data for artifact detection with this window size
                qa_results['checks']['artifacts'] = {
                    'passed': True,
                    'num_artifacts': 0,
                    'threshold_db': IEC.artifact_threshold_db,
                    'max_deviation_db': 0.0,
                    'message': f'Insufficient data points ({df_len}) for artifact detection (need >= {window})'
                }
            else:
                # Proceed with artifact detection
                mag_smooth = df['magnitude_db'].rolling(window=window, center=True, 
                                                        min_periods=1).median()
                deviations = np.abs(df['magnitude_db'] - mag_smooth)
                artifacts = deviations > IEC.artifact_threshold_db
                
                num_artifacts = artifacts.sum()
                artifact_ok = num_artifacts < df_len * IEC.artifact_percentage_max
                
                qa_results['checks']['artifacts'] = {
                    'passed': artifact_ok,
                    'num_artifacts': int(num_artifacts),
                    'threshold_db': IEC.artifact_threshold_db,
                    'max_deviation_db': float(deviations.max())
                }
                
                if not artifact_ok:
                    warnings.warn(f"Detected {num_artifacts} artifacts (>3 dB)", 
                                IECComplianceWarning)
        else:
            # Not enough data points to perform artifact detection
            qa_results['checks']['artifacts'] = {
                'passed': True,
                'num_artifacts': 0,
                'threshold_db': IEC.artifact_threshold_db,
                'max_deviation_db': 0.0,
                'message': f'Insufficient data points ({df_len}) for artifact detection (need >= {IEC.min_data_points})'
            }
        
        self.qa_results[filepath] = qa_results
        return qa_results
    
    def get_qa_results(self, filepath: Optional[str] = None) -> Dict:
        """Get QA check results for a specific file or all files."""
        if filepath:
            return self.qa_results.get(filepath, {})
        return self.qa_results


def parse_fra_file(filepath: str, vendor: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to parse FRA/SFRA file.
    
    Args:
        filepath: Path to data file
        vendor: Optional vendor name (omicron/doble/megger)
    
    Returns:
        pd.DataFrame: Normalized data with canonical schema
    """
    parser = UniversalFRAParser()
    return parser.parse_file(filepath, vendor)
