# FRA Diagnostics Platform - API Documentation

**Version**: 1.0.0  
**Standard**: IEC 60076-18:2012  
**Last Updated**: 2025-10-13

---

## Table of Contents

1. [Overview](#overview)
2. [Core Modules](#core-modules)
3. [Parser API](#parser-api)
4. [Simulator API](#simulator-api)
5. [AI Ensemble API](#ai-ensemble-api)
6. [Report Generator API](#report-generator-api)
7. [Web Application API](#web-application-api)
8. [Data Formats](#data-formats)
9. [Configuration](#configuration)
10. [Error Handling](#error-handling)

---

## Overview

The FRA Diagnostics Platform provides a comprehensive API for analyzing Frequency Response Analysis data from power transformers. The platform supports multiple vendor formats and uses AI-based fault classification.

### Key Features

- **Multi-vendor support**: Omicron, Doble, Megger, Generic formats
- **AI Ensemble**: CNN + ResNet18 + One-Class SVM
- **IEC Compliance**: Full IEC 60076-18 compliance checking
- **Report Generation**: Professional PDF reports
- **Rate Limiting**: Built-in DoS protection

### Quick Start

```python
from parser import UniversalFRAParser
from ai_ensemble import load_models
from report_generator import generate_iec_report

# Parse FRA data
parser = UniversalFRAParser()
df = parser.parse_file('data.csv', vendor='omicron')

# Run AI analysis
ensemble = load_models()
prediction = ensemble.predict(df)

# Generate report
qa_results = parser.get_qa_results('data.csv')
report_path = generate_iec_report(df, prediction, qa_results)
```

---

## Core Modules

### Module: `parser.py`

Universal parser for multi-vendor FRA data files.

### Module: `simulator.py`

Lumped-parameter transformer model for synthetic data generation.

### Module: `ai_ensemble.py`

Multi-model AI ensemble for fault classification.

### Module: `report_generator.py`

IEC-compliant PDF report generation.

### Module: `app.py`

Streamlit web application dashboard.

---

## Parser API

### Class: `UniversalFRAParser`

Main parser class for FRA/SFRA data.

#### Methods

##### `parse_file(filepath: str, vendor: Optional[str] = None) -> pd.DataFrame`

Parse FRA data file with automatic vendor detection.

**Parameters:**
- `filepath` (str): Path to FRA data file
- `vendor` (Optional[str]): Vendor identifier ('omicron', 'doble', 'megger', 'generic') or None for auto-detect

**Returns:**
- `pd.DataFrame`: Normalized data with canonical schema

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `FRAParserError`: If parsing fails

**Example:**
```python
parser = UniversalFRAParser()
df = parser.parse_file('measurement.csv', vendor='omicron')
print(f"Parsed {len(df)} data points")
```

##### `get_qa_results(filepath: Optional[str] = None) -> Dict`

Get IEC 60076-18 compliance check results.

**Parameters:**
- `filepath` (Optional[str]): Path to file, or None for all results

**Returns:**
- `Dict`: QA check results with frequency range, artifacts, etc.

**Example:**
```python
qa_results = parser.get_qa_results('measurement.csv')
if qa_results['checks']['frequency_range']['passed']:
    print("✓ Frequency range compliant")
```

### Canonical Data Schema

All parsed data follows this schema:

| Column | Type | Description |
|--------|------|-------------|
| `frequency_hz` | float | Frequency in Hertz (20 Hz - 2 MHz) |
| `magnitude_db` | float | Magnitude in decibels |
| `phase_deg` | float | Phase angle in degrees (-180 to 180) |
| `test_id` | str | Test identifier |
| `vendor` | str | Equipment vendor |
| `timestamp` | datetime | Measurement timestamp |

---

## Simulator API

### Class: `TransformerSimulator`

Lumped-parameter model for generating synthetic FRA signatures.

#### Constructor

```python
TransformerSimulator(
    n_sections: int = 75,
    base_R: float = 0.1,
    base_L: float = 1e-3,
    base_C: float = 10e-12,
    freq_points: int = 1000,
    seed: Optional[int] = 42
)
```

**Parameters:**
- `n_sections`: Number of ladder sections (10-1000)
- `base_R`: Base resistance per section in Ω (0.001-100)
- `base_L`: Base inductance per section in H (1e-6 to 1e-1)
- `base_C`: Base capacitance per section in F (1e-15 to 1e-9)
- `freq_points`: Frequency points (10-100000)
- `seed`: Random seed for reproducibility

**Raises:**
- `TypeError`: If parameters are wrong type
- `ValueError`: If parameters outside valid range

#### Methods

##### `generate_normal_signature(add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`

Generate healthy transformer signature.

**Returns:**
- Tuple of (frequency, magnitude, phase) arrays

**Example:**
```python
sim = TransformerSimulator(n_sections=75, seed=42)
freq, mag, phase = sim.generate_normal_signature()
```

##### `generate_axial_deformation(severity: float = 0.15) -> Tuple`
##### `generate_radial_deformation(severity: float = 0.15) -> Tuple`
##### `generate_interturn_short(severity: float = 0.30) -> Tuple`
##### `generate_core_grounding(severity: float = 0.25) -> Tuple`
##### `generate_tapchanger_fault(severity: float = 0.15) -> Tuple`

Generate specific fault signatures.

**Parameters:**
- `severity` (float): Fault severity (0.0-1.0)

---

## AI Ensemble API

### Function: `load_models(save_dir: str = 'models', device: str = 'cpu') -> FRAEnsemble`

Load trained AI ensemble models.

**Parameters:**
- `save_dir`: Directory containing saved models
- `device`: Computation device ('cpu' or 'cuda')

**Returns:**
- `FRAEnsemble`: Loaded ensemble model

**Raises:**
- `FileNotFoundError`: If models not found
- `Exception`: If loading fails

### Class: `FRAEnsemble`

Multi-model ensemble for fault classification.

#### Method: `predict(sample_df: pd.DataFrame) -> Dict`

Predict fault type for FRA sample.

**Parameters:**
- `sample_df`: DataFrame with FRA data (canonical schema)

**Returns:**
- Dictionary with prediction results:

```python
{
    'predicted_fault': str,          # Fault type
    'confidence': float,              # 0.0-1.0
    'probabilities': Dict[str, float], # All fault probabilities
    'cnn_probs': Dict[str, float],    # CNN predictions
    'resnet_probs': Dict[str, float], # ResNet predictions
    'svm_score': float,               # Anomaly score
    'uncertainty': float              # Shannon entropy
}
```

**Example:**
```python
ensemble = load_models()
prediction = ensemble.predict(df)
print(f"Fault: {prediction['predicted_fault']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

---

## Report Generator API

### Function: `generate_iec_report(...) -> str`

Generate IEC-compliant PDF report.

**Parameters:**
```python
generate_iec_report(
    df: pd.DataFrame,                      # FRA data
    prediction_result: Dict,               # AI predictions
    qa_results: Dict,                      # QA check results
    test_id: str = "FRA_TEST_001",        # Test identifier
    transformer_details: Optional[Dict] = None,  # Transformer metadata
    output_path: str = "fra_report.pdf"   # Output path
) -> str
```

**Returns:**
- Path to generated PDF file

**Example:**
```python
report_path = generate_iec_report(
    df=parsed_data,
    prediction_result=prediction,
    qa_results=qa_results,
    test_id="XFMR_001_2025",
    transformer_details={
        'Serial Number': 'ABC123',
        'Rated Power': '50 MVA',
        'Voltage': '132/33 kV'
    }
)
```

---

## Web Application API

### Streamlit Dashboard

Access the web interface:

```bash
streamlit run app.py
```

Default URL: `http://localhost:8501`

### Health Check Endpoint

```
GET /_stcore/health
```

Returns HTTP 200 if application is running.

### Rate Limiting

- **Default Limit**: 10 uploads per 60 seconds per client
- **Identification**: Session-based (IP hash when available)
- **Response**: HTTP 429-equivalent with wait time

**Configuration:**
```python
# In config.py
TESTING.rate_limit_max_uploads = 10
TESTING.rate_limit_window_seconds = 60
```

---

## Data Formats

### Supported Input Formats

#### 1. Omicron FRANEO CSV

```csv
Frequency (Hz),Magnitude (dB),Phase (deg)
20.0,-45.2,10.5
25.0,-44.8,12.3
...
```

#### 2. Omicron FRANEO XML

```xml
<?xml version="1.0"?>
<FRA>
  <Measurement>
    <Point frequency="20" magnitude="-45.2" phase="10.5"/>
    ...
  </Measurement>
</FRA>
```

#### 3. Generic CSV

```csv
frequency_hz,magnitude_db,phase_deg
20.0,-45.2,10.5
...
```

### Output Data Format

JSON-compatible dictionary structure:

```json
{
  "predicted_fault": "normal",
  "confidence": 0.95,
  "probabilities": {
    "normal": 0.95,
    "axial_deformation": 0.02,
    "radial_deformation": 0.01,
    "interturn_short": 0.01,
    "core_grounding": 0.005,
    "tapchanger_fault": 0.005
  },
  "svm_score": 0.85,
  "uncertainty": 0.12
}
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FRA_LOG_DIR` | Log directory | `./logs` |
| `FRA_MODEL_DIR` | AI models directory | `./models` |
| `FRA_TEMP_DIR` | Temporary files | `./temp` |
| `FRA_DATA_DIR` | Data directory | `./data` |
| `FRA_LOG_LEVEL` | Logging level | `INFO` |
| `FRA_REQUIRE_AUTH` | Enable authentication | `false` |
| `FRA_AUTH_SECRET` | Auth secret key | - |
| `FRA_DEVICE` | Computation device | `cpu` |

### Configuration File

All settings in `config.py`:

```python
from config import IEC, SIM, PARSER, APP, ML, TESTING

# IEC Standards
print(f"Frequency range: {IEC.freq_min} - {IEC.freq_max} Hz")

# Simulator config
print(f"Default sections: {SIM.default_n_sections}")

# Application config
print(f"Max upload: {APP.max_upload_size_mb} MB")
```

---

## Error Handling

### Exception Hierarchy

```
Exception
├── FRAParserError          # Parsing failures
├── IECComplianceWarning    # IEC standard violations
├── FileNotFoundError       # Missing files
├── ValueError              # Invalid parameters
└── TypeError               # Type mismatches
```

### Error Response Format

```python
try:
    df = parser.parse_file('data.csv')
except FRAParserError as e:
    print(f"Parse error: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| `FILE_TOO_LARGE` | File exceeds size limit | Reduce file size or increase limit |
| `INVALID_FORMAT` | Unsupported file format | Convert to supported format |
| `FREQ_OUT_OF_RANGE` | Frequency outside IEC range | Check measurement setup |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Wait and retry |
| `MODEL_NOT_FOUND` | AI models missing | Train models first |

---

## Best Practices

### 1. File Handling

```python
# Always use context managers
with tempfile.NamedTemporaryFile() as tmp:
    tmp.write(data)
    df = parser.parse_file(tmp.name)
```

### 2. Error Handling

```python
# Catch specific exceptions
try:
    prediction = ensemble.predict(df)
except FileNotFoundError:
    # Model not trained
    train_models()
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
```

### 3. Memory Management

```python
# Explicitly close matplotlib figures
fig, ax = plt.subplots()
try:
    # ... plotting code ...
finally:
    plt.close(fig)
```

### 4. Logging

```python
# Use request IDs for tracing
request_id = str(uuid.uuid4())[:8]
logger.info(f"[{request_id}] Processing file: {filename}")
```

---

## Performance Guidelines

### Benchmarks

| Operation | Expected Time | Limit |
|-----------|--------------|-------|
| File parsing | < 2s | 5s |
| AI inference | < 1s | 2s |
| Report generation | < 5s | 10s |

### Optimization Tips

1. **Use caching**: Models are cached automatically
2. **Batch processing**: Process multiple files in parallel
3. **Memory limits**: Keep files < 50 MB
4. **Subsample data**: DTW automatically subsamples > 500 points

---

## Support & Contact

- **Documentation**: See `README.md` and `ARCHITECTURE.md`
- **Issues**: Report bugs via GitHub Issues
- **Standard Reference**: IEC 60076-18:2012

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-10-13 | Initial release with full API |
| 0.9.0 | 2025-10-01 | Beta release |

---

**End of API Documentation**
