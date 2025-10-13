# System Architecture
## FRA Diagnostics Platform - SIH 2025 PS 25190

This document describes the system architecture, components, and data flow of the Transformer FRA Diagnostics Platform.

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│                    (Streamlit Dashboard)                         │
└───────────────┬─────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ File Upload  │  │ Data Parser  │  │  QA Validator        │  │
│  │  & Validation│  │ (Multi-vendor)│  │ (IEC 60076-18)       │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────────────┘  │
│         │                  │                  │                   │
│         └──────────────────┴──────────────────┘                  │
│                            │                                      │
└────────────────────────────┼──────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AI Ensemble Layer                           │
│  ┌──────────┐    ┌──────────┐    ┌─────────────────────────┐   │
│  │  1D CNN  │───▶│ ResNet18 │───▶│    Voting Ensemble      │   │
│  │(Sequence)│    │ (Images) │    │(Weighted: 0.4/0.4/0.2)  │   │
│  └──────────┘    └──────────┘    └────────┬────────────────┘   │
│                                             │                     │
│  ┌──────────────────────────────────────┐  │                     │
│  │      One-Class SVM                   │──┘                     │
│  │    (Anomaly Detection)               │                        │
│  └──────────────────────────────────────┘                        │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Report Generation Layer                       │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │  PDF Generator   │    │    Visualization Engine          │   │
│  │ (IEC Compliant)  │    │   (Bode Plots, Charts)          │   │
│  └──────────────────┘    └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Component Breakdown

### 1. **Parser Module** (`parser.py`)
- **Purpose**: Universal FRA data parser for multiple vendors
- **Vendors Supported**:
  - Omicron FRANEO (CSV/XML)
  - Doble SFRA Suite (TXT)
  - Megger FRAX (DAT)
  - Generic formats
- **Key Features**:
  - Auto-format detection
  - IEC 60076-18 compliance checking
  - Data normalization to canonical schema
  - Artifact detection (>3dB deviations)

**Data Flow:**
```
Raw File → Format Detection → Vendor-Specific Parser → 
→ Normalization → QA Checks → Canonical DataFrame
```

### 2. **Simulator Module** (`simulator.py`)
- **Purpose**: Generate synthetic FRA data for model training
- **Model**: Lumped-parameter ladder network (R-L-C sections)
- **Fault Types**:
  1. Normal (healthy transformer)
  2. Axial deformation (+15-20% inductance)
  3. Radial deformation (+15-20% capacitance)
  4. Inter-turn short (reduced R/L)
  5. Core grounding (low-freq capacitance increase)
  6. Tap-changer fault (impedance discontinuity)
- **Physics-Based**:
  ```
  H(jω) = ∏[Z_C(i) / (Z_L(i) + Z_R(i) + Z_C(i))]
  ```

### 3. **AI Ensemble** (`ai_ensemble.py`)
**Three-Model Architecture:**

#### a) **1D CNN**
- Input: 3-channel sequence (freq, magnitude, phase)
- Architecture: Conv1D → BatchNorm → MaxPool → GAP → FC
- Purpose: Raw sequence pattern recognition

#### b) **ResNet18** (Transfer Learning)
- Input: Grayscale Bode plot images (224x224)
- Pre-trained: ImageNet weights
- Purpose: Visual pattern recognition in frequency response

#### c) **One-Class SVM**
- Features: Multi-band energy, resonance peaks, DTW distance
- Purpose: Baseline-free anomaly detection
- Trained on: Normal samples only

**Ensemble Strategy:**
```python
P_ensemble = 0.4 * P_CNN + 0.4 * P_ResNet + 0.2 * P_SVM
```

**Uncertainty Quantification:**
- Shannon entropy over probability distributions
- Confidence = max(P_ensemble)

### 4. **Web Application** (`app.py`)
- **Framework**: Streamlit
- **Features**:
  - File upload with validation
  - Real-time analysis
  - Interactive Bode plots (Plotly)
  - Fault probability visualization
  - IEC compliance dashboard
  - PDF report download

### 5. **Report Generator** (`report_generator.py`)
- **Standard**: IEC 60076-18 compliant
- **Library**: ReportLab
- **Sections**:
  1. Cover page with test metadata
  2. Executive summary
  3. Detailed analysis with model breakdown
  4. Annotated Bode plots
  5. Natural language explanations
  6. Recommendations
  7. IEC compliance status

### 6. **Configuration** (`config.py`)
- Centralized configuration management
- Environment variable support
- Dataclass-based settings:
  - `IEC`: IEC 60076-18 standards
  - `SIM`: Simulator parameters
  - `PARSER`: Parser settings
  - `APP`: Web app configuration
  - `ML`: Machine learning hyperparameters
  - `SECURITY`: Security settings
  - `LOGGING`: Logging configuration

---

## 🔄 Data Flow Diagram

### Complete Analysis Pipeline

```
┌──────────────┐
│  User Uploads│
│  FRA File    │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│  File Validation                      │
│  - Size check (< 50MB)               │
│  - Extension check (.csv/.xml/.txt)  │
│  - Magic byte validation             │
│  - XXE attack prevention             │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  UniversalFRAParser                   │
│  1. Detect format & vendor           │
│  2. Parse raw data                   │
│  3. Normalize to canonical schema    │
│  4. Run QA checks                    │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  IEC 60076-18 QA Checks              │
│  ✓ Frequency range (20Hz - 2MHz)    │
│  ✓ Log-scale spacing                 │
│  ✓ Artifact detection (>3dB)        │
│  ✓ Data point count (>10)           │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Feature Preparation                  │
│  - Sequence (freq, mag, phase)       │
│  - Bode plot image generation        │
│  - Engineered features (SVM)         │
└──────┬───────────────────────────────┘
       │
       ├─────────────┬──────────────┬──────────────┐
       │             │              │               │
       ▼             ▼              ▼               ▼
  ┌────────┐   ┌──────────┐  ┌─────────┐   ┌──────────┐
  │ 1D CNN │   │ ResNet18 │  │ One-Class│   │ Baseline │
  │Sequence│   │  Image   │  │   SVM   │   │ Compare  │
  └───┬────┘   └────┬─────┘  └────┬────┘   └────┬─────┘
      │             │             │              │
      └─────────────┴──────┬──────┴──────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Ensemble Voting       │
              │ (Weighted Combination)  │
              └────────┬────────────────┘
                       │
                       ▼
              ┌─────────────────────────┐
              │   Post-Processing       │
              │ - Confidence calc       │
              │ - Uncertainty (entropy) │
              │ - Severity assessment   │
              └────────┬────────────────┘
                       │
                       ▼
              ┌─────────────────────────┐
              │   Report Generation     │
              │ - PDF with plots        │
              │ - IEC compliant format  │
              │ - Recommendations       │
              └─────────────────────────┘
```

---

## 🗄️ Data Schema

### Canonical FRA Data Format

```python
{
    'frequency_hz': float,      # 20 Hz to 2 MHz
    'magnitude_db': float,      # dB scale
    'phase_deg': float,         # -180 to 180 degrees
    'test_id': str,             # Unique test identifier
    'vendor': str,              # omicron/doble/megger/generic
    'timestamp': datetime       # Test timestamp
}
```

### Prediction Result Format

```python
{
    'predicted_fault': str,             # Fault class name
    'confidence': float,                # 0.0 to 1.0
    'probabilities': Dict[str, float],  # All class probabilities
    'cnn_probs': Dict[str, float],      # CNN predictions
    'resnet_probs': Dict[str, float],   # ResNet predictions
    'svm_score': float,                 # Anomaly score
    'uncertainty': float,               # Shannon entropy
    'features': Dict[str, float]        # Engineered features
}
```

### QA Results Format

```python
{
    'filepath': str,
    'test_id': str,
    'vendor': str,
    'checks': {
        'frequency_range': {
            'passed': bool,
            'min_freq': float,
            'max_freq': float,
            'message': str
        },
        'frequency_grid': {
            'passed': bool,
            'log_spacing_std': float,
            'num_points': int
        },
        'artifacts': {
            'passed': bool,
            'num_artifacts': int,
            'threshold_db': float
        }
    }
}
```

---

## 🔐 Security Architecture

### Defense in Depth

1. **Input Layer**:
   - File size limits (50MB)
   - Extension whitelist
   - Magic byte validation
   - XXE attack prevention (defusedxml)

2. **Processing Layer**:
   - Non-root user (Docker)
   - Resource limits (CPU/memory)
   - Timeout protection
   - Sandboxed execution

3. **Output Layer**:
   - Path traversal prevention
   - Sanitized filenames
   - Secure temp file handling

### Authentication (Optional)
```python
# Enable in production
FRA_REQUIRE_AUTH=true
```

Uses Streamlit's authentication framework.

---

## 📈 Scalability Considerations

### Current Limitations
- Single-instance deployment
- In-memory caching only
- No distributed processing

### Future Enhancements
1. **Horizontal Scaling**:
   - Load balancer + multiple instances
   - Shared model storage (S3/NFS)
   - Distributed caching (Redis)

2. **Database Integration**:
   - PostgreSQL for result persistence
   - Analysis history tracking
   - User management

3. **Batch Processing**:
   - Celery task queue
   - Asynchronous analysis
   - Background jobs

4. **API Layer**:
   - REST API (FastAPI)
   - GraphQL endpoint
   - WebSocket for real-time updates

---

## 🧪 Testing Strategy

### Unit Tests
- `test_parser.py`: Parser functionality
- `test_simulator.py`: Simulator physics
- `test_ai_ensemble.py`: Model components

### Integration Tests (Planned)
- End-to-end analysis pipeline
- Multi-vendor file handling
- Report generation

### Performance Tests (Planned)
- Load testing (100+ concurrent users)
- Large file handling (>10MB)
- Model inference latency

---

## 📊 Performance Metrics

### Typical Performance
- **File parsing**: <1s for 1000-point dataset
- **AI inference**: 2-5s on CPU
- **Report generation**: 3-7s including plots
- **Total pipeline**: <15s end-to-end

### Resource Usage
- **Memory**: 2-4GB typical, 8GB peak
- **CPU**: 1-2 cores active during inference
- **Disk**: 500MB for models, variable for temp files

---

## 🔮 Future Roadmap

### Phase 2 (Q2 2025)
- [ ] REST API layer
- [ ] Database integration
- [ ] Multi-language support
- [ ] Advanced explainability (Grad-CAM)

### Phase 3 (Q3 2025)
- [ ] Real-time streaming analysis
- [ ] Mobile app (React Native)
- [ ] Cloud deployment (AWS/Azure)
- [ ] Multi-tenant support

### Phase 4 (Q4 2025)
- [ ] Federated learning
- [ ] Edge deployment optimization
- [ ] Integration with SCADA systems
- [ ] Predictive maintenance features

---

## 📚 References

- **IEC 60076-18:2012**: Power transformers - Measurement of frequency response
- **IEEE C57.149-2012**: Guide for transformer FRA interpretation
- **CIGRE WG A2.26**: Mechanical condition assessment of transformer windings using FRA

---

## 👥 Component Ownership

| Component | Primary | Secondary |
|-----------|---------|-----------|
| Parser | Backend Team | Data Team |
| Simulator | ML Team | Domain Experts |
| AI Ensemble | ML Team | Research Team |
| Web App | Frontend Team | DevOps |
| Reports | Document Team | Backend Team |
| Infrastructure | DevOps | Backend Team |

---

## 🆘 Contact & Support

For architecture questions or system design discussions:
- Technical Lead: [Email]
- Architecture Review: Weekly Thursdays
- Documentation: This file + inline code comments
