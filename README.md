<div align="center">

<!-- Logo -->
<img src="assets/logo.png" alt="AI FRA UNIFY Logo" width="300"/>

# AI FRA UNIFY

### Intelligent Transformer Diagnostics

**Smart India Hackathon 2025 | Problem Statement 25190**

<p align="center">
  <strong>Revolutionizing Power Grid Reliability with AI-Powered FRA Analysis</strong>
</p>

---

[![Watch the video](https://raw.githubusercontent.com/jineshgoud45/aifra/main/assets/thumbnail.png)](https://youtu.be/FzC_YIjqDT4)

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [AI Models](#-ai-models)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [API Documentation](#-api-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 🎯 **Core Capabilities**

- **🤖 AI Ensemble Classification**
  - 1D CNN for sequence analysis
  - ResNet18 for Bode plot image classification
  - One-Class SVM for anomaly detection
  - Weighted voting with uncertainty quantification

- **📊 Multi-Vendor Support**
  - Omicron FRANEO (CSV & XML)
  - Doble SFRA suites (TXT/DAT)
  - Megger FRAX analyzers (DAT)
  - Generic format auto-detection

- **✅ IEC 60076-18 Compliance**
  - Frequency range validation (20 Hz - 2 MHz)
  - Log-scale grid verification
  - Artifact detection (>3 dB threshold)
  - Channel symmetry checking

- **📈 Interactive Visualizations**
  - Plotly-powered Bode plots
  - Probability distributions
  - Frequency band heatmaps
  - Explainable AI visualizations

- **📄 Professional Reporting**
  - IEC-compliant PDF reports
  - Executive summaries
  - Natural language explanations
  - Technical recommendations

### 🛡️ **Production Features**

- ✅ Thread-safe model loading
- ✅ IP-based rate limiting (DoS protection)
- ✅ Comprehensive input validation
- ✅ 85%+ test coverage
- ✅ Type hints throughout
- ✅ Docker deployment ready
- ✅ CI/CD with GitHub Actions
- ✅ Request tracking & observability

---

## 🚀 Quick Start

### **Option 1: Docker (Recommended)**

```bash
# Build and run with Docker
docker build -t fra-diagnostics .
docker run -p 8501:8501 fra-diagnostics

# Or use docker-compose
docker-compose up
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### **Option 2: Local Installation**

```bash
# Clone repository
git clone https://github.com/jineshgoud45/fra-diagnostics.git
cd fra-diagnostics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### **Quick Test**

```python
from parser import UniversalFRAParser
from simulator import TransformerSimulator

# Generate synthetic data
sim = TransformerSimulator()
freq, mag, phase = sim.generate_axial_deformation()

# Parse and validate
parser = UniversalFRAParser()
df = parser.create_dataframe(freq, mag, phase)
qa_results = parser._perform_qa_checks(df, 'test.csv')

print(f"✅ Frequency range: {qa_results['checks']['frequency_range']['passed']}")
```

---

## 📦 Installation

### Prerequisites

- **Python**: 3.9 or higher
- **OS**: Linux, macOS, or Windows
- **Memory**: 4GB RAM minimum (8GB recommended for AI models)
- **Disk**: 2GB free space

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest --cov=. --cov-report=html

# Check code quality
black .
isort .
mypy . --ignore-missing-imports
flake8 .
```

### Environment Configuration

Create `.env` file:

```bash
# Optional configuration
FRA_LOG_LEVEL=INFO
FRA_MODEL_DIR=./models
FRA_TEMP_DIR=./temp
FRA_DEVICE=cpu  # or 'cuda' for GPU
FRA_ENABLE_RATE_LIMIT=true
```

---

## 📖 Usage

### Web Interface

1. **Upload FRA Data**: Drag-drop or browse CSV/XML/TXT files
2. **Auto-Detection**: System identifies vendor and format
3. **AI Analysis**: Ensemble predicts fault type with confidence
4. **Review Results**: Interactive visualizations and metrics
5. **Generate Report**: Download IEC-compliant PDF

### Python API

```python
from ai_ensemble import load_models
from parser import parse_fra_file

# Load AI models
ensemble = load_models(save_dir='models', device='cpu')

# Parse data
df = parse_fra_file('path/to/measurement.csv', vendor='omicron')

# Predict fault
result = ensemble.predict(df)

print(f"Predicted Fault: {result['predicted_fault']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

### CLI Usage

```bash
# Parse file
python predict_cli.py --input data.csv --vendor omicron

# Generate synthetic dataset
python simulator.py --samples 1000 --output synthetic_data/

# Export to ONNX
python onnx_export.py --model-dir models --output models/ensemble.onnx
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web Interface                  │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
         ┌──────▼──────┐            ┌──────▼──────┐
         │   Parser    │            │  Simulator  │
         │  (Multi-    │            │  (Physics   │
         │   Vendor)   │            │   Model)    │
         └──────┬──────┘            └─────────────┘
                │
         ┌──────▼──────┐
         │ IEC 60076-18│
         │ QA Checks   │
         └──────┬──────┘
                │
         ┌──────▼──────────────────────────────┐
         │        AI Ensemble                   │
         ├──────────────┬──────────────────────┤
         │  1D CNN      │  ResNet18   │  SVM   │
         │ (Sequence)   │  (Image)    │(Anomaly)│
         └──────┬───────┴──────┬──────┴────┬───┘
                │              │           │
                └──────────────┼───────────┘
                               │
                        ┌──────▼──────┐
                        │   Report    │
                        │  Generator  │
                        └─────────────┘
```

---

## 🤖 AI Models

### Ensemble Architecture

| Model | Purpose | Accuracy | Weight |
|-------|---------|----------|--------|
| **1D CNN** | Raw sequence classification | 89% | 0.4 |
| **ResNet18** | Bode plot image analysis | 91% | 0.4 |
| **One-Class SVM** | Anomaly detection | 85% | 0.2 |

### Fault Classes

1. **Normal** - Healthy transformer
2. **Axial Deformation** - Winding displacement
3. **Radial Deformation** - Winding compression
4. **Inter-turn Short** - Turn-to-turn faults
5. **Core Grounding** - Core insulation issues
6. **Tap-changer Fault** - Contact problems

### Training

```bash
# Train ensemble from scratch
python -m ai_ensemble \
    --train-data data/train.csv \
    --test-data data/test.csv \
    --epochs 50 \
    --batch-size 32 \
    --device cuda

# Monitor with TensorBoard
tensorboard --logdir runs/
```

---

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest -v

# With coverage
pytest --cov=. --cov-report=html --cov-report=term

# Specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m security      # Security tests
pytest -m property      # Property-based tests

# Performance benchmarks
pytest tests/test_integration.py::test_parse_performance_benchmark
```

### Test Coverage

Current coverage: **85%+**

```bash
# Generate coverage report
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

---

## 🚢 Deployment

### Docker Production Build

```bash
# Build optimized image
DOCKER_BUILDKIT=1 docker build -t fra-diagnostics:latest .

# Run with custom config
docker run -d \
  -p 8501:8501 \
  -e FRA_LOG_LEVEL=INFO \
  -e FRA_DEVICE=cpu \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --name fra-app \
  fra-diagnostics:latest
```

### Raspberry Pi / Edge Devices

```bash
# Use deployment script
chmod +x pi_deploy.sh
./pi_deploy.sh

# Or manual deployment
python onnx_export.py
python predict_cli.py --use-onnx --input test.csv
```

### Cloud Deployment

- **AWS**: Use ECS or Lambda with container image
- **Azure**: Deploy to App Service or Container Instances
- **GCP**: Use Cloud Run or GKE
- **Heroku**: Deploy with `heroku.yml`

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

---

## 📚 API Documentation

Comprehensive API documentation available:

- **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** - Complete API reference
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design details
- **[CHANGELOG.md](CHANGELOG.md)** - Version history

### Key Modules

- `parser.py` - Multi-vendor FRA data parsing
- `simulator.py` - Physics-based fault simulation
- `ai_ensemble.py` - ML model ensemble
- `report_generator.py` - PDF report generation
- `app.py` - Streamlit web interface

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Steps

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### Code Quality Standards

- ✅ Type hints required
- ✅ Docstrings with examples
- ✅ 85%+ test coverage
- ✅ Pass all CI/CD checks
- ✅ Follow PEP 8 style guide

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- PyTorch: BSD-style license
- Streamlit: Apache License 2.0
- ReportLab: BSD-style license

---

## 🙏 Acknowledgments

- **IEC 60076-18** standard for FRA guidelines
- **Smart India Hackathon 2025** for problem statement
- PyTorch and Streamlit communities
- All contributors and testers

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/jineshgoud45/fra-diagnostics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jineshgoud45/fra-diagnostics/discussions)
- **Email**: 23eg109a16@anurag.edu.in OR 24eg509a01@anurag.edu.in

---

## 🌟 Star History

If this project helped you, please ⭐ star it on GitHub!

---

<p align="center">
  <b>Built with ❤️ for Smart India Hackathon 2025</b>
  <br>
  <sub>Production-ready • IEC-compliant • Open Source</sub>
</p>
