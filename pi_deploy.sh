#!/bin/bash
################################################################################
# Raspberry Pi Deployment Script for FRA Diagnostics
# SIH 2025 PS 25190
#
# This script sets up a Raspberry Pi (3B+, 4, or 5) for edge-based FRA fault
# prediction using ONNX models. No GPU required - optimized for ARM CPU.
#
# Usage:
#   chmod +x pi_deploy.sh
#   ./pi_deploy.sh
#
# Requirements:
#   - Raspberry Pi OS (32-bit or 64-bit)
#   - Python 3.7 or higher
#   - Internet connection for initial setup
#   - ~2GB free space for dependencies
################################################################################

set -e  # Exit on error

echo "========================================================================"
echo "FRA Diagnostics - Raspberry Pi Edge Deployment Setup"
echo "SIH 2025 PS 25190"
echo "========================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running on ARM architecture
ARCH=$(uname -m)
if [[ ! "$ARCH" =~ ^(arm|aarch64) ]]; then
    echo -e "${YELLOW}Warning: This doesn't appear to be ARM architecture (detected: $ARCH)${NC}"
    echo "This script is optimized for Raspberry Pi. Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled."
        exit 0
    fi
fi

# Check if running on Raspberry Pi
if [ -f /proc/device-tree/model ]; then
    PI_MODEL=$(cat /proc/device-tree/model)
    echo "Detected: $PI_MODEL"
else
    echo -e "${YELLOW}Warning: Could not detect Raspberry Pi model${NC}"
fi

# Check available memory
TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
echo "Available RAM: ${TOTAL_MEM} MB"

if [ "$TOTAL_MEM" -lt 1024 ]; then
    echo -e "${RED}Warning: Less than 1GB RAM detected. Performance may be limited.${NC}"
    echo "Configuring swap space..."
    
    # Check current swap
    CURRENT_SWAP=$(free -m | awk '/^Swap:/{print $2}')
    
    if [ "$CURRENT_SWAP" -lt 2048 ]; then
        echo "Current swap: ${CURRENT_SWAP} MB (recommended: 2048 MB)"
        echo "Would you like to increase swap to 2GB? (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo "Backing up swap configuration..."
            sudo cp /etc/dphys-swapfile /etc/dphys-swapfile.backup.$(date +%Y%m%d_%H%M%S)
            
            sudo dphys-swapfile swapoff || {
                echo -e "${RED}Error: Failed to turn off swap${NC}"
                echo "Restoring backup..."
                sudo cp /etc/dphys-swapfile.backup.* /etc/dphys-swapfile 2>/dev/null || true
                exit 1
            }
            
            # SECURITY FIX: Create backup with -i.bak before modifying
            sudo sed -i.bak "s/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/" /etc/dphys-swapfile || {
                echo -e "${RED}Error: Failed to modify swap configuration${NC}"
                echo "Restoring backup..."
                sudo cp /etc/dphys-swapfile.backup.* /etc/dphys-swapfile 2>/dev/null || true
                exit 1
            }
            
            sudo dphys-swapfile setup || {
                echo -e "${RED}Error: Failed to setup new swap${NC}"
                echo "Restoring backup..."
                sudo cp /etc/dphys-swapfile.backup.* /etc/dphys-swapfile 2>/dev/null || true
                sudo dphys-swapfile setup
                sudo dphys-swapfile swapon
                exit 1
            }
            
            sudo dphys-swapfile swapon || {
                echo -e "${RED}Error: Failed to turn on swap${NC}"
                exit 1
            }
            
            echo -e "${GREEN}✓ Swap increased to 2GB (backup saved)${NC}"
        fi
    fi
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $PYTHON_VERSION"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 7) else 1)"; then
    echo -e "${RED}Error: Python 3.7 or higher required${NC}"
    exit 1
fi

# Check available disk space
AVAILABLE_SPACE=$(df -BM . | awk 'NR==2 {print $4}' | sed 's/M//')
echo "Available disk space: ${AVAILABLE_SPACE} MB"

if [ "$AVAILABLE_SPACE" -lt 2048 ]; then
    echo -e "${RED}Error: Less than 2GB free space available${NC}"
    echo "Please free up disk space and try again."
    exit 1
fi

# Update system
echo ""
echo "Step 1: Updating system packages..."
echo "------------------------------------------------------------------------"
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo ""
echo "Step 2: Installing system dependencies..."
echo "------------------------------------------------------------------------"
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libopenblas-dev \
    libatlas-base-dev \
    libjpeg-dev \
    zlib1g-dev

echo -e "${GREEN}✓ System dependencies installed${NC}"

# Create virtual environment
echo ""
echo "Step 3: Creating Python virtual environment..."
echo "------------------------------------------------------------------------"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
echo ""
echo "Step 4: Installing Python packages (this may take 10-15 minutes)..."
echo "------------------------------------------------------------------------"

# Install numpy first (required by many packages)
echo "Installing numpy..."
pip install numpy>=1.18.0

# Install pandas
echo "Installing pandas..."
pip install pandas>=1.0.0

# Install ONNX Runtime (CPU version for ARM)
echo "Installing ONNX Runtime (CPU)..."
pip install onnxruntime>=1.12.0

# Install scikit-learn
echo "Installing scikit-learn..."
pip install scikit-learn>=1.0.0

echo -e "${GREEN}✓ Core dependencies installed${NC}"

# Optional: Install additional packages if needed
echo ""
echo "Step 5: Installing optional packages..."
echo "------------------------------------------------------------------------"
pip install scipy>=1.5.0 || echo "scipy installation failed (optional)"

echo -e "${GREEN}✓ Setup complete!${NC}"

# Create models directory
echo ""
echo "Step 6: Setting up directories..."
echo "------------------------------------------------------------------------"
mkdir -p models
mkdir -p data
mkdir -p results

echo -e "${GREEN}✓ Directories created${NC}"

# Test installation
echo ""
echo "Step 7: Testing installation..."
echo "------------------------------------------------------------------------"

python3 << EOF
import sys
print(f"Python: {sys.version}")

try:
    import numpy as np
    print(f"✓ numpy {np.__version__}")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print(f"✓ pandas {pd.__version__}")
except ImportError as e:
    print(f"✗ pandas import failed: {e}")
    sys.exit(1)

try:
    import onnxruntime as ort
    print(f"✓ onnxruntime {ort.__version__}")
    
    # Test inference
    import numpy as np
    session = ort.InferenceSession(None, providers=['CPUExecutionProvider'])
    print(f"✓ ONNX Runtime providers: {ort.get_available_providers()}")
except ImportError as e:
    print(f"✗ onnxruntime import failed: {e}")
    sys.exit(1)

try:
    import sklearn
    print(f"✓ scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"✗ scikit-learn import failed: {e}")
    sys.exit(1)

print("\n✓ All required packages installed successfully!")
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Installation test passed!${NC}"
else
    echo -e "${RED}✗ Installation test failed${NC}"
    exit 1
fi

# Print deployment instructions
echo ""
echo "========================================================================"
echo "DEPLOYMENT COMPLETE!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Copy trained models to the 'models/' directory:"
echo "   - cnn_model.onnx"
echo "   - svm_model.pkl"
echo "   - feature_extractor.pkl"
echo "   - fault_mapping.pkl"
echo ""
echo "2. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Run prediction on a data file:"
echo "   python3 predict_cli.py --file data/test.csv"
echo ""
echo "4. For help:"
echo "   python3 predict_cli.py --help"
echo ""
echo "Performance tips:"
echo "  - Expected inference time: 20-50ms on Pi 4"
echo "  - Use --quiet flag for faster output"
echo "  - Models are optimized for CPU"
echo ""
echo "========================================================================"

# Create a simple test script
cat > test_inference.sh << 'TESTEOF'
#!/bin/bash
# Quick inference test script
echo "Testing FRA prediction on Raspberry Pi..."
python3 predict_cli.py --file data/test.csv --quiet
echo "Test complete!"
TESTEOF

chmod +x test_inference.sh

echo ""
echo -e "${GREEN}✓ Raspberry Pi is ready for edge deployment!${NC}"
echo ""
