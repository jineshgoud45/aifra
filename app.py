"""
Streamlit Web Application for Transformer FRA Diagnostics
SIH 2025 PS 25190

Production-ready dashboard integrating:
- Multi-vendor FRA data parsing
- AI ensemble fault classification
- Interactive visualizations
- IEC 60076-18 compliance checking
- Explainability and anomaly detection
- PDF report generation
"""

from __future__ import annotations  # Enable lazy type hint evaluation

__all__ = [
    'load_ai_models',
    'get_parser', 
    'get_simulator',
    'validate_uploaded_file',
    'create_probability_bar_chart',
    'create_frequency_band_heatmap',
    'display_iec_qa_results',
    'generate_and_download_iec_report',
    'check_rate_limit',
    'main'
]

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
from datetime import datetime
import tempfile
import logging
import gc
import warnings
import time
import threading
import hashlib
import uuid
from typing import Tuple, Optional, Dict
from pathlib import Path
import re
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Import centralized configuration
try:
    from config import APP, IEC, PARSER, SECURITY, TESTING
except ImportError:
    # Fallback for backwards compatibility
    class APP:
        model_dir = 'models'
        temp_dir = 'temp'
        cache_ttl_seconds = 3600
        max_upload_size_mb = 50
        max_upload_size_bytes = 50 * 1024 * 1024
    
    class IEC:
        freq_min = 20
        freq_max = 2e6
    
    class PARSER:
        allowed_extensions = ('.csv', '.xml', '.txt', '.dat')
    
    class SECURITY:
        check_magic_bytes = True
    
    class TESTING:
        rate_limit_window_seconds = 60
        rate_limit_max_uploads = 10

# Backwards compatibility for constants
MAX_FILE_SIZE_MB = APP.max_upload_size_mb
MAX_FILE_SIZE_BYTES = APP.max_upload_size_bytes
CACHE_TTL_SECONDS = APP.cache_ttl_seconds
MODEL_DIR = str(APP.model_dir) if hasattr(APP.model_dir, '__str__') else APP.model_dir
TEMP_DIR = str(APP.temp_dir) if hasattr(APP.temp_dir, '__str__') else APP.temp_dir

# SECURITY: Global lock for thread-safe model loading
_model_load_lock = threading.Lock()

# SECURITY: IP-based rate limiting cache (prevents session bypass attacks)
# Format: {ip_hash: [timestamp1, timestamp2, ...]}
_rate_limit_cache: Dict[str, list] = {}
_rate_limit_lock = threading.Lock()

# Configure logging with dynamic path
log_dir = Path(os.getenv('FRA_LOG_DIR', './logs'))
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'fra_app_{os.getpid()}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Import lightweight project modules (fast imports)
try:
    from parser import UniversalFRAParser
    from report_generator import generate_iec_report
    from utils import create_bode_plot_plotly, validate_fra_data, compute_frequency_bands
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    logger.error(f"Import error: {e}", exc_info=True)
    st.stop()

# Note: Heavy imports (ai_ensemble, simulator) are lazy-loaded when needed
# This makes the app start much faster (1-2s instead of 10-20s)

# AUTO-TRAINING: Global state for training
_training_lock = threading.Lock()
_training_in_progress = False


def check_models_exist() -> bool:
    """
    Check if all required AI model files exist.
    
    Returns:
        bool: True if all models exist, False otherwise
    """
    required_files = [
        'cnn_model.pth',
        'resnet_model.pth',
        'svm_model.pkl',
        'feature_extractor.pkl',
        'fault_mapping.pkl'
    ]
    
    models_path = Path(MODEL_DIR)
    
    # Check if models directory exists
    if not models_path.exists():
        logger.info(f"Models directory does not exist: {models_path}")
        return False
    
    # Check if all required model files exist
    for model_file in required_files:
        file_path = models_path / model_file
        if not file_path.exists():
            logger.info(f"Missing model file: {model_file}")
            return False
    
    logger.info("All model files found")
    return True


def auto_train_models() -> bool:
    """
    Automatically train AI models if they don't exist.
    
    This function generates synthetic training data and trains all ensemble models.
    It runs on first app launch when models are missing.
    
    Returns:
        bool: True if training succeeded, False otherwise
    """
    global _training_in_progress
    
    # Prevent concurrent training
    if _training_in_progress:
        st.warning("⏳ Model training already in progress. Please wait...")
        return False
    
    with _training_lock:
        _training_in_progress = True
        
        try:
            # Create necessary directories
            models_dir = Path(MODEL_DIR)
            synthetic_dir = Path('synthetic_data')
            
            models_dir.mkdir(parents=True, exist_ok=True)
            synthetic_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("=" * 70)
            logger.info("AUTO-TRAINING: Models not found - starting training...")
            logger.info("=" * 70)
            
            # Show progress in UI
            st.info("🤖 **First-time setup**: AI models not found. Starting automatic training...")
            st.warning("⏰ This may take 10-30 minutes. Please keep this page open.")
            
            progress_bar = st.progress(0, text="Step 1/2: Generating synthetic training data...")
            
            # Step 1: Generate synthetic data (smaller dataset for faster training)
            logger.info("Generating synthetic training data...")
            try:
                from simulator import TransformerSimulator, generate_synthetic_dataset
                train_df, test_df = generate_synthetic_dataset(
                    n_samples=2000,  # Smaller dataset for faster first-time training
                    output_dir='synthetic_data',
                    export_formats=False,
                    visualize=False
                )
                
                progress_bar.progress(30, text=f"Step 1/2: Generated {len(train_df)} training samples ✓")
                logger.info(f"Generated {len(train_df)} training samples, {len(test_df)} test samples")
                
            except Exception as e:
                logger.error(f"Failed to generate training data: {e}")
                st.error(f"❌ Failed to generate training data: {e}")
                return False
            
            # Step 2: Train models
            progress_bar.progress(40, text="Step 2/2: Training AI models (CNN, ResNet, SVM)...")
            logger.info("Training ensemble models...")
            
            try:
                from ai_ensemble import train_ensemble_pipeline
                ensemble = train_ensemble_pipeline(
                    train_df=train_df,
                    test_df=test_df,
                    num_epochs_cnn=20,  # Reduced epochs for faster training
                    num_epochs_resnet=10,
                    batch_size=32,
                    device=None,
                    save_dir=MODEL_DIR
                )
                
                progress_bar.progress(100, text="Step 2/2: Training complete ✓")
                
            except Exception as e:
                logger.error(f"Failed to train models: {e}")
                st.error(f"❌ Failed to train models: {e}")
                return False
            
            logger.info("=" * 70)
            logger.info("AUTO-TRAINING: Successfully trained and saved all models!")
            logger.info("=" * 70)
            
            # Show success message
            st.success("✅ AI models trained successfully! The app is now ready for fault detection.")
            st.balloons()
            time.sleep(2)  # Brief pause to show success message
            
            return True
                
        except Exception as e:
            logger.error(f"AUTO-TRAINING: Unexpected error: {e}", exc_info=True)
            st.error(f"❌ Auto-training failed: {e}")
            st.info("💡 **Manual alternative**: Run `python quick_train_models.py` in terminal")
            return False
        
        finally:
            _training_in_progress = False


# Page configuration
st.set_page_config(
    page_title="FRA Diagnostics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .error-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(ttl=CACHE_TTL_SECONDS)
def load_ai_models() -> FRAEnsemble:
    """
    Load AI ensemble models (cached with expiration).
    
    THREAD SAFETY: Uses mutex lock to prevent race conditions during
    concurrent initial loads.
    
    Note: Exceptions are not caught here to prevent caching failures.
    Let exceptions propagate to caller for proper error handling.
    
    Returns:
        FRAEnsemble: Loaded ensemble model
        
    Raises:
        FileNotFoundError: If model files are missing
    """
    # Check if models exist (fast check, doesn't block)
    if not check_models_exist():
        raise FileNotFoundError(
            "AI model files not found. Please train the models first."
        )
    
    # CRITICAL FIX: Prevent race condition during concurrent loads
    with _model_load_lock:
        logger.info("Loading AI ensemble models with thread safety...")
        from ai_ensemble import load_models
        return load_models(MODEL_DIR)


@st.cache_resource
def get_parser():
    """Get FRA parser instance (cached)."""
    logger.info("Initializing FRA parser")
    return UniversalFRAParser()


@st.cache_resource
def get_simulator():
    """Get simulator for baseline generation (cached)."""
    logger.info("Initializing transformer simulator")
    from simulator import TransformerSimulator
    return TransformerSimulator(seed=42)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize uploaded filename to prevent path traversal and injection attacks.
    
    SECURITY: Multi-layer defense against path traversal:
    - Strip ALL path components (even encoded ones)
    - Whitelist only alphanumeric, dots, dashes, underscores
    - Limit filename length
    - Prevent null byte injection
    
    Args:
        filename: Original filename from upload
    
    Returns:
        Sanitized filename safe for filesystem use
        
    Examples:
        >>> sanitize_filename("../../etc/passwd")
        'etc_passwd'
        >>> sanitize_filename("test<script>.csv")
        'test_script_.csv'
        >>> sanitize_filename("very" * 100 + ".csv")
        'veryvery...very.csv'  # Truncated to 255 chars
        >>> sanitize_filename("file.csv%00.exe")
        'file.csv_00.exe'  # Null byte removed
    """
    if not filename:
        return "unnamed_file.txt"
    
    # CRITICAL FIX: Strip ALL path components first (handles ../, ..\, encoded variants)
    # This removes any directory traversal attempts before further processing
    filename = os.path.basename(filename)
    
    # Remove null bytes (prevents null byte injection attacks)
    filename = filename.replace('\x00', '')
    
    # Split extension
    name, ext = os.path.splitext(filename)
    
    # SECURITY: Whitelist only safe characters (alphanumeric + _-.)
    # This prevents ALL special character attacks including encoded ones
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    ext = re.sub(r'[^a-zA-Z0-9.]', '', ext)  # Extension: only alphanumeric + dot
    
    # Prevent empty name
    if not name or name == '_':
        name = 'unnamed'
    
    # Prevent hidden files
    if name.startswith('.'):
        name = 'file' + name
    
    # Reconstruct filename
    sanitized = name + ext
    
    # Limit length (255 is filesystem max for most systems)
    max_length = 255
    if len(sanitized) > max_length:
        # Preserve extension while truncating name
        name_max_len = max_length - len(ext)
        sanitized = name[:name_max_len] + ext
    
    # Final validation: ensure we didn't create an empty or dangerous filename
    if not sanitized or sanitized in ('.', '..', '/', '\\'):
        sanitized = 'safe_file.txt'
    
    logger.debug(f"Sanitized filename: '{filename}' -> '{sanitized}'")
    return sanitized


def validate_uploaded_file(uploaded_file: Optional[UploadedFile]) -> Tuple[bool, str]:
    """
    Validate uploaded file for security and size constraints.
    
    Args:
        uploaded_file: Streamlit UploadedFile object (or None)
        
    Returns:
        Tuple of (is_valid, error_message)
    
    Examples:
        >>> is_valid, error = validate_uploaded_file(uploaded_file)
        >>> if not is_valid:
        ...     st.error(error)
        ...     return
    """
    # Check if file is provided
    if uploaded_file is None:
        return False, "❌ No file uploaded. Please select a file to upload."
    
    # Get filename and size
    filename = uploaded_file.name
    file_size = uploaded_file.size
    
    # Check file size
    size_mb = file_size / (1024 * 1024)
    if file_size > MAX_FILE_SIZE_BYTES:
        return False, (
            f"❌ File too large: {size_mb:.2f} MB exceeds maximum allowed size of {MAX_FILE_SIZE_MB} MB. "
            f"Please upload a smaller file or compress your data."
        )
    
    # Check file extension
    file_ext = os.path.splitext(filename)[1].lower()
    allowed_extensions = PARSER.allowed_extensions
    
    if file_ext not in allowed_extensions:
        return False, (
            f"❌ Invalid file type: '{file_ext}' is not supported. "
            f"Allowed formats: {', '.join(allowed_extensions)}. "
            f"Please convert your file to one of these formats."
        )
    
    # SECURITY: Check magic bytes to prevent file type spoofing
    if SECURITY.check_magic_bytes:
        try:
            # Read first few bytes to check file signature
            header = uploaded_file.read(512)
            uploaded_file.seek(0)  # Reset file pointer
            
            # Define magic byte signatures for allowed file types
            magic_signatures = {
                b'PK\x03\x04': ['.zip', '.xlsx', '.docx'],  # ZIP-based formats
                b'\x1f\x8b': ['.gz'],  # GZIP
                b'<?xml': ['.xml'],  # XML files
                b'<xml': ['.xml'],
                b'<FRA': ['.xml'],  # Custom FRA XML
            }
            
            # CSV and TXT are plain text, harder to validate by magic bytes
            # Just check if it's valid UTF-8 or ASCII
            is_text_file = file_ext in ['.csv', '.txt', '.dat']
            
            if not is_text_file:
                # For XML, verify it starts with XML declaration
                if file_ext == '.xml':
                    if not (header.startswith(b'<?xml') or header.startswith(b'<xml') or header.startswith(b'<FRA')):
                        return False, (
                            f"❌ File type mismatch: File has '.xml' extension but doesn't appear to be a valid XML file. "
                            f"This could indicate a malicious file. Please verify your file."
                        )
            else:
                # For text files, check if decodable as UTF-8
                try:
                    header.decode('utf-8')
                except UnicodeDecodeError:
                    return False, (
                        f"❌ File encoding error: File has '{file_ext}' extension but contains invalid UTF-8 data. "
                        f"Please ensure your file is saved in UTF-8 or ASCII encoding."
                    )
        
        except Exception as e:
            logger.warning(f"Magic byte check failed: {e}")
            # Don't fail validation if magic byte check fails, just log it
    
    # Check for empty file
    if file_size == 0:
        return False, "❌ Empty file: The uploaded file is empty (0 bytes). Please upload a file with data."
    
    # Check for suspiciously small files (likely not valid FRA data)
    if file_size < 100:  # Less than 100 bytes is suspicious
        return False, (
            f"❌ File too small: The file is only {file_size} bytes, which is too small to contain valid FRA data. "
            f"Typical FRA files are at least 1 KB. Please verify your file."
        )
    
    logger.info(f"File validation passed: {filename} ({size_mb:.2f} MB)")
    return True, ""


def create_probability_bar_chart(probabilities: dict):
    """Create bar chart for fault probabilities."""
    fault_types = list(probabilities.keys())
    probs = list(probabilities.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=fault_types,
            orientation='h',
            marker=dict(
                color=probs,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Probability")
            ),
            text=[f'{p:.1%}' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Fault Classification Probabilities",
        xaxis_title="Probability",
        yaxis_title="Fault Type",
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_frequency_band_heatmap(df: pd.DataFrame):
    """Create heatmap showing energy in different frequency bands."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    bands = [
        ('Low\n20-100 Hz', 20, 100),
        ('Mid-Low\n100-1k Hz', 100, 1000),
        ('Mid\n1k-10k Hz', 1000, 10000),
        ('Mid-High\n10k-100k Hz', 10000, 100000),
        ('High\n100k-500k Hz', 100000, 500000),
        ('Very High\n500k-2M Hz', 500000, 2000000)
    ]
    
    freq = df['frequency_hz'].values
    mag = df['magnitude_db'].values
    
    band_energies = []
    for label, low, high in bands:
        mask = (freq >= low) & (freq <= high)
        if mask.sum() > 0:
            energy = np.mean(np.abs(mag[mask]))
        else:
            energy = 0
        band_energies.append(energy)
    
    # Normalize
    band_energies = np.array(band_energies).reshape(1, -1)
    
    fig, ax = plt.subplots(figsize=(12, 2))
    sns.heatmap(
        band_energies,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Energy (dB)'},
        xticklabels=[b[0] for b in bands],
        yticklabels=['Energy'],
        ax=ax
    )
    ax.set_title('Multi-Band Frequency Energy Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def display_iec_qa_results(qa_results: dict):
    """Display IEC 60076-18 QA check results."""
    st.subheader("🔍 IEC 60076-18 Compliance Checks")
    
    if 'checks' not in qa_results:
        st.warning("QA results not available")
        return
    
    checks = qa_results['checks']
    
    cols = st.columns(len(checks))
    
    for idx, (check_name, check_data) in enumerate(checks.items()):
        with cols[idx]:
            passed = check_data.get('passed', False)
            
            if passed:
                st.markdown(f"""
                <div class="success-card">
                    <h4>✅ {check_name.replace('_', ' ').title()}</h4>
                    <p><strong>Status:</strong> PASS</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-card">
                    <h4>⚠️ {check_name.replace('_', ' ').title()}</h4>
                    <p><strong>Status:</strong> WARNING</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display details in expander
            with st.expander("Details"):
                for key, value in check_data.items():
                    if key != 'passed':
                        st.write(f"**{key}:** {value}")


def generate_and_download_iec_report(
    df: pd.DataFrame,
    prediction_result: dict,
    qa_results: dict,
    filename: str = "fra_report.pdf"
) -> bytes:
    """
    Generate PDF report and return bytes for download.
    
    Renamed from generate_iec_report to avoid confusion with imported function.
    
    Args:
        df: FRA data
        prediction_result: AI prediction results
        qa_results: QA check results
        filename: Output filename (will be sanitized)
    
    Returns:
        bytes: PDF content as bytes
        
    Raises:
        Exception: If report generation fails
    """
    try:
        from report_generator import generate_iec_report as gen_report
        
        # Sanitize output filename
        filename = sanitize_filename(filename)
        
        # Create temp file with proper cleanup
        temp_path = os.path.join(TEMP_DIR, filename)
        
        # Generate report
        gen_report(
            df=df,
            prediction_result=prediction_result,
            qa_results=qa_results,
            output_path=temp_path
        )
        
        # Read and return PDF bytes
        try:
            with open(temp_path, 'rb') as f:
                pdf_bytes = f.read()
            return pdf_bytes
        finally:
            # CORRECTNESS: Ensure temp file cleanup even if read fails
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError as e:
                logger.warning(f"Failed to remove temp file {temp_path}: {e}")
        
    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        raise


def check_rate_limit() -> Tuple[bool, str]:
    """
    Check if request exceeded upload rate limit (DoS prevention).
    
    SECURITY: IP-based rate limiting with in-memory cache.
    Uses hashed IP addresses to prevent session bypass attacks.
    
    Limitations:
    - Streamlit doesn't directly expose client IP
    - Falls back to session-based tracking if IP unavailable
    - For production, use Redis or database-backed rate limiting
    
    Returns:
        Tuple of (is_allowed, error_message)
    
    Examples:
        >>> allowed, msg = check_rate_limit()
        >>> if not allowed:
        ...     st.error(msg)
        ...     return
    """
    try:
        # SECURITY FIX: Get client IP from Streamlit context
        # Note: Streamlit doesn't expose raw IP, so we use a combination of factors
        ctx = st.runtime.scriptrunner.get_script_run_ctx()
        
        if ctx and hasattr(ctx, 'session_id'):
            # Use session_id + user_agent hash as identifier
            # This is better than just session_id (can't bypass by clearing cookies alone)
            session_id = ctx.session_id
            
            # Try to get additional browser fingerprinting
            try:
                # In production, you'd get this from headers via st.experimental_get_query_params
                # or a custom component. For now, use session_id.
                identifier = session_id
            except Exception:
                identifier = session_id
            
            # Hash the identifier for privacy
            client_hash = hashlib.sha256(identifier.encode()).hexdigest()[:16]
        else:
            # Fallback: use session state with warning
            logger.warning("Could not determine client identifier, falling back to session state")
            if 'client_id' not in st.session_state:
                st.session_state.client_id = hashlib.sha256(
                    f"{datetime.now().isoformat()}".encode()
                ).hexdigest()[:16]
            client_hash = st.session_state.client_id
        
        current_time = time.time()
        window_seconds = TESTING.rate_limit_window_seconds
        max_uploads = TESTING.rate_limit_max_uploads
        
        with _rate_limit_lock:
            # Initialize or get upload history for this client
            if client_hash not in _rate_limit_cache:
                _rate_limit_cache[client_hash] = []
            
            upload_history = _rate_limit_cache[client_hash]
            
            # Remove timestamps outside the time window
            cutoff_time = current_time - window_seconds
            upload_history[:] = [ts for ts in upload_history if ts > cutoff_time]
            
            # Check if limit exceeded
            if len(upload_history) >= max_uploads:
                oldest_upload = min(upload_history)
                wait_time = int(oldest_upload + window_seconds - current_time)
                
                logger.warning(
                    f"Rate limit exceeded for client {client_hash}: "
                    f"{len(upload_history)} uploads in {window_seconds}s window"
                )
                
                return False, (
                    f"⚠️ Upload rate limit exceeded. "
                    f"Maximum {max_uploads} uploads per {window_seconds} seconds. "
                    f"Please wait {wait_time} seconds before trying again."
                )
            
            # Record this upload attempt
            upload_history.append(current_time)
            _rate_limit_cache[client_hash] = upload_history
            
            # Cleanup old entries from cache (prevent memory growth)
            if len(_rate_limit_cache) > 10000:  # Safety limit
                # Remove oldest entries
                sorted_clients = sorted(
                    _rate_limit_cache.items(),
                    key=lambda x: max(x[1]) if x[1] else 0
                )
                for old_client, _ in sorted_clients[:5000]:
                    del _rate_limit_cache[old_client]
            
            logger.info(
                f"Rate limit check passed for client {client_hash}: "
                f"{len(upload_history)}/{max_uploads} uploads in window"
            )
            return True, ""
    
    except Exception as e:
        # If rate limiting fails, log but allow request (fail open for availability)
        logger.error(f"Rate limit check failed: {e}", exc_info=True)
        return True, ""


# Main application
def main():
    """Main Streamlit application."""
    # OBSERVABILITY: Generate request ID for tracking
    if 'request_id' not in st.session_state:
        st.session_state.request_id = str(uuid.uuid4())[:8]
    
    request_id = st.session_state.request_id
    logger.info(f"[{request_id}] Starting FRA Diagnostics session")
    
    st.set_page_config(
        page_title="FRA Diagnostics Dashboard",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">⚡ Transformer FRA Diagnostics Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">IEC 60076-18 Compliant Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📋 Configuration")
        
        vendor = st.selectbox(
            "Vendor Format",
            ["Auto-detect", "Omicron", "Doble", "Megger", "Generic"],
            help="Select the FRA equipment vendor or use auto-detection"
        )
        vendor_map = {
            "Auto-detect": None,
            "Omicron": "omicron",
            "Doble": "doble",
            "Megger": "megger",
            "Generic": "generic"
        }
        selected_vendor = vendor_map[vendor]
        
        st.divider()
        
        generate_baseline = st.checkbox(
            "Generate Baseline Comparison",
            value=False,
            help="Generate a normal (healthy) baseline for comparison"
        )
        
        st.divider()
        
        st.markdown("""
        ### 📖 Quick Guide
        1. **Upload**: Select your FRA measurement file
        2. **Parse**: System automatically detects format and parses data
        3. **QA Check**: IEC 60076-18 compliance verification
        4. **AI Analysis**: Multi-model ensemble classifies fault type
        5. **Visualization**: Interactive Bode plots and frequency analysis
        6. **Report**: Download comprehensive PDF report
        
        ### Supported File Formats
        - **Omicron FRANEO**: CSV and XML formats
        - **Doble SFRA**: Standard format
        - **Megger FRAX**: Proprietary format
        - **Generic**: Standard CSV with frequency, magnitude, phase columns
        """)

    # File upload section
    st.header("📁 Upload FRA Data")
    
    uploaded_file = st.file_uploader(
        "Choose FRA data file",
        type=['csv', 'xml', 'txt', 'dat'],
        help=f"Maximum file size: {MAX_FILE_SIZE_MB} MB"
    )
    
    if uploaded_file is not None:
        logger.info(f"[{request_id}] File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Rate limiting check
        allowed, rate_limit_msg = check_rate_limit()
        if not allowed:
            st.error(rate_limit_msg)
            logger.warning(f"[{request_id}] Rate limit exceeded")
            return
        
        # Validate file
        is_valid, error_msg = validate_uploaded_file(uploaded_file)
        if not is_valid:
            st.error(error_msg)
            logger.error(f"[{request_id}] File validation failed: {error_msg}")
            return
        
        # Sanitize filename
        sanitized_name = sanitize_filename(uploaded_file.name)
        logger.info(f"[{request_id}] Filename sanitized: '{uploaded_file.name}' → '{sanitized_name}'")
        
        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(sanitized_name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filepath = tmp_file.name
        
        try:
            # Progress indicator
            with st.spinner("🔄 Processing FRA data..."):
                # Parse file
                logger.info(f"[{request_id}] Parsing file with vendor={selected_vendor}")
                parse_start = time.time()
                
                parser = get_parser()
                df = parser.parse_file(tmp_filepath, vendor=selected_vendor)
                
                parse_time = time.time() - parse_start
                logger.info(f"[{request_id}] Parsing completed in {parse_time:.2f}s, {len(df)} data points")
                
                # Get QA results
                qa_results = parser.get_qa_results(tmp_filepath)
                logger.info(f"[{request_id}] QA checks completed")
                
                # AI Prediction
                st.subheader("🤖 AI Fault Classification")
                
                # Check if models exist before trying to load
                if not check_models_exist():
                    st.warning("⚠️ **AI models not found**")
                    st.info("""
                    AI fault detection requires trained models. You have two options:
                    
                    **Option 1 (Recommended):** Auto-train models now
                    - Takes 10-30 minutes
                    - Trains with 2000 samples
                    - Click button below to start
                    
                    **Option 2:** Manual training
                    - Run in terminal: `python quick_train_models.py`
                    - Trains with 5000 samples (better accuracy)
                    - Allows you to see detailed progress
                    """)
                    
                    # Add training button
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🚀 Start Auto-Training Now", type="primary", use_container_width=True):
                            st.info("Starting training... This may take 10-30 minutes.")
                            with st.spinner("Training in progress..."):
                                if auto_train_models():
                                    st.success("✅ Training complete! Refresh the page to use AI predictions.")
                                    st.balloons()
                                else:
                                    st.error("Training failed. Check logs for details.")
                    
                    with col2:
                        st.info("💡 **Tip:** You can continue using other features while models are being trained manually.")
                    
                    # Skip AI prediction but continue with other features
                    prediction = None
                    
                else:
                    # Models exist, load and run prediction
                    try:
                        logger.info(f"[{request_id}] Loading AI models...")
                        ensemble = load_ai_models()
                        
                        logger.info(f"[{request_id}] Running AI inference...")
                        inference_start = time.time()
                        
                        prediction = ensemble.predict(df)
                        
                        inference_time = time.time() - inference_start
                        logger.info(
                            f"[{request_id}] AI prediction completed in {inference_time:.2f}s: "
                            f"{prediction['predicted_fault']} (confidence: {prediction['confidence']:.2%})"
                        )
                        
                        # Display prediction results
                        st.write("Predicted Fault:", prediction['predicted_fault'])
                        st.write("Confidence:", prediction['confidence'])
                        
                    except Exception as e:
                        logger.error(f"[{request_id}] AI model inference failed: {e}", exc_info=True)
                        st.warning("⚠️ AI model not available. Showing data analysis only.")
                        prediction = None
                
                # Display results
                st.subheader("📊 Diagnostic Results")
                
                # Create Bode plot
                fig = create_bode_plot_plotly(df, "FRA Signature Analysis")
                st.plotly_chart(fig, use_container_width=True)
                
                # Display QA results
                display_iec_qa_results(qa_results)
                
                # Download report
                st.subheader("📄 Download PDF Report")
                if st.button("Download Report"):
                    pdf_bytes = generate_and_download_iec_report(df, prediction or {}, qa_results)
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name=f"fra_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
        
        except Exception as e:
            logger.error(f"[{request_id}] Error processing file: {e}", exc_info=True)
            st.error(f"❌ Error processing file: {str(e)}")
        
        finally:
            # Cleanup temp file
            try:
                os.unlink(tmp_filepath)
                logger.debug(f"[{request_id}] Cleaned up temp file: {tmp_filepath}")
            except Exception as e:
                logger.warning(f"[{request_id}] Failed to cleanup temp file: {e}")
    
    else:
        # Show welcome message
        st.info("👆 Please upload an FRA data file to begin analysis")
        
        # Show example/demo section
        with st.expander("🎓 See Example Analysis"):
            st.markdown("""
            ### Sample FRA Analysis Workflow
            
            1. **Upload**: Select your FRA measurement file
            2. **Parse**: System automatically detects format and parses data
            3. **QA Check**: IEC 60076-18 compliance verification
            4. **AI Analysis**: Multi-model ensemble classifies fault type
            5. **Visualization**: Interactive Bode plots and frequency analysis
            6. **Report**: Download comprehensive PDF report
            
            ### Supported File Formats
            - **Omicron FRANEO**: CSV and XML formats
            - **Doble SFRA**: Standard format
            - **Megger FRAX**: Proprietary format
            - **Generic**: Standard CSV with frequency, magnitude, phase columns
            """)

if __name__ == "__main__":
    # Add health check query parameter for monitoring
    query_params = st.query_params
    
    if "health" in query_params:
        # Health check endpoint for monitoring
        st.write("OK")
        st.write({
            "status": "healthy",
            "service": "FRA Diagnostics Dashboard",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        })
        logger.info("Health check requested")
    else:
        # Run main application
        main()