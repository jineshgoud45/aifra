"""
Shared Utility Functions for FRA Diagnostics
SIH 2025 PS 25190

Common utility functions used across multiple modules to avoid code duplication.
Includes Bode plot generation, data validation, and helper functions.
"""

__all__ = [
    'generate_bode_plot_matplotlib',
    'create_bode_plot_plotly',
    'validate_fra_data',
    'compute_frequency_bands'
]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Constants
IEC_FREQ_MIN = 20  # Hz
IEC_FREQ_MAX = 2e6  # Hz


def generate_bode_plot_matplotlib(
    freq: np.ndarray,
    mag: np.ndarray,
    phase: np.ndarray,
    title: str = "FRA Bode Plot",
    figsize: Tuple[int, int] = (6, 6),
    dpi: int = 75
) -> Image.Image:
    """
    Generate Bode plot as PIL Image using Matplotlib.
    
    Properly manages figure cleanup to prevent memory leaks.
    
    Args:
        freq: Frequency array in Hz
        mag: Magnitude array in dB
        phase: Phase array in degrees
        title: Plot title
        figsize: Figure size in inches
        dpi: Resolution in dots per inch
    
    Returns:
        PIL Image object
    """
    fig = None
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Magnitude plot
        ax1.semilogx(freq, mag, 'k-', linewidth=1.5)
        ax1.set_ylabel('Magnitude (dB)', fontsize=10)
        ax1.set_title(title, fontsize=12, fontweight='bold')
        ax1.grid(True, which='both', alpha=0.3)
        ax1.set_xlim(IEC_FREQ_MIN, IEC_FREQ_MAX)
        
        # Phase plot
        ax2.semilogx(freq, phase, 'k-', linewidth=1.5)
        ax2.set_xlabel('Frequency (Hz)', fontsize=10)
        ax2.set_ylabel('Phase (deg)', fontsize=10)
        ax2.grid(True, which='both', alpha=0.3)
        ax2.set_xlim(IEC_FREQ_MIN, IEC_FREQ_MAX)
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        
        # Load as PIL Image
        image = Image.open(buf).convert('L')
        
        return image
        
    finally:
        # CRITICAL: Always close figure to prevent memory leaks
        if fig is not None:
            plt.close(fig)


def create_bode_plot_plotly(
    df: pd.DataFrame,
    title: str = "FRA Bode Plot",
    baseline_df: Optional[pd.DataFrame] = None
) -> go.Figure:
    """
    Create interactive Bode plot using Plotly.
    
    Args:
        df: FRA data DataFrame with columns: frequency_hz, magnitude_db, phase_deg
        title: Plot title
        baseline_df: Optional baseline DataFrame for comparison
    
    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Magnitude Response', 'Phase Response'),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )
    
    # Magnitude plot
    fig.add_trace(
        go.Scatter(
            x=df['frequency_hz'],
            y=df['magnitude_db'],
            mode='lines',
            name='Measured',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Frequency</b>: %{x:.2f} Hz<br><b>Magnitude</b>: %{y:.2f} dB<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Baseline overlay if provided
    if baseline_df is not None:
        fig.add_trace(
            go.Scatter(
                x=baseline_df['frequency_hz'],
                y=baseline_df['magnitude_db'],
                mode='lines',
                name='Normal Baseline',
                line=dict(color='#2ca02c', width=2, dash='dash'),
                hovertemplate='<b>Frequency</b>: %{x:.2f} Hz<br><b>Magnitude</b>: %{y:.2f} dB<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Phase plot
    fig.add_trace(
        go.Scatter(
            x=df['frequency_hz'],
            y=df['phase_deg'],
            mode='lines',
            name='Measured',
            line=dict(color='#ff7f0e', width=2),
            showlegend=False,
            hovertemplate='<b>Frequency</b>: %{x:.2f} Hz<br><b>Phase</b>: %{y:.2f}°<extra></extra>'
        ),
        row=2, col=1
    )
    
    if baseline_df is not None:
        fig.add_trace(
            go.Scatter(
                x=baseline_df['frequency_hz'],
                y=baseline_df['phase_deg'],
                mode='lines',
                name='Normal Baseline',
                line=dict(color='#d62728', width=2, dash='dash'),
                showlegend=False,
                hovertemplate='<b>Frequency</b>: %{x:.2f} Hz<br><b>Phase</b>: %{y:.2f}°<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Update axes
    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
    fig.update_yaxes(title_text="Phase (degrees)", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f77b4')),
        height=700,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def validate_fra_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate FRA data DataFrame for required columns and data quality.
    
    Args:
        df: FRA data DataFrame
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_columns = ['frequency_hz', 'magnitude_db', 'phase_deg']
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    # Check for empty data
    if len(df) == 0:
        return False, "DataFrame is empty"
    
    # Check for NaN values
    if df[required_columns].isnull().any().any():
        return False, "Data contains NaN values"
    
    # Check frequency range
    if df['frequency_hz'].min() <= 0:
        return False, "Frequency values must be positive"
    
    return True, ""


def compute_frequency_bands(df: pd.DataFrame) -> dict:
    """
    Compute energy in different frequency bands.
    
    Args:
        df: FRA data DataFrame
    
    Returns:
        Dictionary with band names and energy values
    """
    bands = {
        'Low (20-100 Hz)': (20, 100),
        'Mid-Low (100-1k Hz)': (100, 1000),
        'Mid (1k-10k Hz)': (1000, 10000),
        'Mid-High (10k-100k Hz)': (10000, 100000),
        'High (100k-500k Hz)': (100000, 500000),
        'Very High (500k-2M Hz)': (500000, 2000000)
    }
    
    freq = df['frequency_hz'].values
    mag = df['magnitude_db'].values
    
    band_energies = {}
    
    for label, (low, high) in bands.items():
        mask = (freq >= low) & (freq <= high)
        if mask.sum() > 0:
            energy = np.mean(np.abs(mag[mask]))
        else:
            energy = 0
        band_energies[label] = energy
    
    return band_energies
