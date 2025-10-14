"""
AI Ensemble for Transformer FRA Fault Classification
SIH 2025 PS 25190

Implements a multi-model ensemble combining:
1. 1D CNN for raw frequency/magnitude/phase sequence classification
2. ResNet18 for Bode plot image classification (transfer learning)
3. One-class SVM for anomaly detection (baseline-free)

Fault Classes:
    0: Normal
    1: Axial Deformation
    2: Radial Deformation
    3: Inter-turn Short
    4: Core Grounding
    5: Tap-changer Fault

Ensemble Strategy:
    Weighted voting: 0.4*CNN + 0.4*ResNet + 0.2*SVM
    Uncertainty: Shannon entropy over probability distributions
    
Explainability:
    - Grad-CAM for CNN and ResNet
    - Feature importance for SVM
    - Multi-band energy analysis
"""

__all__ = [
    'FRADataset',
    'CNN1D',
    'ResNetClassifier',
    'FeatureExtractor',
    'FRAEnsemble',
    'prepare_data_loaders',
    'train_cnn',
    'train_resnet',
    'train_svm',
    'save_models',
    'load_models',
    'plot_confusion_matrix',
    'train_ensemble_pipeline'
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score,
    accuracy_score
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
from typing import Tuple, Dict, List, Optional
import warnings
from datetime import datetime
import pickle
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Import centralized configuration
try:
    from config import ML
except ImportError:
    # Fallback for standalone usage
    class ML:
        dtw_max_sequence_length = 500
        dtw_early_termination_threshold = 1000.0
        dtw_subsample_points = 500

class FRADataset(Dataset):
    """
    PyTorch Dataset for FRA data.
    
    Loads frequency, magnitude, and phase sequences from DataFrame
    and prepares them for neural network training.
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        mode: str = 'sequence',
        transform: Optional[callable] = None,
        augment: bool = False,
        max_length: int = 1000
    ):
        """
        Initialize FRA dataset.
        
        Args:
            dataframe: DataFrame with FRA data
            mode: 'sequence' for 1D CNN, 'image' for ResNet
            transform: Optional transform for images
            augment: Whether to apply data augmentation
            max_length: Maximum sequence length (pad/truncate to this)
        """
        self.mode = mode
        self.transform = transform
        self.augment = augment
        self.max_length = max_length
        
        # Get unique samples
        self.sample_ids = dataframe['sample_id'].unique()
        self.dataframe = dataframe
        
        # Create fault type mapping
        self.fault_types = sorted(dataframe['fault_type'].unique())
        self.fault_to_idx = {fault: idx for idx, fault in enumerate(self.fault_types)}
        self.idx_to_fault = {idx: fault for fault, idx in self.fault_to_idx.items()}
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample_id = self.sample_ids[idx]
        sample_data = self.dataframe[self.dataframe['sample_id'] == sample_id]
        
        # Get fault label
        fault_type = sample_data['fault_type'].iloc[0]
        label = self.fault_to_idx[fault_type]
        
        if self.mode == 'sequence':
            # Extract sequences for 1D CNN
            freq = sample_data['frequency_hz'].values
            mag = sample_data['magnitude_db'].values
            phase = sample_data['phase_deg'].values
            
            # Normalize frequency to [0, 1]
            freq_norm = (np.log10(freq) - np.log10(20)) / (np.log10(2e6) - np.log10(20))
            
            # Stack as 3-channel sequence (like RGB for 1D)
            sequence = np.stack([freq_norm, mag, phase], axis=0)
            
            # FIX: Pad or truncate to fixed length for batching
            current_length = sequence.shape[1]
            if current_length < self.max_length:
                # Pad with zeros
                padding = np.zeros((3, self.max_length - current_length))
                sequence = np.concatenate([sequence, padding], axis=1)
            elif current_length > self.max_length:
                # Truncate to max_length
                sequence = sequence[:, :self.max_length]
            
            # Data augmentation
            if self.augment:
                # Add Gaussian noise
                noise_level = np.random.uniform(0.01, 0.05)
                sequence[1] += np.random.normal(0, noise_level, sequence[1].shape)  # mag
                sequence[2] += np.random.normal(0, noise_level * 0.5, sequence[2].shape)  # phase
            
            return torch.FloatTensor(sequence), label
            
        elif self.mode == 'image':
            # Generate Bode plot image for ResNet
            image = self._generate_bode_image(sample_data)
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
    
    def _generate_bode_image(self, sample_data: pd.DataFrame, size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """
        Generate Bode plot image from FRA data.
        
        Args:
            sample_data: DataFrame for single sample
            size: Output image size
        
        Returns:
            PIL Image in grayscale
        """
        fig = None
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
            
            freq = sample_data['frequency_hz'].values
            mag = sample_data['magnitude_db'].values
            phase = sample_data['phase_deg'].values
            
            # Magnitude plot
            ax1.semilogx(freq, mag, 'k-', linewidth=1.5)
            ax1.set_ylabel('Magnitude (dB)', fontsize=10)
            ax1.grid(True, which='both', alpha=0.3)
            ax1.set_xlim(20, 2e6)
            
            # Phase plot
            ax2.semilogx(freq, phase, 'k-', linewidth=1.5)
            ax2.set_xlabel('Frequency (Hz)', fontsize=10)
            ax2.set_ylabel('Phase (deg)', fontsize=10)
            ax2.grid(True, which='both', alpha=0.3)
            ax2.set_xlim(20, 2e6)
            
            plt.tight_layout()
            
            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=75, bbox_inches='tight')
            buf.seek(0)
            
            # Load as PIL Image and convert to grayscale
            image = Image.open(buf).convert('L')
            image = image.resize(size, Image.LANCZOS)
            
            return image
        finally:
            # CRITICAL: Always close figure to prevent memory leaks
            if fig is not None:
                plt.close(fig)


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for FRA sequence classification.
    
    Architecture:
        Input: [batch, 3, seq_len] - frequency, magnitude, phase
        Conv1D layers with batch norm and dropout
        Global average pooling
        Fully connected classifier
    """
    
    def __init__(self, input_channels: int = 3, num_classes: int = 6, seq_len: int = 1000):
        """
        Initialize 1D CNN.
        
        Args:
            input_channels: Number of input channels (3 for freq/mag/phase)
            num_classes: Number of fault classes
            seq_len: Sequence length
        """
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, 3, seq_len]
        
        Returns:
            Logits [batch, num_classes]
        """
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Fully connected
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for visualization/analysis.
        
        Args:
            x: Input tensor
        
        Returns:
            Feature tensor before classification
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x).squeeze(-1)
        return x


class ResNetClassifier(nn.Module):
    """
    ResNet18-based classifier for Bode plot images.
    
    Uses transfer learning from ImageNet pre-trained model.
    Fine-tunes final layers for FRA fault classification.
    
    Note: Converts grayscale input to 3-channel to leverage pretrained weights.
    """
    
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        """
        Initialize ResNet classifier.
        
        Args:
            num_classes: Number of fault classes
            pretrained: Use ImageNet pre-trained weights
        """
        super(ResNetClassifier, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Keep original conv1 for 3-channel input (we'll convert grayscale to RGB)
        # This preserves pretrained weights for better transfer learning
        
        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [batch, 1, 224, 224] (grayscale)
        
        Returns:
            Logits [batch, num_classes]
        """
        # Convert grayscale to 3-channel by repeating
        # This allows us to use pretrained weights from ImageNet
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        return self.resnet(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before final classification.
        
        Args:
            x: Input images
        
        Returns:
            Feature tensor
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


class FeatureExtractor:
    """
    Extract engineered features from FRA data for SVM.
    
    Features:
        - Multi-band energies (20-100 Hz, 100k-500k Hz, etc.)
        - Resonance peak locations and magnitudes
        - DTW distances from baseline
        - Statistical moments
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        self.baseline_signature = None
        self.scaler = StandardScaler()
        
    def extract_features(self, sample_data: pd.DataFrame) -> np.ndarray:
        """
        Extract engineered features from FRA sample.
        
        Features include:
        - Multi-band energies across 6 frequency ranges
        - Resonance peak locations and magnitudes
        - DTW distances from baseline (if available)
        - Statistical moments (mean, std, skewness, kurtosis)
        
        Args:
            sample_data: DataFrame for single sample with columns:
                        frequency_hz, magnitude_db, phase_deg
        
        Returns:
            np.ndarray: Feature vector of shape (n_features,)
            
        Raises:
            ValueError: If sample_data is empty or missing required columns
            
        Examples:
            >>> extractor = FeatureExtractor()
            >>> df = pd.DataFrame({
            ...     'frequency_hz': [20, 100, 1000],
            ...     'magnitude_db': [0.5, 1.2, 0.8],
            ...     'phase_deg': [-10, -20, -30]
            ... })
            >>> features = extractor.extract_features(df)
            >>> features.shape
            (15,)  # or appropriate feature count
        """
        if len(sample_data) == 0:
            raise ValueError("sample_data is empty")
        
        required_cols = ['frequency_hz', 'magnitude_db', 'phase_deg']
        if not all(col in sample_data.columns for col in required_cols):
            raise ValueError(f"sample_data missing required columns: {required_cols}")
        
        freq = sample_data['frequency_hz'].values
        mag = sample_data['magnitude_db'].values
        phase = sample_data['phase_deg'].values
        
        features = []
        
        # Multi-band energies (magnitude)
        bands = [
            (20, 100),          # Low frequency
            (100, 1000),        # Mid-low
            (1000, 10000),      # Mid
            (10000, 100000),    # Mid-high
            (100000, 500000),   # High
            (500000, 2000000)   # Very high
        ]
        
        for low, high in bands:
            mask = (freq >= low) & (freq <= high)
            if mask.sum() > 0:
                features.append(np.mean(mag[mask]))
                features.append(np.std(mag[mask]))
                features.append(np.mean(phase[mask]))
            else:
                features.extend([0, 0, 0])
        
        # Resonance detection (peaks in magnitude)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(mag, prominence=2.0)
        
        if len(peaks) > 0:
            # Top 3 resonance peaks
            top_peaks = sorted(zip(mag[peaks], freq[peaks]), reverse=True)[:3]
            for peak_mag, peak_freq in top_peaks:
                features.extend([peak_mag, np.log10(peak_freq)])
            
            # Pad if fewer than 3 peaks
            for _ in range(3 - len(top_peaks)):
                features.extend([0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0])
        
        # Statistical moments
        features.extend([
            np.mean(mag),
            np.std(mag),
            np.min(mag),
            np.max(mag),
            np.mean(phase),
            np.std(phase)
        ])
        
        # DTW distance from baseline (if available)
        if self.baseline_signature is not None:
            dtw_dist = self._compute_dtw(mag, self.baseline_signature)
            features.append(dtw_dist)
        else:
            features.append(0)
        
        return np.array(features)
    
    def _compute_dtw(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        Compute Dynamic Time Warping distance.
        
        PERFORMANCE: Optimized with early termination and optional subsampling
        to handle large sequences efficiently.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
        
        Returns:
            DTW distance
        """
        # PERFORMANCE: Subsample if sequences are too long
        if len(seq1) > ML.dtw_max_sequence_length or len(seq2) > ML.dtw_max_sequence_length:
            # Subsample to dtw_subsample_points
            indices1 = np.linspace(0, len(seq1)-1, ML.dtw_subsample_points, dtype=int)
            indices2 = np.linspace(0, len(seq2)-1, ML.dtw_subsample_points, dtype=int)
            seq1 = seq1[indices1]
            seq2 = seq2[indices2]
        
        n, m = len(seq1), len(seq2)
        dtw = np.full((n+1, m+1), np.inf)
        dtw[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(seq1[i-1] - seq2[j-1])
                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
                
                # PERFORMANCE: Early termination if cost explodes
                if dtw[i, j] > ML.dtw_early_termination_threshold:
                    return dtw[i, j]
        
        return dtw[n, m]
    
    def set_baseline(self, baseline_data: pd.DataFrame):
        """
        Set baseline signature for DTW computation.
        
        Args:
            baseline_data: DataFrame with normal/healthy signature
        """
        self.baseline_signature = baseline_data['magnitude_db'].values
    
    def fit_scaler(self, features: np.ndarray):
        """
        Fit StandardScaler on training features.
        
        Args:
            features: Training feature matrix [n_samples, n_features]
        """
        self.scaler.fit(features)
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            features: Feature matrix
        
        Returns:
            Scaled features
        """
        return self.scaler.transform(features)


def prepare_data_loaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 32,
    mode: str = 'sequence'
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare PyTorch DataLoaders for training and testing.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        batch_size: Batch size
        mode: 'sequence' or 'image'
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    if mode == 'image':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        transform = None
    
    train_dataset = FRADataset(train_df, mode=mode, transform=transform, augment=True, max_length=1000)
    test_dataset = FRADataset(test_df, mode=mode, transform=transform, augment=False, max_length=1000)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, test_loader, train_dataset.fault_to_idx


def train_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu'
) -> Dict[str, List[float]]:
    """
    Train 1D CNN model.
    
    Args:
        model: CNN model
        train_loader: Training data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device ('cpu' or 'cuda')
    
    Returns:
        Dict with training history (losses and accuracies per epoch)
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    history = {'loss': [], 'accuracy': []}
    
    logger.info(f"Training 1D CNN on {device}...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Acc: {accuracy:.2f}%")
    
    logger.info("Training complete!")
    return history


def train_resnet(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 30,
    learning_rate: float = 0.0001,
    device: str = 'cpu'
) -> Dict:
    """
    Train ResNet model with transfer learning.
    
    Args:
        model: ResNet model
        train_loader: Training data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate (lower for fine-tuning)
        device: Device ('cpu' or 'cuda')
    
    Returns:
        Training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    history = {'loss': [], 'accuracy': []}
    
    logger.info(f"Training ResNet18 on {device}...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Acc: {accuracy:.2f}%")
    
    logger.info("Training complete!")
    return history


def train_svm(
    train_df: pd.DataFrame,
    feature_extractor: FeatureExtractor,
    nu: float = 0.1,
    gamma: str = 'auto'
) -> OneClassSVM:
    """
    Train One-Class SVM for anomaly detection.
    
    Fits on 'normal' samples only for baseline-free detection.
    
    Args:
        train_df: Training DataFrame
        feature_extractor: FeatureExtractor instance
        nu: Anomaly fraction parameter
        gamma: Kernel coefficient
    
    Returns:
        Trained SVM model
    """
    logger.info("Training One-Class SVM...")
    
    # Extract features for normal samples only
    normal_samples = train_df[train_df['fault_type'] == 'normal']
    sample_ids = normal_samples['sample_id'].unique()
    
    features_list = []
    for sample_id in sample_ids:
        sample_data = normal_samples[normal_samples['sample_id'] == sample_id]
        features = feature_extractor.extract_features(sample_data)
        features_list.append(features)
    
    features_matrix = np.array(features_list)
    
    # Fit scaler
    feature_extractor.fit_scaler(features_matrix)
    features_scaled = feature_extractor.transform(features_matrix)
    
    # Train SVM
    svm = OneClassSVM(nu=nu, gamma=gamma, kernel='rbf')
    svm.fit(features_scaled)
    
    logger.info(f"SVM trained on {len(sample_ids)} normal samples")
    return svm


class FRAEnsemble:
    """
    Ensemble model combining CNN, ResNet, and SVM.
    
    Weighted voting: 0.4*CNN + 0.4*ResNet + 0.2*SVM
    """
    
    def __init__(
        self,
        cnn_model: nn.Module,
        resnet_model: nn.Module,
        svm_model: OneClassSVM,
        feature_extractor: FeatureExtractor,
        fault_mapping: Dict,
        weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
        device: str = 'cpu'
    ):
        """
        Initialize ensemble.
        
        Args:
            cnn_model: Trained CNN model
            resnet_model: Trained ResNet model
            svm_model: Trained SVM model
            feature_extractor: Feature extractor for SVM
            fault_mapping: Fault type to index mapping
            weights: Ensemble weights (CNN, ResNet, SVM)
            device: Device for inference
        """
        self.cnn_model = cnn_model.to(device)
        self.resnet_model = resnet_model.to(device)
        self.svm_model = svm_model
        self.feature_extractor = feature_extractor
        self.fault_mapping = fault_mapping
        self.idx_to_fault = {v: k for k, v in fault_mapping.items()}
        self.weights = weights
        self.device = device
        
        self.cnn_model.eval()
        self.resnet_model.eval()
    
    def predict(self, sample_df: pd.DataFrame) -> Dict:
        """
        Predict fault type for a single sample using ensemble.
        
        Args:
            sample_df: DataFrame for single FRA sample
        
        Returns:
            Dictionary with prediction results
        """
        # Prepare data
        dataset_seq = FRADataset(sample_df, mode='sequence', max_length=1000)
        dataset_img = FRADataset(sample_df, mode='image', transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]), max_length=1000)
        
        # CNN prediction
        seq_data, _ = dataset_seq[0]
        seq_data = seq_data.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            cnn_logits = self.cnn_model(seq_data)
            cnn_probs = F.softmax(cnn_logits, dim=1).cpu().numpy()[0]
        
        # ResNet prediction
        img_data, _ = dataset_img[0]
        img_data = img_data.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            resnet_logits = self.resnet_model(img_data)
            resnet_probs = F.softmax(resnet_logits, dim=1).cpu().numpy()[0]
        
        # SVM prediction (anomaly score)
        features = self.feature_extractor.extract_features(sample_df)
        features_scaled = self.feature_extractor.transform(features.reshape(1, -1))
        svm_score = self.svm_model.decision_function(features_scaled)[0]
        
        # Convert SVM score to probabilities
        # Positive score = normal, negative = anomaly
        svm_probs = np.zeros(len(self.fault_mapping))
        if svm_score > 0:
            # Likely normal
            normal_idx = self.fault_mapping.get('normal', 0)
            svm_probs[normal_idx] = 1.0
        else:
            # Anomaly - distribute among fault classes
            fault_indices = [i for i, fault in self.idx_to_fault.items() if fault != 'normal']
            svm_probs[fault_indices] = 1.0 / len(fault_indices)
        
        # Ensemble voting
        ensemble_probs = (
            self.weights[0] * cnn_probs +
            self.weights[1] * resnet_probs +
            self.weights[2] * svm_probs
        )
        
        # Normalize
        ensemble_probs = ensemble_probs / ensemble_probs.sum()
        
        # Uncertainty (Shannon entropy)
        entropy = -np.sum(ensemble_probs * np.log(ensemble_probs + 1e-10))
        max_entropy = np.log(len(self.fault_mapping))
        uncertainty = entropy / max_entropy
        
        predicted_idx = np.argmax(ensemble_probs)
        predicted_fault = self.idx_to_fault[predicted_idx]
        confidence = ensemble_probs[predicted_idx]
        
        return {
            'predicted_fault': predicted_fault,
            'confidence': float(confidence),
            'uncertainty': float(uncertainty),
            'probabilities': {self.idx_to_fault[i]: float(p) for i, p in enumerate(ensemble_probs)},
            'cnn_probs': {self.idx_to_fault[i]: float(p) for i, p in enumerate(cnn_probs)},
            'resnet_probs': {self.idx_to_fault[i]: float(p) for i, p in enumerate(resnet_probs)},
            'svm_score': float(svm_score)
        }
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """
        Evaluate ensemble on test dataset.
        
        Args:
            test_df: Test DataFrame
        
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating ensemble...")
        
        sample_ids = test_df['sample_id'].unique()
        predictions = []
        true_labels = []
        
        for sample_id in sample_ids:
            sample_data = test_df[test_df['sample_id'] == sample_id]
            true_fault = sample_data['fault_type'].iloc[0]
            
            result = self.predict(sample_data)
            
            predictions.append(result['predicted_fault'])
            true_labels.append(true_fault)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(
            true_labels,
            predictions,
            labels=sorted(self.fault_mapping.keys())
        )
        
        # Classification report
        report = classification_report(
            true_labels,
            predictions,
            target_names=sorted(self.fault_mapping.keys()),
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'num_samples': len(sample_ids)
        }


def save_models(
    cnn_model: nn.Module,
    resnet_model: nn.Module,
    svm_model: OneClassSVM,
    feature_extractor: FeatureExtractor,
    fault_mapping: Dict,
    save_dir: str = 'models'
):
    """
    Save all trained models.
    
    Args:
        cnn_model: Trained CNN
        resnet_model: Trained ResNet
        svm_model: Trained SVM
        feature_extractor: Feature extractor
        fault_mapping: Fault type mapping
        save_dir: Directory to save models
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save PyTorch models
    torch.save(cnn_model.state_dict(), os.path.join(save_dir, 'cnn_model.pth'))
    torch.save(resnet_model.state_dict(), os.path.join(save_dir, 'resnet_model.pth'))
    
    # Export to ONNX
    try:
        dummy_input_cnn = torch.randn(1, 3, 1000)
        torch.onnx.export(
            cnn_model.cpu(),
            dummy_input_cnn,
            os.path.join(save_dir, 'cnn_model.onnx'),
            input_names=['input'],
            output_names=['output']
        )
        
        dummy_input_resnet = torch.randn(1, 1, 224, 224)
        torch.onnx.export(
            resnet_model.cpu(),
            dummy_input_resnet,
            os.path.join(save_dir, 'resnet_model.onnx'),
            input_names=['input'],
            output_names=['output']
        )
    except Exception as e:
        logger.warning(f"Warning: ONNX export failed: {e}")
    
    # Save SVM and feature extractor
    with open(os.path.join(save_dir, 'svm_model.pkl'), 'wb') as f:
        pickle.dump(svm_model, f)
    
    with open(os.path.join(save_dir, 'feature_extractor.pkl'), 'wb') as f:
        pickle.dump(feature_extractor, f)
    
    # Save fault mapping
    with open(os.path.join(save_dir, 'fault_mapping.pkl'), 'wb') as f:
        pickle.dump(fault_mapping, f)
    
    logger.info(f"Models saved to {save_dir}/")


def load_models(save_dir: str = 'models', device: str = 'cpu') -> FRAEnsemble:
    """
    Load trained models and create ensemble.
    
    Args:
        save_dir: Directory containing saved models
        device: Device for inference
    
    Returns:
        FRAEnsemble instance
    """
    # Load fault mapping
    with open(os.path.join(save_dir, 'fault_mapping.pkl'), 'rb') as f:
        fault_mapping = pickle.load(f)
    
    num_classes = len(fault_mapping)
    
    # Load CNN
    cnn_model = CNN1D(num_classes=num_classes)
    cnn_model.load_state_dict(torch.load(
        os.path.join(save_dir, 'cnn_model.pth'),
        map_location=device
    ))
    
    # Load ResNet
    resnet_model = ResNetClassifier(num_classes=num_classes, pretrained=False)
    resnet_model.load_state_dict(torch.load(
        os.path.join(save_dir, 'resnet_model.pth'),
        map_location=device
    ))
    
    # Load SVM
    with open(os.path.join(save_dir, 'svm_model.pkl'), 'rb') as f:
        svm_model = pickle.load(f)
    
    # Load feature extractor
    with open(os.path.join(save_dir, 'feature_extractor.pkl'), 'rb') as f:
        feature_extractor = pickle.load(f)
    
    ensemble = FRAEnsemble(
        cnn_model,
        resnet_model,
        svm_model,
        feature_extractor,
        fault_mapping,
        device=device
    )
    
    logger.info(f"Models loaded from {save_dir}/")
    return ensemble


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], save_path: Optional[str] = None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        labels: Class labels
        save_path: Path to save plot
    """
    fig = None
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=labels,
               yticklabels=labels,
               ylabel='True Label',
               xlabel='Predicted Label',
               title='Confusion Matrix')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
    finally:
        # CRITICAL: Always close figure to prevent memory leaks
        if fig is not None:
            plt.close(fig)
    
    plt.show()


def train_ensemble_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_epochs_cnn: int = 50,
    num_epochs_resnet: int = 30,
    batch_size: int = 32,
    device: str = None,
    save_dir: str = 'models'
) -> FRAEnsemble:
    """
    Complete training pipeline for ensemble.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        num_epochs_cnn: Epochs for CNN
        num_epochs_resnet: Epochs for ResNet
        batch_size: Batch size
        device: Device (auto-detect if None)
        save_dir: Directory to save models
    
    Returns:
        Trained FRAEnsemble
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info("="*70)
    logger.info("FRA Fault Classification - Ensemble Training Pipeline")
    logger.info("SIH 2025 PS 25190")
    logger.info("="*70)
    logger.info(f"Device: {device}")
    logger.info(f"Training samples: {len(train_df['sample_id'].unique())}")
    logger.info(f"Testing samples: {len(test_df['sample_id'].unique())}")
    logger.info("")
    
    # Get fault mapping
    fault_types = sorted(train_df['fault_type'].unique())
    fault_mapping = {fault: idx for idx, fault in enumerate(fault_types)}
    num_classes = len(fault_types)
    
    logger.info(f"Fault classes ({num_classes}): {', '.join(fault_types)}")
    logger.info("")
    
    # Prepare data loaders
    logger.info("Preparing data loaders...")
    train_loader_seq, test_loader_seq, _ = prepare_data_loaders(
        train_df, test_df, batch_size=batch_size, mode='sequence'
    )
    train_loader_img, test_loader_img, _ = prepare_data_loaders(
        train_df, test_df, batch_size=batch_size, mode='image'
    )
    
    # Train CNN
    logger.info("\n" + "="*70)
    logger.info("Training 1D CNN")
    logger.info("="*70)
    cnn_model = CNN1D(num_classes=num_classes)
    train_cnn(cnn_model, train_loader_seq, num_epochs=num_epochs_cnn, device=device)
    
    # Train ResNet
    logger.info("\n" + "="*70)
    logger.info("Training ResNet18")
    logger.info("="*70)
    resnet_model = ResNetClassifier(num_classes=num_classes, pretrained=True)
    train_resnet(resnet_model, train_loader_img, num_epochs=num_epochs_resnet, device=device)
    
    # Train SVM
    logger.info("\n" + "="*70)
    logger.info("Training One-Class SVM")
    logger.info("="*70)
    feature_extractor = FeatureExtractor()
    
    # Set baseline from normal samples
    normal_data = train_df[train_df['fault_type'] == 'normal']
    if len(normal_data) > 0:
        baseline_sample = normal_data[normal_data['sample_id'] == normal_data['sample_id'].iloc[0]]
        feature_extractor.set_baseline(baseline_sample)
    
    svm_model = train_svm(train_df, feature_extractor)
    
    # Create ensemble
    logger.info("\n" + "="*70)
    logger.info("Creating Ensemble")
    logger.info("="*70)
    ensemble = FRAEnsemble(
        cnn_model, resnet_model, svm_model,
        feature_extractor, fault_mapping, device=device
    )
    
    # Evaluate
    logger.info("\n" + "="*70)
    logger.info("Evaluation")
    logger.info("="*70)
    metrics = ensemble.evaluate(test_df)
    
    logger.info(f"\nAccuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"F1 Score: {metrics['f1_score']*100:.2f}%")
    
    # Save models
    logger.info("\n" + "="*70)
    logger.info("Saving Models")
    logger.info("="*70)
    save_models(cnn_model, resnet_model, svm_model, feature_extractor, fault_mapping, save_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        sorted(fault_types),
        save_path=os.path.join(save_dir, 'confusion_matrix.png')
    )
    
    logger.info("\n" + "="*70)
    logger.info("Training Complete!")
    logger.info("="*70)
    
    return ensemble


if __name__ == '__main__':
    # Example usage
    logger.info("Loading synthetic data...")
    
    # Assume data generated by simulator
    if os.path.exists('synthetic_data/train_dataset.csv'):
        train_df = pd.read_csv('synthetic_data/train_dataset.csv')
        test_df = pd.read_csv('synthetic_data/test_dataset.csv')
        
        # Train ensemble
        ensemble = train_ensemble_pipeline(
            train_df, test_df,
            num_epochs_cnn=50,
            num_epochs_resnet=30,
            batch_size=32
        )
        
        logger.info("\nEnsemble training complete!")
        logger.info("Models saved in 'models/' directory")
    else:
        logger.info("Error: Synthetic data not found.")
        logger.info("Please run simulator.py first to generate training data.")
