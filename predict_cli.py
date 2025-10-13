#!/usr/bin/env python3
"""
CLI FRA Fault Predictor for Edge Deployment
SIH 2025 PS 25190

Lightweight command-line interface for running FRA fault predictions
on edge devices (Raspberry Pi) using ONNX models.

No GUI dependencies - pure CLI with text output.
Optimized for ARM processors with minimal memory footprint.

Usage:
    python predict_cli.py --file data.csv --vendor omicron
    python predict_cli.py --file data.xml --output results.json
"""

import argparse
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
import pickle
import os
import sys
from datetime import datetime
from typing import Dict, Tuple


class EdgePredictor:
    """
    Lightweight FRA fault predictor for edge devices.
    
    Uses ONNX models for fast CPU inference without GPU dependency.
    """
    
    FAULT_TYPES = [
        'axial_deformation',
        'core_grounding',
        'interturn_short',
        'normal',
        'radial_deformation',
        'tapchanger_fault'
    ]
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize edge predictor.
        
        Args:
            models_dir: Directory containing ONNX models and artifacts
        """
        self.models_dir = models_dir
        
        # Load ONNX models
        print("Loading ONNX models...")
        self.cnn_session = None
        self.resnet_session = None
        
        cnn_path = os.path.join(models_dir, 'cnn_model.onnx')
        if os.path.exists(cnn_path):
            self.cnn_session = ort.InferenceSession(cnn_path)
            print(f"✓ CNN model loaded")
        
        resnet_path = os.path.join(models_dir, 'resnet_model.onnx')
        if os.path.exists(resnet_path):
            self.resnet_session = ort.InferenceSession(resnet_path)
            print(f"✓ ResNet model loaded")
        
        # Load SVM and feature extractor
        svm_path = os.path.join(models_dir, 'svm_model.pkl')
        if os.path.exists(svm_path):
            with open(svm_path, 'rb') as f:
                self.svm_model = pickle.load(f)
            print(f"✓ SVM model loaded")
        else:
            self.svm_model = None
        
        feature_path = os.path.join(models_dir, 'feature_extractor.pkl')
        if os.path.exists(feature_path):
            with open(feature_path, 'rb') as f:
                self.feature_extractor = pickle.load(f)
            print(f"✓ Feature extractor loaded")
        else:
            self.feature_extractor = None
        
        # Load fault mapping
        mapping_path = os.path.join(models_dir, 'fault_mapping.pkl')
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                self.fault_mapping = pickle.load(f)
                self.idx_to_fault = {v: k for k, v in self.fault_mapping.items()}
        else:
            # Use default mapping
            self.fault_mapping = {fault: idx for idx, fault in enumerate(self.FAULT_TYPES)}
            self.idx_to_fault = {idx: fault for fault, idx in self.fault_mapping.items()}
        
        print("✓ All models loaded successfully\n")
    
    def parse_data_file(self, filepath: str, vendor: str = None) -> pd.DataFrame:
        """
        Parse FRA data file.
        
        Args:
            filepath: Path to data file
            vendor: Vendor name (omicron/doble/megger/generic)
        
        Returns:
            Parsed DataFrame
        """
        print(f"Parsing file: {filepath}")
        
        # Simple parser without full dependency on parser module
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.csv':
            df = pd.read_csv(filepath, comment='#')
        elif ext == '.txt':
            df = pd.read_csv(filepath, sep='\t', comment='#')
        else:
            df = pd.read_csv(filepath, comment='#')
        
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Find required columns
        freq_col = None
        mag_col = None
        phase_col = None
        
        for col in df.columns:
            if 'freq' in col:
                freq_col = col
            elif 'mag' in col or 'db' in col:
                mag_col = col
            elif 'phase' in col or 'deg' in col:
                phase_col = col
        
        if not all([freq_col, mag_col, phase_col]):
            raise ValueError("Could not find required columns (frequency, magnitude, phase)")
        
        # Create standardized DataFrame
        result = pd.DataFrame({
            'frequency_hz': pd.to_numeric(df[freq_col], errors='coerce'),
            'magnitude_db': pd.to_numeric(df[mag_col], errors='coerce'),
            'phase_deg': pd.to_numeric(df[phase_col], errors='coerce')
        })
        
        result = result.dropna()
        
        print(f"✓ Parsed {len(result)} data points\n")
        
        return result
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Run fault prediction on FRA data.
        
        Args:
            df: FRA data DataFrame
        
        Returns:
            Prediction results dictionary
        """
        print("Running AI prediction...")
        
        # Prepare sequence data for CNN
        freq = df['frequency_hz'].values
        mag = df['magnitude_db'].values
        phase = df['phase_deg'].values
        
        # Normalize frequency
        freq_norm = (np.log10(freq) - np.log10(20)) / (np.log10(2e6) - np.log10(20))
        
        # Stack as 3-channel sequence
        sequence = np.stack([freq_norm, mag, phase], axis=0)
        sequence = sequence.reshape(1, 3, len(freq)).astype(np.float32)
        
        # CNN prediction
        cnn_probs = None
        if self.cnn_session:
            cnn_input = {self.cnn_session.get_inputs()[0].name: sequence}
            cnn_output = self.cnn_session.run(None, cnn_input)[0]
            cnn_probs = self._softmax(cnn_output[0])
        
        # SVM prediction (simplified - just use decision function)
        svm_score = 0.0
        if self.svm_model and self.feature_extractor:
            features = self.feature_extractor.extract_features(df)
            features_scaled = self.feature_extractor.scaler.transform(features.reshape(1, -1))
            svm_score = float(self.svm_model.decision_function(features_scaled)[0])
        
        # Ensemble prediction (simplified - mainly CNN + SVM)
        if cnn_probs is not None:
            # Weight: 0.8 CNN + 0.2 SVM contribution
            ensemble_probs = cnn_probs * 0.8
            
            # Add SVM contribution
            if svm_score > 0:
                normal_idx = self.fault_mapping.get('normal', 0)
                ensemble_probs[normal_idx] += 0.2
            else:
                # Distribute SVM anomaly weight
                fault_indices = [i for i, fault in self.idx_to_fault.items() if fault != 'normal']
                for idx in fault_indices:
                    ensemble_probs[idx] += 0.2 / len(fault_indices)
            
            # Normalize
            ensemble_probs = ensemble_probs / ensemble_probs.sum()
        else:
            raise RuntimeError("No models available for prediction")
        
        # Get prediction
        predicted_idx = int(np.argmax(ensemble_probs))
        predicted_fault = self.idx_to_fault[predicted_idx]
        confidence = float(ensemble_probs[predicted_idx])
        
        # Calculate uncertainty (Shannon entropy)
        entropy = -np.sum(ensemble_probs * np.log(ensemble_probs + 1e-10))
        max_entropy = np.log(len(self.fault_mapping))
        uncertainty = float(entropy / max_entropy)
        
        results = {
            'predicted_fault': predicted_fault,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'svm_score': svm_score,
            'probabilities': {
                self.idx_to_fault[i]: float(p) 
                for i, p in enumerate(ensemble_probs)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        print("✓ Prediction complete\n")
        
        return results
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


def print_results(results: Dict):
    """Print prediction results in formatted text."""
    print("="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print()
    print(f"Predicted Fault:  {results['predicted_fault'].replace('_', ' ').title()}")
    print(f"Confidence:       {results['confidence']:.2%}")
    print(f"Uncertainty:      {results['uncertainty']:.2f}")
    print(f"SVM Score:        {results['svm_score']:.3f}")
    print()
    print("Top 3 Fault Probabilities:")
    print("-" * 60)
    
    sorted_probs = sorted(results['probabilities'].items(), key=lambda x: x[1], reverse=True)
    for i, (fault, prob) in enumerate(sorted_probs[:3], 1):
        print(f"  {i}. {fault.replace('_', ' ').title():<30} {prob:.2%}")
    
    print()
    print("="*60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='FRA Fault Predictor - Edge Deployment CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_cli.py --file test_data.csv
  python predict_cli.py --file data.xml --vendor omicron --output result.json
  python predict_cli.py --file fra.txt --models ./my_models
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        required=True,
        help='Path to FRA data file (CSV, TXT, XML)'
    )
    
    parser.add_argument(
        '--vendor', '-v',
        choices=['omicron', 'doble', 'megger', 'generic'],
        help='Equipment vendor (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--models', '-m',
        default='models',
        help='Directory containing ONNX models (default: models/)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output JSON file path (optional)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode - minimal output'
    )
    
    args = parser.parse_args()
    
    # Header
    if not args.quiet:
        print("\n" + "="*60)
        print("FRA FAULT PREDICTOR - EDGE DEPLOYMENT")
        print("SIH 2025 PS 25190")
        print("="*60)
        print()
    
    try:
        # Initialize predictor
        predictor = EdgePredictor(models_dir=args.models)
        
        # Parse data file
        df = predictor.parse_data_file(args.file, args.vendor)
        
        # Run prediction
        results = predictor.predict(df)
        
        # Print results
        if not args.quiet:
            print_results(results)
        else:
            print(f"{results['predicted_fault']},{results['confidence']:.4f}")
        
        # Save to JSON if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            if not args.quiet:
                print(f"\n✓ Results saved to {args.output}")
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
