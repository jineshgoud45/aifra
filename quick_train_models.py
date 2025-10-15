#!/usr/bin/env python3
"""
Quick Model Training Script
Generates synthetic data and trains AI models for FRA fault detection
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("FRA AI Model Training - Quick Setup")
print("SIH 2025 PS 25190")
print("=" * 70)
print()

# Import modules
print("Importing modules...")
from simulator import generate_synthetic_dataset
from ai_ensemble import train_ensemble_pipeline

# Step 1: Generate synthetic training data
print("\n" + "=" * 70)
print("STEP 1: Generating Synthetic Training Data")
print("=" * 70)
print("This will create 5000 samples (4000 train, 1000 test)")
print("Fault types: normal, axial_deformation, radial_deformation,")
print("             interturn_short, core_grounding, tapchanger_fault")
print()

try:
    # Create necessary directories with proper checks
    print("Checking/creating directories...")
    
    synthetic_dir = Path('synthetic_data')
    models_dir = Path('models')
    
    # Create synthetic_data directory if it doesn't exist
    if not synthetic_dir.exists():
        print(f"  Creating {synthetic_dir}...")
        synthetic_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"  {synthetic_dir} already exists")
    
    # Create models directory if it doesn't exist
    if not models_dir.exists():
        print(f"  Creating {models_dir}...")
        models_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"  {models_dir} already exists")
    
    print()
    
    train_df, test_df = generate_synthetic_dataset(
        n_samples=5000,
        output_dir='synthetic_data',
        export_formats=False,  # Skip vendor format export for speed
        visualize=False  # Skip visualization for speed
    )
    print(f"Training data: {len(train_df)} samples")
    print(f"Testing data: {len(test_df)} samples")
except Exception as e:
    print(f"Error generating data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Train AI ensemble models
print("\n" + "=" * 70)
print("STEP 2: Training AI Models")
print("=" * 70)
print("This will train 3 models:")
print("  1. 1D CNN (50 epochs)")
print("  2. ResNet18 (30 epochs)")
print("  3. One-Class SVM")
print()
print("This may take 10-30 minutes depending on your hardware...")
print()

try:
    # Verify models directory exists before training
    models_dir = Path('models')
    if not models_dir.exists():
        print(f"Creating {models_dir} directory...")
        models_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the ensemble
    ensemble = train_ensemble_pipeline(
        train_df=train_df,
        test_df=test_df,
        num_epochs_cnn=50,
        num_epochs_resnet=30,
        batch_size=32,
        device=None,  # Auto-detect (cuda if available, else cpu)
        save_dir='models'
    )
    
    print("\n" + "=" * 70)
    print("SUCCESS: All models trained and saved!")
    print("=" * 70)
    print()
    print("Model files saved in ./models/:")
    print("  - cnn_model.pth")
    print("  - resnet_model.pth")
    print("  - svm_model.pkl")
    print("  - feature_extractor.pkl")
    print("  - fault_mapping.pkl")
    print("  - confusion_matrix.png")
    print()
    print("You can now run the Streamlit app with AI inference!")
    print("   Run: streamlit run app.py")
    print()
    
except Exception as e:
    print(f"\nError during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
