"""
ONNX Export and Validation Utility
SIH 2025 PS 25190

Exports trained PyTorch models to ONNX format for edge deployment.
Validates that ONNX inference matches PyTorch output within tolerance (1e-4).

ONNX Runtime provides:
- Cross-platform inference (Windows, Linux, ARM)
- CPU optimization (no GPU required)
- Smaller footprint for embedded devices (Raspberry Pi)
- Faster inference (<50ms on CPU)
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import os
from typing import Tuple, Dict
import time


def export_cnn_to_onnx(
    model_path: str = 'models/cnn_model.pth',
    output_path: str = 'models/cnn_model.onnx',
    input_shape: Tuple[int, int, int] = (1, 3, 1000),
    opset_version: int = 12
) -> str:
    """
    Export 1D CNN model to ONNX format.
    
    Args:
        model_path: Path to PyTorch model weights
        output_path: Output ONNX file path
        input_shape: Model input shape (batch, channels, sequence_length)
        opset_version: ONNX opset version
    
    Returns:
        Path to exported ONNX model
    """
    from ai_ensemble import CNN1D
    
    print(f"Exporting CNN model to ONNX...")
    
    # Load PyTorch model
    model = CNN1D(num_classes=6)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"✓ CNN exported to {output_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified")
    
    return output_path


def export_resnet_to_onnx(
    model_path: str = 'models/resnet_model.pth',
    output_path: str = 'models/resnet_model.onnx',
    input_shape: Tuple[int, int, int, int] = (1, 1, 224, 224),
    opset_version: int = 12
) -> str:
    """
    Export ResNet18 model to ONNX format.
    
    Args:
        model_path: Path to PyTorch model weights
        output_path: Output ONNX file path
        input_shape: Model input shape (batch, channels, height, width)
        opset_version: ONNX opset version
    
    Returns:
        Path to exported ONNX model
    """
    from ai_ensemble import ResNetClassifier
    
    print(f"Exporting ResNet model to ONNX...")
    
    # Load PyTorch model
    model = ResNetClassifier(num_classes=6, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"✓ ResNet exported to {output_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified")
    
    return output_path


def validate_onnx_accuracy(
    pytorch_model,
    onnx_path: str,
    test_inputs: torch.Tensor,
    tolerance: float = 1e-4
) -> Dict:
    """
    Validate ONNX model accuracy against PyTorch model.
    
    Args:
        pytorch_model: PyTorch model instance
        onnx_path: Path to ONNX model
        test_inputs: Test input tensors
        tolerance: Maximum allowed difference
    
    Returns:
        Validation results dictionary
    """
    print("\nValidating ONNX model accuracy...")
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_start = time.time()
        pytorch_output = pytorch_model(test_inputs)
        pytorch_time = time.time() - pytorch_start
    
    # ONNX inference
    ort_session = ort.InferenceSession(onnx_path)
    onnx_inputs = {ort_session.get_inputs()[0].name: test_inputs.numpy()}
    
    onnx_start = time.time()
    onnx_output = ort_session.run(None, onnx_inputs)[0]
    onnx_time = time.time() - onnx_start
    
    # Compare outputs
    pytorch_np = pytorch_output.numpy()
    max_diff = np.max(np.abs(pytorch_np - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_np - onnx_output))
    
    passed = max_diff < tolerance
    
    results = {
        'passed': passed,
        'max_difference': float(max_diff),
        'mean_difference': float(mean_diff),
        'tolerance': tolerance,
        'pytorch_time_ms': pytorch_time * 1000,
        'onnx_time_ms': onnx_time * 1000,
        'speedup': pytorch_time / onnx_time if onnx_time > 0 else 0
    }
    
    print(f"  Max difference: {max_diff:.2e} (tolerance: {tolerance:.2e})")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  PyTorch inference: {results['pytorch_time_ms']:.2f} ms")
    print(f"  ONNX inference: {results['onnx_time_ms']:.2f} ms")
    print(f"  Speedup: {results['speedup']:.2f}x")
    
    if passed:
        print("✓ ONNX model accuracy validated!")
    else:
        print("✗ ONNX model accuracy check FAILED!")
    
    return results


def export_all_models(models_dir: str = 'models') -> Dict[str, str]:
    """
    Export all models to ONNX format and validate.
    
    Args:
        models_dir: Directory containing PyTorch models
    
    Returns:
        Dictionary mapping model names to ONNX paths
    """
    print("="*70)
    print("ONNX Model Export and Validation")
    print("="*70)
    print()
    
    exported = {}
    
    # Export CNN
    cnn_pytorch = os.path.join(models_dir, 'cnn_model.pth')
    cnn_onnx = os.path.join(models_dir, 'cnn_model.onnx')
    
    if os.path.exists(cnn_pytorch):
        export_cnn_to_onnx(cnn_pytorch, cnn_onnx)
        exported['cnn'] = cnn_onnx
        
        # Validate
        from ai_ensemble import CNN1D
        model = CNN1D(num_classes=6)
        model.load_state_dict(torch.load(cnn_pytorch, map_location='cpu'))
        
        test_input = torch.randn(5, 3, 1000)  # Batch of 5
        validate_onnx_accuracy(model, cnn_onnx, test_input)
    else:
        print(f"⚠ CNN model not found: {cnn_pytorch}")
    
    print()
    
    # Export ResNet
    resnet_pytorch = os.path.join(models_dir, 'resnet_model.pth')
    resnet_onnx = os.path.join(models_dir, 'resnet_model.onnx')
    
    if os.path.exists(resnet_pytorch):
        export_resnet_to_onnx(resnet_pytorch, resnet_onnx)
        exported['resnet'] = resnet_onnx
        
        # Validate
        from ai_ensemble import ResNetClassifier
        model = ResNetClassifier(num_classes=6, pretrained=False)
        model.load_state_dict(torch.load(resnet_pytorch, map_location='cpu'))
        
        test_input = torch.randn(5, 1, 224, 224)  # Batch of 5
        validate_onnx_accuracy(model, resnet_onnx, test_input)
    else:
        print(f"⚠ ResNet model not found: {resnet_pytorch}")
    
    print()
    print("="*70)
    print(f"Exported {len(exported)} models to ONNX format")
    print("="*70)
    
    return exported


def test_onnx_inference_speed(
    onnx_path: str,
    input_shape: Tuple,
    num_iterations: int = 100
) -> Dict:
    """
    Test ONNX inference speed.
    
    Args:
        onnx_path: Path to ONNX model
        input_shape: Input tensor shape
        num_iterations: Number of inference iterations
    
    Returns:
        Timing statistics
    """
    print(f"\nTesting ONNX inference speed ({num_iterations} iterations)...")
    
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    
    # Warm-up
    dummy_input = {input_name: np.random.randn(*input_shape).astype(np.float32)}
    for _ in range(10):
        ort_session.run(None, dummy_input)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        dummy_input = {input_name: np.random.randn(*input_shape).astype(np.float32)}
        start = time.time()
        ort_session.run(None, dummy_input)
        times.append(time.time() - start)
    
    times = np.array(times) * 1000  # Convert to ms
    
    results = {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99))
    }
    
    print(f"  Mean: {results['mean_ms']:.2f} ms")
    print(f"  Std: {results['std_ms']:.2f} ms")
    print(f"  Min: {results['min_ms']:.2f} ms")
    print(f"  Max: {results['max_ms']:.2f} ms")
    print(f"  P95: {results['p95_ms']:.2f} ms")
    
    if results['p95_ms'] < 50:
        print("✓ Inference time <50ms requirement MET!")
    else:
        print("⚠ Inference time >50ms (consider model optimization)")
    
    return results


if __name__ == '__main__':
    # Export all models
    exported_models = export_all_models()
    
    # Test inference speed
    if 'cnn' in exported_models:
        print("\n" + "="*70)
        print("CNN Inference Speed Test")
        print("="*70)
        test_onnx_inference_speed(exported_models['cnn'], (1, 3, 1000))
    
    if 'resnet' in exported_models:
        print("\n" + "="*70)
        print("ResNet Inference Speed Test")
        print("="*70)
        test_onnx_inference_speed(exported_models['resnet'], (1, 1, 224, 224))
    
    print("\n✓ ONNX export and validation complete!")
