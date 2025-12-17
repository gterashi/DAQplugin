#!/usr/bin/env python3
"""
Test ONNX inference pipeline without ChimeraX.

This script tests the core computation components (excluding ChimeraX-specific
volume resampling) to verify the ONNX model and inference work correctly.
"""

import sys
import numpy as np
from pathlib import Path

# Add the src directory directly to avoid ChimeraX imports from __init__.py
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir / "daqcolor" / "src"))


def test_onnx_model():
    """Test ONNX model loading and inference."""
    print("=" * 60)
    print("Testing ONNX Model Inference")
    print("=" * 60)
    
    from onnx_model import DAQOnnxModel, get_model_path, get_device_info
    
    # Check device info
    print("\nDevice Information:")
    info = get_device_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Load model
    model_path = get_model_path()
    print(f"\nLoading model from: {model_path}")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run: python scripts/export_onnx.py")
        return False
    
    model = DAQOnnxModel(str(model_path), device="cpu")
    print(f"Model loaded: {model}")
    print(f"  Active device: {model.device}")
    print(f"  Input name: {model.input_name}")
    print(f"  Output names: {model.output_names}")
    
    # Test inference with random data
    print("\nTesting inference with random patches...")
    n_patches = 100
    patches = np.random.randn(n_patches, 1, 11, 11, 11).astype(np.float32)
    
    aa_probs, atom_probs, ss_probs = model.predict(patches)
    
    print(f"  Input shape: {patches.shape}")
    print(f"  AA probs shape: {aa_probs.shape}")
    print(f"  Atom probs shape: {atom_probs.shape}")
    print(f"  SS probs shape: {ss_probs.shape}")
    
    # Verify outputs are valid probabilities
    assert aa_probs.shape == (n_patches, 20), f"Expected (100, 20), got {aa_probs.shape}"
    assert atom_probs.shape == (n_patches, 6), f"Expected (100, 6), got {atom_probs.shape}"
    assert ss_probs.shape == (n_patches, 3), f"Expected (100, 3), got {ss_probs.shape}"
    
    # Check probabilities sum to ~1
    aa_sum = aa_probs.sum(axis=1)
    atom_sum = atom_probs.sum(axis=1)
    ss_sum = ss_probs.sum(axis=1)
    
    assert np.allclose(aa_sum, 1.0, atol=1e-5), f"AA probs don't sum to 1: {aa_sum[:5]}"
    assert np.allclose(atom_sum, 1.0, atol=1e-5), f"Atom probs don't sum to 1: {atom_sum[:5]}"
    assert np.allclose(ss_sum, 1.0, atol=1e-5), f"SS probs don't sum to 1: {ss_sum[:5]}"
    
    print("  Probability sums verified!")
    
    # Test batched inference
    print("\nTesting batched inference...")
    n_patches = 1000
    patches = np.random.randn(n_patches, 1, 11, 11, 11).astype(np.float32)
    
    aa_probs, atom_probs, ss_probs = model.predict_batched(patches, batch_size=256)
    
    assert aa_probs.shape == (n_patches, 20), f"Expected ({n_patches}, 20), got {aa_probs.shape}"
    print(f"  Batched inference successful for {n_patches} patches")
    
    print("\n✓ ONNX model tests PASSED")
    return True


def test_compute_functions():
    """Test compute utility functions."""
    print("\n" + "=" * 60)
    print("Testing Compute Functions")
    print("=" * 60)
    
    from compute import (
        find_contour_cutoff,
        normalize_volume,
        extract_threshold_points,
        extract_patches,
        compute_log_ratio_scores,
    )
    
    # Create dummy volume data
    print("\nTesting with synthetic volume data...")
    vol = np.random.rand(50, 50, 50).astype(np.float32) * 0.5
    # Add some high-density region
    vol[20:30, 20:30, 20:30] = 0.8
    
    origin = (0.0, 0.0, 0.0)
    step = (1.0, 1.0, 1.0)
    
    # Test contour cutoff
    cutoff = find_contour_cutoff(vol, c=0.95)
    print(f"  Contour cutoff (c=0.95): {cutoff:.4f}")
    
    # Test normalization
    vol_norm = normalize_volume(vol)
    print(f"  Normalized volume range: [{vol_norm.min():.4f}, {vol_norm.max():.4f}]")
    assert vol_norm.min() >= 0.0 and vol_norm.max() <= 1.0
    
    # Test point extraction
    points = extract_threshold_points(vol, origin, step, contour=0.5, stride=2)
    print(f"  Extracted points: {points.shape[0]} (above contour 0.5)")
    assert points.shape[1] == 3
    
    if points.shape[0] == 0:
        print("  Warning: No points above threshold, adjusting...")
        points = extract_threshold_points(vol, origin, step, contour=0.3, stride=2)
        print(f"  Extracted points: {points.shape[0]} (above contour 0.3)")
    
    # Test patch extraction
    if points.shape[0] > 0:
        patches = extract_patches(vol_norm, points[:min(100, len(points))], origin, step, patch_size=11)
        print(f"  Extracted patches shape: {patches.shape}")
        assert patches.shape[1] == 1  # channel
        assert patches.shape[2] == patches.shape[3] == patches.shape[4] == 11
    
    # Test log-ratio score computation
    n = 50
    points_test = np.random.rand(n, 3).astype(np.float32) * 10
    aa_probs = np.random.rand(n, 20).astype(np.float32)
    aa_probs /= aa_probs.sum(axis=1, keepdims=True)
    atom_probs = np.random.rand(n, 6).astype(np.float32)
    atom_probs /= atom_probs.sum(axis=1, keepdims=True)
    ss_probs = np.random.rand(n, 3).astype(np.float32)
    ss_probs /= ss_probs.sum(axis=1, keepdims=True)
    
    scores = compute_log_ratio_scores(points_test, aa_probs, atom_probs, ss_probs)
    print(f"  Log-ratio scores shape: {scores.shape}")
    assert scores.shape == (n, 32), f"Expected ({n}, 32), got {scores.shape}"
    
    print("\n✓ Compute function tests PASSED")
    return True


def test_with_example_mrc():
    """Test with actual example MRC file (without ChimeraX)."""
    print("\n" + "=" * 60)
    print("Testing with Example MRC File")
    print("=" * 60)
    
    try:
        import mrcfile
    except ImportError:
        print("  mrcfile not installed, skipping MRC test")
        return True
    
    from onnx_model import DAQOnnxModel, get_model_path
    from compute import (
        normalize_volume,
        extract_threshold_points,
        extract_patches,
        compute_log_ratio_scores,
    )
    
    # Find example MRC file
    example_mrc = Path(__file__).parent.parent / "DAQ" / "example" / "2566_3J6B_9.mrc"
    
    if not example_mrc.exists():
        print(f"  Example MRC not found at {example_mrc}, skipping")
        return True
    
    print(f"\nLoading MRC file: {example_mrc}")
    
    with mrcfile.open(str(example_mrc), permissive=True) as mrc:
        vol = mrc.data.astype(np.float32)
        voxel_size = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z])
        origin = np.array([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z])
    
    print(f"  Volume shape: {vol.shape}")
    print(f"  Voxel size: {voxel_size}")
    print(f"  Origin: {origin}")
    
    # Normalize
    vol_norm = normalize_volume(vol)
    print(f"  Normalized range: [{vol_norm.min():.4f}, {vol_norm.max():.4f}]")
    
    # Extract points
    step = tuple(voxel_size)
    origin_tuple = tuple(origin)
    points = extract_threshold_points(vol, origin_tuple, step, contour=0.0, stride=2, max_points=10000)
    print(f"  Extracted {points.shape[0]} points")
    
    if points.shape[0] == 0:
        print("  Warning: No points found, check contour threshold")
        return True
    
    # Extract patches
    n_sample = min(500, points.shape[0])
    patches = extract_patches(vol_norm, points[:n_sample], origin_tuple, step, patch_size=11)
    print(f"  Extracted {patches.shape[0]} patches")
    
    # Run inference
    model_path = get_model_path()
    if not model_path.exists():
        print(f"  ONNX model not found, skipping inference")
        return True
    
    print(f"\nRunning ONNX inference on {n_sample} patches...")
    model = DAQOnnxModel(str(model_path), device="cpu")
    aa_probs, atom_probs, ss_probs = model.predict_batched(patches, batch_size=256)
    
    print(f"  AA probs: mean={aa_probs.mean():.4f}, std={aa_probs.std():.4f}")
    print(f"  Atom probs: mean={atom_probs.mean():.4f}, std={atom_probs.std():.4f}")
    print(f"  SS probs: mean={ss_probs.mean():.4f}, std={ss_probs.std():.4f}")
    
    # Compute scores
    scores = compute_log_ratio_scores(points[:n_sample], aa_probs, atom_probs, ss_probs)
    print(f"\n  Output scores shape: {scores.shape}")
    
    # Check score statistics
    aa_scores = scores[:, 3:23]
    print(f"  AA log-ratio scores: mean={aa_scores.mean():.4f}, std={aa_scores.std():.4f}")
    
    print("\n✓ Example MRC test PASSED")
    return True


def main():
    print("\n" + "=" * 60)
    print("DAQ ONNX Integration Tests")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_onnx_model()
    all_passed &= test_compute_functions()
    all_passed &= test_with_example_mrc()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
