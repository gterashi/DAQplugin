#!/usr/bin/env python3
"""
Export DAQ PyTorch model to ONNX format.

This script converts the DAQ ResNet model from PyTorch (.pth) to ONNX format
for use in ChimeraX plugin without requiring PyTorch installation.

Usage:
    python export_onnx.py [--input <model.pth>] [--output <model.onnx>] [--verify]

Requirements:
    - PyTorch
    - ONNX
    - numpy

Note: Run this script in an environment with PyTorch installed (e.g., conda env).
      The resulting ONNX file can be used with onnxruntime in ChimeraX.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def get_default_paths():
    """Get default model paths relative to this script."""
    script_dir = Path(__file__).parent.parent
    input_path = script_dir / "DAQ" / "best_model" / "qa_model" / "Multimodel.pth"
    output_path = script_dir / "daqcolor" / "data" / "Multimodel.onnx"
    return input_path, output_path


def load_pytorch_model(ckpt_path, voxel_size=11, device="cpu"):
    """Load the DAQ PyTorch model."""
    # Import the model architecture
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from DAQ.models.resnet import resnet18 as resnet18_multi

    # Create model
    model = resnet18_multi(sample_size=voxel_size)
    model = nn.DataParallel(model)

    # Load checkpoint
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Remove DataParallel wrapper for export
    if hasattr(model, "module"):
        model = model.module

    return model.to(device)


def export_to_onnx(model, output_path, voxel_size=11, opset_version=14):
    """Export PyTorch model to ONNX format."""
    # Create dummy input: (batch, channels, depth, height, width)
    # DAQ uses 11x11x11 patches with 1 channel
    dummy_input = torch.randn(1, 1, voxel_size, voxel_size, voxel_size)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting model to: {output_path}")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Opset version: {opset_version}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["patches"],
        output_names=["aa_logits", "atom_logits", "ss_logits"],
        dynamic_axes={
            "patches": {0: "batch_size"},
            "aa_logits": {0: "batch_size"},
            "atom_logits": {0: "batch_size"},
            "ss_logits": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )

    print(f"Successfully exported to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return output_path


def verify_onnx_model(onnx_path, pytorch_model, voxel_size=11, rtol=1e-4, atol=1e-5):
    """Verify ONNX model produces same outputs as PyTorch model."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as e:
        print(f"Warning: Cannot verify ONNX model - {e}")
        print("Install onnx and onnxruntime to enable verification.")
        return False

    print("\nVerifying ONNX model output matches PyTorch...")

    # Load ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("  ONNX model structure is valid.")

    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )

    # Test with random inputs
    test_batch_sizes = [1, 4, 16]

    for batch_size in test_batch_sizes:
        # Create test input
        test_input = torch.randn(batch_size, 1, voxel_size, voxel_size, voxel_size)

        # PyTorch inference
        pytorch_model.eval()
        with torch.no_grad():
            pt_outputs = pytorch_model(test_input)

        # ONNX Runtime inference
        ort_inputs = {"patches": test_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)

        # Compare outputs
        all_close = True
        for i, (pt_out, ort_out) in enumerate(zip(pt_outputs, ort_outputs)):
            pt_np = pt_out.numpy()
            if not np.allclose(pt_np, ort_out, rtol=rtol, atol=atol):
                max_diff = np.abs(pt_np - ort_out).max()
                print(f"  Output {i} mismatch (batch={batch_size}): max_diff={max_diff}")
                all_close = False

        if all_close:
            print(f"  Batch size {batch_size}: PASSED")
        else:
            print(f"  Batch size {batch_size}: FAILED")
            return False

    print("Verification PASSED: ONNX outputs match PyTorch outputs.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export DAQ PyTorch model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    default_input, default_output = get_default_paths()

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(default_input),
        help=f"Input PyTorch model path (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(default_output),
        help=f"Output ONNX model path (default: {default_output})",
    )
    parser.add_argument(
        "--voxel-size",
        type=int,
        default=11,
        help="Voxel size for patch input (default: 11)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ONNX output matches PyTorch output",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for PyTorch model (default: cpu)",
    )

    args = parser.parse_args()

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input model not found: {input_path}")
        sys.exit(1)

    # Load PyTorch model
    model = load_pytorch_model(input_path, args.voxel_size, args.device)

    # Export to ONNX
    onnx_path = export_to_onnx(model, args.output, args.voxel_size, args.opset)

    # Verify if requested
    if args.verify:
        success = verify_onnx_model(onnx_path, model, args.voxel_size)
        if not success:
            sys.exit(1)

    print("\nDone! ONNX model ready for use in ChimeraX plugin.")


if __name__ == "__main__":
    main()
