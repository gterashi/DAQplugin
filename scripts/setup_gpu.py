#!/usr/bin/env python3
"""
Helper script to set up GPU support for DAQ Score computation.

This script helps users:
1. Check if NVIDIA GPU and CUDA are available
2. Install onnxruntime-gpu if needed
3. Verify the installation

Usage:
    python scripts/setup_gpu.py [--check-only]
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, capture=True, check=False):
    """Run a shell command and return the result."""
    try:
        if capture:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=True, check=check)
            return result.returncode == 0, "", ""
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def check_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    print("🔍 Checking for NVIDIA GPU...")
    success, stdout, stderr = run_command("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
    
    if success and stdout.strip():
        print("✅ NVIDIA GPU detected:")
        for line in stdout.strip().split('\n'):
            print(f"   {line}")
        return True
    else:
        print("❌ NVIDIA GPU not detected")
        print("   Make sure NVIDIA drivers are installed")
        print("   Try running: nvidia-smi")
        return False


def check_cuda():
    """Check if CUDA is available."""
    print("\n🔍 Checking CUDA installation...")
    success, stdout, stderr = run_command("nvcc --version")
    
    if success:
        # Extract CUDA version
        for line in stdout.split('\n'):
            if 'release' in line.lower():
                print(f"✅ CUDA detected: {line.strip()}")
                return True
    
    print("⚠️  CUDA compiler (nvcc) not found")
    print("   CUDA may still work if runtime libraries are installed")
    return None  # Uncertain


def check_current_onnxruntime():
    """Check which version of onnxruntime is installed."""
    print("\n🔍 Checking current ONNX Runtime installation...")
    
    try:
        import onnxruntime as ort
        print(f"📦 onnxruntime version: {ort.__version__}")
        
        providers = ort.get_available_providers()
        print(f"   Available providers: {', '.join(providers)}")
        
        if "CUDAExecutionProvider" in providers:
            print("✅ GPU support is already enabled!")
            return "gpu"
        else:
            print("💻 Currently using CPU-only version")
            return "cpu"
    except ImportError:
        print("❌ onnxruntime is not installed")
        return None


def install_onnxruntime_gpu():
    """Install onnxruntime-gpu."""
    print("\n🚀 Installing onnxruntime-gpu...")
    print("   This will uninstall onnxruntime and install onnxruntime-gpu")
    
    # Uninstall CPU version
    print("\n   Step 1: Uninstalling onnxruntime...")
    success, _, _ = run_command("pip uninstall -y onnxruntime", capture=False)
    
    if not success:
        print("⚠️  Warning: Could not uninstall onnxruntime (may not be installed)")
    
    # Install GPU version
    print("\n   Step 2: Installing onnxruntime-gpu...")
    success, _, _ = run_command("pip install onnxruntime-gpu", capture=False)
    
    if success:
        print("\n✅ onnxruntime-gpu installation completed!")
        return True
    else:
        print("\n❌ Failed to install onnxruntime-gpu")
        return False


def verify_installation():
    """Verify that GPU support is working."""
    print("\n🔍 Verifying GPU support...")
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        
        if "CUDAExecutionProvider" in providers:
            print("✅ GPU support is enabled!")
            print(f"   Available providers: {', '.join(providers)}")
            print("\n💡 You can now use GPU acceleration:")
            print("   daqscore compute #1 device cuda")
            return True
        else:
            print("❌ GPU support is not available")
            print(f"   Available providers: {', '.join(providers)}")
            print("\n   Possible issues:")
            print("   - CUDA libraries not found")
            print("   - Incompatible CUDA version")
            print("   - Driver issues")
            return False
    except ImportError:
        print("❌ onnxruntime is not installed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Set up GPU support for DAQ Score computation"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check system status, don't install anything"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("DAQ Score GPU Setup Helper")
    print("=" * 70)
    
    # Step 1: Check NVIDIA GPU
    has_gpu = check_nvidia_gpu()
    
    if not has_gpu:
        print("\n❌ Cannot proceed without NVIDIA GPU")
        print("\n   If you have an NVIDIA GPU:")
        print("   1. Install NVIDIA drivers")
        print("   2. Reboot your system")
        print("   3. Run this script again")
        sys.exit(1)
    
    # Step 2: Check CUDA (optional)
    check_cuda()
    
    # Step 3: Check current onnxruntime
    current_status = check_current_onnxruntime()
    
    if current_status == "gpu":
        print("\n✨ GPU support is already configured!")
        print("\n   Usage:")
        print("   daqscore compute #1 device cuda")
        sys.exit(0)
    
    if args.check_only:
        print("\n💡 To enable GPU support, run without --check-only flag")
        sys.exit(0)
    
    # Step 4: Ask for confirmation
    if current_status == "cpu":
        print("\n⚠️  This will replace onnxruntime with onnxruntime-gpu")
    
    response = input("\nProceed with installation? [y/N]: ")
    if response.lower() != 'y':
        print("Installation cancelled")
        sys.exit(0)
    
    # Step 5: Install
    if install_onnxruntime_gpu():
        # Step 6: Verify
        verify_installation()
    else:
        print("\n❌ Installation failed")
        print("\n   Manual installation:")
        print("   pip uninstall onnxruntime")
        print("   pip install onnxruntime-gpu")
        sys.exit(1)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
