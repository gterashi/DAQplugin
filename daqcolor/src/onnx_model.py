# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
ONNX Runtime inference wrapper for DAQ model.

This module provides GPU-aware inference using ONNX Runtime,
with automatic fallback to CPU when GPU is not available.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Compute softmax along specified axis."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def get_execution_providers(device: str = "auto") -> List[str]:
    """
    Get ONNX execution providers based on device setting.
    
    Parameters
    ----------
    device : str
        Device selection: "auto", "cpu", or "cuda"
        
    Returns
    -------
    list of str
        List of execution provider names in priority order
        
    Raises
    ------
    RuntimeError
        If device="cuda" but CUDA is not available
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        raise ImportError(
            "onnxruntime is not installed.\n"
            "Install with:\n"
            "  pip install onnxruntime         (CPU only)\n"
            "  pip install onnxruntime-gpu     (GPU support)\n"
        )
    
    if device == "cpu":
        return ["CPUExecutionProvider"]
    elif device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDA execution provider not available.\n\n"
                "To enable GPU support:\n"
                "  1. Ensure NVIDIA GPU and CUDA are installed (check with: nvidia-smi)\n"
                "  2. Uninstall CPU-only version: pip uninstall onnxruntime\n"
                "  3. Install GPU version: pip install onnxruntime-gpu\n"
                "  4. Verify with: daqscore info\n\n"
                "Note: onnxruntime-gpu requires CUDA 11.x or 12.x\n"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:  # auto
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers


def get_device_info() -> dict:
    """
    Get information about available ONNX execution providers.
    
    Returns
    -------
    dict
        Dictionary with device availability information
    """
    info = {
        "onnxruntime_installed": False,
        "version": None,
        "available_providers": [],
        "cuda_available": False,
        "recommended_device": "cpu",
    }
    
    try:
        import onnxruntime as ort
        info["onnxruntime_installed"] = True
        info["version"] = ort.__version__
        info["available_providers"] = ort.get_available_providers()
        info["cuda_available"] = "CUDAExecutionProvider" in info["available_providers"]
        info["recommended_device"] = "cuda" if info["cuda_available"] else "cpu"
    except ImportError:
        pass
    
    return info


class DAQOnnxModel:
    """
    ONNX Runtime wrapper for DAQ model inference.
    
    This class handles loading the ONNX model and running batched inference
    with automatic GPU/CPU selection.
    
    Parameters
    ----------
    model_path : str or Path
        Path to the ONNX model file
    device : str, optional
        Device selection: "auto" (default), "cpu", or "cuda"
        
    Attributes
    ----------
    device : str
        The actual device being used ("cpu" or "cuda")
    session : onnxruntime.InferenceSession
        The ONNX Runtime inference session
    """
    
    def __init__(self, model_path: str, device: str = "auto", verbose: bool = False):
        import onnxruntime as ort
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        
        # Get execution providers
        self.providers = get_execution_providers(device)
        
        # Create inference session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if not verbose:
            # Suppress ONNX Runtime warnings
            sess_options.log_severity_level = 3
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=self.providers,
        )
        
        # Determine actual device being used
        active_provider = self.session.get_providers()[0]
        self.device = "cuda" if "CUDA" in active_provider else "cpu"
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]
    
    def __repr__(self) -> str:
        return (
            f"DAQOnnxModel(model={self.model_path.name}, "
            f"device={self.device}, providers={self.providers})"
        )
    
    def predict(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on a batch of patches.
        
        Parameters
        ----------
        patches : np.ndarray
            Input patches with shape (N, 1, 11, 11, 11), float32
            
        Returns
        -------
        tuple of np.ndarray
            (aa_probs, atom_probs, ss_probs) - softmax probabilities
            aa_probs: (N, 20) amino acid probabilities
            atom_probs: (N, 6) atom type probabilities
            ss_probs: (N, 3) secondary structure probabilities
        """
        # Ensure correct dtype and shape
        if patches.dtype != np.float32:
            patches = patches.astype(np.float32)
        
        if patches.ndim == 4:
            # Add channel dimension: (N, D, H, W) -> (N, 1, D, H, W)
            patches = patches[:, np.newaxis, :, :, :]
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: patches})
        
        # Apply softmax to logits
        aa_probs = softmax(outputs[0], axis=1)
        atom_probs = softmax(outputs[1], axis=1)
        ss_probs = softmax(outputs[2], axis=1)
        
        return aa_probs, atom_probs, ss_probs
    
    def predict_batched(
        self,
        patches: np.ndarray,
        batch_size: int = 512,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run batched inference on patches.
        
        Parameters
        ----------
        patches : np.ndarray
            Input patches with shape (N, 1, 11, 11, 11)
        batch_size : int
            Batch size for inference (default: 512)
        progress_callback : callable, optional
            Function called with (current, total) for progress updates
            
        Returns
        -------
        tuple of np.ndarray
            (aa_probs, atom_probs, ss_probs) concatenated for all patches
        """
        N = patches.shape[0]
        aa_all, atom_all, ss_all = [], [], []
        
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch = patches[i:end]
            
            aa, atom, ss = self.predict(batch)
            aa_all.append(aa)
            atom_all.append(atom)
            ss_all.append(ss)
            
            if progress_callback is not None:
                progress_callback(end, N)
        
        return (
            np.concatenate(aa_all, axis=0),
            np.concatenate(atom_all, axis=0),
            np.concatenate(ss_all, axis=0),
        )


def get_model_path() -> Path:
    """
    Get the path to the ONNX model, checking multiple locations.
    
    Search order:
    1. Plugin data/ directory (bundled with plugin)
    2. User's home directory ~/.chimerax/daq_model/Multimodel.onnx
    3. Environment variable DAQ_MODEL_PATH
    
    Returns
    -------
    Path
        Path to Multimodel.onnx
        
    Raises
    ------
    FileNotFoundError
        If the model cannot be found in any location
    """
    import os
    
    # Possible model locations
    candidates = []
    
    # 1. Plugin data/ directory
    module_dir = Path(__file__).parent.parent
    candidates.append(module_dir / "data" / "Multimodel.onnx")
    
    # 2. User's ChimeraX config directory
    home = Path.home()
    candidates.append(home / ".chimerax" / "daq_model" / "Multimodel.onnx")
    
    # 3. Environment variable
    env_path = os.environ.get("DAQ_MODEL_PATH")
    if env_path:
        candidates.insert(0, Path(env_path))
    
    # Return first existing path
    for path in candidates:
        if path.exists():
            return path
    
    # Return the default path even if it doesn't exist
    # (caller will handle the error with a helpful message)
    return candidates[0]


def load_model(model_path: Optional[str] = None, device: str = "auto", verbose: bool = False) -> DAQOnnxModel:
    """
    Load the DAQ ONNX model.
    
    Parameters
    ----------
    model_path : str, optional
        Path to ONNX model. If None, uses bundled model.
    device : str
        Device selection: "auto", "cpu", or "cuda"
    verbose : bool
        If True, show detailed ONNX Runtime logs
        
    Returns
    -------
    DAQOnnxModel
        Loaded model ready for inference
        
    Raises
    ------
    FileNotFoundError
        If the model file cannot be found
    """
    if model_path is None:
        model_path = get_model_path()
    
    model_path = Path(model_path)
    if not model_path.exists():
        home_path = Path.home() / ".chimerax" / "daq_model" / "Multimodel.onnx"
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}\n\n"
            f"The DAQ score computation requires the ONNX model file.\n"
            f"Please install it by either:\n"
            f"  1. Copying Multimodel.onnx to: {model_path}\n"
            f"  2. Or copying to: {home_path}\n"
            f"  3. Or setting DAQ_MODEL_PATH environment variable\n\n"
            f"Generate the model using: python scripts/export_onnx.py"
        )
    
    return DAQOnnxModel(model_path, device=device, verbose=verbose)
