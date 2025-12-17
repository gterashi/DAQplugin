# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
ONNX Runtime inference wrapper for DAQ model.

This module provides CPU inference using ONNX Runtime.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Compute softmax along specified axis."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class DAQOnnxModel:
    """
    ONNX Runtime wrapper for DAQ model inference.
    
    This class handles loading the ONNX model and running batched inference.
    
    Parameters
    ----------
    model_path : str or Path
        Path to the ONNX model file
        
    Attributes
    ----------
    session : onnxruntime.InferenceSession
        The ONNX Runtime inference session
    """
    
    def __init__(self, model_path: str, verbose: bool = False):
        import onnxruntime as ort
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        
        # Create inference session with CPU provider
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if not verbose:
            # Suppress ONNX Runtime warnings
            sess_options.log_severity_level = 3
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]
    
    def __repr__(self) -> str:
        return f"DAQOnnxModel(model={self.model_path.name})"
    
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


def load_model(model_path: Optional[str] = None, verbose: bool = False) -> DAQOnnxModel:
    """
    Load the DAQ ONNX model.
    
    Parameters
    ----------
    model_path : str, optional
        Path to ONNX model. If None, uses bundled model.
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
    
    return DAQOnnxModel(model_path, verbose=verbose)
