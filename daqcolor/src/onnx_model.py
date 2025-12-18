# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
ONNX Runtime inference wrapper for DAQ model.

This module provides CPU inference using ONNX Runtime.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

# URL to download the model file
MODEL_URL = "https://huggingface.co/zhtronics/DAQscore/resolve/main/Multimodel.onnx"
MODEL_FILENAME = "Multimodel.onnx"


def download_model(dest_path: Path, url: str = MODEL_URL) -> bool:
    """
    Download the ONNX model file.
    
    Parameters
    ----------
    dest_path : Path
        Destination path for the model file
    url : str
        URL to download from
        
    Returns
    -------
    bool
        True if download successful, False otherwise
    """
    import urllib.request
    import sys
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading DAQ model from {url}...")
    print(f"Destination: {dest_path}")
    
    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, str(dest_path), reporthook=report_progress)
        print("\nDownload complete!")
        return True
        
    except Exception as e:
        print(f"\nDownload failed: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Remove partial download
        return False


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


def get_model_path(auto_download: bool = True) -> Path:
    """
    Get the path to the ONNX model, checking multiple locations.
    If not found and auto_download is True, downloads to user directory.
    
    Search order:
    1. Environment variable DAQ_MODEL_PATH
    2. Plugin data/ directory (installed package layout)
    3. Plugin data/ directory (development layout)
    4. User's home directory ~/.chimerax/daq_model/Multimodel.onnx
    
    Parameters
    ----------
    auto_download : bool
        If True, automatically download the model if not found
    
    Returns
    -------
    Path
        Path to Multimodel.onnx
    """
    import os
    
    # Possible model locations
    candidates = []
    
    # 1. Environment variable (highest priority)
    env_path = os.environ.get("DAQ_MODEL_PATH")
    if env_path:
        candidates.append(Path(env_path))
    
    # 2. Installed package layout: data/ is sibling to module
    # When installed: chimerax/daqcolor/onnx_model.py -> chimerax/daqcolor/data/
    module_dir = Path(__file__).parent
    candidates.append(module_dir / "data" / MODEL_FILENAME)
    
    # 3. Development layout: src/ and data/ are siblings under daqcolor/
    # In dev: daqcolor/src/onnx_model.py -> daqcolor/data/
    candidates.append(module_dir.parent / "data" / MODEL_FILENAME)
    
    # 4. User's ChimeraX config directory (also download destination)
    home = Path.home()
    user_model_path = home / ".chimerax" / "daq_model" / MODEL_FILENAME
    candidates.append(user_model_path)
    
    # Return first existing path
    for path in candidates:
        if path.exists():
            return path
    
    # Model not found - try to download if enabled
    if auto_download:
        print(f"DAQ model not found in any of the expected locations.")
        print(f"Attempting to download...")
        if download_model(user_model_path):
            return user_model_path
    
    # Return the user path for error message
    return user_model_path


def load_model(model_path: Optional[str] = None, verbose: bool = False) -> DAQOnnxModel:
    """
    Load the DAQ ONNX model.
    
    If model is not found, attempts to download it automatically.
    
    Parameters
    ----------
    model_path : str, optional
        Path to ONNX model. If None, uses bundled model or downloads.
    verbose : bool
        If True, show detailed ONNX Runtime logs
        
    Returns
    -------
    DAQOnnxModel
        Loaded model ready for inference
        
    Raises
    ------
    FileNotFoundError
        If the model file cannot be found or downloaded
    """
    if model_path is None:
        model_path = get_model_path(auto_download=True)
    
    model_path = Path(model_path)
    if not model_path.exists():
        home_path = Path.home() / ".chimerax" / "daq_model" / MODEL_FILENAME
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}\n\n"
            f"The DAQ score computation requires the ONNX model file.\n"
            f"Automatic download failed. Please install manually:\n"
            f"  1. Download from: {MODEL_URL}\n"
            f"  2. Save to: {home_path}\n"
            f"  3. Or set DAQ_MODEL_PATH environment variable"
        )
    
    return DAQOnnxModel(model_path, verbose=verbose)
