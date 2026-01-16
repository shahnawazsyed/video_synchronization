"""
utils.py
---------
Helper utilities for file handling, plotting, and logging.
"""

import os
import logging
import numpy as np
import subprocess
from scipy.io import wavfile
from typing import Dict, List, Tuple

def ffmpeg_exists() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def detect_outliers(pairwise: Dict[Tuple[str, str], Tuple[float, float]], 
                   optimized: Dict[str, float], 
                   threshold: float = 0.5) -> List[Tuple[str, str, float, float, float]]:
    """
    Find pairwise estimates that disagree strongly with optimized solution.
    
    Args:
        pairwise: Dict of (fileA, fileB) -> (offset, confidence)
        optimized: Dict of filename -> optimized offset
        threshold: Flag pairs with error > this many seconds (default 0.5s)
    
    Returns:
        List of (fileA, fileB, measured_offset, expected_offset, error) tuples
    """
    outliers = []
    print(f"\nChecking for outliers (threshold={threshold}s)...")
    
    for (file_a, file_b), (d_measured, conf) in pairwise.items():
        d_expected = optimized[file_b] - optimized[file_a]
        error = abs(d_measured - d_expected)
        
        if error > threshold:
            outliers.append((file_a, file_b, d_measured, d_expected, error))
            print(f"  ⚠️  {file_a} <-> {file_b}:")
            print(f"      measured={d_measured:.3f}s, expected={d_expected:.3f}s, "
                  f"error={error:.3f}s, conf={conf:.3f}")
    
    if not outliers:
        print("  ✓ No outliers detected")
    else:
        print(f"\n  Found {len(outliers)} outlier pair(s)")
    
    return outliers

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def setup_logger(name: str = "sync", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a simple logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p

def load_audio(path: str) -> Tuple[np.ndarray, int]:
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2**31
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr

def apply_offset(signal: np.ndarray, offset_sec: float, sample_rate: int) -> np.ndarray:
    """
    Apply time offset to a signal by shifting samples.
    
    Args:
        signal: Input audio signal
        offset_sec: Offset in seconds (positive = shift right, negative = shift left)
        sample_rate: Sample rate in Hz
        
    Returns:
        Shifted signal (same length as input, padded with zeros)
    """
    offset_samples = int(round(offset_sec * sample_rate))
    
    if offset_samples == 0:
        return signal.copy()
    elif offset_samples > 0:
        # Shift right (pad at beginning)
        return np.pad(signal, (offset_samples, 0), mode='constant')[:len(signal)]
    else:
        # Shift left (pad at end)
        return np.pad(signal, (0, -offset_samples), mode='constant')[:len(signal)]
