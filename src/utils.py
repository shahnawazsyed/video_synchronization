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
from typing import Tuple

def ffmpeg_exists() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False



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