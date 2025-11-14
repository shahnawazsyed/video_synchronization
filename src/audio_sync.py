"""
audio_sync.py
--------------
FFT + GCC-PHAT implementation to estimate inter-stream time offsets.

Functions:
- compute_gcc_phat: core FFT + PHAT cross-correlation returning offset + confidence
- estimate_offsets_gccphat: load audio files in directory and compute offsets vs reference
"""
import os
import numpy as np
from typing import Dict, Tuple, List
from scipy.signal import fftconvolve
from scipy.fft import fft, ifft

from .utils import next_pow2, load_audio


def compute_gcc_phat(sig_a: np.ndarray, sig_b: np.ndarray, fs: int) -> Tuple[float, float]:
    """
    Compute time offset between two signals using GCC-PHAT.
    Returns:
        (offset_seconds, confidence_score)
    offset_seconds is the amount to add to sig_b timestamps to align to sig_a.
    """
    a = sig_a - np.mean(sig_a)
    b = sig_b - np.mean(sig_b)

    max_len = max(len(a), len(b))
    n = next_pow2(2 * max_len)

    A = fft(a, n=n)
    B = fft(b, n=n)
    R = A * np.conj(B)
    denom = np.abs(R)
    denom[denom < 1e-8] = 1e-8
    R_phat = R / denom

    cc = np.real(ifft(R_phat))
    cc = np.concatenate((cc[-(n//2):], cc[:n//2]))

    lag_idx = np.argmax(np.abs(cc))
    lags = np.arange(-n//2, n//2)
    lag = lags[lag_idx]
    offset_seconds = lag / float(fs)

    peak = np.abs(cc[lag_idx])
    window = int(0.01 * fs)
    exclude_start = max(0, lag_idx - window)
    exclude_end = min(len(cc), lag_idx + window)
    mag = np.abs(cc)
    if exclude_end <= exclude_start:
        noise_floor = np.mean(mag)
    else:
        noise_vals = np.concatenate((mag[:exclude_start], mag[exclude_end:]))
        noise_floor = np.mean(noise_vals) if noise_vals.size > 0 else np.mean(mag)
    if noise_floor < 1e-8:
        confidence = 1.0
    else:
        confidence = float(peak / (noise_floor + 1e-8))
    confidence = confidence / (confidence + 1.0)

    return offset_seconds, confidence

def estimate_offsets_gccphat(audio_dir: str) -> Dict[str, float]:
    """
    Compute offsets (seconds) for all WAV files in audio_dir relative to the first file.
    Returns mapping { filename.wav : offset_seconds } where offset_seconds is how much
    to add to that file's timestamps to align with the reference.
    """
    wavs = sorted([f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")])
    if not wavs:
        raise FileNotFoundError(f"No WAV files found in {audio_dir}")

    ref_fname = wavs[0]
    ref_path = os.path.join(audio_dir, ref_fname)
    ref_sig, ref_sr = load_audio(ref_path)

    offsets: Dict[str, float] = {}
    offsets[ref_fname] = 0.0

    for w in wavs[1:]:
        path = os.path.join(audio_dir, w)
        sig, sr = load_audio(path)
        if sr != ref_sr:
            duration = len(sig) / sr
            new_len = int(round(duration * ref_sr))
            sig = np.interp(np.linspace(0, len(sig), new_len, endpoint=False),
                            np.arange(len(sig)), sig).astype(np.float32)
            sr = ref_sr
        off_s, conf = compute_gcc_phat(ref_sig, sig, ref_sr)
        offsets[w] = float(off_s)
    return offsets