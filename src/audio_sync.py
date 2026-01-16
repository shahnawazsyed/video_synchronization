"""
audio_sync.py
--------------
FFT + GCC-PHAT implementation to estimate inter-stream time offsets.
Enhanced with pairwise alignment and global optimization for robustness.

Functions:
- compute_gcc_phat: core FFT + PHAT cross-correlation returning offset + confidence
- estimate_offsets_gccphat: (legacy) load audio files and compute offsets vs reference
- compute_pairwise_offsets: compute all pairwise offsets between files
- optimize_offsets: find globally consistent offsets using least-squares
- estimate_offsets_robust: main entry point for robust pairwise sync
"""
import os
import warnings
import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.signal import butter, sosfilt
from scipy.fft import fft, ifft
from scipy.optimize import least_squares

from .utils import next_pow2, load_audio, detect_outliers

def compute_gcc_phat(sig_a: np.ndarray, sig_b: np.ndarray, fs: int, 
                     max_offset_sec: float = 10.0, 
                     window_sec: Optional[float] = None) -> Tuple[float, float]:
    """
    Compute time offset between two signals using GCC-PHAT.
    
    Args:
        sig_a: Reference signal
        sig_b: Signal to align
        fs: Sample rate
        max_offset_sec: Maximum expected offset in seconds (default 10s)
        window_sec: If provided, use only first N seconds for speed (default: use all)
    
    Returns:
        (offset_seconds, confidence_score)
    offset_seconds is the amount to add to sig_b timestamps to align to sig_a.
    """
    sos = butter(4, [300, 5000], btype='bandpass', fs=fs, output='sos')
    sig_a = sosfilt(sos, sig_a)
    sig_b = sosfilt(sos, sig_b)
    # Windowing for speed
    if window_sec is not None:
        window_samples = int(window_sec * fs)
        sig_a = sig_a[:window_samples]
        sig_b = sig_b[:window_samples]
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
    # Constrain search to realistic lag range
    max_lag_samples = int(max_offset_sec * fs)
    center = n // 2
    search_start = max(0, center - max_lag_samples)
    search_end = min(len(cc), center + max_lag_samples)
    # Search only within constrained window
    search_region = cc[search_start:search_end]
    lag_idx_local = np.argmax(np.abs(search_region))
    lag_idx = search_start + lag_idx_local
    lags = np.arange(-n//2, n//2)
    lag = lags[lag_idx]
    offset_seconds = lag / float(fs)
    # Sub-sample precision via parabolic interpolation
    if 0 < lag_idx < len(cc) - 1:
        y1, y2, y3 = cc[lag_idx-1], cc[lag_idx], cc[lag_idx+1]
        denom_interp = 2*y2 - y1 - y3
        if abs(denom_interp) > 1e-8:
            delta = 0.5 * (y3 - y1) / denom_interp
            offset_seconds += delta / float(fs)
    # Confidence scoring
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
    
    # Verbose mode: show top correlation peaks for debugging
    verbose = False  # Set to True for diagnostics
    if verbose:
        # Find top 3 peaks
        mag_sorted_idx = np.argsort(mag)[::-1]
        print(f"\n  Top 3 correlation peaks:")
        for i in range(min(3, len(mag_sorted_idx))):
            idx = mag_sorted_idx[i]
            peak_lag = lags[idx]
            peak_offset = peak_lag / float(fs)
            peak_mag = mag[idx]
            marker = "***" if idx == lag_idx else ""
            print(f"    Peak {i+1}: offset={peak_offset:+.3f}s, mag={peak_mag:.3f} {marker}")
    
    # Quality warnings
    if confidence < 0.3:
        warnings.warn(f"Low confidence ({confidence:.2f}) - sync may be unreliable")
    if abs(offset_seconds) > max_offset_sec * 0.9:
        warnings.warn(f"Offset ({offset_seconds:.2f}s) near search boundary - may be truncated")
    return -offset_seconds, confidence

def compute_pairwise_offsets(audio_dir: str, 
                            max_offset_sec: float = 10.0,
                            window_sec: Optional[float] = 30.0,
                            min_confidence: float = 0.0) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Compute offsets between all pairs of WAV files in directory.
    Args:
        audio_dir: Directory containing WAV files
        max_offset_sec: Maximum expected offset between files (default 10s)
        window_sec: Use only first N seconds for speed (default 30s, None=use all)
        min_confidence: Skip pairs with confidence below this threshold (default 0.0)
    
    Returns:
        Dict mapping (fileA, fileB) -> (offset_seconds, confidence)
        where offset_seconds is how much to add to fileB to align with fileA
    """
    wavs = sorted([f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")])
    if not wavs:
        raise FileNotFoundError(f"No WAV files found in {audio_dir}")
    if len(wavs) < 2:
        raise ValueError(f"Need at least 2 files for pairwise sync, found {len(wavs)}")
    
    print(f"Loading {len(wavs)} audio files...")
    # Load all files once
    signals = {}
    sample_rates = {}
    for w in wavs:
        path = os.path.join(audio_dir, w)
        sig, sr = load_audio(path)
        signals[w] = sig
        sample_rates[w] = sr
    # Use most common sample rate as reference
    ref_sr = max(set(sample_rates.values()), key=list(sample_rates.values()).count)
    # Resample all to reference rate
    for w in wavs:
        if sample_rates[w] != ref_sr:
            sig = signals[w]
            sr = sample_rates[w]
            duration = len(sig) / sr
            new_len = int(round(duration * ref_sr))
            signals[w] = np.interp(
                np.linspace(0, len(sig), new_len, endpoint=False),
                np.arange(len(sig)), 
                sig
            ).astype(np.float32)
            sample_rates[w] = ref_sr
    print(f"Computing {len(wavs) * (len(wavs) - 1) // 2} pairwise offsets...")
    pairwise = {}
    skipped = 0
    for i, file_a in enumerate(wavs):
        for j, file_b in enumerate(wavs[i+1:], start=i+1):
            sig_a = signals[file_a]
            sig_b = signals[file_b]
            offset, conf = compute_gcc_phat(
                sig_a, sig_b, ref_sr, 
                max_offset_sec=max_offset_sec,
                window_sec=window_sec
            )
            if conf >= min_confidence:
                pairwise[(file_a, file_b)] = (offset, conf)
                print(f"  {file_a} <-> {file_b}: offset={offset:.3f}s, confidence={conf:.3f}")
            else:
                skipped += 1
                print(f"  {file_a} <-> {file_b}: SKIPPED (confidence={conf:.3f} < {min_confidence})")
    if skipped > 0:
        print(f"\nSkipped {skipped} pairs due to low confidence")
    return pairwise

def optimize_offsets(pairwise: Dict[Tuple[str, str], Tuple[float, float]], 
                     wavs: List[str]) -> Dict[str, float]:
    """
    Find globally consistent offsets using weighted least-squares optimization.
    
    For each pair (A, B) with measured offset d_AB and confidence w_AB:
    We want: offset_B - offset_A ≈ d_AB
    
    Minimizes: Σ w_AB * (offset_B - offset_A - d_AB)²
    
    Args:
        pairwise: Dict of (fileA, fileB) -> (offset, confidence)
        wavs: List of all wav filenames
    
    Returns:
        Dict mapping filename -> optimized offset (first file anchored at 0.0)
    """
    if not pairwise:
        raise ValueError("No pairwise offsets provided for optimization")
    n = len(wavs)
    file_to_idx = {w: i for i, w in enumerate(wavs)}
    def residuals(offsets):
        res = []
        for (file_a, file_b), (d_ab, conf) in pairwise.items():
            i = file_to_idx[file_a]
            j = file_to_idx[file_b]
            # offset_B - offset_A should equal d_AB
            error = offsets[j] - offsets[i] - d_ab
            # Weight by square root of confidence (since we're minimizing squared errors)
            res.append(np.sqrt(conf) * error)
        return np.array(res)
    x0 = np.zeros(n)
    result = least_squares(residuals, x0, loss='soft_l1', f_scale=0.1, verbose=0) 
    offsets_opt = result.x - result.x[0] # Anchor first file to 0 (arbitrary reference frame)
    final_residuals = residuals(result.x)
    rmse = np.sqrt(np.mean(final_residuals**2))
    print(f"\nOptimization complete:")
    print(f"  RMSE: {rmse:.4f}s")
    print(f"  Max residual: {np.max(np.abs(final_residuals)):.4f}s")
    return {w: float(offsets_opt[i]) for i, w in enumerate(wavs)}

def estimate_offsets_robust(audio_dir: str, 
                           max_offset_sec: float = 10.0,
                           window_sec: Optional[float] = 30.0,
                           min_confidence: float = 0.2,
                           outlier_threshold: float = 0.5) -> Dict[str, float]:
    """
    Robust offset estimation using pairwise alignment + global optimization.
    
    This is the main entry point for the improved synchronization approach.
    It's more robust than single-reference alignment when all files have
    some degradation (clipping, noise, etc).
    
    Args:
        audio_dir: Directory containing WAV files
        max_offset_sec: Maximum expected offset between files (default 10s)
        window_sec: Use only first N seconds for speed (default 30s, None=use all)
        min_confidence: Skip pairs with confidence below this (default 0.2)
        outlier_threshold: Flag inconsistent pairs above this error (default 0.5s)
    
    Returns:
        Dict mapping filename -> offset_seconds (first file anchored at 0.0)
        Add these offsets to each file's timestamps to align them.
    """
    wavs = sorted([f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")])
    if len(wavs) == 0:
        raise ValueError(f"No WAV files found in {audio_dir}")
        
    # Step 1: Compute all pairwise offsets
    pairwise = compute_pairwise_offsets(
        audio_dir, 
        max_offset_sec=max_offset_sec,
        window_sec=window_sec,
        min_confidence=min_confidence
    )
    if not pairwise:
        raise ValueError("No valid pairwise offsets found - all pairs below confidence threshold")
    # Step 2: Optimize for global consistency
    print("\nOptimizing for global consistency...")
    optimized = optimize_offsets(pairwise, wavs)
    # Step 3: Detect outliers
    outliers = detect_outliers(pairwise, optimized, threshold=outlier_threshold)
    # Step 4: Report results
    print("\n" + "="*60)
    print("FINAL SYNCHRONIZED OFFSETS:")
    print("="*60)
    for w in wavs:
        print(f"  {w}: {optimized[w]:+.3f}s")
    print("="*60)
    
    return optimized
