"""
verify_sync.py
---------------
Evaluates synchronization accuracy and produces diagnostic visualizations.

Functions:
- evaluate_synchronization: measures alignment error and confidence.
- plot_waveform_alignment: displays overlayed audio signals for verification.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from scipy.signal import correlate
import json

from .utils import load_audio
from .audio_sync import compute_gcc_phat


def evaluate_synchronization(audio_dir: str, offsets: Dict[str, float], output_dir: str, 
                            segment_duration: float = 10.0, max_offset_sec: float = 10.0):
    """
    Evaluate how well videos are synchronized using audio correlation.
    
    Produces:
    - Cross-correlation plots for each pair
    - Confidence scores and statistics
    - JSON report with detailed metrics

    Args:
        audio_dir: Directory containing extracted audio.
        offsets: Dictionary of {filename.wav: offset_seconds}.
        output_dir: Directory for saving evaluation results.
        segment_duration: Duration (sec) of audio segment to analyze (default 10s).
        max_offset_sec: Maximum offset for correlation search (default 10s).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    wavs = sorted([f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")])
    if not wavs:
        raise FileNotFoundError(f"No WAV files found in {audio_dir}")
    
    # Load reference audio (first file)
    ref_fname = wavs[0]
    ref_path = os.path.join(audio_dir, ref_fname)
    ref_sig, ref_sr = load_audio(ref_path)
    
    # Extract segment for analysis
    segment_samples = int(segment_duration * ref_sr)
    ref_segment = ref_sig[:min(segment_samples, len(ref_sig))]
    
    results = {
        "reference": ref_fname,
        "sample_rate": ref_sr,
        "segment_duration": segment_duration,
        "files": {}
    }
    
    print(f"\n{'='*60}")
    print(f"Synchronization Evaluation Report")
    print(f"{'='*60}")
    print(f"Reference: {ref_fname}")
    print(f"Sample Rate: {ref_sr} Hz")
    print(f"Analysis Segment: {segment_duration}s")
    print(f"{'='*60}\n")
    
    for wav in wavs:
        path = os.path.join(audio_dir, wav)
        sig, sr = load_audio(path)
        
        # Resample if needed
        if sr != ref_sr:
            duration = len(sig) / sr
            new_len = int(round(duration * ref_sr))
            sig = np.interp(np.linspace(0, len(sig), new_len, endpoint=False),
                          np.arange(len(sig)), sig).astype(np.float32)
            sr = ref_sr
        
        # Extract segment
        segment = sig[:min(segment_samples, len(sig))]
        
        # Compute correlation metrics
        offset_s, confidence = compute_gcc_phat(ref_segment, segment, ref_sr, max_offset_sec)
        applied_offset = offsets.get(wav, 0.0)
        
        # Compute residual error after applied offset
        residual_error = abs(offset_s - applied_offset)
        
        # Compute SNR-like metric
        aligned_segment = _apply_offset(segment, applied_offset, ref_sr)
        min_len = min(len(ref_segment), len(aligned_segment))
        correlation = np.corrcoef(ref_segment[:min_len], aligned_segment[:min_len])[0, 1]
        
        results["files"][wav] = {
            "measured_offset": float(offset_s),
            "applied_offset": float(applied_offset),
            "residual_error": float(residual_error),
            "confidence": float(confidence),
            "correlation": float(correlation)
        }
        
        # Print summary
        status = "✓ GOOD" if residual_error < 0.1 and confidence > 0.5 else "⚠ CHECK"
        print(f"{status} {wav}")
        print(f"  Measured Offset: {offset_s:+.3f}s")
        print(f"  Applied Offset:  {applied_offset:+.3f}s")
        print(f"  Residual Error:  {residual_error:.3f}s")
        print(f"  Confidence:      {confidence:.3f}")
        print(f"  Correlation:     {correlation:.3f}")
        print()
        
        # Generate waveform plot
        if wav != ref_fname:
            plot_path = os.path.join(output_dir, f"alignment_{wav.replace('.wav', '')}.png")
            plot_waveform_alignment(ref_path, path, applied_offset, output_path=plot_path)
    
    # Save JSON report
    report_path = os.path.join(output_dir, "sync_evaluation.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"{'='*60}")
    print(f"Report saved to: {report_path}")
    print(f"Plots saved to: {output_dir}/")
    print(f"{'='*60}\n")
    
    return results


def plot_waveform_alignment(ref_audio: str, target_audio: str, offset: float, 
                           output_path: str = None, duration: float = 5.0):
    """
    Plot overlayed waveforms of two synchronized audio tracks.

    Args:
        ref_audio: Path to reference audio file.
        target_audio: Path to target audio file.
        offset: Estimated offset (seconds) applied for alignment.
        output_path: Path to save plot (if None, displays interactively).
        duration: Duration (sec) of waveform to plot (default 5s).
    """
    # Load audio
    ref_sig, ref_sr = load_audio(ref_audio)
    tgt_sig, tgt_sr = load_audio(target_audio)
    
    # Resample target if needed
    if tgt_sr != ref_sr:
        duration_tgt = len(tgt_sig) / tgt_sr
        new_len = int(round(duration_tgt * ref_sr))
        tgt_sig = np.interp(np.linspace(0, len(tgt_sig), new_len, endpoint=False),
                           np.arange(len(tgt_sig)), tgt_sig).astype(np.float32)
        tgt_sr = ref_sr
    
    # Apply offset to target
    aligned_tgt = _apply_offset(tgt_sig, offset, ref_sr)
    
    # Extract segments for plotting
    plot_samples = int(duration * ref_sr)
    ref_plot = ref_sig[:min(plot_samples, len(ref_sig))]
    tgt_plot = aligned_tgt[:min(plot_samples, len(aligned_tgt))]
    
    # Normalize for visualization
    ref_plot = ref_plot / (np.max(np.abs(ref_plot)) + 1e-8)
    tgt_plot = tgt_plot / (np.max(np.abs(tgt_plot)) + 1e-8)
    
    # Create time axis
    time_ref = np.arange(len(ref_plot)) / ref_sr
    time_tgt = np.arange(len(tgt_plot)) / ref_sr
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Reference waveform
    axes[0].plot(time_ref, ref_plot, color='blue', alpha=0.7, linewidth=0.5)
    axes[0].set_title(f'Reference Audio: {os.path.basename(ref_audio)}', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, duration)
    
    # Plot 2: Target waveform (aligned)
    axes[1].plot(time_tgt, tgt_plot, color='red', alpha=0.7, linewidth=0.5)
    axes[1].set_title(f'Target Audio (aligned, offset={offset:+.3f}s): {os.path.basename(target_audio)}', 
                     fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, duration)
    
    # Plot 3: Overlay
    min_len = min(len(ref_plot), len(tgt_plot))
    axes[2].plot(time_ref[:min_len], ref_plot[:min_len], color='blue', alpha=0.5, 
                linewidth=0.5, label='Reference')
    axes[2].plot(time_tgt[:min_len], tgt_plot[:min_len], color='red', alpha=0.5, 
                linewidth=0.5, label='Target (aligned)')
    axes[2].set_title('Overlay (Aligned)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, duration)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _apply_offset(signal: np.ndarray, offset_sec: float, sample_rate: int) -> np.ndarray:
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
        return np.pad(signal, (0, -offset_samples), mode='constant')[-offset_samples:]