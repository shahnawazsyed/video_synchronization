"""
verify_sync.py
---------------
Evaluates synchronization accuracy and produces diagnostic visualizations.
Enhanced with ground truth comparison when available.

Functions:
- load_ground_truth: loads ground truth offsets from JSON
- evaluate_synchronization: measures alignment error and confidence
- evaluate_with_ground_truth: compares predicted vs ground truth offsets
- plot_waveform_alignment: displays overlayed audio signals for verification
- plot_offset_comparison: visualizes predicted vs ground truth offsets
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from scipy.signal import correlate
import json

from .utils import load_audio, apply_offset
from .audio_sync import compute_gcc_phat


def load_ground_truth(gt_path: str) -> Dict[str, float]:
    """
    Load ground truth offsets from JSON file.

    Supports multiple formats, including:
    - List of dicts:
        [
            {"video": "F1C3LR.mp4", "augmented_file": "F1C3LR_aug.mp4", "time_offset_sec": 1.28},
            {"video": "F1C12LR.mp4", "augmented_file": "F1C12LR_aug.mp4", "time_offset_sec": -0.46},
            ...
        ]
    - Dict of {filename: offset}
    - Dict with nested "offsets" key

    Args:
        gt_path: Path to ground_truth.json file

    Returns:
        Dict mapping filename (augmented_file with .mp4 replaced by .wav) -> true offset in seconds
    """
    import json, os

    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    # Case 1: list of dicts with augmented_file + time_offset_sec
    if isinstance(gt_data, list):
        offsets = {}
        for entry in gt_data:
            if not isinstance(entry, dict):
                continue
            # Prefer augmented_file if present
            fname = (
                entry.get("augmented_file")
                or entry.get("output_file")
                or entry.get("video")
            )
            offset = entry.get("time_offset_sec")
            if fname is None or offset is None:
                continue

            # Convert .mp4 to .wav for matching against predicted audio filenames
            if fname.endswith(".mp4"):
                fname = fname.replace(".mp4", ".wav")
            offsets[os.path.basename(fname)] = float(offset)

        if offsets:
            return offsets

    # Case 2: dict with direct filename → offset mapping
    if isinstance(gt_data, dict):
        if all(isinstance(v, (int, float)) for v in gt_data.values()):
            return {os.path.basename(k): float(v) for k, v in gt_data.items()}
        elif "offsets" in gt_data and isinstance(gt_data["offsets"], dict):
            return {os.path.basename(k): float(v) for k, v in gt_data["offsets"].items()}

    raise ValueError(f"Unexpected ground truth format in {gt_path}")



def evaluate_with_ground_truth(predicted_offsets: Dict[str, float],
                               ground_truth_path: str,
                               output_dir: str,
                               runtime_sec: Optional[float] = None) -> Dict:
    """
    Compare predicted offsets against ground truth and generate metrics.
    
    Args:
        predicted_offsets: Dict of {filename: predicted_offset}
        ground_truth_path: Path to ground_truth.json
        output_dir: Directory to save evaluation results
    
    Returns:
        Dict with detailed accuracy metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ground truth
    try:
        gt_offsets = load_ground_truth(ground_truth_path)
    except FileNotFoundError:
        print(f"Warning: Ground truth file not found at {ground_truth_path}")
        return None
    except Exception as e:
        print(f"Warning: Could not load ground truth: {e}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Ground Truth Comparison")
    print(f"{'='*60}\n")
    
    # Find common files
    common_files = set(predicted_offsets.keys()) & set(gt_offsets.keys())
    if not common_files:
        print("Warning: No common files between predicted and ground truth")
        return None
    
    # Adjust predicted offsets to align with ground truth reference frame
    # The sync algorithm anchors the first file (alphabetically) to 0.0, but that file
    # has its own ground truth offset. We need to shift all predictions by this amount.
    sorted_files = sorted(predicted_offsets.keys())
    if sorted_files:
        reference_file = sorted_files[0]  # First file alphabetically
        if reference_file in gt_offsets:
            reference_gt_offset = gt_offsets[reference_file]
            print(f"Adjusting predictions: Reference file '{reference_file}' has GT offset {reference_gt_offset:+.3f}s")
            print(f"Shifting all predictions by {reference_gt_offset:+.3f}s to align with GT reference frame\n")
            
            # Adjust all predictions
            adjusted_offsets = {
                fname: offset + reference_gt_offset 
                for fname, offset in predicted_offsets.items()
            }
            predicted_offsets = adjusted_offsets
    
    # Compute metrics
    errors = []
    results = {
        "ground_truth_file": ground_truth_path,
        "num_files": len(common_files),
        "files": {}
    }
    
    print(f"{'File':<30} {'Predicted':<12} {'Ground Truth':<12} {'Error':<10} {'Status'}")
    print("-" * 80)
    
    for fname in sorted(common_files):
        pred = predicted_offsets[fname]
        true = gt_offsets[fname]
        error = abs(pred - true)
        errors.append(error)
        
        # Determine status
        if error < 0.05:
            status = "✓ EXCELLENT"
        elif error < 0.1:
            status = "✓ GOOD"
        elif error < 0.5:
            status = "⚠ OK"
        else:
            status = "✗ POOR"
        
        results["files"][fname] = {
            "predicted_offset": float(pred),
            "ground_truth_offset": float(true),
            "absolute_error": float(error)
        }
        
        print(f"{fname:<30} {pred:>+10.3f}s  {true:>+10.3f}s  {error:>8.3f}s  {status}")
    
    # Overall statistics
    errors = np.array(errors)
    results["statistics"] = {
        "mean_absolute_error": float(np.mean(errors)),
        "median_absolute_error": float(np.median(errors)),
        "std_error": float(np.std(errors)),
        "max_error": float(np.max(errors)),
        "min_error": float(np.min(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "accuracy_50ms": float(np.mean(errors < 0.05) * 100),  # % within 50ms
        "accuracy_100ms": float(np.mean(errors < 0.1) * 100),   # % within 100ms
        "accuracy_500ms": float(np.mean(errors < 0.5) * 100)    # % within 500ms
    }
    
    # Add runtime if provided
    if runtime_sec is not None:
        results["statistics"]["runtime_sec"] = float(runtime_sec)
    
    print(f"\n{'='*60}")
    print(f"Overall Statistics:")
    print(f"{'='*60}")
    print(f"  Mean Absolute Error:   {results['statistics']['mean_absolute_error']:.4f}s")
    print(f"  Median Absolute Error: {results['statistics']['median_absolute_error']:.4f}s")
    print(f"  Std Dev:               {results['statistics']['std_error']:.4f}s")
    print(f"  RMSE:                  {results['statistics']['rmse']:.4f}s")
    print(f"  Max Error:             {results['statistics']['max_error']:.4f}s")
    
    if runtime_sec is not None:
        print(f"\n  Synchronization Runtime: {runtime_sec:.2f}s")
    
    print(f"\n  Accuracy (< 50ms):     {results['statistics']['accuracy_50ms']:.1f}%")
    print(f"  Accuracy (< 100ms):    {results['statistics']['accuracy_100ms']:.1f}%")
    print(f"  Accuracy (< 500ms):    {results['statistics']['accuracy_500ms']:.1f}%")
    print(f"{'='*60}\n")
    
    # Save results
    report_path = os.path.join(output_dir, "ground_truth_comparison.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Ground truth comparison saved to: {report_path}\n")
    
    # Generate visualization
    plot_path = os.path.join(output_dir, "offset_comparison.png")
    plot_offset_comparison(predicted_offsets, gt_offsets, common_files, plot_path)
    
    return results


def plot_offset_comparison(predicted: Dict[str, float], 
                          ground_truth: Dict[str, float],
                          common_files: set,
                          output_path: str):
    """
    Create visualization comparing predicted vs ground truth offsets.
    
    Args:
        predicted: Dict of predicted offsets
        ground_truth: Dict of ground truth offsets
        common_files: Set of filenames to compare
        output_path: Path to save plot
    """
    files = sorted(common_files)
    pred_vals = [predicted[f] for f in files]
    true_vals = [ground_truth[f] for f in files]
    errors = [abs(p - t) for p, t in zip(pred_vals, true_vals)]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Scatter plot (predicted vs true)
    ax = axes[0, 0]
    ax.scatter(true_vals, pred_vals, alpha=0.6, s=80)
    
    # Add diagonal reference line
    all_vals = true_vals + pred_vals
    min_val, max_val = min(all_vals), max(all_vals)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=2, label='Perfect')
    
    ax.set_xlabel('Ground Truth Offset (s)', fontsize=11)
    ax.set_ylabel('Predicted Offset (s)', fontsize=11)
    ax.set_title('Predicted vs Ground Truth Offsets', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Error distribution
    ax = axes[0, 1]
    ax.hist(errors, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.3f}s')
    ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.3f}s')
    ax.set_xlabel('Absolute Error (s)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Error by file
    ax = axes[1, 0]
    x_pos = np.arange(len(files))
    colors = ['green' if e < 0.1 else 'orange' if e < 0.5 else 'red' for e in errors]
    ax.bar(x_pos, errors, alpha=0.7, color=colors, edgecolor='black')
    ax.set_xlabel('File Index', fontsize=11)
    ax.set_ylabel('Absolute Error (s)', fontsize=11)
    ax.set_title('Error by File', fontsize=12, fontweight='bold')
    ax.axhline(0.05, color='green', linestyle='--', alpha=0.5, linewidth=1, label='50ms threshold')
    ax.axhline(0.1, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='100ms threshold')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, linewidth=1, label='500ms threshold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Offsets comparison (bar chart)
    ax = axes[1, 1]
    x_pos = np.arange(len(files))
    width = 0.35
    ax.bar(x_pos - width/2, true_vals, width, alpha=0.7, label='Ground Truth', color='blue', edgecolor='black')
    ax.bar(x_pos + width/2, pred_vals, width, alpha=0.7, label='Predicted', color='red', edgecolor='black')
    ax.set_xlabel('File Index', fontsize=11)
    ax.set_ylabel('Offset (s)', fontsize=11)
    ax.set_title('Offset Values Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {output_path}")


def evaluate_synchronization(audio_dir: str, offsets: Dict[str, float], output_dir: str, 
                            segment_duration: float = 10.0, max_offset_sec: float = 10.0,
                            ground_truth_path: Optional[str] = None):
    """
    Evaluate how well videos are synchronized using audio correlation.
    
    Produces:
    - Cross-correlation plots for each pair
    - Confidence scores and statistics
    - JSON report with detailed metrics
    - Ground truth comparison (if ground_truth_path provided)

    Args:
        audio_dir: Directory containing extracted audio.
        offsets: Dictionary of {filename.wav: offset_seconds}.
        output_dir: Directory for saving evaluation results.
        segment_duration: Duration (sec) of audio segment to analyze (default 10s).
        max_offset_sec: Maximum offset for correlation search (default 10s).
        ground_truth_path: Optional path to ground_truth.json for accuracy evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ground truth comparison first (if available)
    gt_results = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        gt_results = evaluate_with_ground_truth(offsets, ground_truth_path, output_dir)
    
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
        "ground_truth_comparison": gt_results,
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
        aligned_segment = apply_offset(segment, applied_offset, ref_sr)
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
    aligned_tgt = apply_offset(tgt_sig, offset, ref_sr)
    
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