"""
Alternative synchronization strategy using weighted voting from multiple correlation windows
This helps resolve ambiguous cases where GCC-PHAT finds false peaks
"""
import numpy as np
from scipy.io import wavfile
from src.audio_sync import compute_gcc_phat
import os
from typing import Dict, List, Tuple
import itertools

def multi_window_sync(audio_dir: str, 
                      selected_files: List[str],
                      max_offset_sec: float = 7.0,
                      window_sizes: List[float] = [240.0, 270.0, 300.0, 320.0],
                      min_confidence: float = 0.15) -> Dict[str, float]:
    """
    Synchronize using multiple window sizes and vote on best offset.
    This helps resolve false correlation peaks by checking consistency across windows.
    
    Args:
        audio_dir: Directory with WAV files
        selected_files: List of audio files to sync
        max_offset_sec: Maximum offset to search
        window_sizes: Different window sizes to try (seconds)
        min_confidence: Minimum confidence threshold
    
    Returns:
        Dictionary mapping filename to offset in seconds
    """
    # Get all pairwise combinations
    pairs = list(itertools.combinations(selected_files, 2))
    
    # For each pair, compute offset with multiple window sizes
    pair_votes = {}
    
    print(f"Multi-window synchronization using {len(window_sizes)} window sizes...")
    print(f"Window sizes: {window_sizes}")
    print()
    
    for file1, file2 in pairs:
        path1 = os.path.join(audio_dir, file1)
        path2 = os.path.join(audio_dir, file2)
        
        # Load audio files
        fs1, sig1 = wavfile.read(path1)
        fs2, sig2 = wavfile.read(path2)
        
        if fs1 != fs2:
            print(f"  ⚠️  Sample rate mismatch: {file1}={fs1}, {file2}={fs2}")
            continue
        
        # Convert to mono if stereo
        if len(sig1.shape) == 2:
            sig1 = sig1.mean(axis=1)
        if len(sig2.shape) == 2:
            sig2 = sig2.mean(axis=1)
        
        offsets = []
        confidences = []
        
        for window_sec in window_sizes:
            offset, confidence = compute_gcc_phat(
                sig1, sig2, fs1,
                max_offset_sec=max_offset_sec,
                window_sec=window_sec
            )
            
            if confidence >= min_confidence:
                offsets.append(offset)
                confidences.append(confidence)
        
        if len(offsets) > 0:
            # Weighted average by confidence
            weights = np.array(confidences)
            weights = weights / np.sum(weights)
            final_offset = np.average(offsets, weights=weights)
            final_conf = np.mean(confidences)
            
            # Check consistency
            std_dev = np.std(offsets)
            
            print(f"  {file1} <-> {file2}:")
            print(f"    Offsets across windows: {[f'{o:.3f}s' for o in offsets]}")
            print(f"    Std dev: {std_dev:.3f}s")
            print(f"    Final offset: {final_offset:.3f}s (confidence: {final_conf:.3f})")
            
            if std_dev > 0.5:
                print(f"    ⚠️  High variance - ambiguous correlation!")
            
            pair_votes[(file1, file2)] = (final_offset, final_conf, std_dev)
        else:
            print(f"  {file1} <-> {file2}: No valid correlation")
            pair_votes[(file1, file2)] = (0.0, 0.0, 999.0)
    
    print()
    
    # Use least squares to find globally consistent offsets
    from scipy.optimize import lsq_linear
    
    # Build constraint matrix
    n_files = len(selected_files)
    file_to_idx = {f: i for i, f in enumerate(selected_files)}
    
    A = []
    b = []
    weights_list = []
    
    for (file1, file2), (offset, conf, std) in pair_votes.items():
        if std < 999.0:  # Valid measurement
            # offset(file2) - offset(file1) = measured_offset
            row = np.zeros(n_files)
            row[file_to_idx[file2]] = 1.0
            row[file_to_idx[file1]] = -1.0
            A.append(row)
            b.append(offset)
            # Weight by confidence and consistency (low std = high weight)
            weight = conf / (1.0 + std)
            weights_list.append(weight)
    
    if len(A) == 0:
        print("❌ No valid pairwise measurements!")
        return {f: 0.0 for f in selected_files}
    
    A = np.array(A)
    b = np.array(b)
    weights_array = np.array(weights_list)
    
    # Weight the system
    W = np.diag(weights_array)
    A_weighted = W @ A
    b_weighted = W @ b
    
    # Solve weighted least squares
    # Pin first file to 0.0
    result = lsq_linear(A_weighted, b_weighted, bounds=(-max_offset_sec, max_offset_sec))
    offsets_array = result.x
    
    # Set reference file to 0
    offsets_array -= offsets_array[0]
    
    # Create result dictionary
    result_dict = {selected_files[i]: float(offsets_array[i]) for i in range(n_files)}
    
    print("Multi-window synchronized offsets:")
    for f, offset in result_dict.items():
        print(f"  {f}: {offset:+.3f}s")
    
    return result_dict


if __name__ == "__main__":
    # Test with the new video files
    audio_dir = "data/audio"
    selected_files = ["3_video_b.wav", "3_video_h.wav", "3_video_r.wav", "3_video_v.wav"]
    
    result = multi_window_sync(audio_dir, selected_files)
    
    print("\n" + "="*80)
    print("Synchronization Results (Manual inspection required)")
    print("="*80)
    
    print(f"\n{'File':<25} {'Offset':<12}")
    print("-" * 40)
    
    for f in selected_files:
        offset = result.get(f, 0.0)
        print(f"{f:<25} {offset:+.3f}s")
