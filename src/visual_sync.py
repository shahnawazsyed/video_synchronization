"""
Visual-based video synchronization using motion detection.

This module synchronizes videos by correlating motion patterns across
different camera views. Even though cameras may have different angles,
the TIMING of motion events (walking, sitting, gestures) should be the same.

Approach:
1. Extract motion energy over time from each video
2. Create a "motion timeseries" for each video  
3. Cross-correlate motion timeseries to find offsets
"""
import os
import itertools
import cv2
import numpy as np

# Force non-interactive backend for matplotlib (required for background threads)
import matplotlib
matplotlib.use('Agg')

from scipy.signal import correlate, resample
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional

def extract_motion_energy(video_path: str, 
                          downsample: int = 4,
                          blur_size: int = 5,
                          center_crop: bool = True,
                          step: int = 3) -> Tuple[np.ndarray, float]:
    """
    Extract motion energy timeseries from a video with optimization.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    motion_energy = []
    prev_gray = None
    frame_idx = 0
    
    while True:
        if step > 1 and frame_idx > 0:
            for _ in range(step - 1):
                if not cap.grab():
                    break
                frame_idx += 1
                
        ret, frame = cap.read()
        if not ret:
            break
        
        if center_crop:
            h, w = frame.shape[:2]
            start_y, end_y = int(h * 0.25), int(h * 0.75)
            start_x, end_x = int(w * 0.25), int(w * 0.75)
            frame = frame[start_y:end_y, start_x:end_x]
        
        h, w = frame.shape[:2]
        small_frame = cv2.resize(frame, (w // downsample, h // downsample))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        if blur_size > 0:
            gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            energy = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)
            motion_energy.append(energy)
        else:
            motion_energy.append(0.0)
        
        prev_gray = gray
        frame_idx += 1
        
        if frame_idx % 300 == 0:
            print(f"  {os.path.basename(video_path)}: {frame_idx}/{total_frames}", end="\r")
    
    cap.release()
    effective_fps = fps / step
    return np.array(motion_energy), effective_fps

def smooth_motion_signal(signal: np.ndarray, 
                         fps: float,
                         window_sec: float = 0.2) -> np.ndarray:
    """Smooth the motion signal to reduce noise."""
    window_frames = int(window_sec * fps)
    if window_frames < 1:
        window_frames = 1
    return uniform_filter1d(signal, window_frames)

def correlate_motion_signals(sig1: np.ndarray, sig2: np.ndarray,
                             fps: float,
                             max_offset_sec: float = 20.0) -> Tuple[float, float]:
    """Find time offset between two motion signals using cross-correlation."""
    sig1_norm = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-10)
    sig2_norm = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-10)
    
    cc = correlate(sig1_norm, sig2_norm, mode='full')
    max_lag_frames = int(max_offset_sec * fps)
    center = len(sig2_norm) - 1
    search_start = max(0, center - max_lag_frames)
    search_end = min(len(cc), center + max_lag_frames)
    
    search_region = cc[search_start:search_end]
    if len(search_region) == 0:
        return 0.0, 0.0

    lag_idx_local = np.argmax(search_region)
    lag_idx = search_start + lag_idx_local
    lag_frames = lag_idx - center
    offset_seconds = lag_frames / fps
    
    peak = cc[lag_idx]
    mean_cc = np.mean(np.abs(cc))
    std_cc = np.std(cc)
    confidence = (peak - mean_cc) / (std_cc + 1e-10)
    confidence = float(np.clip(confidence / 10.0, 0, 1))
    
    return offset_seconds, confidence

def visualize_motion_signals(motion_signals: Dict[str, np.ndarray],
                             fps: float,
                             output_path: str):
    """Create a visualization of motion signals."""
    import matplotlib.pyplot as plt
    n = len(motion_signals)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3*n), sharex=True)
    if n == 1: axes = [axes]
    
    for ax, (name, signal) in zip(axes, motion_signals.items()):
        time = np.arange(len(signal)) / fps
        ax.plot(time, signal, 'b-', linewidth=0.5)
        ax.set_ylabel('Motion')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def sync_videos_by_motion(video_dir: str,
                          selected_files: List[str],
                          max_offset_sec: float = 20.0,
                          output_dir: Optional[str] = None) -> Dict[str, float]:
    """Main function to synchronize videos by motion."""
    print("="*60)
    print("VISUAL (MOTION) SYNCHRONIZATION")
    print("="*60)
    
    print(f"Step 1: Extracting motion energy from {len(selected_files)} videos...")
    motion_signals = {}
    fps_dict = {}
    
    def process_one(fname):
        path = os.path.join(video_dir, fname)
        motion, eff_fps = extract_motion_energy(path, step=3)
        motion = smooth_motion_signal(motion, eff_fps)
        return fname, motion, eff_fps

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_one, selected_files))
    
    for fname, motion, eff_fps in results:
        motion_signals[fname] = motion
        fps_dict[fname] = eff_fps
        print(f"  {fname}: {len(motion)} samples @ {eff_fps:.2f} fps")
    
    target_fps = 10.0
    for fname in selected_files:
        orig_fps = fps_dict[fname]
        sig = motion_signals[fname]
        if orig_fps > 0:
            new_len = int(len(sig) * target_fps / orig_fps)
            motion_signals[fname] = resample(sig, new_len)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        visualize_motion_signals(motion_signals, target_fps, 
                                 os.path.join(output_dir, "motion_signals.png"))
    
    print("\nStep 2: Computing pairwise correlations...")
    pairwise_offsets = {}
    pairs = list(itertools.combinations(selected_files, 2))
    for f1, f2 in pairs:
        offset, conf = correlate_motion_signals(motion_signals[f1], motion_signals[f2], target_fps, max_offset_sec)
        pairwise_offsets[(f1, f2)] = (offset, conf)
        print(f"  {f1} <-> {f2}: offset={offset:+.2f}s, conf={conf:.3f}")
    
    print("\nStep 3: Global Optimization...")
    n = len(selected_files)
    file_to_idx = {f: i for i, f in enumerate(selected_files)}
    
    def residuals(offsets):
        res = []
        for (f1, f2), (offset, conf) in pairwise_offsets.items():
            i, j = file_to_idx[f1], file_to_idx[f2]
            res.append(np.sqrt(conf) * (offsets[j] - offsets[i] - offset))
        return np.array(res)
    
    x0 = np.zeros(n)
    result = least_squares(residuals, x0, loss='soft_l1', f_scale=0.5)
    offsets_opt = result.x - result.x[0]
    final_offsets = {selected_files[i]: float(offsets_opt[i]) for i in range(n)}
    
    rmse = np.sqrt(np.mean(residuals(result.x)**2))
    print(f"  RMSE: {rmse:.3f}s")
    
    print("\nFINAL OFFSETS:")
    for f in selected_files:
        print(f"  {f}: {final_offsets[f]:+.2f}s")
    
    return final_offsets
