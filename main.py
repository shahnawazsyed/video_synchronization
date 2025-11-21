"""
main.py
---------
Entry point for the multi-camera synchronization project.

This script:
1. Extracts and preprocesses audio from all camera feeds.
2. Estimates time offsets using FFT + GCC-PHAT.
3. Applies offsets to video streams for synchronization.
4. Verifies alignment and exports synchronized outputs.
"""

from src.preprocess import extract_audio_from_videos
from src.audio_sync import estimate_offsets_robust
from src.video_sync import apply_video_offsets
from src.verify_sync import evaluate_with_ground_truth
from src.display_videos import show_video_grid
import os, shutil

#TODO: different datasets

def main():
    """
    Execute the full synchronization pipeline.
    """
    video_dir = "data/augmented/"
    audio_dir = "data/audio/"
    output_dir = "data/synced/"
    # Display videos before synchronization
    print("\n" + "="*60)
    print("STEP 0: BEFORE SYNCHRONIZATION")
    print("="*60)
    print("Displaying desynchronized videos...")
    print("Press 'q' to continue to synchronization pipeline\n")
    show_video_grid(video_dir, title="Before Sync (Desynchronized)", selected_files=["F1C7LR_aug.mp4", "F1C23LR_aug.mp4", "F1C5LR_aug.mp4", "F1C4LR_aug.mp4"])

    # Step 1: Extract and preprocess audio
    extract_audio_from_videos(video_dir, audio_dir)

    # -------------------------------------------------
    # Only keep audio for the selected videos before running sync
    # -------------------------------------------------
    SELECTED_FILES = ["F1C7LR_aug.mp4", "F1C23LR_aug.mp4", "F1C5LR_aug.mp4", "F1C4LR_aug.mp4"]
    selected_audio_dir = os.path.join(audio_dir, "selected")
    os.makedirs(selected_audio_dir, exist_ok=True)
    for vid in SELECTED_FILES:
        wav = vid.replace('.mp4', '.wav')
        src = os.path.join(audio_dir, wav)
        dst = os.path.join(selected_audio_dir, wav)
        if os.path.exists(src):
            shutil.copy(src, dst)
    # -------------------------------------------------

    # Step 2: Estimate offsets between audio tracks
    import time
    start_time = time.time()
    # Run robust offset estimation with optimal parameters (from grid search)
    # Best config: max_offset=7.0s, window=300.0s, min_conf=0.15, outlier_th=0.5
    # Achieves 75% accuracy < 50ms, median error 0.048s
    offsets = estimate_offsets_robust(
        selected_audio_dir,
        max_offset_sec=7.0,       # Captures all offsets up to ±7s
        window_sec=300.0,          # Full video window for complete overlap
        min_confidence=0.15,
        outlier_threshold=0.5      # Increased from 0.5 to handle F1C5 issues
    )
    sync_runtime = time.time() - start_time
    # Convert audio offset keys (wav) to corresponding video filenames (mp4)
    video_offsets = {fname.replace('.wav', '.mp4'): off for fname, off in offsets.items()}
    # Keep only offsets for those selected videos
    video_offsets = {k: v for k, v in video_offsets.items() if k in SELECTED_FILES}
    apply_video_offsets(video_dir, video_offsets, output_dir)
    figures_dir = "outputs/figures/"
    ground_truth_path = "data/augmented/ground_truth.json"   # adjust if your JSON is elsewhere
    evaluate_with_ground_truth(offsets, ground_truth_path, figures_dir, runtime_sec=sync_runtime)
    # Display videos after synchronization
    print("\n" + "="*60)
    print("STEP 5: AFTER SYNCHRONIZATION")
    print("="*60)
    print("Displaying synchronized videos...")
    print("Press 'q' to finish\n")
    show_video_grid(output_dir, title="After Sync (Synchronized)", selected_files=["F1C7LR_aug.mp4", "F1C23LR_aug.mp4", "F1C5LR_aug.mp4", "F1C4LR_aug.mp4"])

if __name__ == "__main__":
    main()
