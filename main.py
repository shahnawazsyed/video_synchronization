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
from src.audio_sync import estimate_offsets_gccphat, estimate_offsets_robust
from src.video_sync import apply_video_offsets
from src.verify_sync import evaluate_synchronization, evaluate_with_ground_truth
from src.display_videos import show_video_grid


#TODO: different datasets
#TODO: add ayca to github


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

    # Step 2: Estimate offsets between audio tracks
    import time
    start_time = time.time()
    offsets = estimate_offsets_robust(
        audio_dir,
        max_offset_sec=5.0,       # Increased to capture all offsets in dataset
        window_sec=90.0,          # Larger window provides more overlap for correlation
        min_confidence=0.15,      
        outlier_threshold=0.5
    )
    sync_runtime = time.time() - start_time
    apply_video_offsets(video_dir, offsets, output_dir)
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
