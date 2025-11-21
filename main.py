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

#TODO: UI outputting videos
#TODO: different datasets
#TODO: output runtime as metric
#TODO: add ayca to github
#TODO: meeting notes


def main():
    """
    Execute the full synchronization pipeline.
    """
    video_dir = "data/augmented/"
    audio_dir = "data/audio/"
    output_dir = "data/synced/"

    # Step 1: Extract and preprocess audio
    extract_audio_from_videos(video_dir, audio_dir)

    # Step 2: Estimate offsets between audio tracks
    # raw = estimate_offsets_gccphat(audio_dir)
    # offsets = {k: v['offset_s'] for k, v in raw.items()}
    # Optimal parameters from grid search (100% accuracy < 50ms)
    offsets = estimate_offsets_robust(
        audio_dir,
        max_offset_sec=5.0,       # Increased to capture all offsets in dataset
        window_sec=90.0,          # Larger window provides more overlap for correlation
        min_confidence=0.15,      
        outlier_threshold=0.5
    )

    # Step 3: Apply offsets to synchronize videos
    apply_video_offsets(video_dir, offsets, output_dir)

    # Step 4: Evaluate synchronization quality
    output_dir = "outputs/figures/"
    ground_truth_path = "data/augmented/ground_truth.json"   # adjust if your JSON is elsewhere
    evaluate_with_ground_truth(offsets, ground_truth_path, output_dir)



if __name__ == "__main__":
    main()
