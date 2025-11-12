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
from src.audio_sync import estimate_offsets_gccphat
from src.video_sync import apply_video_offsets
from src.verify_sync import evaluate_synchronization

def main():
    """
    Execute the full synchronization pipeline.
    """
    video_dir = "data/raw/"
    audio_dir = "data/audio/"
    output_dir = "data/synced/"

    # Step 1: Extract and preprocess audio
    extract_audio_from_videos(video_dir, audio_dir)

    # Step 2: Estimate offsets between audio tracks
    offsets = estimate_offsets_gccphat(audio_dir)

    # Step 3: Apply offsets to synchronize videos
    apply_video_offsets(video_dir, offsets, output_dir)

    # Step 4: Evaluate synchronization quality
    evaluate_synchronization(audio_dir, offsets, output_dir)


if __name__ == "__main__":
    main()
