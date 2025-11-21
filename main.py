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
#TODO: meeting notes


def main():
    """
    Execute the full synchronization pipeline.
    """
    video_dir = "data/augmented/"
    audio_dir = "data/audio/"
    output_dir = "data/synced/"

    # Display videos BEFORE synchronization
    print("\n" + "="*60)
    print("STEP 0: BEFORE SYNCHRONIZATION")
    print("="*60)
    print("Displaying desynchronized videos...")
    print("Press 'q' to continue to synchronization pipeline\n")
    
    # Select specific videos to display (diverse offsets for better visualization)
    selected_videos = ['F1C7LR_aug.mp4', 'F1C23LR_aug.mp4', 'F1C5LR_aug.mp4', 'F1C4LR_aug.mp4']
    show_video_grid(video_dir, title="Before Sync (Desynchronized)", video_selection=selected_videos)

    # Step 1: Extract and preprocess audio (only for selected videos)
    print(f"\nExtracting audio for {len(selected_videos)} selected videos...")
    extract_audio_from_videos(video_dir, audio_dir)
    
    # Filter to only selected audio files
    import os
    import shutil
    selected_audio_dir = "data/audio_selected/"
    os.makedirs(selected_audio_dir, exist_ok=True)
    
    # Copy only selected audio files
    selected_audio = [v.replace('.mp4', '.wav') for v in selected_videos]
    for audio_file in selected_audio:
        src = os.path.join(audio_dir, audio_file)
        dst = os.path.join(selected_audio_dir, audio_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    print(f"Processing only selected videos: {', '.join(selected_videos)}\n")

    # Step 2: Estimate offsets between audio tracks
    # raw = estimate_offsets_gccphat(audio_dir)
    # offsets = {k: v['offset_s'] for k, v in raw.items()}
    # Optimal parameters from grid search (100% accuracy < 50ms)
    import time
    start_time = time.time()
    
    offsets = estimate_offsets_robust(
        selected_audio_dir,  # Use filtered audio directory
        max_offset_sec=5.0,
        window_sec=90.0,
        min_confidence=0.15,      
        outlier_threshold=0.5
    )
    
    sync_runtime = time.time() - start_time

    # Step 3: Apply offsets to synchronize videos (only selected)
    # Filter video directory to only selected videos
    selected_video_dir = "data/augmented_selected/"
    os.makedirs(selected_video_dir, exist_ok=True)
    
    for video_file in selected_videos:
        src = os.path.join(video_dir, video_file)
        dst = os.path.join(selected_video_dir, video_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    apply_video_offsets(selected_video_dir, offsets, output_dir)

    # Step 4: Evaluate synchronization quality
    figures_dir = "outputs/figures/"
    ground_truth_path = "data/augmented/ground_truth.json"   # adjust if your JSON is elsewhere
    evaluate_with_ground_truth(offsets, ground_truth_path, figures_dir, runtime_sec=sync_runtime)

    # Display videos AFTER synchronization
    print("\n" + "="*60)
    print("STEP 5: AFTER SYNCHRONIZATION")
    print("="*60)
    print("Displaying synchronized videos...")
    print("Press 'q' to finish\n")
    show_video_grid(output_dir, title="After Sync (Synchronized)", video_selection=selected_videos)



if __name__ == "__main__":
    main()
