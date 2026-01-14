"""
main.py
---------
Entry point for the multi-camera synchronization project.

This script:
1. Displays videos before synchronization
2. Estimates time offsets using visual (motion) or audio sync
3. Applies offsets to video streams for synchronization
4. Displays synchronized videos for manual inspection
"""

import os
import shutil
import src.config as config
from src.video_sync import apply_video_offsets
from src.display_videos import show_video_grid
from src.visual_sync import sync_videos_by_motion

def run_synchronization():
    """Execute the full video synchronization pipeline."""
    video_dir = config.VIDEO_DIR
    output_dir = config.OUTPUT_DIR
    selected_videos = config.SELECTED_VIDEOS
    
    # --- STEP 0: PRE-SYNC VISUALIZATION ---
    print("\n" + "="*60)
    print("STEP 0: INITIAL VISUALIZATION")
    print("="*60)
    print("Displaying original videos. Press 'q' to proceed to sync.\n")
    show_video_grid(video_dir, title="Pre-Sync Check", selected_files=selected_videos)

    # --- STEP 1: OFFSET ESTIMATION ---
    print("\n" + "="*60)
    print(f"STEP 1: {config.SYNC_METHOD.upper()} OFFSET ESTIMATION")
    print("="*60)
    
    if config.SYNC_METHOD == "visual":
        video_offsets = sync_videos_by_motion(
            video_dir, 
            selected_videos,
            max_offset_sec=20.0,
            output_dir=config.VISUAL_SYNC_OUTPUT_DIR
        )
    else:
        # Audio-based sync
        from src.preprocess import extract_audio_from_videos
        from src.audio_sync import estimate_offsets_robust
        
        audio_dir = config.AUDIO_DIR
        print("Extracting audio tracks...")
        extract_audio_from_videos(video_dir, audio_dir)

        # Prepare directory for selected audio files
        selected_audio_dir = os.path.join(audio_dir, "selected")
        os.makedirs(selected_audio_dir, exist_ok=True)
        for vid in selected_videos:
            base_name = os.path.splitext(vid)[0]
            wav_file = f"{base_name}.wav"
            src_path = os.path.join(audio_dir, wav_file)
            if os.path.exists(src_path):
                shutil.copy(src_path, os.path.join(selected_audio_dir, wav_file))

        print("Estimating offsets using GCC-PHAT...")
        raw_offsets = estimate_offsets_robust(
            selected_audio_dir,
            max_offset_sec=15.0,
            window_sec=300.0,
            min_confidence=0.15,
            outlier_threshold=0.5
        )
        
        # Map audio results back to video files
        video_offsets = {}
        for fname, offset in raw_offsets.items():
            base_name = fname.replace('.wav', '')
            for vid in selected_videos:
                if vid.startswith(base_name):
                    video_offsets[vid] = offset
                    break

    # --- STEP 2: OFFSET APPLICATION ---
    print("\n" + "="*60)
    print("STEP 2: APPLYING SYNCHRONIZATION")
    print("="*60)
    apply_video_offsets(video_dir, video_offsets, output_dir)
    
    # --- STEP 3: POST-SYNC VERIFICATION ---
    print("\n" + "="*60)
    print("STEP 3: SYNCHRONIZATION VERIFICATION")
    print("="*60)
    print("Displaying synced results. Press 'q' to exit.\n")
    
    # Verify existance of synced files
    output_files = []
    for vid in selected_videos:
        base_name = os.path.splitext(vid)[0]
        for ext in ['.mp4', '.mov']:
            if os.path.exists(os.path.join(output_dir, f"{base_name}{ext}")):
                output_files.append(f"{base_name}{ext}")
                break
    
    show_video_grid(output_dir, title="Post-Sync Verification", 
                   selected_files=output_files if output_files else None)

if __name__ == "__main__":
    run_synchronization()
