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
from src.video_sync import apply_video_offsets
from src.display_videos import show_video_grid
from src.visual_sync import sync_videos_by_motion
import src.config as config

def main():
    """
    Execute the full synchronization pipeline.
    """
    video_dir = config.VIDEO_DIR
    output_dir = config.OUTPUT_DIR
    
    # Display videos before synchronization
    print("\n" + "="*60)
    print("STEP 0: BEFORE SYNCHRONIZATION")
    print("="*60)
    print("Displaying desynchronized videos...")
    print("Press 'q' to continue to synchronization pipeline\n")
    show_video_grid(video_dir, title="Before Sync (Desynchronized)", selected_files=config.SELECTED_VIDEOS)

    # Estimate offsets
    if config.SYNC_METHOD == "visual":
        print("\n" + "="*60)
        print("STEP 1: VISUAL (MOTION) SYNCHRONIZATION")
        print("="*60)
        
        offsets = sync_videos_by_motion(
            video_dir, 
            config.SELECTED_VIDEOS,
            max_offset_sec=20.0,
            output_dir=config.VISUAL_SYNC_OUTPUT_DIR
        )
        video_offsets = offsets
        
    else:
        # Audio-based sync (fallback)
        from src.preprocess import extract_audio_from_videos
        from src.audio_sync import estimate_offsets_robust
        
        audio_dir = config.AUDIO_DIR
        print("\n" + "="*60)
        print("STEP 1: EXTRACTING AUDIO")
        print("="*60)
        extract_audio_from_videos(video_dir, audio_dir)

        selected_audio_dir = os.path.join(audio_dir, "selected")
        os.makedirs(selected_audio_dir, exist_ok=True)
        for vid in config.SELECTED_VIDEOS:
            base_name = os.path.splitext(vid)[0]
            wav = base_name + '.wav'
            src = os.path.join(audio_dir, wav)
            dst = os.path.join(selected_audio_dir, wav)
            if os.path.exists(src):
                shutil.copy(src, dst)

        print("\n" + "="*60)
        print("STEP 2: ESTIMATING AUDIO OFFSETS")
        print("="*60)
        
        offsets = estimate_offsets_robust(
            selected_audio_dir,
            max_offset_sec=15.0,
            window_sec=300.0,
            min_confidence=0.15,
            outlier_threshold=0.5
        )
        
        # Convert audio offset keys (wav) to corresponding video filenames
        video_offsets = {}
        for fname, off in offsets.items():
            base_name = fname.replace('.wav', '')
            # Find the matching video file (could be .mp4 or .mov)
            for vid in config.SELECTED_VIDEOS:
                if vid.startswith(base_name):
                    video_offsets[vid] = off
                    break

    # Apply offsets to videos
    print("\n" + "="*60)
    print("STEP 2: APPLYING VIDEO OFFSETS")
    print("="*60)
    
    apply_video_offsets(video_dir, video_offsets, output_dir)
    
    # Display videos after synchronization
    print("\n" + "="*60)
    print("STEP 3: AFTER SYNCHRONIZATION")
    print("="*60)
    print("Displaying synchronized videos for manual inspection...")
    print("Press 'q' to finish\n")
    
    # Get output filenames
    output_files = []
    for vid in config.SELECTED_VIDEOS:
        base_name = os.path.splitext(vid)[0]
        for ext in ['.mp4', '.mov']:
            if os.path.exists(os.path.join(output_dir, base_name + ext)):
                output_files.append(base_name + ext)
                break
    
    show_video_grid(output_dir, title="After Sync (Synchronized)", 
                   selected_files=output_files if output_files else None)

if __name__ == "__main__":
    main()
