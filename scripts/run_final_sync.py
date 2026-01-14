"""
run_final_sync.py
-----------------
Runs the final Synchronization pipeline headlessly (no UI interaction required).
"""
import os
import shutil
from src.visual_sync import sync_videos_by_motion
from src.video_sync import apply_video_offsets

def run():
    video_dir = "data/raw/"
    output_dir = "data/synced/"
    
    files = [
        "3_video_b.mov",
        "3_video_h.mp4", 
        "3_video_r.mp4",
        "3_video_v.mp4"
    ]
    
    print("\n" + "="*60)
    print("STEP 1: VISUAL (MOTION) SYNCHRONIZATION")
    print("="*60)
    
    offsets = sync_videos_by_motion(
        video_dir, 
        files,
        max_offset_sec=20.0,
        output_dir="outputs/visual_sync"
    )
    
    print("\n" + "="*60)
    print("STEP 2: APPLYING VIDEO OFFSETS (Re-encoding with padding)")
    print("="*60)
    
    # Ensure clean slate
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    success = apply_video_offsets(video_dir, offsets, output_dir)
    
    if success:
        print("\n" + "="*60)
        print("SYNC COMPLETE!")
        print(f"Videos saved to: {output_dir}")
        print("="*60)
    else:
        print("\nSYNC FAILED during video generation.")

if __name__ == "__main__":
    run()
