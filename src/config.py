"""
Configuration settings for the video synchronization project.
"""
import os

# Video files to synchronize
SELECTED_VIDEOS = [
    "3_video_b.mov",
    "3_video_h.mp4",
    "3_video_r.mp4",
    "3_video_v.mp4"
]

# Sync method: "visual" (motion-based) or "audio" (GCC-PHAT)
SYNC_METHOD = "visual"

# Directories
VIDEO_DIR = "data/raw/"
OUTPUT_DIR = "data/synced/"
AUDIO_DIR = "data/audio/"
VISUAL_SYNC_OUTPUT_DIR = "outputs/visual_sync"

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(VISUAL_SYNC_OUTPUT_DIR, exist_ok=True)
